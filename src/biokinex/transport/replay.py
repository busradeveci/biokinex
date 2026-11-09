"""Recording and deterministic replay of telemetry streams.

A recording is a text file with one JSON telemetry packet per line (JSON Lines). Replaying the same
file always produces the same frames, because filters use sensor timestamps, never the wall clock.
Pacing only changes when packets are delivered, not what the pipeline computes.

Only single-line payloads can be recorded, so binary packets are not supported by this format.
"""

from __future__ import annotations

import asyncio
import json
import math
from os import PathLike
from types import TracebackType
from typing import IO, Final, Self

from biokinex.core.clock import Clock, MonotonicClock, Sleep
from biokinex.core.errors import TransportError
from biokinex.transport.base import TelemetrySource

__all__ = ["DEFAULT_MAX_LINE_BYTES", "RecordingSource", "ReplayRecorder", "ReplaySource"]

DEFAULT_MAX_LINE_BYTES: Final = 1 << 20
"""Longest accepted line; far above any valid packet, but it stops one bad line using all memory."""


class ReplaySource:
    """Reads payloads from a recording, one line at a time.

    Args:
        path: The recording to read.
        speed: ``None`` replays as fast as possible. Otherwise packets are delivered at their
            recorded sensor times divided by ``speed``, so ``1.0`` is real time.
        clock: Time source used for pacing.
        sleep: Coroutine used to wait; tests pass a fake to avoid real sleeping.
        max_line_bytes: Longest accepted line.
    """

    __slots__ = ("_clock", "_file", "_max_line_bytes", "_origin", "_sleep", "_speed")

    def __init__(
        self,
        path: str | PathLike[str],
        *,
        speed: float | None = None,
        clock: Clock | None = None,
        sleep: Sleep = asyncio.sleep,
        max_line_bytes: int = DEFAULT_MAX_LINE_BYTES,
    ) -> None:
        if speed is not None and not (math.isfinite(speed) and speed > 0.0):
            msg = f"speed must be finite and positive, got {speed!r}"
            raise ValueError(msg)
        if max_line_bytes < 1:
            msg = f"max_line_bytes must be >= 1, got {max_line_bytes}"
            raise ValueError(msg)
        self._speed = speed
        self._clock = clock if clock is not None else MonotonicClock()
        self._sleep = sleep
        self._max_line_bytes = max_line_bytes
        self._origin: tuple[float, float] | None = None
        # The file stays open across many calls, so a with-block cannot own it.
        self._file: IO[bytes] | None = open(path, "rb")  # noqa: SIM115

    async def receive(self) -> bytes | None:
        """Return the next non-blank line, or ``None`` at the end of the recording.

        Raises:
            TransportError: If a line is longer than ``max_line_bytes``.
        """
        payload = self._next_line()
        if payload is None:
            await self.close()
            return None
        if self._speed is not None:
            await self._pace(payload, self._speed)
        return payload

    async def close(self) -> None:
        """Close the file; later calls to :meth:`receive` return ``None``."""
        if self._file is not None:
            self._file.close()
            self._file = None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()

    def _next_line(self) -> bytes | None:
        """Read lines until one has content; local file reads are short, so they run inline."""
        while self._file is not None:
            line = self._file.readline(self._max_line_bytes + 1)
            if not line:
                return None
            if len(line) > self._max_line_bytes and not line.endswith(b"\n"):
                msg = f"recording line is longer than {self._max_line_bytes} bytes"
                raise TransportError(msg)
            payload = line.rstrip(b"\r\n")
            if payload.strip():
                return payload
        return None

    async def _pace(self, payload: bytes, speed: float) -> None:
        """Wait until this packet is due, based on its first sensor timestamp."""
        packet_time = _packet_time(payload)
        if packet_time is None:
            # A line that cannot be read is still delivered, so the validator can count it.
            return
        now = self._clock.now()
        if self._origin is None:
            self._origin = (packet_time, now)
            return
        first_packet_time, first_clock_time = self._origin
        due = first_clock_time + (packet_time - first_packet_time) / speed
        if due > now:
            await self._sleep(due - now)


class ReplayRecorder:
    """Writes payloads to a new recording file.

    The file is opened in exclusive mode, so an earlier recording is never overwritten by mistake.

    Raises:
        FileExistsError: If ``path`` already exists.
    """

    __slots__ = ("_file", "_written")

    def __init__(self, path: str | PathLike[str]) -> None:
        # The file stays open across many calls, so a with-block cannot own it.
        self._file: IO[bytes] | None = open(path, "xb")  # noqa: SIM115
        self._written = 0

    @property
    def written(self) -> int:
        """Number of payloads written."""
        return self._written

    def write(self, payload: bytes) -> None:
        """Append one payload as a line.

        Raises:
            ValueError: If the payload is blank or contains a line break.
            TransportError: If the recorder is closed.
        """
        if self._file is None:
            msg = "recorder is closed"
            raise TransportError(msg)
        # A line break would split the payload, and a blank line would be skipped on replay.
        if b"\n" in payload or b"\r" in payload or not payload.strip():
            msg = "only non-blank, single-line payloads can be recorded"
            raise ValueError(msg)
        self._file.write(payload + b"\n")
        self._written += 1

    def close(self) -> None:
        """Flush and close the file; calling it twice is safe."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class RecordingSource:
    """Passes payloads through from another source and records each one.

    A payload that cannot be recorded is still delivered, because recording must never change
    what the pipeline receives.
    """

    __slots__ = ("_recorder", "_skipped", "_source")

    def __init__(self, source: TelemetrySource, recorder: ReplayRecorder) -> None:
        self._source = source
        self._recorder = recorder
        self._skipped = 0

    @property
    def skipped(self) -> int:
        """Number of payloads delivered but not recorded."""
        return self._skipped

    async def receive(self) -> bytes | None:
        """Receive from the wrapped source and record the payload."""
        payload = await self._source.receive()
        if payload is None:
            return None
        try:
            self._recorder.write(payload)
        except ValueError:
            self._skipped += 1
        return payload

    async def close(self) -> None:
        """Close the wrapped source and the recorder."""
        await self._source.close()
        self._recorder.close()


def _packet_time(payload: bytes) -> float | None:
    """Return the first sample timestamp of a JSON packet, or ``None`` if it cannot be read."""
    try:
        packet = json.loads(payload)
    except ValueError:
        return None
    if not isinstance(packet, dict):
        return None
    samples = packet.get("samples")
    if not (isinstance(samples, list) and samples and isinstance(samples[0], dict)):
        return None
    timestamp = samples[0].get("timestamp")
    # bool is a subclass of int, so it must be excluded explicitly.
    if isinstance(timestamp, bool) or not isinstance(timestamp, int | float):
        return None
    try:
        value = float(timestamp)
    except OverflowError:
        # JSON integers have no size limit, so a huge one cannot become a float.
        return None
    return value if math.isfinite(value) else None
