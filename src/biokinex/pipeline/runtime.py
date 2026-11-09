"""The async shell around the synchronous processor.

The runtime connects a telemetry source, a codec, one :class:`PipelineProcessor` per device, and a
frame sink. It runs two tasks on one event loop:

- the receive loop, which decodes each payload, processes it, and sends the frames;
- the watchdog loop, which sends ``STALE`` frames for devices that went silent.

All DSP stays in the processor, so the runtime only moves data and counts what happened. No
error from one bad packet or one device can stop the stream for the others.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from biokinex.core.clock import Clock, MonotonicClock, Sleep
from biokinex.core.errors import BioKineXError, TelemetryError, TransportError
from biokinex.observability.metrics import LatencyHistogram, LatencySummary
from biokinex.pipeline.processor import PipelineProcessor
from biokinex.pipeline.watchdog import Watchdog
from biokinex.schemas.config import PipelineConfig
from biokinex.schemas.pose import PoseFrame
from biokinex.transport.base import Codec, FrameSink, TelemetrySource
from biokinex.transport.codecs import JsonCodec

__all__ = ["DEFAULT_MAX_DEVICES", "WATCHDOG_CHECKS_PER_TIMEOUT", "PipelineRuntime", "RuntimeStats"]

DEFAULT_MAX_DEVICES: Final = 16
"""Upper bound on devices per runtime, so packets with made-up device IDs cannot exhaust memory."""

WATCHDOG_CHECKS_PER_TIMEOUT: Final = 4
"""Watchdog checks per timeout; a silent device is reported within 1.25 timeouts."""


@dataclass(frozen=True, slots=True)
class RuntimeStats:
    """Totals since the runtime was created.

    Attributes:
        packets: Payloads received from the source.
        decode_errors: Payloads the codec rejected.
        rejected_devices: Packets dropped because ``max_devices`` was reached.
        processing_errors: Packets that raised inside the processor; the device was reset.
        frames_sent: Frames handed to the sink, including stale frames.
        stale_frames: Stale frames sent by the watchdog.
        send_errors: Frames the sink refused.
        devices: Device IDs in the order they were first seen.
        latency: Time from receiving a payload to encoding its last frame.
    """

    packets: int
    decode_errors: int
    rejected_devices: int
    processing_errors: int
    frames_sent: int
    stale_frames: int
    send_errors: int
    devices: tuple[str, ...]
    latency: LatencySummary


class PipelineRuntime:
    """Streams telemetry from a source to a sink until the source ends.

    The caller owns the source and the sink: the runtime never closes the sink, and it closes the
    source only when :meth:`stop` is called.

    Args:
        config: Configuration shared by every device's processor.
        source: Where telemetry payloads come from.
        sink: Where encoded frames go.
        codec: Wire format for both directions; JSON by default.
        clock: Time source for the watchdog and latency measurement.
        sleep: Async sleep used by the watchdog loop; tests may pass a fake.
        max_devices: Largest number of devices served at the same time.
    """

    __slots__ = (
        "_clock",
        "_codec",
        "_config",
        "_counts",
        "_latency",
        "_max_devices",
        "_processors",
        "_running",
        "_sink",
        "_sleep",
        "_source",
        "_watchdogs",
    )

    def __init__(
        self,
        config: PipelineConfig,
        source: TelemetrySource,
        sink: FrameSink,
        *,
        codec: Codec | None = None,
        clock: Clock | None = None,
        sleep: Sleep = asyncio.sleep,
        max_devices: int = DEFAULT_MAX_DEVICES,
    ) -> None:
        if max_devices < 1:
            msg = f"max_devices must be >= 1, got {max_devices}"
            raise ValueError(msg)
        self._config = config
        self._source = source
        self._sink = sink
        self._codec: Codec = codec if codec is not None else JsonCodec()
        self._clock: Clock = clock if clock is not None else MonotonicClock()
        self._sleep = sleep
        self._max_devices = max_devices
        self._processors: dict[str, PipelineProcessor] = {}
        self._watchdogs: dict[str, Watchdog] = {}
        self._latency = LatencyHistogram()
        self._running = False
        self._counts = dict.fromkeys(
            (
                "packets",
                "decode_errors",
                "rejected_devices",
                "processing_errors",
                "frames_sent",
                "stale_frames",
                "send_errors",
            ),
            0,
        )

    @property
    def processors(self) -> Mapping[str, PipelineProcessor]:
        """The processor of each device, as a read-only view."""
        return MappingProxyType(self._processors)

    def stats(self) -> RuntimeStats:
        """Return totals since the runtime was created."""
        return RuntimeStats(
            **self._counts,
            devices=tuple(self._processors),
            latency=self._latency.summary(),
        )

    async def run(self) -> RuntimeStats:
        """Process the stream until the source ends, then return the totals.

        Raises:
            RuntimeError: If the runtime is already running.
        """
        if self._running:
            msg = "runtime is already running"
            raise RuntimeError(msg)
        self._running = True
        watcher = asyncio.create_task(self._watch(), name="biokinex-watchdog")
        try:
            while (payload := await self._source.receive()) is not None:
                await self.handle(payload)
        finally:
            watcher.cancel()
            # asyncio.wait never re-raises the task's cancellation, but still lets our own
            # cancellation through, so the caller can always stop us.
            await asyncio.wait([watcher])
            self._running = False
        return self.stats()

    async def stop(self) -> None:
        """Close the source, so :meth:`run` finishes after the queued payloads."""
        await self._source.close()

    async def handle(self, payload: bytes) -> tuple[PoseFrame, ...]:
        """Process one payload and send its frames; return the frames that were built."""
        counts = self._counts
        counts["packets"] += 1
        started = self._clock.now()
        try:
            batch = self._codec.decode_packet(payload)
        except TelemetryError:
            counts["decode_errors"] += 1
            return ()
        processor = self._processor_for(batch.device_id)
        if processor is None:
            counts["rejected_devices"] += 1
            return ()
        try:
            frames = processor.process(batch).frames
        except (BioKineXError, ValueError):
            # A failure here is a bug or a broken frame, never valid data, so the device starts
            # fresh and the other devices keep running.
            counts["processing_errors"] += 1
            processor.reset()
            return ()
        encoded = [self._codec.encode_frame(frame) for frame in frames]
        self._latency.record(self._clock.now() - started)
        for payload_out in encoded:
            await self._send(payload_out)
        if frames:
            self._watchdogs[batch.device_id].feed(frames[-1])
        return frames

    async def check_watchdogs(self) -> tuple[PoseFrame, ...]:
        """Send a stale frame for every device whose warning is due; return those frames."""
        stale = tuple(
            frame
            for watchdog in self._watchdogs.values()
            if (frame := watchdog.check()) is not None
        )
        for frame in stale:
            await self._send(self._codec.encode_frame(frame))
            self._counts["stale_frames"] += 1
        return stale

    def _processor_for(self, device_id: str) -> PipelineProcessor | None:
        """Return the device's processor, creating it if there is room for a new device."""
        processor = self._processors.get(device_id)
        if processor is None and len(self._processors) < self._max_devices:
            processor = PipelineProcessor(self._config)
            self._processors[device_id] = processor
            self._watchdogs[device_id] = Watchdog(self._config.watchdog_timeout_s, self._clock)
        return processor

    async def _send(self, payload: bytes) -> None:
        try:
            await self._sink.send(payload)
        except TransportError:
            # A broken sink must not stop processing, because the watchdogs depend on it.
            self._counts["send_errors"] += 1
        else:
            self._counts["frames_sent"] += 1

    async def _watch(self) -> None:
        interval = self._config.watchdog_timeout_s / WATCHDOG_CHECKS_PER_TIMEOUT
        while True:
            await self._sleep(interval)
            await self.check_watchdogs()
