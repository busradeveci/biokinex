"""Injectable time sources.

DSP filters always use sensor timestamps. A :class:`Clock` is only used by the runtime, for example
to detect stale input or to measure processing latency. Injecting the clock lets tests control time
exactly instead of sleeping.
"""

from __future__ import annotations

import math
import time
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, TypeAlias, runtime_checkable

Sleep: TypeAlias = Callable[[float], Awaitable[Any]]
"""An async sleep such as :func:`asyncio.sleep`; tests pass a fake that moves a manual clock."""


@runtime_checkable
class Clock(Protocol):
    """A monotonic time source measured in seconds."""

    def now(self) -> float:
        """Return the current time in seconds; the value must never decrease."""
        ...


class MonotonicClock:
    """Real clock backed by :func:`time.perf_counter`."""

    __slots__ = ()

    def now(self) -> float:
        """Return the current monotonic time in seconds."""
        # perf_counter is monotonic and has the highest available resolution on every platform.
        return time.perf_counter()


class ManualClock:
    """Clock that only moves when the caller advances it, for deterministic tests and replay."""

    __slots__ = ("_now",)

    def __init__(self, start: float = 0.0) -> None:
        if not math.isfinite(start):
            msg = f"start must be finite, got {start!r}"
            raise ValueError(msg)
        self._now = float(start)

    def now(self) -> float:
        """Return the current manual time in seconds."""
        return self._now

    def advance(self, seconds: float) -> float:
        """Move time forward by ``seconds`` and return the new time."""
        if not math.isfinite(seconds) or seconds < 0.0:
            msg = f"seconds must be finite and non-negative, got {seconds!r}"
            raise ValueError(msg)
        self._now += seconds
        return self._now

    def set(self, timestamp: float) -> float:
        """Jump to ``timestamp`` and return it; time may not go backwards."""
        if not math.isfinite(timestamp) or timestamp < self._now:
            msg = f"timestamp must be finite and >= {self._now!r}, got {timestamp!r}"
            raise ValueError(msg)
        self._now = float(timestamp)
        return self._now
