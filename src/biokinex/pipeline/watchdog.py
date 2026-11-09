"""Staleness detection for one device.

If a sensor cable breaks or a network link fails, no new frames arrive, but a controller would
keep acting on the last frame it received. The watchdog notices the silence and produces a copy of
the last frame marked ``STALE``, so the controller can enter a safe state.

The watchdog uses the runtime clock, not sensor timestamps, because it must detect that sensor time
has stopped moving.
"""

from __future__ import annotations

import math

from biokinex.core.clock import Clock
from biokinex.schemas.pose import FrameStatus, PoseFrame

__all__ = ["Watchdog"]


class Watchdog:
    """Reports a stale frame when no fresh frame has arrived for ``timeout_s`` seconds.

    While the device stays silent, a stale frame is reported again after each further timeout.
    UDP may lose any single datagram, so repeating the warning makes sure a controller hears it.

    Args:
        timeout_s: Silence after which the device counts as stale.
        clock: Monotonic time source of the runtime.
    """

    __slots__ = ("_clock", "_deadline", "_last_frame", "_last_fresh", "_timeout_s")

    def __init__(self, timeout_s: float, clock: Clock) -> None:
        if not (math.isfinite(timeout_s) and timeout_s > 0.0):
            msg = f"timeout_s must be finite and positive, got {timeout_s!r}"
            raise ValueError(msg)
        self._timeout_s = timeout_s
        self._clock = clock
        self._last_frame: PoseFrame | None = None
        self._last_fresh = -math.inf
        self._deadline = math.inf

    @property
    def timeout_s(self) -> float:
        """Silence after which the device counts as stale."""
        return self._timeout_s

    @property
    def is_stale(self) -> bool:
        """Whether the device has been silent for at least the timeout."""
        return self._last_frame is not None and (
            self._clock.now() - self._last_fresh >= self._timeout_s
        )

    def feed(self, frame: PoseFrame) -> None:
        """Record a fresh frame; this restarts the timeout."""
        self._last_frame = frame
        self._last_fresh = self._clock.now()
        self._deadline = self._last_fresh + self._timeout_s

    def check(self) -> PoseFrame | None:
        """Return a stale frame if a warning is due now, otherwise ``None``.

        Nothing is reported before the first frame, because there is no state to report yet.
        """
        if self._last_frame is None:
            return None
        now = self._clock.now()
        if now < self._deadline:
            return None
        self._deadline = now + self._timeout_s
        # Only the status changes, so the copy keeps the last known values for the controller.
        return self._last_frame.model_copy(update={"status": FrameStatus.STALE})

    def reset(self) -> None:
        """Forget the device, for example after it was removed on purpose."""
        self._last_frame = None
        self._last_fresh = -math.inf
        self._deadline = math.inf
