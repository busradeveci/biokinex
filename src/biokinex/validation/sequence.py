"""Sequence number tracking for one device stream.

Devices count samples with an unsigned 32-bit counter that wraps from ``2**32 - 1`` back to zero.
Comparisons use serial number arithmetic (RFC 1982): the distance from the last accepted number is
taken modulo ``2**32``, so a wrap looks like a normal step forward.
"""

from __future__ import annotations

from enum import Enum
from typing import Final, NamedTuple

from biokinex.schemas.common import MAX_SEQUENCE

__all__ = [
    "DEFAULT_MAX_FORWARD_JUMP",
    "DEFAULT_REORDER_WINDOW",
    "SEQUENCE_MODULUS",
    "SequenceCheck",
    "SequenceStatus",
    "SequenceTracker",
]

SEQUENCE_MODULUS: Final = MAX_SEQUENCE + 1
_HALF_RANGE: Final = SEQUENCE_MODULUS // 2

DEFAULT_REORDER_WINDOW: Final = 64
"""How far back a sequence number may be and still count as a late packet, not a restart."""

DEFAULT_MAX_FORWARD_JUMP: Final = 1000
"""Largest forward step still treated as packet loss; about two seconds at 500 Hz."""


class SequenceStatus(Enum):
    """How a sequence number relates to the last accepted one."""

    FIRST = "first"
    """No sample has been accepted yet."""

    IN_ORDER = "in_order"
    """Exactly one step forward."""

    GAP = "gap"
    """A small step forward; the numbers in between were lost."""

    DUPLICATE = "duplicate"
    """The same number as the last accepted sample."""

    LATE = "late"
    """Slightly behind; the sample arrived after newer ones and is too old to use."""

    RESYNC = "resync"
    """A jump too large to be loss or reordering, which usually means the device restarted."""


class SequenceCheck(NamedTuple):
    """Result of classifying one sequence number."""

    status: SequenceStatus
    missing: int
    """Number of lost samples before this one; only non-zero for ``GAP``."""


class SequenceTracker:
    """Classifies the sequence numbers of one device stream.

    :meth:`classify` never changes state; call :meth:`advance` once a sample is really accepted.
    Keeping these steps apart lets the caller reject a sample for other reasons, such as a bad
    timestamp, without moving the tracker.
    """

    __slots__ = ("_last", "_max_forward_jump", "_reorder_window")

    def __init__(
        self,
        *,
        reorder_window: int = DEFAULT_REORDER_WINDOW,
        max_forward_jump: int = DEFAULT_MAX_FORWARD_JUMP,
    ) -> None:
        # Both ranges must stay inside half the counter, or a step could be read both ways.
        if not 0 <= reorder_window < _HALF_RANGE:
            msg = f"reorder_window must be in [0, {_HALF_RANGE}), got {reorder_window}"
            raise ValueError(msg)
        if not 1 <= max_forward_jump < _HALF_RANGE:
            msg = f"max_forward_jump must be in [1, {_HALF_RANGE}), got {max_forward_jump}"
            raise ValueError(msg)
        self._reorder_window = reorder_window
        self._max_forward_jump = max_forward_jump
        self._last: int | None = None

    @property
    def last(self) -> int | None:
        """The last accepted sequence number, or ``None`` before the first one."""
        return self._last

    def classify(self, sequence: int) -> SequenceCheck:
        """Classify ``sequence`` against the last accepted number without changing state."""
        if self._last is None:
            return SequenceCheck(SequenceStatus.FIRST, 0)
        forward = (sequence - self._last) % SEQUENCE_MODULUS
        if forward == 0:
            return SequenceCheck(SequenceStatus.DUPLICATE, 0)
        if forward == 1:
            return SequenceCheck(SequenceStatus.IN_ORDER, 0)
        if forward <= self._max_forward_jump:
            return SequenceCheck(SequenceStatus.GAP, forward - 1)
        if SEQUENCE_MODULUS - forward <= self._reorder_window:
            return SequenceCheck(SequenceStatus.LATE, 0)
        return SequenceCheck(SequenceStatus.RESYNC, 0)

    def advance(self, sequence: int) -> None:
        """Record ``sequence`` as the last accepted number."""
        if not 0 <= sequence <= MAX_SEQUENCE:
            msg = f"sequence must be in [0, {MAX_SEQUENCE}], got {sequence}"
            raise ValueError(msg)
        self._last = sequence

    def reset(self) -> None:
        """Forget the stream history, so the next number is classified as ``FIRST``."""
        self._last = None
