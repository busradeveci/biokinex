"""Fixed-size history of recent output frames, with zero-copy access.

Controllers and monitors often need the last few hundred frames, for example to check stability or
draw a plot. Building those arrays from a list of frames would allocate on every call. Instead, the
history keeps one ring buffer per channel, and a window is a set of read-only NumPy views into them.
"""

from __future__ import annotations

from dataclasses import dataclass

from biokinex.buffers.ring import RingBuffer
from biokinex.core.types import QUATERNION_SIZE, VECTOR3_SIZE, FloatArray

__all__ = ["FrameHistory", "HistoryLayout", "HistoryWindow"]


@dataclass(frozen=True, slots=True)
class HistoryLayout:
    """Which channels the stored frames have."""

    orientation: bool
    angular_velocity: bool
    acceleration: bool
    joint_names: tuple[str, ...]


@dataclass(frozen=True, slots=True, eq=False)
class HistoryWindow:
    """The newest frames, oldest first, as read-only views.

    The views point into live storage and stay valid only until the next frame is added. Copy them
    if you need to keep the values longer.
    """

    timestamps: FloatArray
    orientation: FloatArray | None
    angular_velocity: FloatArray | None
    acceleration: FloatArray | None
    joint_names: tuple[str, ...]
    joint_angles: FloatArray | None


class FrameHistory:
    """Keeps the newest ``capacity`` output frames of one device.

    All stored frames share one layout. When the layout changes, the history is cleared, because
    rows with different channels cannot share one array.
    """

    __slots__ = (
        "_acceleration",
        "_angular_velocity",
        "_capacity",
        "_joint_angles",
        "_layout",
        "_orientation",
        "_timestamps",
    )

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._timestamps = RingBuffer(capacity)
        self._layout: HistoryLayout | None = None
        self._orientation: RingBuffer | None = None
        self._angular_velocity: RingBuffer | None = None
        self._acceleration: RingBuffer | None = None
        self._joint_angles: RingBuffer | None = None

    @property
    def capacity(self) -> int:
        """Largest number of frames kept."""
        return self._capacity

    @property
    def layout(self) -> HistoryLayout | None:
        """The layout of the stored frames, or ``None`` before the first frame."""
        return self._layout

    def __len__(self) -> int:
        return len(self._timestamps)

    def append(
        self,
        layout: HistoryLayout,
        timestamp: float,
        *,
        orientation: FloatArray | None,
        angular_velocity: FloatArray | None,
        acceleration: FloatArray | None,
        joint_angles: FloatArray | None,
    ) -> None:
        """Store one frame; the values must match ``layout``."""
        if layout != self._layout:
            self._configure(layout)
        self._timestamps.append(timestamp)
        _store(self._orientation, orientation)
        _store(self._angular_velocity, angular_velocity)
        _store(self._acceleration, acceleration)
        _store(self._joint_angles, joint_angles)

    def window(self, count: int | None = None) -> HistoryWindow:
        """Return the newest ``count`` frames, or all stored frames when ``count`` is ``None``.

        Raises:
            ValueError: If ``count`` is negative or larger than the number of stored frames.
        """
        timestamps = self._timestamps.latest(count)
        size = timestamps.shape[0]
        return HistoryWindow(
            timestamps=timestamps,
            orientation=_latest(self._orientation, size),
            angular_velocity=_latest(self._angular_velocity, size),
            acceleration=_latest(self._acceleration, size),
            joint_names=() if self._layout is None else self._layout.joint_names,
            joint_angles=_latest(self._joint_angles, size),
        )

    def clear(self) -> None:
        """Remove every frame and forget the layout."""
        self._timestamps.clear()
        self._layout = None
        self._orientation = self._angular_velocity = self._acceleration = None
        self._joint_angles = None

    def _configure(self, layout: HistoryLayout) -> None:
        """Start an empty history for a new layout."""
        self._timestamps.clear()
        self._layout = layout
        # Buffers are allocated only on a layout change, which is rare, never per frame.
        self._orientation = self._buffer(layout.orientation, (QUATERNION_SIZE,))
        self._angular_velocity = self._buffer(layout.angular_velocity, (VECTOR3_SIZE,))
        self._acceleration = self._buffer(layout.acceleration, (VECTOR3_SIZE,))
        joint_count = len(layout.joint_names)
        self._joint_angles = self._buffer(joint_count > 0, (joint_count,))

    def _buffer(self, present: bool, row_shape: tuple[int, ...]) -> RingBuffer | None:
        return RingBuffer(self._capacity, row_shape) if present else None


def _store(buffer: RingBuffer | None, value: FloatArray | None) -> None:
    """Append ``value`` to ``buffer``, checking that both are present or both are absent."""
    if buffer is None and value is None:
        return
    if buffer is None or value is None:
        msg = "frame channels do not match the history layout"
        raise ValueError(msg)
    buffer.append(value)


def _latest(buffer: RingBuffer | None, count: int) -> FloatArray | None:
    return None if buffer is None else buffer.latest(count)
