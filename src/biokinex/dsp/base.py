"""Common interface and helpers for stateful, sample-by-sample filters.

Every filter consumes one sample at a time with its sensor timestamp. Filters use the time between
samples instead of a fixed rate, so jitter in the device clock and filled gaps are handled
correctly, and recorded data replays with identical results.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Protocol

import numpy as np
import numpy.typing as npt

from biokinex.core.types import FloatArray

__all__ = ["Filter", "FilterChain", "apply_to_batch", "check_timestamp", "elapsed"]


class Filter(Protocol):
    """A stateful filter that processes one sample per call.

    Vector filters take and return arrays of a fixed shape. Orientation filters take and return
    unit quaternions ``(w, x, y, z)``. Implementations return a new array on every call, so callers
    may keep or change the result safely.
    """

    def step(self, value: npt.ArrayLike, timestamp: float) -> FloatArray:
        """Filter one sample taken at ``timestamp`` seconds and return the filtered value."""
        ...

    def reset(self) -> None:
        """Forget all history, so the next sample starts the filter fresh."""
        ...


def elapsed(previous: float, current: float) -> float:
    """Return ``current - previous`` after checking that time moves forward.

    Raises:
        ValueError: If ``current`` is not finite or is not later than ``previous``.
    """
    # A zero or negative step would divide by zero or reverse a filter, so it is a caller bug.
    if not math.isfinite(current) or current <= previous:
        msg = f"timestamp must be finite and later than {previous!r}, got {current!r}"
        raise ValueError(msg)
    return current - previous


def check_timestamp(timestamp: float) -> float:
    """Return ``timestamp`` after checking that it is finite.

    Raises:
        ValueError: If ``timestamp`` is NaN or infinite.
    """
    if not math.isfinite(timestamp):
        msg = f"timestamp must be finite, got {timestamp!r}"
        raise ValueError(msg)
    return timestamp


class FilterChain:
    """Runs several filters in order, feeding each output into the next filter."""

    __slots__ = ("_filters",)

    def __init__(self, filters: Iterable[Filter] = ()) -> None:
        self._filters = tuple(filters)

    @property
    def filters(self) -> tuple[Filter, ...]:
        """The filters in the order they run."""
        return self._filters

    def __len__(self) -> int:
        return len(self._filters)

    def step(self, value: npt.ArrayLike, timestamp: float) -> FloatArray:
        """Pass one sample through every filter in order."""
        # Copying keeps an empty chain from handing the caller's own array back to them.
        result = np.array(value, dtype=np.float64)
        for item in self._filters:
            result = item.step(result, timestamp)
        return result

    def reset(self) -> None:
        """Reset every filter in the chain."""
        for item in self._filters:
            item.reset()


def apply_to_batch(filter_: Filter, values: npt.ArrayLike, timestamps: npt.ArrayLike) -> FloatArray:
    """Run ``filter_`` over each row of ``values`` and stack the results.

    Args:
        filter_: Any :class:`Filter`, including a :class:`FilterChain`.
        values: Samples with shape ``(N, ...)``.
        timestamps: Sample times with shape ``(N,)``.
    """
    rows = np.asarray(values, dtype=np.float64)
    times = np.asarray(timestamps, dtype=np.float64)
    if rows.ndim == 0 or times.shape != (rows.shape[0],):
        msg = (
            "need values of shape (N, ...) and timestamps of shape (N,), "
            f"got {rows.shape} and {times.shape}"
        )
        raise ValueError(msg)
    output = np.empty_like(rows)
    for index in range(rows.shape[0]):
        output[index] = filter_.step(rows[index], float(times[index]))
    return output
