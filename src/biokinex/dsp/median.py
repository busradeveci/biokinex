"""Sliding median filter for removing short spikes.

A median ignores a single wild sample completely, while a low-pass filter would smear it over many
samples. It also keeps step edges sharp. The price is a delay of ``(window - 1) / 2`` samples on
trends, so small windows (3 or 5) are the usual choice for spike removal.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from biokinex.buffers.ring import RingBuffer
from biokinex.core.types import FloatArray
from biokinex.dsp.base import check_timestamp, elapsed
from biokinex.schemas.config import MedianConfig

__all__ = ["MedianFilter"]


class MedianFilter:
    """Median of the newest ``window`` samples, for each element on its own.

    Until the window is full, the median of the newest odd number of samples is used. With an odd
    count the median is always one of the real input values, never an average of two.
    """

    __slots__ = ("_buffer", "_timestamp", "_window")

    def __init__(self, window: int = 5) -> None:
        if window < 1 or window % 2 == 0:
            msg = f"window must be a positive odd number, got {window}"
            raise ValueError(msg)
        self._window = window
        self._buffer: RingBuffer | None = None
        self._timestamp = -math.inf

    @classmethod
    def from_config(cls, config: MedianConfig) -> MedianFilter:
        """Build a filter from validated configuration."""
        return cls(config.window)

    @property
    def window(self) -> int:
        """Number of samples the median is taken over."""
        return self._window

    def step(self, value: npt.ArrayLike, timestamp: float) -> FloatArray:
        """Filter one sample and return a new array with the median."""
        sample = np.asarray(value, dtype=np.float64)
        if self._buffer is None:
            self._timestamp = check_timestamp(timestamp)
            # The buffer is created on the first sample, because only then is the shape known.
            self._buffer = RingBuffer(self._window, sample.shape)
        else:
            if sample.shape != self._buffer.row_shape:
                msg = (
                    f"sample shape {sample.shape} does not match "
                    f"filter shape {self._buffer.row_shape}"
                )
                raise ValueError(msg)
            elapsed(self._timestamp, timestamp)
            self._timestamp = timestamp
        self._buffer.append(sample)

        count = len(self._buffer)
        odd_count = count if count % 2 == 1 else count - 1
        # The ring buffer gives a zero-copy view of the newest samples; only the sort copies.
        # For small windows, sorting and taking the middle row is much faster than np.median.
        return np.asarray(np.sort(self._buffer.latest(odd_count), axis=0)[odd_count // 2])

    def reset(self) -> None:
        """Forget all history; the next sample may have a new shape."""
        self._buffer = None
        self._timestamp = -math.inf
