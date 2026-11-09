"""Allocation-free latency measurement.

A runtime that streams for hours cannot keep every measurement. The histogram counts values in fixed
bins spaced evenly on a log scale, so its memory never grows and each percentile is accurate to
one bin width (about 12% with the default 20 bins per decade).
"""

from __future__ import annotations

import bisect
import itertools
import math
from dataclasses import dataclass

__all__ = ["LatencyHistogram", "LatencySummary"]


@dataclass(frozen=True, slots=True)
class LatencySummary:
    """Latency figures in seconds; all are zero before the first measurement."""

    count: int
    mean: float
    p50: float
    p99: float
    max: float


class LatencyHistogram:
    """Counts durations in log-spaced bins between ``min_s`` and ``max_s``.

    Values below ``min_s`` or above ``max_s`` are still counted, in an underflow or an overflow
    bin, so no measurement is ever lost.
    """

    __slots__ = ("_counts", "_edges", "_max", "_total")

    def __init__(self, min_s: float = 1e-6, max_s: float = 1.0, bins_per_decade: int = 20) -> None:
        if not (math.isfinite(min_s) and math.isfinite(max_s) and 0.0 < min_s < max_s):
            msg = f"need finite 0 < min_s < max_s, got {min_s!r} and {max_s!r}"
            raise ValueError(msg)
        if bins_per_decade < 1:
            msg = f"bins_per_decade must be >= 1, got {bins_per_decade}"
            raise ValueError(msg)
        bins = math.ceil(math.log10(max_s / min_s) * bins_per_decade)
        step = math.log10(max_s / min_s) / bins
        self._edges = tuple(min_s * 10.0 ** (step * index) for index in range(bins + 1))
        # One extra slot on each side holds the underflow and overflow counts.
        self._counts = [0] * (bins + 2)
        self._total = 0.0
        self._max = 0.0

    @property
    def count(self) -> int:
        """Number of recorded durations."""
        return sum(self._counts)

    def record(self, seconds: float) -> None:
        """Count one duration; negative or non-finite values are treated as zero."""
        value = seconds if math.isfinite(seconds) and seconds > 0.0 else 0.0
        self._counts[bisect.bisect_right(self._edges, value)] += 1
        self._total += value
        self._max = max(self._max, value)

    def percentile(self, fraction: float) -> float:
        """Return an upper bound for the given fraction of durations, for example ``0.99``.

        The result is the upper edge of the bin that holds the percentile, so it never
        understates the real latency.
        """
        if not 0.0 <= fraction <= 1.0:
            msg = f"fraction must be in [0, 1], got {fraction!r}"
            raise ValueError(msg)
        count = self.count
        if count == 0:
            return 0.0
        target = max(1, math.ceil(fraction * count))
        index = next(
            i for i, seen in enumerate(itertools.accumulate(self._counts)) if seen >= target
        )
        # The overflow bin has no upper edge, so the largest value seen is used instead.
        upper = self._edges[index] if index < len(self._edges) else self._max
        return min(upper, self._max)

    def summary(self) -> LatencySummary:
        """Return count, mean, median, 99th percentile, and maximum."""
        count = self.count
        return LatencySummary(
            count=count,
            mean=self._total / count if count else 0.0,
            p50=self.percentile(0.5),
            p99=self.percentile(0.99),
            max=self._max,
        )

    def reset(self) -> None:
        """Forget all measurements; the bins are kept."""
        self._counts = [0] * len(self._counts)
        self._total = 0.0
        self._max = 0.0
