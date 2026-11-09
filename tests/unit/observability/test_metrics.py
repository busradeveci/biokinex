"""Tests for biokinex.observability.metrics."""

from __future__ import annotations

import math
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from biokinex.observability.metrics import LatencyHistogram, LatencySummary


def test_empty_histogram() -> None:
    histogram = LatencyHistogram()
    assert histogram.count == 0
    assert histogram.percentile(0.99) == 0.0
    assert histogram.summary() == LatencySummary(count=0, mean=0.0, p50=0.0, p99=0.0, max=0.0)


def test_summary_of_known_values() -> None:
    histogram = LatencyHistogram()
    for value in [10e-6] * 98 + [500e-6, 2e-3]:
        histogram.record(value)
    summary = histogram.summary()
    assert summary.count == 100
    assert summary.mean == pytest.approx((98 * 10e-6 + 500e-6 + 2e-3) / 100)
    assert summary.p50 == pytest.approx(10e-6, rel=0.13)
    assert summary.p99 == pytest.approx(500e-6, rel=0.13)
    assert summary.max == 2e-3


def test_percentile_never_understates() -> None:
    histogram = LatencyHistogram()
    histogram.record(123e-6)
    assert 123e-6 <= histogram.percentile(1.0) <= 123e-6 * 1.13


def test_values_outside_the_range_are_kept() -> None:
    histogram = LatencyHistogram(min_s=1e-3, max_s=1e-1)
    histogram.record(1e-5)
    histogram.record(5.0)
    assert histogram.count == 2
    assert histogram.percentile(0.5) == pytest.approx(1e-3)
    assert histogram.percentile(1.0) == 5.0


@pytest.mark.parametrize("value", [-1.0, math.nan, math.inf])
def test_invalid_durations_count_as_zero(value: float) -> None:
    histogram = LatencyHistogram()
    histogram.record(value)
    assert histogram.summary() == LatencySummary(count=1, mean=0.0, p50=0.0, p99=0.0, max=0.0)


def test_reset_keeps_bins() -> None:
    histogram = LatencyHistogram()
    histogram.record(1e-4)
    histogram.reset()
    assert histogram.count == 0
    histogram.record(2e-4)
    assert histogram.summary().max == 2e-4


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"min_s": 0.0}, "min_s"),
        ({"min_s": 1.0, "max_s": 0.5}, "min_s"),
        ({"max_s": math.inf}, "min_s"),
        ({"bins_per_decade": 0}, "bins_per_decade"),
    ],
)
def test_rejects_invalid_layout(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        LatencyHistogram(**kwargs)


@pytest.mark.parametrize("fraction", [-0.1, 1.1])
def test_rejects_invalid_fraction(fraction: float) -> None:
    with pytest.raises(ValueError, match="fraction"):
        LatencyHistogram().percentile(fraction)


@given(st.lists(st.floats(min_value=1e-7, max_value=10.0), min_size=1, max_size=200))
def test_percentile_bounds_the_true_value(values: list[float]) -> None:
    histogram = LatencyHistogram()
    for value in values:
        histogram.record(value)
    ordered = sorted(values)
    true_p99 = ordered[math.ceil(0.99 * len(ordered)) - 1]
    estimate = histogram.percentile(0.99)
    assert true_p99 <= estimate * (1 + 1e-12)
    assert estimate <= max(values)
