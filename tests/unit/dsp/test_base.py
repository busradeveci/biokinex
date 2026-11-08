"""Tests for biokinex.dsp.base."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pytest
from numpy.testing import assert_allclose

from biokinex.core.types import FloatArray
from biokinex.dsp.base import Filter, FilterChain, apply_to_batch, check_timestamp, elapsed
from biokinex.dsp.one_euro import OneEuroFilter


class Affine:
    """Minimal filter for testing composition: returns ``scale * value + offset``."""

    def __init__(self, scale: float, offset: float) -> None:
        self.scale = scale
        self.offset = offset
        self.resets = 0

    def step(self, value: npt.ArrayLike, timestamp: float) -> FloatArray:
        return np.asarray(self.scale * np.asarray(value, dtype=np.float64) + self.offset)

    def reset(self) -> None:
        self.resets += 1


def test_elapsed() -> None:
    assert elapsed(1.0, 1.25) == 0.25
    assert elapsed(-math.inf, 0.0) == math.inf


@pytest.mark.parametrize("current", [1.0, 0.5, math.nan, math.inf])
def test_elapsed_rejects_non_increasing_or_non_finite(current: float) -> None:
    with pytest.raises(ValueError, match="timestamp must be finite and later"):
        elapsed(1.0, current)


def test_check_timestamp() -> None:
    assert check_timestamp(3.5) == 3.5
    with pytest.raises(ValueError, match="timestamp must be finite"):
        check_timestamp(math.nan)


class TestFilterChain:
    def test_runs_filters_in_order(self) -> None:
        chain = FilterChain([Affine(2.0, 0.0), Affine(1.0, 1.0)])
        assert_allclose(chain.step([1.0, 2.0], 0.0), [3.0, 5.0])
        assert len(chain) == 2

    def test_empty_chain_returns_a_copy(self) -> None:
        value = np.array([1.0, 2.0])
        result = FilterChain().step(value, 0.0)
        assert_allclose(result, value)
        assert not np.shares_memory(result, value)

    def test_reset_reaches_every_filter(self) -> None:
        filters = [Affine(1.0, 0.0), Affine(1.0, 0.0)]
        FilterChain(filters).reset()
        assert [f.resets for f in filters] == [1, 1]

    def test_satisfies_filter_protocol(self) -> None:
        chain: Filter = FilterChain([OneEuroFilter()])
        assert_allclose(chain.step([4.0], 0.0), [4.0])
        assert isinstance(FilterChain().filters, tuple)


class TestApplyToBatch:
    def test_applies_row_by_row(self) -> None:
        result = apply_to_batch(Affine(10.0, 0.0), [[1.0, 2.0], [3.0, 4.0]], [0.0, 0.1])
        assert_allclose(result, [[10.0, 20.0], [30.0, 40.0]])

    def test_matches_manual_stepping(self) -> None:
        values = np.sin(np.linspace(0.0, 3.0, 50))
        times = np.arange(50) * 0.01
        manual = OneEuroFilter(2.0, 0.1)
        expected = [manual.step(v, t) for v, t in zip(values, times, strict=True)]
        assert_allclose(apply_to_batch(OneEuroFilter(2.0, 0.1), values, times), expected)

    @pytest.mark.parametrize(
        ("values", "timestamps"),
        [(1.0, [0.0]), ([1.0, 2.0], [0.0]), ([[1.0], [2.0]], [[0.0, 1.0]])],
    )
    def test_rejects_mismatched_shapes(self, values: object, timestamps: object) -> None:
        with pytest.raises(ValueError, match="need values of shape"):
            apply_to_batch(Affine(1.0, 0.0), values, timestamps)
