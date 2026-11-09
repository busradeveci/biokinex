"""Tests for biokinex.dsp.median."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_allclose, assert_array_equal

from biokinex.dsp.median import MedianFilter
from biokinex.schemas.config import MedianConfig

DT = 0.002


def run(filter_: MedianFilter, values: np.ndarray | list[float] | list[list[float]]) -> np.ndarray:
    return np.array([filter_.step(v, i * DT) for i, v in enumerate(values)])


class TestMedianFilter:
    def test_removes_single_spike(self) -> None:
        output = run(MedianFilter(3), [1.0, 1.0, 1.0, 50.0, 1.0, 1.0])
        assert_array_equal(output, np.ones(6))

    def test_removes_two_sample_spike_with_window_five(self) -> None:
        output = run(MedianFilter(5), [0.0] * 5 + [9.0, 9.0] + [0.0] * 5)
        assert_array_equal(output, np.zeros(12))

    def test_warm_up_uses_odd_counts(self) -> None:
        # Two samples give the newest one, never their average.
        assert_array_equal(run(MedianFilter(5), [1.0, 4.0, 2.0, 8.0]), [1.0, 4.0, 2.0, 4.0])

    def test_step_edge_stays_sharp_with_known_delay(self) -> None:
        output = run(MedianFilter(5), [0.0] * 6 + [1.0] * 6)
        # The window of 5 needs 3 new values to switch, so the edge moves by 2 samples.
        assert_array_equal(output, [0.0] * 8 + [1.0] * 4)

    def test_window_one_passes_through(self) -> None:
        values = [3.0, -1.0, 8.0]
        assert_array_equal(run(MedianFilter(1), values), values)

    def test_channels_are_independent(self) -> None:
        output = run(MedianFilter(3), [[0.0, 5.0], [9.0, 5.0], [0.0, -9.0], [0.0, 5.0]])
        assert_array_equal(output[-1], [0.0, 5.0])

    @given(
        st.integers(min_value=0, max_value=4).map(lambda k: 2 * k + 1),
        st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=1, max_size=40),
    )
    def test_output_is_a_recent_input_value(self, window: int, values: list[float]) -> None:
        output = run(MedianFilter(window), values)
        for index, result in enumerate(output):
            recent = values[max(0, index - window + 1) : index + 1]
            assert result in recent
            assert min(recent) <= result <= max(recent)

    def test_output_does_not_share_memory_with_buffer(self) -> None:
        filter_ = MedianFilter(3)
        first = filter_.step([1.0, 2.0], 0.0)
        first[:] = 100.0
        assert_allclose(filter_.step([1.0, 2.0], DT), [1.0, 2.0])

    @pytest.mark.parametrize("window", [0, -3, 2, 4])
    def test_rejects_bad_window(self, window: int) -> None:
        with pytest.raises(ValueError, match="positive odd"):
            MedianFilter(window)

    def test_rejects_shape_change(self) -> None:
        filter_ = MedianFilter(3)
        filter_.step([1.0, 2.0], 0.0)
        with pytest.raises(ValueError, match="does not match"):
            filter_.step([1.0], DT)

    @pytest.mark.parametrize("bad_time", [0.0, math.nan])
    def test_rejects_time_that_does_not_move_forward(self, bad_time: float) -> None:
        filter_ = MedianFilter(3)
        filter_.step(1.0, 0.0)
        with pytest.raises(ValueError, match="timestamp must be finite and later"):
            filter_.step(1.0, bad_time)

    def test_reset_allows_new_shape(self) -> None:
        filter_ = MedianFilter(3)
        run(filter_, [1.0, 2.0, 3.0])
        filter_.reset()
        assert_array_equal(filter_.step([5.0, 6.0], 0.0), [5.0, 6.0])

    def test_from_config(self) -> None:
        assert MedianFilter.from_config(MedianConfig(window=7)).window == 7
