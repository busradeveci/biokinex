"""Tests for biokinex.dsp.biquad."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_allclose

from biokinex.dsp.biquad import BiquadCoefficients, BiquadFilter
from biokinex.schemas.config import LowPassConfig, NotchConfig

RATE_HZ = 500.0
DT = 1.0 / RATE_HZ
NYQUIST_HZ = 0.5 * RATE_HZ


def run(filter_: BiquadFilter, values: np.ndarray) -> np.ndarray:
    """Feed ``values`` at the design rate and return the stacked outputs."""
    return np.array([filter_.step(v, i * DT) for i, v in enumerate(values)])


def sine(frequency_hz: float, count: int) -> np.ndarray:
    return np.sin(2.0 * math.pi * frequency_hz * np.arange(count) * DT)


def reference_filter(c: BiquadCoefficients, values: np.ndarray) -> np.ndarray:
    """Plain difference equation, with history set as if the first value had always been there."""
    x1 = x2 = values[0]
    y1 = y2 = c.dc_gain * values[0]
    out = []
    for x in values:
        y = c.b0 * x + c.b1 * x1 + c.b2 * x2 - c.a1 * y1 - c.a2 * y2
        x2, x1, y2, y1 = x1, x, y1, y
        out.append(y)
    return np.array(out)


class TestCoefficients:
    def test_butterworth_at_quarter_rate(self) -> None:
        # At fs/4, cos(w0) is 0, which gives the well-known half-band Butterworth values.
        c = BiquadCoefficients.low_pass(RATE_HZ / 4.0, RATE_HZ)
        alpha = 1.0 / math.sqrt(2.0)
        assert_allclose(
            [c.b0, c.b1, c.b2, c.a1, c.a2],
            [0.5 / (1 + alpha), 1 / (1 + alpha), 0.5 / (1 + alpha), 0.0, (1 - alpha) / (1 + alpha)],
            atol=1e-15,
        )

    @pytest.mark.parametrize("q", [0.5, 1.0 / math.sqrt(2.0), 2.0])
    @pytest.mark.parametrize("cutoff_hz", [1.0, 20.0, 200.0])
    def test_low_pass_response(self, cutoff_hz: float, q: float) -> None:
        c = BiquadCoefficients.low_pass(cutoff_hz, RATE_HZ, q)
        assert abs(c.frequency_response(0.0, RATE_HZ)) == pytest.approx(1.0)
        assert abs(c.frequency_response(cutoff_hz, RATE_HZ)) == pytest.approx(q)
        assert abs(c.frequency_response(NYQUIST_HZ, RATE_HZ)) == pytest.approx(0.0, abs=1e-12)
        assert c.dc_gain == pytest.approx(1.0)

    def test_butterworth_is_minus_3_db_at_cutoff(self) -> None:
        c = BiquadCoefficients.low_pass(30.0, RATE_HZ)
        gain_db = 20.0 * math.log10(abs(c.frequency_response(30.0, RATE_HZ)))
        assert gain_db == pytest.approx(-3.0103, abs=1e-3)

    @pytest.mark.parametrize("center_hz", [10.0, 50.0, 120.0])
    def test_notch_response(self, center_hz: float) -> None:
        c = BiquadCoefficients.notch(center_hz, RATE_HZ, q=5.0)
        assert abs(c.frequency_response(center_hz, RATE_HZ)) == pytest.approx(0.0, abs=1e-12)
        assert abs(c.frequency_response(0.0, RATE_HZ)) == pytest.approx(1.0)
        assert abs(c.frequency_response(NYQUIST_HZ, RATE_HZ)) == pytest.approx(1.0)

    def test_higher_q_makes_a_narrower_notch(self) -> None:
        wide = BiquadCoefficients.notch(50.0, RATE_HZ, q=1.0)
        narrow = BiquadCoefficients.notch(50.0, RATE_HZ, q=10.0)
        near_hz = 55.0
        assert abs(narrow.frequency_response(near_hz, RATE_HZ)) > abs(
            wide.frequency_response(near_hz, RATE_HZ)
        )

    @pytest.mark.parametrize(
        ("frequency_hz", "rate_hz", "q", "match"),
        [
            (0.0, RATE_HZ, 1.0, "frequency"),
            (NYQUIST_HZ, RATE_HZ, 1.0, "frequency"),
            (-5.0, RATE_HZ, 1.0, "frequency"),
            (10.0, 0.0, 1.0, "sample_rate_hz"),
            (10.0, math.inf, 1.0, "sample_rate_hz"),
            (10.0, RATE_HZ, 0.0, "q must"),
            (10.0, RATE_HZ, math.nan, "q must"),
        ],
    )
    def test_rejects_invalid_design(
        self, frequency_hz: float, rate_hz: float, q: float, match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            BiquadCoefficients.low_pass(frequency_hz, rate_hz, q)
        with pytest.raises(ValueError, match=match):
            BiquadCoefficients.notch(frequency_hz, rate_hz, q)

    @given(
        frequency_hz=st.floats(min_value=0.01, max_value=NYQUIST_HZ - 0.01),
        q=st.floats(min_value=0.05, max_value=50.0),
        notch=st.booleans(),
    )
    def test_design_is_always_stable(self, frequency_hz: float, q: float, notch: bool) -> None:
        design = BiquadCoefficients.notch if notch else BiquadCoefficients.low_pass
        c = design(frequency_hz, RATE_HZ, q)
        poles = np.roots([1.0, c.a1, c.a2])
        assert np.all(np.abs(poles) < 1.0)


class TestBiquadFilter:
    def test_constant_input_has_no_start_up_transient(self) -> None:
        for c in (
            BiquadCoefficients.low_pass(5.0, RATE_HZ),
            BiquadCoefficients.notch(50.0, RATE_HZ, q=2.0),
        ):
            gravity = np.full((200, 3), [0.0, 0.0, 9.80665])
            assert_allclose(run(BiquadFilter(c), gravity), gravity, rtol=1e-12)

    @given(
        st.lists(st.floats(min_value=-100.0, max_value=100.0), min_size=1, max_size=60),
        st.booleans(),
    )
    def test_matches_difference_equation(self, values: list[float], notch: bool) -> None:
        c = (
            BiquadCoefficients.notch(40.0, RATE_HZ, q=3.0)
            if notch
            else BiquadCoefficients.low_pass(25.0, RATE_HZ, q=0.9)
        )
        array = np.array(values)
        assert_allclose(run(BiquadFilter(c), array), reference_filter(c, array), atol=1e-9)

    def test_low_pass_sine_at_cutoff_is_minus_3_db(self) -> None:
        filter_ = BiquadFilter(BiquadCoefficients.low_pass(20.0, RATE_HZ))
        output = run(filter_, sine(20.0, 2000))
        assert np.max(np.abs(output[1000:])) == pytest.approx(1.0 / math.sqrt(2.0), rel=1e-3)

    def test_notch_removes_its_center_and_keeps_other_tones(self) -> None:
        filter_ = BiquadFilter(BiquadCoefficients.notch(50.0, RATE_HZ, q=5.0))
        hum = run(filter_, sine(50.0, 3000))
        assert np.max(np.abs(hum[2000:])) < 1e-3
        filter_.reset()
        slow = run(filter_, sine(2.0, 3000))
        assert np.max(np.abs(slow[2000:])) == pytest.approx(1.0, abs=1e-2)

    def test_channels_are_independent(self) -> None:
        c = BiquadCoefficients.low_pass(15.0, RATE_HZ)
        a, b = sine(3.0, 300), sine(70.0, 300)
        combined = run(BiquadFilter(c), np.stack([a, b], axis=1))
        assert_allclose(combined[:, 0], run(BiquadFilter(c), a))
        assert_allclose(combined[:, 1], run(BiquadFilter(c), b))

    def test_changing_an_output_does_not_change_the_state(self) -> None:
        c = BiquadCoefficients.low_pass(10.0, RATE_HZ)
        filter_ = BiquadFilter(c)
        first = filter_.step([1.0, 2.0], 0.0)
        first[:] = 99.0
        second = filter_.step([3.0, 4.0], DT)
        assert_allclose(second, reference_filter(c, np.array([[1.0, 2.0], [3.0, 4.0]]))[1])

    def test_rejects_shape_change(self) -> None:
        filter_ = BiquadFilter(BiquadCoefficients.low_pass(10.0, RATE_HZ))
        filter_.step([1.0, 2.0, 3.0], 0.0)
        with pytest.raises(ValueError, match="does not match"):
            filter_.step([1.0, 2.0], DT)

    @pytest.mark.parametrize("bad_time", [0.0, -1.0, math.nan])
    def test_rejects_time_that_does_not_move_forward(self, bad_time: float) -> None:
        filter_ = BiquadFilter(BiquadCoefficients.low_pass(10.0, RATE_HZ))
        filter_.step(1.0, 0.0)
        with pytest.raises(ValueError, match="timestamp must be finite and later"):
            filter_.step(1.0, bad_time)

    def test_rejects_non_finite_first_timestamp(self) -> None:
        filter_ = BiquadFilter(BiquadCoefficients.low_pass(10.0, RATE_HZ))
        with pytest.raises(ValueError, match="finite"):
            filter_.step(1.0, math.inf)

    def test_reset_starts_fresh(self) -> None:
        filter_ = BiquadFilter(BiquadCoefficients.low_pass(10.0, RATE_HZ))
        run(filter_, sine(5.0, 50))
        filter_.reset()
        assert_allclose(filter_.step([7.0, 7.0], 0.0), [7.0, 7.0])

    def test_from_config(self) -> None:
        low = BiquadFilter.from_config(LowPassConfig(cutoff_hz=12.0, q=0.8), RATE_HZ)
        assert low.coefficients == BiquadCoefficients.low_pass(12.0, RATE_HZ, 0.8)
        notch = BiquadFilter.from_config(NotchConfig(center_hz=50.0, q=4.0), RATE_HZ)
        assert notch.coefficients == BiquadCoefficients.notch(50.0, RATE_HZ, 4.0)
