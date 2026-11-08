"""Tests for biokinex.dsp.one_euro."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_allclose

from biokinex.core import quaternion as quat
from biokinex.dsp.one_euro import OneEuroFilter, OrientationOneEuroFilter, smoothing_factor
from biokinex.schemas.config import OneEuroConfig

RATE_HZ = 500.0
DT = 1.0 / RATE_HZ
Z_AXIS = np.array([0.0, 0.0, 1.0])


def run(filter_: OneEuroFilter, values: np.ndarray, dt: float = DT) -> np.ndarray:
    """Feed ``values`` at a fixed rate and return the stacked outputs."""
    return np.array([filter_.step(v, i * dt) for i, v in enumerate(values)])


class TestSmoothingFactor:
    def test_known_value(self) -> None:
        r = 2.0 * math.pi * 10.0 * 0.01
        assert_allclose(smoothing_factor(10.0, 0.01), r / (1.0 + r))

    def test_is_vectorized(self) -> None:
        result = smoothing_factor(np.array([1.0, 100.0]), DT)
        assert result.shape == (2,)
        assert 0.0 < result[0] < result[1] < 1.0


class TestParameters:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"min_cutoff_hz": 0.0},
            {"min_cutoff_hz": math.inf},
            {"beta": -0.1},
            {"beta": math.nan},
            {"derivative_cutoff_hz": 0.0},
        ],
    )
    @pytest.mark.parametrize("filter_type", [OneEuroFilter, OrientationOneEuroFilter])
    def test_rejects_invalid_tuning(self, filter_type: type[Any], kwargs: dict[str, float]) -> None:
        with pytest.raises(ValueError, match="must be finite"):
            filter_type(**kwargs)

    @pytest.mark.parametrize("filter_type", [OneEuroFilter, OrientationOneEuroFilter])
    def test_from_config(self, filter_type: type[Any]) -> None:
        config = OneEuroConfig(min_cutoff_hz=2.0, beta=0.5, derivative_cutoff_hz=3.0)
        built = filter_type.from_config(config)
        assert (built._min_cutoff_hz, built._beta, built._derivative_cutoff_hz) == (2.0, 0.5, 3.0)


class TestVectorFilter:
    def test_first_sample_passes_through(self) -> None:
        assert_allclose(OneEuroFilter().step([1.0, -2.0, 3.0], 5.0), [1.0, -2.0, 3.0])

    def test_constant_input_stays_constant(self) -> None:
        output = run(OneEuroFilter(1.0, 0.5), np.full((100, 3), 2.5))
        assert_allclose(output, 2.5)

    def test_beta_zero_is_first_order_low_pass(self) -> None:
        alpha = float(smoothing_factor(5.0, DT))
        output = run(OneEuroFilter(min_cutoff_hz=5.0), np.r_[0.0, np.ones(20)])
        expected = 1.0 - (1.0 - alpha) ** np.arange(21)
        assert_allclose(output, expected, atol=1e-12)

    def test_ramp_lag_matches_closed_form(self) -> None:
        # For a first-order low-pass driven by a ramp, the steady-state lag is v / (2 pi fc).
        speed = 3.0
        ramp = speed * np.arange(5_000) * DT
        output = run(OneEuroFilter(min_cutoff_hz=2.0), ramp)
        assert_allclose(ramp[-1] - output[-1], speed / (2.0 * math.pi * 2.0), rtol=1e-6)

    def test_beta_raises_cutoff_with_speed(self) -> None:
        speed, min_cutoff, beta = 3.0, 2.0, 0.5
        ramp = speed * np.arange(5_000) * DT
        output = run(OneEuroFilter(min_cutoff, beta, derivative_cutoff_hz=1.0), ramp)
        effective_cutoff = min_cutoff + beta * speed
        expected_lag = speed / (2.0 * math.pi * effective_cutoff)
        assert_allclose(ramp[-1] - output[-1], expected_lag, rtol=1e-6)

    def test_reduces_jitter_at_rest(self) -> None:
        rng = np.random.default_rng(seed=3)
        noisy = 1.0 + rng.normal(scale=0.05, size=(2_000, 3))
        output = run(OneEuroFilter(min_cutoff_hz=1.0, beta=0.01), noisy)
        assert np.std(output[500:]) < 0.1 * np.std(noisy[500:])

    def test_channels_are_independent(self) -> None:
        values = np.column_stack([np.zeros(200), np.linspace(0.0, 10.0, 200)])
        output = run(OneEuroFilter(1.0, 1.0), values)
        assert_allclose(output[:, 0], 0.0)

    def test_handles_variable_time_steps(self) -> None:
        filter_ = OneEuroFilter(min_cutoff_hz=4.0)
        filter_.step(0.0, 0.0)
        first = float(filter_.step(1.0, 0.01))
        alpha_1 = float(smoothing_factor(4.0, 0.01))
        assert_allclose(first, alpha_1)
        second = float(filter_.step(1.0, 0.05))
        alpha_2 = float(smoothing_factor(4.0, 0.04))
        assert_allclose(second, first + alpha_2 * (1.0 - first))

    def test_scalar_input(self) -> None:
        filter_ = OneEuroFilter()
        assert filter_.step(1.0, 0.0).shape == ()

    def test_result_is_a_copy(self) -> None:
        filter_ = OneEuroFilter()
        result = filter_.step([1.0], 0.0)
        result[0] = 100.0
        assert_allclose(filter_.step([1.0], DT), [1.0])

    def test_does_not_keep_reference_to_input(self) -> None:
        filter_ = OneEuroFilter()
        sample = np.array([1.0])
        filter_.step(sample, 0.0)
        sample[0] = 100.0
        assert_allclose(filter_.step([1.0], DT), [1.0])

    def test_rejects_shape_change(self) -> None:
        filter_ = OneEuroFilter()
        filter_.step([1.0, 2.0], 0.0)
        with pytest.raises(ValueError, match="does not match filter shape"):
            filter_.step([1.0, 2.0, 3.0], DT)

    @pytest.mark.parametrize("timestamp", [0.0, -1.0, math.nan])
    def test_rejects_time_not_moving_forward(self, timestamp: float) -> None:
        filter_ = OneEuroFilter()
        filter_.step(0.0, 0.0)
        with pytest.raises(ValueError, match="timestamp"):
            filter_.step(1.0, timestamp)

    def test_rejects_non_finite_first_timestamp(self) -> None:
        with pytest.raises(ValueError, match="timestamp must be finite"):
            OneEuroFilter().step(0.0, math.inf)

    def test_reset_starts_fresh(self) -> None:
        filter_ = OneEuroFilter()
        filter_.step([0.0, 0.0], 0.0)
        filter_.step([1.0, 1.0], DT)
        filter_.reset()
        assert_allclose(filter_.step([5.0, 5.0, 5.0], 0.0), [5.0, 5.0, 5.0])

    @given(
        st.lists(st.floats(-100.0, 100.0), min_size=2, max_size=50),
        st.floats(0.1, 50.0),
        st.floats(0.0, 10.0),
    )
    def test_output_stays_within_input_range(
        self, values: list[float], min_cutoff: float, beta: float
    ) -> None:
        # Each step is a weighted average of the old output and the new input.
        output = run(OneEuroFilter(min_cutoff, beta), np.array(values))
        assert np.all(output >= min(values) - 1e-9)
        assert np.all(output <= max(values) + 1e-9)


class TestOrientationFilter:
    def test_first_sample_passes_through_normalized(self) -> None:
        result = OrientationOneEuroFilter().step([2.0, 0.0, 0.0, 0.0], 0.0)
        assert_allclose(result, [1.0, 0.0, 0.0, 0.0])

    def test_constant_orientation_stays_constant(self) -> None:
        filter_ = OrientationOneEuroFilter(1.0, 0.5)
        q = quat.from_euler_zyx([0.3, -0.4, 1.2])
        for i in range(100):
            result = filter_.step(q, i * DT)
        assert float(quat.angle_between(result, q)) < 1e-12

    @pytest.mark.parametrize("beta", [0.0, 0.8])
    def test_single_axis_matches_scalar_filter(self, beta: float) -> None:
        """Rotation about one fixed axis must behave exactly like the scalar filter on the angle."""
        angles = 0.8 * np.sin(np.linspace(0.0, 6.0, 1_000)) + 0.02 * np.cos(np.arange(1_000))
        scalar = OneEuroFilter(1.5, beta, 1.0)
        orientation = OrientationOneEuroFilter(1.5, beta, 1.0)
        for i, angle in enumerate(angles):
            expected = float(scalar.step(angle, i * DT))
            result = orientation.step(quat.from_axis_angle(Z_AXIS, angle), i * DT)
            assert_allclose(quat.to_rotation_vector(result)[2], expected, atol=1e-9)

    def test_sign_flips_in_input_do_not_disturb_output(self) -> None:
        smooth = OrientationOneEuroFilter(2.0)
        flipping = OrientationOneEuroFilter(2.0)
        for i in range(200):
            q = quat.from_axis_angle([1.0, 1.0, 0.0], 0.002 * i)
            expected = smooth.step(q, i * DT)
            result = flipping.step(q if i % 2 else -q, i * DT)
            assert float(quat.angle_between(result, expected)) < 1e-12

    def test_reduces_jitter_at_rest(self) -> None:
        rng = np.random.default_rng(seed=11)
        truth = quat.from_euler_zyx([0.2, 0.1, -0.5])
        filter_ = OrientationOneEuroFilter(min_cutoff_hz=1.0, beta=0.01)
        raw_errors, filtered_errors = [], []
        for i in range(2_000):
            noisy = quat.multiply(truth, quat.from_rotation_vector(rng.normal(scale=0.02, size=3)))
            result = filter_.step(noisy, i * DT)
            if i >= 500:
                raw_errors.append(float(quat.angle_between(noisy, truth)))
                filtered_errors.append(float(quat.angle_between(result, truth)))
        assert np.mean(filtered_errors) < 0.1 * np.mean(raw_errors)

    def test_output_is_unit_over_long_runs(self) -> None:
        rng = np.random.default_rng(seed=5)
        filter_ = OrientationOneEuroFilter(3.0, 0.2)
        q = quat.identity()
        for i in range(5_000):
            q = quat.integrate(q, rng.normal(scale=2.0, size=3), DT)
            result = filter_.step(q, i * DT)
        assert bool(quat.is_unit(result, atol=1e-12))

    def test_rejects_batch_input(self) -> None:
        with pytest.raises(ValueError, match="single quaternion"):
            OrientationOneEuroFilter().step(np.tile([1.0, 0.0, 0.0, 0.0], (2, 1)), 0.0)

    def test_rejects_time_not_moving_forward(self) -> None:
        filter_ = OrientationOneEuroFilter()
        filter_.step(quat.identity(), 1.0)
        with pytest.raises(ValueError, match="timestamp"):
            filter_.step(quat.identity(), 1.0)

    def test_reset_starts_fresh(self) -> None:
        filter_ = OrientationOneEuroFilter()
        filter_.step(quat.identity(), 0.0)
        filter_.step(quat.identity(), DT)
        filter_.reset()
        q = quat.from_euler_zyx([1.0, 0.0, 0.0])
        assert_allclose(filter_.step(q, 0.0), q)
