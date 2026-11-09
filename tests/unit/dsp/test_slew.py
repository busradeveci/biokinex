"""Tests for biokinex.dsp.slew."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_allclose

from biokinex.core import quaternion as quat
from biokinex.dsp.slew import OrientationSlewRateLimiter, SlewRateLimiter
from biokinex.schemas.config import SlewRateConfig

DT = 0.002
Z_AXIS = np.array([0.0, 0.0, 1.0])
X_AXIS = np.array([1.0, 0.0, 0.0])

finite = st.floats(min_value=-100.0, max_value=100.0)
quaternions = st.lists(st.floats(min_value=-1.0, max_value=1.0), min_size=4, max_size=4).filter(
    lambda q: np.linalg.norm(q) > 0.1
)


class TestParameters:
    @pytest.mark.parametrize("limiter", [SlewRateLimiter, OrientationSlewRateLimiter])
    @pytest.mark.parametrize("max_rate", [0.0, -1.0, math.inf, math.nan])
    def test_rejects_bad_rate(self, limiter: type[Any], max_rate: float) -> None:
        with pytest.raises(ValueError, match="max_rate"):
            limiter(max_rate)

    @pytest.mark.parametrize("limiter", [SlewRateLimiter, OrientationSlewRateLimiter])
    def test_from_config(self, limiter: type[Any]) -> None:
        assert limiter.from_config(SlewRateConfig(max_rate=3.5)).max_rate == 3.5


class TestSlewRateLimiter:
    def test_first_sample_passes_through(self) -> None:
        assert_allclose(SlewRateLimiter(1.0).step([5.0, -5.0], 0.0), [5.0, -5.0])

    def test_small_changes_pass_unchanged(self) -> None:
        limiter = SlewRateLimiter(10.0)
        limiter.step(0.0, 0.0)
        assert limiter.step(0.015, DT) == pytest.approx(0.015)

    def test_step_becomes_a_ramp(self) -> None:
        limiter = SlewRateLimiter(100.0)
        limiter.step(0.0, 0.0)
        outputs = [float(limiter.step(1.0, i * DT)) for i in range(1, 8)]
        assert_allclose(outputs, [0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0])

    def test_each_element_has_its_own_limit(self) -> None:
        limiter = SlewRateLimiter(100.0)
        limiter.step([0.0, 0.0, 0.0], 0.0)
        assert_allclose(limiter.step([1.0, -1.0, 0.1], DT), [0.2, -0.2, 0.1])

    def test_limit_scales_with_elapsed_time(self) -> None:
        limiter = SlewRateLimiter(100.0)
        limiter.step(0.0, 0.0)
        assert limiter.step(10.0, 0.05) == pytest.approx(5.0)

    @given(st.lists(st.tuples(finite, st.floats(min_value=1e-4, max_value=0.1)), max_size=40))
    def test_change_never_exceeds_limit(self, steps: list[tuple[float, float]]) -> None:
        limiter = SlewRateLimiter(50.0)
        previous = float(limiter.step(0.0, 0.0))
        time = 0.0
        for value, dt in steps:
            time += dt
            current = float(limiter.step(value, time))
            assert abs(current - previous) <= 50.0 * dt * (1 + 1e-9) + 1e-12
            previous = current

    def test_output_is_a_copy(self) -> None:
        limiter = SlewRateLimiter(100.0)
        first = limiter.step([0.0], 0.0)
        first[:] = 50.0
        assert_allclose(limiter.step([0.0], DT), [0.0])

    def test_rejects_shape_change(self) -> None:
        limiter = SlewRateLimiter(1.0)
        limiter.step([0.0, 0.0], 0.0)
        with pytest.raises(ValueError, match="does not match"):
            limiter.step([0.0], DT)

    def test_rejects_time_that_does_not_move_forward(self) -> None:
        limiter = SlewRateLimiter(1.0)
        limiter.step(0.0, 1.0)
        with pytest.raises(ValueError, match="timestamp must be finite and later"):
            limiter.step(0.0, 1.0)

    def test_reset_passes_next_sample_through(self) -> None:
        limiter = SlewRateLimiter(1.0)
        limiter.step(0.0, 0.0)
        limiter.reset()
        assert limiter.step(100.0, 0.0) == pytest.approx(100.0)


class TestOrientationSlewRateLimiter:
    def test_first_orientation_passes_through_normalized(self) -> None:
        output = OrientationSlewRateLimiter(1.0).step([2.0, 0.0, 0.0, 0.0], 0.0)
        assert_allclose(output, [1.0, 0.0, 0.0, 0.0])

    def test_large_rotation_is_limited_along_its_axis(self) -> None:
        limiter = OrientationSlewRateLimiter(math.pi)
        limiter.step(quat.identity(), 0.0)
        target = quat.from_axis_angle(Z_AXIS, 1.0)
        output = limiter.step(target, 0.1)
        assert_allclose(output, quat.from_axis_angle(Z_AXIS, 0.1 * math.pi), atol=1e-12)

    def test_small_rotation_passes_unchanged(self) -> None:
        limiter = OrientationSlewRateLimiter(10.0)
        limiter.step(quat.identity(), 0.0)
        target = quat.from_axis_angle(X_AXIS, 0.01)
        assert_allclose(limiter.step(target, DT), target, atol=1e-15)

    def test_keeps_previous_hemisphere(self) -> None:
        limiter = OrientationSlewRateLimiter(10.0)
        limiter.step(quat.identity(), 0.0)
        target = quat.from_axis_angle(X_AXIS, 0.01)
        assert_allclose(limiter.step(-target, DT), target, atol=1e-15)

    def test_reaches_target_after_enough_time(self) -> None:
        limiter = OrientationSlewRateLimiter(1.0)
        limiter.step(quat.identity(), 0.0)
        target = quat.from_axis_angle(Z_AXIS, 0.5)
        outputs = [limiter.step(target, i * 0.1) for i in range(1, 10)]
        assert quat.angle_between(outputs[3], target) == pytest.approx(0.1, abs=1e-12)
        assert quat.angle_between(outputs[-1], target) == pytest.approx(0.0, abs=1e-7)

    @given(st.lists(st.tuples(quaternions, st.floats(min_value=1e-4, max_value=0.1)), max_size=25))
    def test_angular_speed_never_exceeds_limit(
        self, steps: list[tuple[list[float], float]]
    ) -> None:
        limiter = OrientationSlewRateLimiter(2.0)
        previous = limiter.step(quat.identity(), 0.0)
        time = 0.0
        for sample, dt in steps:
            time += dt
            current = limiter.step(sample, time)
            assert quat.is_unit(current)
            assert quat.angle_between(previous, current) <= 2.0 * dt + 1e-7
            previous = current

    def test_rejects_wrong_shape(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            OrientationSlewRateLimiter(1.0).step([1.0, 0.0, 0.0], 0.0)

    def test_rejects_time_that_does_not_move_forward(self) -> None:
        limiter = OrientationSlewRateLimiter(1.0)
        limiter.step(quat.identity(), 1.0)
        with pytest.raises(ValueError, match="timestamp must be finite and later"):
            limiter.step(quat.identity(), 0.5)

    def test_reset_passes_next_orientation_through(self) -> None:
        limiter = OrientationSlewRateLimiter(0.1)
        limiter.step(quat.identity(), 0.0)
        limiter.reset()
        target = quat.from_axis_angle(Z_AXIS, 2.0)
        assert_allclose(limiter.step(target, 0.0), target, atol=1e-15)
