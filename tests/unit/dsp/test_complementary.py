"""Tests for biokinex.dsp.complementary."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_allclose

from biokinex.core import quaternion as quat
from biokinex.core.types import STANDARD_GRAVITY, FloatArray
from biokinex.dsp.complementary import ComplementaryFilter, tilt_from_up
from biokinex.schemas.config import ComplementaryConfig

RATE_HZ = 500.0
DT = 1.0 / RATE_HZ
ZERO = np.zeros(3)
G = STANDARD_GRAVITY


def gravity_in_body(roll: float, pitch: float, yaw: float = 0.0) -> FloatArray:
    """Return what a resting accelerometer reads for the given orientation."""
    q = quat.from_euler_zyx([roll, pitch, yaw])
    return quat.rotate_vector(quat.conjugate(q), [0.0, 0.0, G])


def euler(filter_: ComplementaryFilter) -> FloatArray:
    """Return the current estimate as ``(roll, pitch, yaw)``."""
    orientation = filter_.orientation
    assert orientation is not None
    return quat.to_euler_zyx(orientation)


def run(
    filter_: ComplementaryFilter,
    seconds: float,
    gyro: FloatArray,
    accel: FloatArray | None,
    start: float = 0.0,
) -> float:
    """Feed constant readings for ``seconds`` and return the last timestamp."""
    timestamp = start
    for _ in range(round(seconds * RATE_HZ)):
        timestamp += DT
        filter_.step(gyro, accel, timestamp)
    return timestamp


class TestTiltFromUp:
    @given(st.floats(-3.0, 3.0), st.floats(-1.5, 1.5))
    def test_recovers_roll_and_pitch(self, roll: float, pitch: float) -> None:
        up = gravity_in_body(roll, pitch) / G
        assert_allclose(quat.to_euler_zyx(tilt_from_up(up)), [roll, pitch, 0.0], atol=1e-9)

    def test_level_sensor_is_identity(self) -> None:
        assert_allclose(tilt_from_up([0.0, 0.0, 1.0]), quat.identity(), atol=1e-15)


class TestInitialization:
    def test_starts_from_measured_tilt(self) -> None:
        filter_ = ComplementaryFilter()
        filter_.step(ZERO, gravity_in_body(0.3, -0.2), 0.0)
        assert_allclose(euler(filter_), [0.3, -0.2, 0.0], atol=1e-9)
        assert filter_.accel_trusted

    @pytest.mark.parametrize(
        "accel", [None, np.zeros(3), np.array([0.0, 0.0, 2.0 * G])], ids=["none", "zero", "2g"]
    )
    def test_starts_level_without_trusted_accel(self, accel: FloatArray | None) -> None:
        filter_ = ComplementaryFilter()
        assert_allclose(filter_.step(ZERO, accel, 0.0), quat.identity())
        assert not filter_.accel_trusted

    def test_orientation_is_none_before_first_sample(self) -> None:
        assert ComplementaryFilter().orientation is None

    def test_from_config(self) -> None:
        filter_ = ComplementaryFilter.from_config(
            ComplementaryConfig(gain=3.0, accel_tolerance=0.2)
        )
        assert (filter_._gain, filter_._accel_tolerance) == (3.0, 0.2)


class TestFusion:
    def test_gain_zero_is_pure_gyro_integration(self) -> None:
        filter_ = ComplementaryFilter(gain=0.0)
        start = filter_.step(ZERO, gravity_in_body(0.0, 0.0), 0.0)
        omega = np.array([0.3, -0.1, 0.7])
        result = filter_.step(omega, gravity_in_body(0.5, 0.5), DT)
        assert_allclose(result, quat.integrate(start, omega, DT), atol=1e-15)

    def test_converges_to_accelerometer_tilt(self) -> None:
        gain = 2.0
        filter_ = ComplementaryFilter(gain=gain)
        filter_.step(ZERO, gravity_in_body(0.0, 0.0), 0.0)
        tilted = gravity_in_body(0.3, 0.0)
        timestamp = run(filter_, 1.0 / gain, ZERO, tilted)
        # After one time constant about 63% of the error is gone, as for a first-order system.
        assert 0.15 < euler(filter_)[0] < 0.22
        run(filter_, 5.0 / gain, ZERO, tilted, start=timestamp)
        assert_allclose(euler(filter_), [0.3, 0.0, 0.0], atol=3e-3)

    def test_gyro_bias_gives_known_steady_state_error(self) -> None:
        # With proportional correction only, a bias b leaves a tilt error of asin(b / gain).
        gain, bias = 1.0, 0.05
        filter_ = ComplementaryFilter(gain=gain)
        filter_.step(ZERO, gravity_in_body(0.0, 0.0), 0.0)
        run(filter_, 20.0, np.array([bias, 0.0, 0.0]), gravity_in_body(0.0, 0.0))
        assert_allclose(euler(filter_)[0], math.asin(bias / gain), rtol=1e-3)

    def test_without_correction_gyro_bias_drifts(self) -> None:
        filter_ = ComplementaryFilter(gain=0.0)
        filter_.step(ZERO, gravity_in_body(0.0, 0.0), 0.0)
        run(filter_, 2.0, np.array([0.05, 0.0, 0.0]), gravity_in_body(0.0, 0.0))
        assert_allclose(euler(filter_)[0], 0.1, rtol=1e-6)

    def test_yaw_follows_gyro_only(self) -> None:
        filter_ = ComplementaryFilter(gain=5.0)
        filter_.step(ZERO, gravity_in_body(0.0, 0.0), 0.0)
        run(filter_, 1.0, np.array([0.0, 0.0, 0.5]), gravity_in_body(0.0, 0.0))
        assert_allclose(euler(filter_), [0.0, 0.0, 0.5], atol=1e-9)

    def test_tracks_rotation_while_correcting(self) -> None:
        """Rolling at a known rate with matching accelerometer data keeps the error small."""
        filter_ = ComplementaryFilter(gain=2.0)
        rate = 0.5
        filter_.step(ZERO, gravity_in_body(0.0, 0.0), 0.0)
        for i in range(1, 1_001):
            filter_.step([rate, 0.0, 0.0], gravity_in_body(rate * i * DT, 0.0), i * DT)
        assert_allclose(euler(filter_)[0], rate * 1_000 * DT, atol=1e-3)

    def test_ignores_accelerometer_during_strong_motion(self) -> None:
        filter_ = ComplementaryFilter(gain=5.0, accel_tolerance=0.1)
        start = filter_.step(ZERO, gravity_in_body(0.0, 0.0), 0.0)
        omega = np.array([0.0, 0.2, 0.0])
        result = filter_.step(omega, np.array([3.0, 0.0, 1.5 * G]), DT)
        assert not filter_.accel_trusted
        assert_allclose(result, quat.integrate(start, omega, DT), atol=1e-15)

    def test_missing_accelerometer_uses_gyro_only(self) -> None:
        filter_ = ComplementaryFilter(gain=5.0)
        start = filter_.step(ZERO, gravity_in_body(0.2, 0.0), 0.0)
        omega = np.array([0.1, 0.0, 0.0])
        assert_allclose(filter_.step(omega, None, DT), quat.integrate(start, omega, DT))

    def test_result_is_a_copy(self) -> None:
        filter_ = ComplementaryFilter()
        result = filter_.step(ZERO, gravity_in_body(0.0, 0.0), 0.0)
        result[:] = 0.0
        assert_allclose(filter_.orientation, quat.identity())  # type: ignore[arg-type]

    def test_reset_starts_again_from_measured_tilt(self) -> None:
        filter_ = ComplementaryFilter()
        filter_.step(ZERO, gravity_in_body(0.0, 0.0), 0.0)
        filter_.reset()
        assert filter_.orientation is None
        filter_.step(ZERO, gravity_in_body(0.4, 0.0), 0.0)
        assert_allclose(euler(filter_), [0.4, 0.0, 0.0], atol=1e-9)


class TestInputChecks:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"gain": -1.0},
            {"gain": math.inf},
            {"accel_tolerance": 0.0},
            {"accel_tolerance": 1.5},
            {"gravity": 0.0},
        ],
    )
    def test_rejects_invalid_settings(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ValueError, match="must be"):
            ComplementaryFilter(**kwargs)

    @pytest.mark.parametrize(
        ("gyro", "accel", "name"),
        [([0.0, 0.0], None, "angular_velocity"), (ZERO, [0.0, 9.8], "acceleration")],
    )
    def test_rejects_wrong_vector_shape(
        self, gyro: list[float], accel: list[float] | None, name: str
    ) -> None:
        with pytest.raises(ValueError, match=f"{name} must have shape"):
            ComplementaryFilter().step(gyro, accel, 0.0)

    def test_rejects_time_not_moving_forward(self) -> None:
        filter_ = ComplementaryFilter()
        filter_.step(ZERO, None, 1.0)
        with pytest.raises(ValueError, match="timestamp"):
            filter_.step(ZERO, None, 0.5)
