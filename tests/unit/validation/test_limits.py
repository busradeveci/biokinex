"""Tests for biokinex.validation.limits."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from biokinex.core.types import STANDARD_GRAVITY
from biokinex.validation.limits import SensorLimits


def test_defaults_match_16g_2000dps_imu() -> None:
    limits = SensorLimits()
    assert limits.max_acceleration == pytest.approx(16.0 * STANDARD_GRAVITY)
    assert limits.max_angular_velocity == pytest.approx(math.radians(2000.0))
    assert limits.min_joint_angle == -math.pi
    assert limits.max_joint_angle == math.pi


def test_from_datasheet_converts_to_si() -> None:
    limits = SensorLimits.from_datasheet(8.0, 500.0, min_joint_angle=0.0, max_joint_angle=2.6)
    assert limits.max_acceleration == pytest.approx(8.0 * STANDARD_GRAVITY)
    assert limits.max_angular_velocity == pytest.approx(math.radians(500.0))
    assert (limits.min_joint_angle, limits.max_joint_angle) == (0.0, 2.6)


@pytest.mark.parametrize(
    "raw",
    [
        {"max_acceleration": 0.0},
        {"max_angular_velocity": -1.0},
        {"max_acceleration": math.inf},
        {"min_joint_angle": 1.0, "max_joint_angle": 1.0},
        {"min_joint_angle": 2.0, "max_joint_angle": 1.0},
        {"max_temperature": 40.0},
    ],
)
def test_rejects_invalid_limits(raw: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        SensorLimits.model_validate(raw)


def test_joint_range_message() -> None:
    with pytest.raises(ValidationError, match="min_joint_angle must be smaller"):
        SensorLimits(min_joint_angle=1.0, max_joint_angle=0.0)
