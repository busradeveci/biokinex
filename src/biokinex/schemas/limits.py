"""Physical limits that a plausible sensor reading must respect.

The limits live in the schema layer, so the pipeline configuration can include them and load them
from TOML. The validator reads them from here.
"""

from __future__ import annotations

import math
from typing import Final, Self

from pydantic import model_validator

from biokinex.core.types import STANDARD_GRAVITY
from biokinex.schemas.common import FiniteFloat, FrozenModel, PositiveFloat

__all__ = ["SensorLimits"]

_DEFAULT_ACCEL_RANGE_G: Final = 16.0
_DEFAULT_GYRO_RANGE_DPS: Final = 2000.0


class SensorLimits(FrozenModel):
    """Largest physically plausible readings, in SI units.

    Acceleration and angular velocity limits apply to each axis on its own, because MEMS sensors
    specify and saturate their full-scale range per axis. The defaults match a common ±16 g,
    ±2000 °/s IMU.
    """

    max_acceleration: PositiveFloat = _DEFAULT_ACCEL_RANGE_G * STANDARD_GRAVITY
    """Largest absolute acceleration per axis, in m/s²."""

    max_angular_velocity: PositiveFloat = math.radians(_DEFAULT_GYRO_RANGE_DPS)
    """Largest absolute angular velocity per axis, in rad/s."""

    min_joint_angle: FiniteFloat = -math.pi
    """Smallest accepted joint angle, in radians."""

    max_joint_angle: FiniteFloat = math.pi
    """Largest accepted joint angle, in radians."""

    @model_validator(mode="after")
    def _check_joint_range(self) -> Self:
        if self.min_joint_angle >= self.max_joint_angle:
            msg = "min_joint_angle must be smaller than max_joint_angle"
            raise ValueError(msg)
        return self

    @classmethod
    def from_datasheet(
        cls,
        accel_range_g: float,
        gyro_range_dps: float,
        *,
        min_joint_angle: float = -math.pi,
        max_joint_angle: float = math.pi,
    ) -> SensorLimits:
        """Build limits from the full-scale ranges printed in a sensor datasheet."""
        return cls(
            max_acceleration=accel_range_g * STANDARD_GRAVITY,
            max_angular_velocity=math.radians(gyro_range_dps),
            min_joint_angle=min_joint_angle,
            max_joint_angle=max_joint_angle,
        )
