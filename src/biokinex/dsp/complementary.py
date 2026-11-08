"""Mahony-style complementary filter for attitude (tilt) estimation.

Reference: R. Mahony, T. Hamel, J.-M. Pflimlin, "Nonlinear Complementary Filters on the Special
Orthogonal Group", IEEE Transactions on Automatic Control, 2008. This is the proportional-only form.

The gyroscope is accurate over short times but drifts. The accelerometer measures the direction of
gravity, which is noisy but does not drift. Each step integrates the gyroscope and adds a small
correction rate that turns the estimated "up" direction towards the measured one:

    omega_corrected = omega + gain * (measured_up x estimated_up)

Roll and pitch converge with a time constant of about ``1 / gain`` seconds. Yaw is not observable
from gravity, so it follows the gyroscope alone and will drift slowly.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np
import numpy.typing as npt

from biokinex.core import quaternion
from biokinex.core import scalar_quaternion as sq
from biokinex.core.types import STANDARD_GRAVITY, Quaternion
from biokinex.dsp.base import check_timestamp, elapsed
from biokinex.schemas.config import ComplementaryConfig

__all__ = ["ComplementaryFilter", "tilt_from_up"]

_MIN_ACCEL_NORM: Final = 1e-9


def tilt_from_up(up: npt.ArrayLike) -> Quaternion:
    """Return the orientation with zero yaw whose body-frame "up" direction is ``up``.

    At rest an accelerometer measures ``+g`` along world up, so a normalized reading can be passed
    directly.
    """
    x, y, z = np.asarray(up, dtype=np.float64)
    roll = math.atan2(y, z)
    pitch = math.atan2(-x, math.hypot(y, z))
    return quaternion.from_euler_zyx([roll, pitch, 0.0])


class ComplementaryFilter:
    """Fuses body-frame angular velocity and acceleration into a unit quaternion.

    Args:
        gain: Correction gain in 1/s. Zero disables the accelerometer correction.
        accel_tolerance: The accelerometer is ignored for a sample when its magnitude differs from
            ``gravity`` by more than this fraction, because it then measures motion, not tilt.
        gravity: Expected magnitude of gravity in m/s².
    """

    __slots__ = (
        "_accel_tolerance",
        "_accel_trusted",
        "_gain",
        "_gravity",
        "_orientation",
        "_timestamp",
    )

    def __init__(
        self,
        gain: float = 1.0,
        accel_tolerance: float = 0.1,
        gravity: float = STANDARD_GRAVITY,
    ) -> None:
        if not (math.isfinite(gain) and gain >= 0.0):
            msg = f"gain must be finite and non-negative, got {gain!r}"
            raise ValueError(msg)
        if not 0.0 < accel_tolerance <= 1.0:
            msg = f"accel_tolerance must be in (0, 1], got {accel_tolerance!r}"
            raise ValueError(msg)
        if not (math.isfinite(gravity) and gravity > 0.0):
            msg = f"gravity must be finite and positive, got {gravity!r}"
            raise ValueError(msg)
        self._gain = gain
        self._accel_tolerance = accel_tolerance
        self._gravity = gravity
        self._orientation: sq.Quat | None = None
        self._timestamp = -math.inf
        self._accel_trusted = False

    @classmethod
    def from_config(cls, config: ComplementaryConfig) -> ComplementaryFilter:
        """Build a filter from validated configuration."""
        return cls(gain=config.gain, accel_tolerance=config.accel_tolerance)

    @property
    def orientation(self) -> Quaternion | None:
        """A copy of the current estimate, or ``None`` before the first sample."""
        return None if self._orientation is None else sq.to_array(self._orientation)

    @property
    def accel_trusted(self) -> bool:
        """Whether the accelerometer passed the gravity check on the last sample."""
        return self._accel_trusted

    def step(
        self,
        angular_velocity: npt.ArrayLike,
        acceleration: npt.ArrayLike | None,
        timestamp: float,
    ) -> Quaternion:
        """Update the estimate with one sample and return a copy of it.

        Args:
            angular_velocity: Body-frame angular velocity in rad/s, shape ``(3,)``.
            acceleration: Body-frame specific force in m/s², shape ``(3,)``, or ``None`` if the
                sample has no accelerometer data.
            timestamp: Sensor time in seconds.
        """
        # Scalar math on tuples is used here because this runs once per sample on one quaternion.
        wx, wy, wz = sq.vector_from_array(angular_velocity, "angular_velocity")
        measured_up = self._measured_up(acceleration)
        self._accel_trusted = measured_up is not None

        if self._orientation is None:
            self._timestamp = check_timestamp(timestamp)
            # Starting from the measured tilt avoids a slow convergence from an arbitrary guess.
            start = quaternion.identity() if measured_up is None else tilt_from_up(measured_up)
            self._orientation = sq.from_array(start)
            return sq.to_array(self._orientation)

        dt = elapsed(self._timestamp, timestamp)
        if measured_up is not None and self._gain > 0.0:
            mx, my, mz = measured_up
            ex, ey, ez = sq.body_up(self._orientation)
            # The cross product points along the axis that turns the estimate towards the
            # measurement, and its length grows with the tilt error.
            wx += self._gain * (my * ez - mz * ey)
            wy += self._gain * (mz * ex - mx * ez)
            wz += self._gain * (mx * ey - my * ex)
        self._orientation = sq.integrate(self._orientation, (wx, wy, wz), dt)
        self._timestamp = timestamp
        return sq.to_array(self._orientation)

    def reset(self) -> None:
        """Forget the estimate; the next sample starts again from the measured tilt."""
        self._orientation = None
        self._timestamp = -math.inf
        self._accel_trusted = False

    def _measured_up(self, acceleration: npt.ArrayLike | None) -> sq.Vec3 | None:
        """Return the normalized accelerometer reading, or ``None`` if it cannot be trusted."""
        if acceleration is None:
            return None
        ax, ay, az = sq.vector_from_array(acceleration, "acceleration")
        norm = math.sqrt(ax * ax + ay * ay + az * az)
        if norm < _MIN_ACCEL_NORM:
            return None
        if abs(norm - self._gravity) > self._accel_tolerance * self._gravity:
            return None
        return (ax / norm, ay / norm, az / norm)
