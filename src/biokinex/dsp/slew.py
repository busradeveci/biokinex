"""Slew-rate limiters: a final safety stage that caps how fast an output may change.

Even when an upstream filter is tuned badly or a sensor glitches, a limiter guarantees that the
command sent to an actuator never jumps. Large changes are turned into ramps at the maximum rate.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from biokinex.core import scalar_quaternion as sq
from biokinex.core.types import FloatArray, Quaternion
from biokinex.dsp.base import check_timestamp, elapsed
from biokinex.schemas.config import SlewRateConfig

__all__ = ["OrientationSlewRateLimiter", "SlewRateLimiter"]


def _check_max_rate(max_rate: float) -> None:
    if not (math.isfinite(max_rate) and max_rate > 0.0):
        msg = f"max_rate must be finite and positive, got {max_rate!r}"
        raise ValueError(msg)


class SlewRateLimiter:
    """Limits each element of a vector to change by at most ``max_rate`` units per second.

    Each element is limited on its own, because actuators such as joints have separate limits.
    """

    __slots__ = ("_max_rate", "_timestamp", "_value")

    def __init__(self, max_rate: float) -> None:
        _check_max_rate(max_rate)
        self._max_rate = max_rate
        self._value: FloatArray | None = None
        self._timestamp = -math.inf

    @classmethod
    def from_config(cls, config: SlewRateConfig) -> SlewRateLimiter:
        """Build a limiter from validated configuration."""
        return cls(config.max_rate)

    @property
    def max_rate(self) -> float:
        """Largest allowed change per second for each element."""
        return self._max_rate

    def step(self, value: npt.ArrayLike, timestamp: float) -> FloatArray:
        """Limit one sample and return a new array with the limited value."""
        sample = np.array(value, dtype=np.float64)
        if self._value is None:
            # With no previous output there is nothing to limit against.
            self._timestamp = check_timestamp(timestamp)
            self._value = sample
            return sample.copy()
        if sample.shape != self._value.shape:
            msg = f"sample shape {sample.shape} does not match filter shape {self._value.shape}"
            raise ValueError(msg)
        max_step = self._max_rate * elapsed(self._timestamp, timestamp)
        self._value += np.clip(sample - self._value, -max_step, max_step)
        self._timestamp = timestamp
        return self._value.copy()

    def reset(self) -> None:
        """Forget the previous output; the next sample passes through unchanged."""
        self._value = None
        self._timestamp = -math.inf


class OrientationSlewRateLimiter:
    """Limits the angular speed of an orientation stream to ``max_rate`` rad/s.

    The rotation angle between outputs is limited, not each component, so the axis of motion is
    kept and only its speed is reduced.
    """

    __slots__ = ("_max_rate", "_orientation", "_timestamp")

    def __init__(self, max_rate: float) -> None:
        _check_max_rate(max_rate)
        self._max_rate = max_rate
        self._orientation: sq.Quat | None = None
        self._timestamp = -math.inf

    @classmethod
    def from_config(cls, config: SlewRateConfig) -> OrientationSlewRateLimiter:
        """Build a limiter from validated configuration."""
        return cls(config.max_rate)

    @property
    def max_rate(self) -> float:
        """Largest allowed angular speed in rad/s."""
        return self._max_rate

    def step(self, value: npt.ArrayLike, timestamp: float) -> Quaternion:
        """Limit one orientation and return a new unit quaternion."""
        sample = sq.normalize(sq.from_array(value))
        if self._orientation is None:
            self._timestamp = check_timestamp(timestamp)
            self._orientation = sample
            return sq.to_array(sample)

        max_angle = self._max_rate * elapsed(self._timestamp, timestamp)
        dx, dy, dz = sq.relative_rotation_vector(self._orientation, sample)
        angle = math.sqrt(dx * dx + dy * dy + dz * dz)
        if angle > max_angle:
            # Shortening the rotation vector keeps the direction of motion and only slows it down.
            scale = max_angle / angle
            self._orientation = sq.apply_rotation_vector(
                self._orientation, (dx * scale, dy * scale, dz * scale)
            )
        else:
            # Matching the previous sign keeps the output components free of q / -q jumps.
            pw, px, py, pz = self._orientation
            sw, sx, sy, sz = sample
            same_side = pw * sw + px * sx + py * sy + pz * sz >= 0.0
            self._orientation = sample if same_side else (-sw, -sx, -sy, -sz)
        self._timestamp = timestamp
        return sq.to_array(self._orientation)

    def reset(self) -> None:
        """Forget the previous output; the next orientation passes through unchanged."""
        self._orientation = None
        self._timestamp = -math.inf
