"""The 1€ (One-Euro) filter for vectors and for orientations.

Reference: G. Casiez, N. Roussel, D. Vogel, "1€ Filter: A Simple Speed-based Low-pass Filter for
Noisy Input in Interactive Systems", CHI 2012.

The filter is a first-order low-pass whose cutoff rises with the estimated speed of the signal::

    cutoff = min_cutoff + beta * |speed|

At rest the cutoff is low, which removes jitter and tremor. During fast motion the cutoff rises,
which keeps lag small. ``beta = 0`` gives a plain first-order low-pass at ``min_cutoff``.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from biokinex.core import scalar_quaternion as sq
from biokinex.core.types import FloatArray, Quaternion
from biokinex.dsp.base import check_timestamp, elapsed
from biokinex.schemas.config import OneEuroConfig

__all__ = ["OneEuroFilter", "OrientationOneEuroFilter", "smoothing_factor"]


def smoothing_factor(cutoff_hz: npt.ArrayLike, dt: float) -> FloatArray:
    """Return the weight of a new sample in a first-order low-pass, in ``(0, 1)``.

    This is the exact discrete form of an RC filter with time constant ``1 / (2 pi cutoff)``,
    so it stays correct when ``dt`` varies between samples.
    """
    r = 2.0 * math.pi * np.asarray(cutoff_hz, dtype=np.float64) * dt
    return np.asarray(r / (1.0 + r))


def _check_parameters(min_cutoff_hz: float, beta: float, derivative_cutoff_hz: float) -> None:
    """Reject tuning values that would make the filter unstable or meaningless."""
    if not (math.isfinite(min_cutoff_hz) and min_cutoff_hz > 0.0):
        msg = f"min_cutoff_hz must be finite and positive, got {min_cutoff_hz!r}"
        raise ValueError(msg)
    if not (math.isfinite(beta) and beta >= 0.0):
        msg = f"beta must be finite and non-negative, got {beta!r}"
        raise ValueError(msg)
    if not (math.isfinite(derivative_cutoff_hz) and derivative_cutoff_hz > 0.0):
        msg = f"derivative_cutoff_hz must be finite and positive, got {derivative_cutoff_hz!r}"
        raise ValueError(msg)


class OneEuroFilter:
    """One-Euro filter for a fixed-shape vector; every element is filtered on its own.

    The shape is fixed by the first sample after construction or :meth:`reset`.
    """

    __slots__ = (
        "_beta",
        "_derivative",
        "_derivative_cutoff_hz",
        "_min_cutoff_hz",
        "_previous_sample",
        "_timestamp",
        "_value",
    )

    def __init__(
        self,
        min_cutoff_hz: float = 1.0,
        beta: float = 0.0,
        derivative_cutoff_hz: float = 1.0,
    ) -> None:
        _check_parameters(min_cutoff_hz, beta, derivative_cutoff_hz)
        self._min_cutoff_hz = min_cutoff_hz
        self._beta = beta
        self._derivative_cutoff_hz = derivative_cutoff_hz
        self._value: FloatArray | None = None
        self._previous_sample: FloatArray | None = None
        self._derivative: FloatArray | None = None
        self._timestamp = -math.inf

    @classmethod
    def from_config(cls, config: OneEuroConfig) -> OneEuroFilter:
        """Build a filter from validated configuration."""
        return cls(config.min_cutoff_hz, config.beta, config.derivative_cutoff_hz)

    def step(self, value: npt.ArrayLike, timestamp: float) -> FloatArray:
        """Filter one sample and return a new array with the filtered value."""
        sample = np.array(value, dtype=np.float64)
        if self._value is None or self._previous_sample is None or self._derivative is None:
            # The first sample has no history to smooth against, so it passes through unchanged.
            self._timestamp = check_timestamp(timestamp)
            self._value = sample.copy()
            self._previous_sample = sample
            self._derivative = np.zeros_like(sample)
            return sample.copy()
        if sample.shape != self._value.shape:
            msg = f"sample shape {sample.shape} does not match filter shape {self._value.shape}"
            raise ValueError(msg)

        dt = elapsed(self._timestamp, timestamp)
        # Speed comes from consecutive raw samples, as in the reference implementation, so beta
        # scales the real rate of change of the signal.
        raw_derivative = (sample - self._previous_sample) / dt
        self._derivative += smoothing_factor(self._derivative_cutoff_hz, dt) * (
            raw_derivative - self._derivative
        )
        cutoff = self._min_cutoff_hz + self._beta * np.abs(self._derivative)
        self._value += smoothing_factor(cutoff, dt) * (sample - self._value)
        self._previous_sample = sample
        self._timestamp = timestamp
        return self._value.copy()

    def reset(self) -> None:
        """Forget all history; the next sample passes through unchanged."""
        self._value = None
        self._previous_sample = None
        self._derivative = None
        self._timestamp = -math.inf


class OrientationOneEuroFilter:
    """One-Euro filter for unit quaternions, applied on rotation increments.

    Each step measures the rotation from the filtered orientation to the new sample and moves the
    filtered orientation part of the way along that shortest arc (a slerp with the One-Euro
    weight). The speed term uses the size of the angular velocity, not each axis on its own,
    because rotation axes are coupled.
    """

    __slots__ = (
        "_angular_velocity",
        "_beta",
        "_derivative_cutoff_hz",
        "_min_cutoff_hz",
        "_orientation",
        "_previous_sample",
        "_timestamp",
    )

    def __init__(
        self,
        min_cutoff_hz: float = 1.0,
        beta: float = 0.0,
        derivative_cutoff_hz: float = 1.0,
    ) -> None:
        _check_parameters(min_cutoff_hz, beta, derivative_cutoff_hz)
        self._min_cutoff_hz = min_cutoff_hz
        self._beta = beta
        self._derivative_cutoff_hz = derivative_cutoff_hz
        self._orientation: sq.Quat | None = None
        self._previous_sample: sq.Quat | None = None
        self._angular_velocity: sq.Vec3 = (0.0, 0.0, 0.0)
        self._timestamp = -math.inf

    @classmethod
    def from_config(cls, config: OneEuroConfig) -> OrientationOneEuroFilter:
        """Build a filter from validated configuration."""
        return cls(config.min_cutoff_hz, config.beta, config.derivative_cutoff_hz)

    def step(self, value: npt.ArrayLike, timestamp: float) -> Quaternion:
        """Filter one orientation and return a new unit quaternion."""
        # Scalar math on tuples is used here because this runs once per sample on one quaternion.
        sample = sq.normalize(sq.from_array(value))
        if self._orientation is None or self._previous_sample is None:
            self._timestamp = check_timestamp(timestamp)
            self._orientation = sample
            self._previous_sample = sample
            return sq.to_array(sample)

        dt = elapsed(self._timestamp, timestamp)
        rx, ry, rz = sq.relative_rotation_vector(self._previous_sample, sample)
        derivative_weight = _smoothing_weight(self._derivative_cutoff_hz, dt)
        vx, vy, vz = self._angular_velocity
        vx += derivative_weight * (rx / dt - vx)
        vy += derivative_weight * (ry / dt - vy)
        vz += derivative_weight * (rz / dt - vz)
        self._angular_velocity = (vx, vy, vz)

        speed = math.sqrt(vx * vx + vy * vy + vz * vz)
        weight = _smoothing_weight(self._min_cutoff_hz + self._beta * speed, dt)
        dx, dy, dz = sq.relative_rotation_vector(self._orientation, sample)
        self._orientation = sq.apply_rotation_vector(
            self._orientation, (weight * dx, weight * dy, weight * dz)
        )
        self._previous_sample = sample
        self._timestamp = timestamp
        return sq.to_array(self._orientation)

    def reset(self) -> None:
        """Forget all history; the next orientation passes through unchanged."""
        self._orientation = None
        self._previous_sample = None
        self._angular_velocity = (0.0, 0.0, 0.0)
        self._timestamp = -math.inf


def _smoothing_weight(cutoff_hz: float, dt: float) -> float:
    """Return :func:`smoothing_factor` for one cutoff as a plain float."""
    r = 2.0 * math.pi * cutoff_hz * dt
    return r / (1.0 + r)
