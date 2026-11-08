"""Single-quaternion math on plain Python floats, for per-sample hot paths.

The functions in :mod:`biokinex.core.quaternion` accept whole batches, and that flexibility costs a
few microseconds of NumPy overhead per call. Per-sample filters call several of them on one
quaternion at a time, so this module repeats the few operations they need with :mod:`math` on
tuples. Conventions and thresholds are identical, and the tests check both versions against each
other.
"""

from __future__ import annotations

import math
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from biokinex.core.errors import DegenerateQuaternionError
from biokinex.core.quaternion import _NORM_EPSILON, _SMALL_ANGLE
from biokinex.core.types import FloatArray

__all__ = [
    "Quat",
    "Vec3",
    "apply_rotation_vector",
    "body_up",
    "conjugate",
    "from_array",
    "from_rotation_vector",
    "integrate",
    "multiply",
    "normalize",
    "relative_rotation_vector",
    "to_array",
    "to_rotation_vector",
    "vector_from_array",
]

Quat: TypeAlias = tuple[float, float, float, float]
"""One Hamilton quaternion ``(w, x, y, z)``."""

Vec3: TypeAlias = tuple[float, float, float]
"""One vector ``(x, y, z)``."""


def from_array(value: npt.ArrayLike) -> Quat:
    """Convert one quaternion of shape ``(4,)`` to a tuple of floats."""
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (4,):
        msg = f"q must be a single quaternion of shape (4,), got {array.shape}"
        raise ValueError(msg)
    w, x, y, z = array.tolist()
    return (w, x, y, z)


def vector_from_array(value: npt.ArrayLike, name: str) -> Vec3:
    """Convert one vector of shape ``(3,)`` to a tuple of floats."""
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,):
        msg = f"{name} must have shape (3,), got {array.shape}"
        raise ValueError(msg)
    x, y, z = array.tolist()
    return (x, y, z)


def to_array(values: Quat | Vec3) -> FloatArray:
    """Convert a tuple back to a new float64 array."""
    return np.array(values, dtype=np.float64)


def normalize(q: Quat) -> Quat:
    """Scale ``q`` to unit length.

    Raises:
        DegenerateQuaternionError: If ``q`` is zero, nearly zero, or non-finite.
    """
    w, x, y, z = q
    length = math.sqrt(w * w + x * x + y * y + z * z)
    if not (math.isfinite(length) and length >= _NORM_EPSILON):
        msg = "cannot normalize a zero, near-zero, or non-finite quaternion"
        raise DegenerateQuaternionError(msg)
    return (w / length, x / length, y / length, z / length)


def conjugate(q: Quat) -> Quat:
    """Return the conjugate, which is the inverse rotation for a unit quaternion."""
    w, x, y, z = q
    return (w, -x, -y, -z)


def multiply(p: Quat, q: Quat) -> Quat:
    """Return the Hamilton product ``p * q``."""
    pw, px, py, pz = p
    qw, qx, qy, qz = q
    return (
        pw * qw - px * qx - py * qy - pz * qz,
        pw * qx + px * qw + py * qz - pz * qy,
        pw * qy - px * qz + py * qw + pz * qx,
        pw * qz + px * qy - py * qx + pz * qw,
    )


def from_rotation_vector(v: Vec3) -> Quat:
    """Convert a rotation vector ``axis * angle`` to a unit quaternion."""
    vx, vy, vz = v
    angle = math.sqrt(vx * vx + vy * vy + vz * vz)
    # The same Taylor series as the batch version avoids dividing by zero for tiny rotations.
    scale = 0.5 - angle * angle / 48.0 if angle < _SMALL_ANGLE else math.sin(0.5 * angle) / angle
    return (math.cos(0.5 * angle), vx * scale, vy * scale, vz * scale)


def to_rotation_vector(q: Quat) -> Vec3:
    """Convert a unit quaternion to the rotation vector of the shortest equal rotation."""
    w, x, y, z = q
    if w < 0.0:
        w, x, y, z = -w, -x, -y, -z
    length = math.sqrt(x * x + y * y + z * z)
    if length < _SMALL_ANGLE:
        scale = 2.0 / w if w > 0.0 else 2.0
    else:
        scale = 2.0 * math.atan2(length, w) / length
    return (x * scale, y * scale, z * scale)


def relative_rotation_vector(reference: Quat, target: Quat) -> Vec3:
    """Return the shortest body-frame rotation vector that turns ``reference`` into ``target``."""
    return to_rotation_vector(multiply(conjugate(reference), target))


def apply_rotation_vector(reference: Quat, rotation_vector: Vec3) -> Quat:
    """Rotate ``reference`` by a body-frame rotation vector and return a unit quaternion."""
    return normalize(multiply(reference, from_rotation_vector(rotation_vector)))


def integrate(q: Quat, angular_velocity: Vec3, dt: float) -> Quat:
    """Advance ``q`` by a constant body-frame angular velocity in rad/s over ``dt`` seconds."""
    wx, wy, wz = angular_velocity
    return apply_rotation_vector(q, (wx * dt, wy * dt, wz * dt))


def body_up(q: Quat) -> Vec3:
    """Return the world "up" axis ``(0, 0, 1)`` expressed in the body frame of unit ``q``."""
    w, x, y, z = q
    # This is the third row of the rotation matrix, which equals conj(q) applied to world up.
    return (2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y))
