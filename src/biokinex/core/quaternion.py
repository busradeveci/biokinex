"""Hamilton quaternion math for orientation tracking.

Conventions used everywhere in BioKineX:

- Components are stored scalar-first as ``(w, x, y, z)``.
- Multiplication follows the Hamilton rule ``i * j = k``.
- A unit quaternion ``q`` rotates a vector ``v`` as ``q * v * conj(q)``. For sensor orientation,
  this maps vectors from the body frame into the world frame.
- Euler angles are intrinsic ZYX (yaw, then pitch, then roll) and stored as ``(roll, pitch, yaw)``.
- A rotation vector is ``axis * angle``, with the angle in radians.

Every function accepts a single quaternion of shape ``(4,)`` or a batch of shape ``(..., 4)`` and
broadcasts like NumPy. Functions documented as expecting unit quaternions do not renormalize their
input, because the hot path should not pay for work the caller has already done.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np
import numpy.typing as npt

from biokinex.core.errors import DegenerateQuaternionError
from biokinex.core.types import (
    QUATERNION_SIZE,
    VECTOR3_SIZE,
    BoolArray,
    EulerAngles,
    FloatArray,
    Quaternion,
    Vector3,
)

__all__ = [
    "DEFAULT_GIMBAL_LOCK_MARGIN",
    "angle_between",
    "canonicalize",
    "conjugate",
    "ensure_same_hemisphere",
    "from_axis_angle",
    "from_euler_zyx",
    "from_rotation_vector",
    "identity",
    "integrate",
    "is_near_gimbal_lock",
    "is_unit",
    "multiply",
    "nlerp",
    "normalize",
    "rotate_vector",
    "slerp",
    "to_euler_zyx",
    "to_rotation_vector",
    "wrap_angle",
]

DEFAULT_GIMBAL_LOCK_MARGIN: Final = math.radians(2.0)
"""Default distance from ±90° pitch, in radians, at which Euler output is flagged as unreliable."""

_NORM_EPSILON: Final = 1e-12
_SMALL_ANGLE: Final = 1e-8
_SLERP_LINEAR_THRESHOLD: Final = 1.0 - 1e-6
_GIMBAL_LOCK_COS: Final = 1e-8
_TWO_PI: Final = 2.0 * math.pi
_CONJUGATE_SIGNS: Final = np.array([1.0, -1.0, -1.0, -1.0])


def _as_array(value: npt.ArrayLike, size: int, name: str) -> FloatArray:
    """Convert ``value`` to a float64 array and check that its last axis has ``size`` elements."""
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0 or array.shape[-1] != size:
        msg = f"{name} must have shape (..., {size}), got {array.shape}"
        raise ValueError(msg)
    return array


def _dot(p: FloatArray, q: FloatArray) -> FloatArray:
    """Return the 4D dot product over the last axis."""
    return np.asarray(np.sum(p * q, axis=-1))


def wrap_angle(angle: npt.ArrayLike) -> FloatArray:
    """Wrap angles in radians into the interval ``[-pi, pi)``."""
    shifted = np.asarray(angle, dtype=np.float64) + math.pi
    return np.asarray(np.remainder(shifted, _TWO_PI) - math.pi)


def identity() -> Quaternion:
    """Return the identity rotation ``(1, 0, 0, 0)``."""
    return np.array([1.0, 0.0, 0.0, 0.0])


def is_unit(q: npt.ArrayLike, atol: float = 1e-9) -> BoolArray:
    """Return whether each quaternion has unit length within ``atol``."""
    array = _as_array(q, QUATERNION_SIZE, "q")
    return np.asarray(np.abs(np.linalg.norm(array, axis=-1) - 1.0) <= atol)


def normalize(q: npt.ArrayLike) -> Quaternion:
    """Scale each quaternion to unit length.

    Raises:
        DegenerateQuaternionError: If any quaternion is zero, nearly zero, or non-finite.
    """
    array = _as_array(q, QUATERNION_SIZE, "q")
    length = np.linalg.norm(array, axis=-1, keepdims=True)
    if not np.all(np.isfinite(length) & (length >= _NORM_EPSILON)):
        msg = "cannot normalize a zero, near-zero, or non-finite quaternion"
        raise DegenerateQuaternionError(msg)
    return np.asarray(array / length)


def conjugate(q: npt.ArrayLike) -> Quaternion:
    """Return the conjugate, which is the inverse rotation for unit quaternions."""
    return np.asarray(_as_array(q, QUATERNION_SIZE, "q") * _CONJUGATE_SIGNS)


def multiply(p: npt.ArrayLike, q: npt.ArrayLike) -> Quaternion:
    """Return the Hamilton product ``p * q``, which applies ``q`` first and then ``p``."""
    pw, px, py, pz = np.moveaxis(_as_array(p, QUATERNION_SIZE, "p"), -1, 0)
    qw, qx, qy, qz = np.moveaxis(_as_array(q, QUATERNION_SIZE, "q"), -1, 0)
    return np.stack(
        [
            pw * qw - px * qx - py * qy - pz * qz,
            pw * qx + px * qw + py * qz - pz * qy,
            pw * qy - px * qz + py * qw + pz * qx,
            pw * qz + px * qy - py * qx + pz * qw,
        ],
        axis=-1,
    )


def rotate_vector(q: npt.ArrayLike, v: npt.ArrayLike) -> Vector3:
    """Rotate vector(s) ``v`` by unit quaternion(s) ``q``."""
    quat = _as_array(q, QUATERNION_SIZE, "q")
    vec = _as_array(v, VECTOR3_SIZE, "v")
    w = quat[..., :1]
    u = quat[..., 1:]
    # This expanded form of q * v * conj(q) needs two cross products instead of two full products.
    t = 2.0 * np.cross(u, vec)
    return np.asarray(vec + w * t + np.cross(u, t))


def canonicalize(q: npt.ArrayLike) -> Quaternion:
    """Flip each quaternion so that ``w >= 0``, picking one of the two equal representations."""
    array = _as_array(q, QUATERNION_SIZE, "q")
    return np.asarray(np.where(array[..., :1] < 0.0, -array, array))


def ensure_same_hemisphere(q: npt.ArrayLike, reference: npt.ArrayLike) -> Quaternion:
    """Flip ``q`` where needed so that its dot product with ``reference`` is non-negative.

    ``q`` and ``-q`` describe the same rotation; keeping a stream on one side stops filters from
    averaging two opposite representations into nonsense.
    """
    array = _as_array(q, QUATERNION_SIZE, "q")
    ref = _as_array(reference, QUATERNION_SIZE, "reference")
    return np.asarray(np.where(_dot(array, ref)[..., None] < 0.0, -array, array))


def angle_between(p: npt.ArrayLike, q: npt.ArrayLike) -> FloatArray:
    """Return the smallest rotation angle in radians, in ``[0, pi]``, between unit quaternions."""
    delta = multiply(conjugate(p), q)
    vector_length = np.linalg.norm(delta[..., 1:], axis=-1)
    # atan2 stays accurate for tiny angles, where acos(|w|) loses precision.
    return np.asarray(2.0 * np.arctan2(vector_length, np.abs(delta[..., 0])))


def from_rotation_vector(rotation_vector: npt.ArrayLike) -> Quaternion:
    """Convert rotation vector(s) ``axis * angle`` to unit quaternion(s) (the exponential map)."""
    vec = _as_array(rotation_vector, VECTOR3_SIZE, "rotation_vector")
    angle = np.linalg.norm(vec, axis=-1, keepdims=True)
    small = angle < _SMALL_ANGLE
    safe_angle = np.where(small, 1.0, angle)
    # A Taylor series replaces sin(angle / 2) / angle near zero to avoid dividing by zero.
    scale = np.where(small, 0.5 - angle * angle / 48.0, np.sin(0.5 * angle) / safe_angle)
    return np.concatenate([np.cos(0.5 * angle), vec * scale], axis=-1)


def to_rotation_vector(q: npt.ArrayLike) -> Vector3:
    """Convert unit quaternion(s) to rotation vector(s) (the logarithm map).

    The result always describes the shortest rotation, so its length is in ``[0, pi]``.
    """
    quat = canonicalize(q)
    w = quat[..., :1]
    u = quat[..., 1:]
    length = np.linalg.norm(u, axis=-1, keepdims=True)
    small = length < _SMALL_ANGLE
    safe_length = np.where(small, 1.0, length)
    safe_w = np.where(w > 0.0, w, 1.0)
    # Near the identity, angle / length tends to 2 / w, which avoids dividing by zero.
    scale = np.where(small, 2.0 / safe_w, 2.0 * np.arctan2(length, w) / safe_length)
    return np.asarray(u * scale)


def from_axis_angle(axis: npt.ArrayLike, angle: npt.ArrayLike) -> Quaternion:
    """Build unit quaternion(s) that rotate by ``angle`` radians about ``axis``.

    The axis does not need to be unit length, but it must not be zero.
    """
    vec = _as_array(axis, VECTOR3_SIZE, "axis")
    length = np.linalg.norm(vec, axis=-1, keepdims=True)
    if not np.all(np.isfinite(length) & (length >= _NORM_EPSILON)):
        msg = "axis must be finite and non-zero"
        raise ValueError(msg)
    angles = np.asarray(angle, dtype=np.float64)[..., None]
    return from_rotation_vector(vec / length * angles)


def from_euler_zyx(angles: npt.ArrayLike) -> Quaternion:
    """Convert intrinsic ZYX Euler angles ``(roll, pitch, yaw)`` to unit quaternion(s)."""
    half = 0.5 * _as_array(angles, VECTOR3_SIZE, "angles")
    cos_half = np.cos(half)
    sin_half = np.sin(half)
    cr, cp, cy = np.moveaxis(cos_half, -1, 0)
    sr, sp, sy = np.moveaxis(sin_half, -1, 0)
    return np.stack(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        axis=-1,
    )


def to_euler_zyx(q: npt.ArrayLike) -> EulerAngles:
    """Convert unit quaternion(s) to intrinsic ZYX Euler angles ``(roll, pitch, yaw)``.

    Roll and yaw are in ``[-pi, pi]`` and pitch is in ``[-pi/2, pi/2]``. At ±90° pitch, roll and
    yaw turn about the same axis, so roll is set to zero and yaw carries the whole rotation.
    Use :func:`is_near_gimbal_lock` to flag outputs close to that point.
    """
    w, x, y, z = np.moveaxis(_as_array(q, QUATERNION_SIZE, "q"), -1, 0)
    sin_pitch = 2.0 * (w * y - z * x)
    yaw_cos_term = 1.0 - 2.0 * (y * y + z * z)
    yaw_sin_term = 2.0 * (w * z + x * y)
    # Taking cos(pitch) from these terms keeps full precision near ±90°, where arcsin does not.
    cos_pitch = np.hypot(yaw_cos_term, yaw_sin_term)
    pitch = np.arctan2(sin_pitch, cos_pitch)
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    yaw = np.arctan2(yaw_sin_term, yaw_cos_term)

    locked = cos_pitch < _GIMBAL_LOCK_COS
    locked_up = locked & (sin_pitch > 0.0)
    # At +90° only (yaw - roll) is defined, and at -90° only (yaw + roll), so we solve for yaw.
    half_angle = np.arctan2(x, w)
    locked_yaw = wrap_angle(np.where(locked_up, -2.0 * half_angle, 2.0 * half_angle))

    return np.stack(
        [np.where(locked, 0.0, roll), pitch, np.where(locked, locked_yaw, yaw)],
        axis=-1,
    )


def is_near_gimbal_lock(q: npt.ArrayLike, margin: float = DEFAULT_GIMBAL_LOCK_MARGIN) -> BoolArray:
    """Return whether the ZYX pitch of each unit quaternion is within ``margin`` radians of ±90°."""
    if not 0.0 < margin < 0.5 * math.pi:
        msg = f"margin must be in (0, pi/2), got {margin!r}"
        raise ValueError(msg)
    w, x, y, z = np.moveaxis(_as_array(q, QUATERNION_SIZE, "q"), -1, 0)
    # |pitch| >= pi/2 - margin is the same test as |sin(pitch)| >= cos(margin), with no arcsin.
    return np.asarray(np.abs(2.0 * (w * y - z * x)) >= math.cos(margin))


def nlerp(p: npt.ArrayLike, q: npt.ArrayLike, t: float) -> Quaternion:
    """Normalized linear interpolation from ``p`` (``t = 0``) to ``q`` (``t = 1``).

    It is cheaper than :func:`slerp` and accurate for small angles, but its angular speed is not
    constant.
    """
    _check_fraction(t)
    start = _as_array(p, QUATERNION_SIZE, "p")
    end = ensure_same_hemisphere(q, start)
    return normalize((1.0 - t) * start + t * end)


def slerp(p: npt.ArrayLike, q: npt.ArrayLike, t: float) -> Quaternion:
    """Spherical linear interpolation along the shortest arc from ``p`` (``t = 0``) to ``q``."""
    _check_fraction(t)
    start = _as_array(p, QUATERNION_SIZE, "p")
    end = ensure_same_hemisphere(q, start)
    cos_theta = np.clip(_dot(start, end), -1.0, 1.0)[..., None]
    theta = np.arccos(cos_theta)
    # Nearly equal inputs make sin(theta) close to zero, so linear weights are used instead.
    linear = cos_theta > _SLERP_LINEAR_THRESHOLD
    sin_theta = np.where(linear, 1.0, np.sin(theta))
    start_weight = np.where(linear, 1.0 - t, np.sin((1.0 - t) * theta) / sin_theta)
    end_weight = np.where(linear, t, np.sin(t * theta) / sin_theta)
    return normalize(start_weight * start + end_weight * end)


def integrate(q: npt.ArrayLike, angular_velocity: npt.ArrayLike, dt: float) -> Quaternion:
    """Advance orientation ``q`` by a body-frame angular velocity in rad/s over ``dt`` seconds.

    The rate is treated as constant during ``dt``, which makes the step exact for that case.
    """
    if not math.isfinite(dt) or dt < 0.0:
        msg = f"dt must be finite and non-negative, got {dt!r}"
        raise ValueError(msg)
    omega = _as_array(angular_velocity, VECTOR3_SIZE, "angular_velocity")
    # Gyroscopes measure in the body frame, so the increment is applied on the right.
    step = from_rotation_vector(omega * dt)
    return normalize(multiply(q, step))


def _check_fraction(t: float) -> None:
    """Reject interpolation fractions outside ``[0, 1]``."""
    if not 0.0 <= t <= 1.0:
        msg = f"t must be in [0, 1], got {t!r}"
        raise ValueError(msg)
