"""Helpers for filtering orientation on rotation increments instead of raw components.

Averaging quaternion components or Euler angles gives wrong results: Euler angles jump at ±180°,
and ``q`` and ``-q`` are the same rotation. These helpers express the difference between two
orientations as a small body-frame rotation vector, which a filter can scale safely, and then
apply the scaled step back. The result is always a valid unit quaternion.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from biokinex.core import quaternion
from biokinex.core.types import QUATERNION_SIZE, Quaternion, Vector3

__all__ = [
    "angular_velocity_between",
    "apply_rotation_vector",
    "make_continuous",
    "relative_rotation_vector",
]


def relative_rotation_vector(reference: npt.ArrayLike, target: npt.ArrayLike) -> Vector3:
    """Return the shortest body-frame rotation vector that turns ``reference`` into ``target``.

    The result does not depend on the sign of either quaternion, except at exactly 180°: there
    ``v`` and ``-v`` describe the same rotation, and either one may be returned.
    """
    delta = quaternion.multiply(quaternion.conjugate(reference), target)
    return quaternion.to_rotation_vector(delta)


def apply_rotation_vector(reference: npt.ArrayLike, rotation_vector: npt.ArrayLike) -> Quaternion:
    """Rotate ``reference`` by a body-frame rotation vector and return a unit quaternion."""
    step = quaternion.from_rotation_vector(rotation_vector)
    # Normalizing here stops rounding errors from building up over millions of filter steps.
    return quaternion.normalize(quaternion.multiply(reference, step))


def angular_velocity_between(start: npt.ArrayLike, end: npt.ArrayLike, dt: float) -> Vector3:
    """Return the constant body-frame angular velocity in rad/s that moves ``start`` to ``end``."""
    if not math.isfinite(dt) or dt <= 0.0:
        msg = f"dt must be finite and positive, got {dt!r}"
        raise ValueError(msg)
    return np.asarray(relative_rotation_vector(start, end) / dt)


def make_continuous(
    quaternions: npt.ArrayLike, reference: npt.ArrayLike | None = None
) -> Quaternion:
    """Flip signs in a sequence of quaternions so neighbours stay in the same hemisphere.

    The rotations do not change, but the components no longer jump between ``q`` and ``-q``. This
    matters for anything that compares or plots raw components.

    Args:
        quaternions: Sequence with shape ``(N, 4)``.
        reference: Optional quaternion that the first element is matched to, for example the last
            element of the previous batch.
    """
    sequence = np.array(quaternions, dtype=np.float64)
    if sequence.ndim != 2 or sequence.shape[1] != QUATERNION_SIZE:
        msg = f"quaternions must have shape (N, 4), got {sequence.shape}"
        raise ValueError(msg)
    if sequence.shape[0] == 0:
        return sequence
    neighbour_dots = np.sum(sequence[1:] * sequence[:-1], axis=1)
    # Each flip changes the sign of everything after it, so a running product gives every sign.
    steps = np.where(neighbour_dots < 0.0, -1.0, 1.0)
    signs = np.concatenate([[1.0], np.cumprod(steps)])
    if reference is not None and np.dot(sequence[0], np.asarray(reference)) < 0.0:
        signs = -signs
    return np.asarray(sequence * signs[:, None])
