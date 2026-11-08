"""Shared type aliases, index enums, and physical constants.

All arrays in BioKineX use ``float64`` and SI units: seconds, metres, m/s², rad/s, and radians.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Final, TypeAlias

import numpy as np
import numpy.typing as npt

FloatArray: TypeAlias = npt.NDArray[np.float64]
"""A ``float64`` NumPy array of any shape."""

BoolArray: TypeAlias = npt.NDArray[np.bool_]
"""A boolean NumPy array of any shape."""

Quaternion: TypeAlias = FloatArray
"""Hamilton quaternion(s) in scalar-first ``(w, x, y, z)`` order, shape ``(..., 4)``."""

Vector3: TypeAlias = FloatArray
"""Cartesian vector(s) in ``(x, y, z)`` order, shape ``(..., 3)``."""

EulerAngles: TypeAlias = FloatArray
"""Intrinsic ZYX Euler angles stored as ``(roll, pitch, yaw)`` in radians, shape ``(..., 3)``."""

QUATERNION_SIZE: Final = 4
VECTOR3_SIZE: Final = 3

STANDARD_GRAVITY: Final = 9.80665
"""Standard gravity in m/s² (CGPM 1901), used to convert accelerometer readings given in g."""


class Axis(IntEnum):
    """Index of each component in a :data:`Vector3`."""

    X = 0
    Y = 1
    Z = 2


class QuaternionComponent(IntEnum):
    """Index of each component in a :data:`Quaternion`."""

    W = 0
    X = 1
    Y = 2
    Z = 3


class EulerAngle(IntEnum):
    """Index of each angle in an :data:`EulerAngles` array."""

    ROLL = 0
    PITCH = 1
    YAW = 2
