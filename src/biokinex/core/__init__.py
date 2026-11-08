"""Foundational building blocks: type aliases, quaternion math, clocks, and errors.

Quaternion functions live in their own namespace to keep names such as ``multiply`` unambiguous::

    from biokinex.core import quaternion as quat
"""

from biokinex.core import quaternion
from biokinex.core.clock import Clock, ManualClock, MonotonicClock
from biokinex.core.errors import (
    BioKineXError,
    ConfigurationError,
    DegenerateQuaternionError,
    TelemetryError,
    TransportError,
)
from biokinex.core.types import (
    QUATERNION_SIZE,
    STANDARD_GRAVITY,
    VECTOR3_SIZE,
    Axis,
    BoolArray,
    EulerAngle,
    EulerAngles,
    FloatArray,
    Quaternion,
    QuaternionComponent,
    Vector3,
)

__all__ = [
    "QUATERNION_SIZE",
    "STANDARD_GRAVITY",
    "VECTOR3_SIZE",
    "Axis",
    "BioKineXError",
    "BoolArray",
    "Clock",
    "ConfigurationError",
    "DegenerateQuaternionError",
    "EulerAngle",
    "EulerAngles",
    "FloatArray",
    "ManualClock",
    "MonotonicClock",
    "Quaternion",
    "QuaternionComponent",
    "TelemetryError",
    "TransportError",
    "Vector3",
    "quaternion",
]
