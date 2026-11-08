"""Wire schema for incoming telemetry packets.

A packet carries a short run of samples from one device. Devices may report in their native units
(g, deg/s, degrees); :meth:`TelemetryPacket.to_batch` converts everything to SI units, and Euler
orientations to quaternions, so nothing past this boundary needs to know about wire formats.

Example JSON payload::

    {
        "device_id": "imu-left-shank",
        "units": {"acceleration": "g", "angular_velocity": "deg/s", "angle": "deg"},
        "joint_names": ["knee"],
        "samples": [
            {
                "sequence": 41,
                "timestamp": 12.002,
                "acceleration": [0.01, -0.02, 1.0],
                "angular_velocity": [0.5, -1.2, 0.0],
                "joint_angles": [35.0],
            }
        ],
    }
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Final, Literal, Self

import numpy as np
from pydantic import Field, ValidationError, model_validator

from biokinex.core import quaternion
from biokinex.core.errors import TelemetryError
from biokinex.core.types import STANDARD_GRAVITY
from biokinex.schemas.batch import SampleBatch, read_only
from biokinex.schemas.common import (
    MAX_JOINTS,
    FiniteFloat,
    FrozenModel,
    Identifier,
    QuaternionField,
    SequenceNumber,
    Timestamp,
    Vector3Field,
)

__all__ = [
    "MAX_SAMPLES_PER_PACKET",
    "QUATERNION_NORM_TOLERANCE",
    "AccelerationUnit",
    "AngleUnit",
    "AngularVelocityUnit",
    "TelemetryPacket",
    "TelemetrySample",
    "Units",
]

MAX_SAMPLES_PER_PACKET: Final = 256
"""Upper bound on samples per packet, which caps the work and memory one packet can cause."""

QUATERNION_NORM_TOLERANCE: Final = 0.1
"""Largest accepted deviation from unit length for an incoming quaternion.

Small deviations come from sensor rounding and are normalized away. Larger ones point to a corrupted
packet, and normalizing would hide that.
"""

_DEGREES_TO_RADIANS: Final = math.pi / 180.0

_MEASUREMENT_FIELDS: Final = (
    "acceleration",
    "angular_velocity",
    "orientation_quaternion",
    "orientation_euler",
    "joint_angles",
)


class AccelerationUnit(StrEnum):
    """Units a device may use for acceleration."""

    METERS_PER_SECOND_SQUARED = "m/s^2"
    STANDARD_GRAVITY = "g"

    @property
    def si_factor(self) -> float:
        """Multiplier that converts a value in this unit to m/s²."""
        return STANDARD_GRAVITY if self is AccelerationUnit.STANDARD_GRAVITY else 1.0


class AngularVelocityUnit(StrEnum):
    """Units a device may use for angular velocity."""

    RADIANS_PER_SECOND = "rad/s"
    DEGREES_PER_SECOND = "deg/s"

    @property
    def si_factor(self) -> float:
        """Multiplier that converts a value in this unit to rad/s."""
        return _DEGREES_TO_RADIANS if self is AngularVelocityUnit.DEGREES_PER_SECOND else 1.0


class AngleUnit(StrEnum):
    """Units a device may use for Euler and joint angles."""

    RADIANS = "rad"
    DEGREES = "deg"

    @property
    def si_factor(self) -> float:
        """Multiplier that converts a value in this unit to radians."""
        return _DEGREES_TO_RADIANS if self is AngleUnit.DEGREES else 1.0


class Units(FrozenModel):
    """Units used on the wire by one packet; the defaults are SI."""

    acceleration: AccelerationUnit = AccelerationUnit.METERS_PER_SECOND_SQUARED
    angular_velocity: AngularVelocityUnit = AngularVelocityUnit.RADIANS_PER_SECOND
    angle: AngleUnit = AngleUnit.RADIANS


class TelemetrySample(FrozenModel):
    """One time-stamped measurement from a device, in the packet's wire units.

    Orientation may be sent as a quaternion ``(w, x, y, z)`` or as ZYX Euler angles
    ``(roll, pitch, yaw)``, but not both.
    """

    sequence: SequenceNumber
    timestamp: Timestamp
    acceleration: Vector3Field | None = None
    angular_velocity: Vector3Field | None = None
    orientation_quaternion: QuaternionField | None = None
    orientation_euler: Vector3Field | None = None
    joint_angles: tuple[FiniteFloat, ...] | None = Field(default=None, max_length=MAX_JOINTS)

    @model_validator(mode="after")
    def _check_measurements(self) -> Self:
        if self.orientation_quaternion is not None and self.orientation_euler is not None:
            msg = "orientation must be sent as a quaternion or as Euler angles, not both"
            raise ValueError(msg)
        if all(getattr(self, name) is None for name in _MEASUREMENT_FIELDS):
            msg = "a sample must carry at least one measurement"
            raise ValueError(msg)
        if self.orientation_quaternion is not None:
            norm = math.sqrt(sum(c * c for c in self.orientation_quaternion))
            if abs(norm - 1.0) > QUATERNION_NORM_TOLERANCE:
                msg = f"orientation_quaternion norm {norm:.6g} is too far from 1"
                raise ValueError(msg)
        return self

    def layout(self) -> tuple[bool, ...]:
        """Return which measurement fields are present, in a fixed order."""
        return tuple(getattr(self, name) is not None for name in _MEASUREMENT_FIELDS)


class TelemetryPacket(FrozenModel):
    """A validated packet of consecutive samples from a single device.

    Every sample in a packet must carry the same set of measurements, so the packet maps cleanly
    onto fixed-shape arrays.
    """

    schema_version: Literal[1] = 1
    device_id: Identifier
    units: Units = Units()
    joint_names: tuple[Identifier, ...] = Field(default=(), max_length=MAX_JOINTS)
    samples: tuple[TelemetrySample, ...] = Field(min_length=1, max_length=MAX_SAMPLES_PER_PACKET)

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if len(set(self.joint_names)) != len(self.joint_names):
            msg = "joint_names must be unique"
            raise ValueError(msg)

        expected_layout = self.samples[0].layout()
        previous_timestamp = -math.inf
        for index, sample in enumerate(self.samples):
            if sample.layout() != expected_layout:
                msg = f"sample {index} carries different measurements than sample 0"
                raise ValueError(msg)
            # Strict ordering inside a packet is cheap to check here and protects every dt > 0.
            if sample.timestamp <= previous_timestamp:
                msg = f"sample {index} timestamp is not strictly increasing"
                raise ValueError(msg)
            previous_timestamp = sample.timestamp
            joint_count = 0 if sample.joint_angles is None else len(sample.joint_angles)
            if joint_count != len(self.joint_names):
                msg = (
                    f"sample {index} has {joint_count} joint angles "
                    f"but the packet declares {len(self.joint_names)} joint names"
                )
                raise ValueError(msg)
        return self

    @classmethod
    def from_json(cls, data: str | bytes | bytearray) -> TelemetryPacket:
        """Parse and validate a JSON payload.

        Raises:
            TelemetryError: If the payload is not valid JSON or breaks any schema rule.
        """
        try:
            return cls.model_validate_json(data)
        except ValidationError as exc:
            msg = f"invalid telemetry packet ({exc.error_count()} error(s))"
            raise TelemetryError(msg) from exc

    def to_batch(self) -> SampleBatch:
        """Convert the packet to read-only NumPy arrays in SI units."""
        samples = self.samples
        first = samples[0]

        acceleration = None
        if first.acceleration is not None:
            raw = np.array([s.acceleration for s in samples], dtype=np.float64)
            acceleration = read_only(raw * self.units.acceleration.si_factor)

        angular_velocity = None
        if first.angular_velocity is not None:
            raw = np.array([s.angular_velocity for s in samples], dtype=np.float64)
            angular_velocity = read_only(raw * self.units.angular_velocity.si_factor)

        orientation = None
        if first.orientation_quaternion is not None:
            raw = np.array([s.orientation_quaternion for s in samples], dtype=np.float64)
            orientation = read_only(quaternion.normalize(raw))
        elif first.orientation_euler is not None:
            raw = np.array([s.orientation_euler for s in samples], dtype=np.float64)
            orientation = read_only(quaternion.from_euler_zyx(raw * self.units.angle.si_factor))

        joint_angles = None
        if first.joint_angles is not None:
            raw = np.array([s.joint_angles for s in samples], dtype=np.float64)
            joint_angles = read_only(raw * self.units.angle.si_factor)

        return SampleBatch(
            device_id=self.device_id,
            timestamps=read_only(np.array([s.timestamp for s in samples], dtype=np.float64)),
            sequences=read_only(np.array([s.sequence for s in samples], dtype=np.int64)),
            acceleration=acceleration,
            angular_velocity=angular_velocity,
            orientation=orientation,
            joint_names=self.joint_names,
            joint_angles=joint_angles,
        )
