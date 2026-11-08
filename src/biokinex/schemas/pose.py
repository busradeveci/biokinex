"""Output schema for stabilized pose frames sent to downstream controllers.

A frame is validated when it is built, so a NaN or a broken quaternion can never reach an actuator,
even if a bug slips into the DSP stages.
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Any, Final, Self

from pydantic import Field, computed_field, model_validator

from biokinex.core import quaternion
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

__all__ = ["FrameStatus", "PoseFrame"]

UNIT_QUATERNION_TOLERANCE: Final = 1e-6
"""Largest allowed deviation from unit length for an output quaternion."""

_DERIVED_FIELDS: Final = ("euler", "near_gimbal_lock")


class FrameStatus(StrEnum):
    """Health of a pose frame.

    - ``ok``: built from fresh, fully valid input.
    - ``degraded``: usable, but some input was missing, filled in, or corrected.
    - ``stale``: no valid input arrived within the watchdog timeout; controllers should enter a
      safe state.
    """

    OK = "ok"
    DEGRADED = "degraded"
    STALE = "stale"


class PoseFrame(FrozenModel):
    """A stabilized, validated output frame in SI units.

    ``euler`` and ``near_gimbal_lock`` are derived from ``orientation`` and appear in serialized
    output for convenience. They are recomputed on parsing, so they can never disagree with it.
    """

    device_id: Identifier
    sequence: SequenceNumber
    timestamp: Timestamp
    status: FrameStatus
    orientation: QuaternionField | None = None
    angular_velocity: Vector3Field | None = None
    acceleration: Vector3Field | None = None
    joint_names: tuple[Identifier, ...] = Field(default=(), max_length=MAX_JOINTS)
    joint_angles: tuple[FiniteFloat, ...] | None = Field(default=None, max_length=MAX_JOINTS)

    @model_validator(mode="before")
    @classmethod
    def _drop_derived_fields(cls, data: Any) -> Any:
        # Serialized frames include derived fields; dropping them lets frames round-trip safely.
        if isinstance(data, dict):
            return {key: value for key, value in data.items() if key not in _DERIVED_FIELDS}
        return data

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if self.orientation is not None:
            norm = math.sqrt(sum(c * c for c in self.orientation))
            if abs(norm - 1.0) > UNIT_QUATERNION_TOLERANCE:
                msg = f"orientation must be a unit quaternion, got norm {norm:.9g}"
                raise ValueError(msg)
        if self.joint_angles is not None and len(self.joint_angles) != len(self.joint_names):
            msg = (
                f"got {len(self.joint_angles)} joint angles for {len(self.joint_names)} joint names"
            )
            raise ValueError(msg)
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def euler(self) -> tuple[float, float, float] | None:
        """ZYX Euler angles ``(roll, pitch, yaw)`` in radians, derived from ``orientation``."""
        if self.orientation is None:
            return None
        roll, pitch, yaw = quaternion.to_euler_zyx(self.orientation).tolist()
        return (roll, pitch, yaw)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def near_gimbal_lock(self) -> bool:
        """Whether ``euler`` is close to ±90° pitch, where roll and yaw become unreliable."""
        if self.orientation is None:
            return False
        return bool(quaternion.is_near_gimbal_lock(self.orientation))
