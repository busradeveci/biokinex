"""Tests for biokinex.schemas.pose."""

from __future__ import annotations

import json
import math
from typing import Any

import pytest
from numpy.testing import assert_allclose
from pydantic import ValidationError

from biokinex.core import quaternion as quat
from biokinex.schemas.pose import FrameStatus, PoseFrame


def frame_fields(**overrides: Any) -> dict[str, Any]:
    """Build raw fields for a valid frame, with optional overrides."""
    fields: dict[str, Any] = {
        "device_id": "imu-0",
        "sequence": 10,
        "timestamp": 0.02,
        "status": "ok",
        "orientation": tuple(quat.from_euler_zyx([0.1, -0.2, 0.3]).tolist()),
        "angular_velocity": (0.0, 0.1, 0.0),
        "acceleration": (0.0, 0.0, 9.81),
        "joint_names": ("knee",),
        "joint_angles": (0.6,),
    }
    fields.update(overrides)
    return fields


def test_valid_frame() -> None:
    frame = PoseFrame.model_validate(frame_fields())
    assert frame.status is FrameStatus.OK
    assert frame.joint_angles == (0.6,)


def test_minimal_stale_frame() -> None:
    frame = PoseFrame(device_id="imu-0", sequence=0, timestamp=0.0, status=FrameStatus.STALE)
    assert frame.orientation is None
    assert frame.euler is None
    assert frame.near_gimbal_lock is False


def test_euler_is_derived_from_orientation() -> None:
    frame = PoseFrame.model_validate(frame_fields())
    assert frame.euler is not None
    assert_allclose(frame.euler, [0.1, -0.2, 0.3], atol=1e-12)
    assert frame.near_gimbal_lock is False


def test_flags_gimbal_lock() -> None:
    orientation = tuple(quat.from_euler_zyx([0.0, 0.5 * math.pi, 0.0]).tolist())
    frame = PoseFrame.model_validate(frame_fields(orientation=orientation))
    assert frame.near_gimbal_lock is True


def test_serialized_output_includes_derived_fields() -> None:
    dumped = json.loads(PoseFrame.model_validate(frame_fields()).model_dump_json())
    assert dumped["status"] == "ok"
    assert len(dumped["euler"]) == 3
    assert dumped["near_gimbal_lock"] is False


def test_json_round_trip() -> None:
    frame = PoseFrame.model_validate(frame_fields())
    assert PoseFrame.model_validate_json(frame.model_dump_json()) == frame


def test_derived_fields_on_input_are_recomputed() -> None:
    frame = PoseFrame.model_validate(frame_fields(euler=(9.0, 9.0, 9.0), near_gimbal_lock=True))
    assert frame.euler is not None
    assert_allclose(frame.euler, [0.1, -0.2, 0.3], atol=1e-12)
    assert frame.near_gimbal_lock is False


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"orientation": (1.0, 0.0, 0.0, 0.01)}, "unit quaternion"),
        ({"orientation": (0.0, 0.0, 0.0, 0.0)}, "unit quaternion"),
        ({"joint_angles": (0.1, 0.2)}, "2 joint angles for 1 joint names"),
    ],
)
def test_rejects_inconsistent_frames(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        PoseFrame.model_validate(frame_fields(**overrides))


@pytest.mark.parametrize(
    "overrides",
    [
        {"status": "unknown"},
        {"acceleration": (0.0, math.nan, 0.0)},
        {"angular_velocity": (math.inf, 0.0, 0.0)},
        {"timestamp": -1.0},
        {"extra": 1},
    ],
)
def test_rejects_malformed_fields(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        PoseFrame.model_validate(frame_fields(**overrides))


def test_rejects_non_mapping_input() -> None:
    with pytest.raises(ValidationError):
        PoseFrame.model_validate(42)
