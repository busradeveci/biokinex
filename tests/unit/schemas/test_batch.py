"""Tests for biokinex.schemas.batch."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from biokinex.schemas.batch import SampleBatch


def make_batch(n: int = 3, **overrides: Any) -> SampleBatch:
    """Build a valid batch of ``n`` samples, with optional field overrides."""
    fields: dict[str, Any] = {
        "device_id": "imu-0",
        "timestamps": np.arange(n, dtype=np.float64) * 0.002,
        "sequences": np.arange(n, dtype=np.int64),
        "acceleration": np.zeros((n, 3)),
        "angular_velocity": np.zeros((n, 3)),
        "orientation": np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)),
        "joint_names": ("knee",),
        "joint_angles": np.zeros((n, 1)),
    }
    fields.update(overrides)
    return SampleBatch(**fields)


def test_valid_batch() -> None:
    batch = make_batch(4)
    assert len(batch) == 4
    assert batch.device_id == "imu-0"


def test_optional_channels_may_be_absent() -> None:
    batch = make_batch(
        2, acceleration=None, angular_velocity=None, orientation=None, joint_angles=None
    )
    assert len(batch) == 2


def test_select_keeps_rows_in_order() -> None:
    batch = make_batch(4, joint_angles=np.array([[0.0], [1.0], [2.0], [3.0]]))
    selected = batch.select(np.array([3, 1], dtype=np.int64))
    assert len(selected) == 2
    assert selected.sequences.tolist() == [3, 1]
    assert selected.joint_names == ("knee",)
    assert selected.joint_angles is not None
    assert selected.joint_angles.tolist() == [[3.0], [1.0]]
    assert selected.acceleration is not None
    assert not selected.acceleration.flags.writeable


def test_select_keeps_absent_channels_absent() -> None:
    batch = make_batch(3, orientation=None, joint_angles=None, joint_names=())
    selected = batch.select(np.array([0], dtype=np.int64))
    assert selected.orientation is None
    assert selected.joint_angles is None


def test_is_frozen() -> None:
    batch = make_batch()
    with pytest.raises(AttributeError):
        batch.device_id = "other"  # type: ignore[misc]


@pytest.mark.parametrize(
    "overrides",
    [
        {"timestamps": np.zeros(0)},
        {"timestamps": np.zeros((3, 1))},
        {"timestamps": np.zeros(3, dtype=np.float32)},
        {"sequences": np.arange(3, dtype=np.int32)},
        {"sequences": np.arange(4, dtype=np.int64)},
        {"acceleration": np.zeros((3, 2))},
        {"angular_velocity": np.zeros((2, 3))},
        {"orientation": np.zeros((3, 3))},
        {"joint_angles": np.zeros((3, 2))},
        {"joint_names": ("knee", "ankle")},
    ],
    ids=[
        "empty",
        "timestamps_2d",
        "timestamps_float32",
        "sequences_int32",
        "sequences_length",
        "acceleration_width",
        "angular_velocity_length",
        "orientation_width",
        "joint_columns",
        "joint_names_mismatch",
    ],
)
def test_rejects_inconsistent_arrays(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="must"):
        make_batch(**overrides)
