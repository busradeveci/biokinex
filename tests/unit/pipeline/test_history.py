"""Tests for biokinex.pipeline.history."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from biokinex.pipeline.history import FrameHistory, HistoryLayout

FULL = HistoryLayout(
    orientation=True, angular_velocity=True, acceleration=True, joint_names=("knee", "hip")
)
IMU_ONLY = HistoryLayout(
    orientation=False, angular_velocity=True, acceleration=True, joint_names=()
)


def add_full(history: FrameHistory, value: float) -> None:
    history.append(
        FULL,
        value,
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_velocity=np.full(3, value),
        acceleration=np.full(3, -value),
        joint_angles=np.array([value, 2 * value]),
    )


def add_imu(history: FrameHistory, value: float) -> None:
    history.append(
        IMU_ONLY,
        value,
        orientation=None,
        angular_velocity=np.full(3, value),
        acceleration=np.full(3, value),
        joint_angles=None,
    )


def test_empty_history() -> None:
    history = FrameHistory(4)
    window = history.window()
    assert len(history) == 0
    assert history.capacity == 4
    assert history.layout is None
    assert window.timestamps.shape == (0,)
    assert window.orientation is None
    assert window.joint_names == ()


def test_window_holds_newest_frames_oldest_first() -> None:
    history = FrameHistory(3)
    for value in range(5):
        add_full(history, float(value))
    window = history.window()
    assert len(history) == 3
    assert history.layout == FULL
    assert window.timestamps.tolist() == [2.0, 3.0, 4.0]
    assert window.angular_velocity is not None
    assert_allclose(window.angular_velocity[:, 0], [2.0, 3.0, 4.0])
    assert window.acceleration is not None
    assert_allclose(window.acceleration[:, 1], [-2.0, -3.0, -4.0])
    assert window.joint_angles is not None
    assert_allclose(window.joint_angles[-1], [4.0, 8.0])
    assert window.joint_names == ("knee", "hip")
    assert window.orientation is not None
    assert window.orientation.shape == (3, 4)


def test_window_count_and_bounds() -> None:
    history = FrameHistory(8)
    for value in range(4):
        add_imu(history, float(value))
    assert history.window(2).timestamps.tolist() == [2.0, 3.0]
    with pytest.raises(ValueError, match="count must be"):
        history.window(5)


def test_window_is_zero_copy_and_read_only() -> None:
    history = FrameHistory(8)
    add_imu(history, 1.0)
    add_imu(history, 2.0)
    first = history.window()
    second = history.window()
    assert first.acceleration is not None
    assert second.acceleration is not None
    assert np.shares_memory(first.timestamps, second.timestamps)
    assert np.shares_memory(first.acceleration, second.acceleration)
    assert not first.timestamps.flags.writeable
    assert not first.acceleration.flags.writeable


def test_layout_change_clears_history() -> None:
    history = FrameHistory(8)
    add_full(history, 1.0)
    add_imu(history, 2.0)
    window = history.window()
    assert history.layout == IMU_ONLY
    assert window.timestamps.tolist() == [2.0]
    assert window.orientation is None
    assert window.joint_angles is None


def test_rejects_values_that_do_not_match_layout() -> None:
    history = FrameHistory(4)
    with pytest.raises(ValueError, match="do not match"):
        history.append(
            IMU_ONLY,
            0.0,
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
            acceleration=np.zeros(3),
            joint_angles=None,
        )
    with pytest.raises(ValueError, match="do not match"):
        history.append(
            IMU_ONLY,
            0.0,
            orientation=None,
            angular_velocity=None,
            acceleration=np.zeros(3),
            joint_angles=None,
        )


def test_clear_forgets_layout() -> None:
    history = FrameHistory(4)
    add_full(history, 1.0)
    history.clear()
    assert len(history) == 0
    assert history.layout is None
    assert history.window().acceleration is None


def test_memory_does_not_grow_with_frames() -> None:
    history = FrameHistory(16)
    add_imu(history, 0.0)
    window = history.window()
    assert window.acceleration is not None
    storage = window.acceleration.base
    for value in range(1, 200):
        add_imu(history, float(value))
    later = history.window()
    assert later.acceleration is not None
    assert later.acceleration.base is storage
