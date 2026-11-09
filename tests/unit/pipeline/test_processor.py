"""Tests for biokinex.pipeline.processor."""

from __future__ import annotations

import math
import tracemalloc
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_allclose

from biokinex.core import quaternion as quat
from biokinex.core.errors import TelemetryError
from biokinex.core.types import STANDARD_GRAVITY
from biokinex.dsp.base import apply_to_batch
from biokinex.dsp.biquad import BiquadCoefficients, BiquadFilter
from biokinex.dsp.complementary import ComplementaryFilter
from biokinex.pipeline.processor import PipelineProcessor
from biokinex.schemas.batch import SampleBatch
from biokinex.schemas.config import PipelineConfig
from biokinex.schemas.pose import FrameStatus, PoseFrame
from biokinex.schemas.telemetry import TelemetryPacket
from biokinex.validation.validator import Verdict

RATE_HZ = 500.0
DT = 1.0 / RATE_HZ
Z_AXIS = np.array([0.0, 0.0, 1.0])
GRAVITY = np.array([0.0, 0.0, STANDARD_GRAVITY])


def make_config(**fields: Any) -> PipelineConfig:
    """Build a 500 Hz config; fusion is off unless a test turns it on."""
    raw: dict[str, Any] = {"sample_rate_hz": RATE_HZ, "fusion": {"enabled": False}}
    raw.update(fields)
    return PipelineConfig.model_validate(raw)


def make_batch(
    sequences: Sequence[int],
    *,
    acceleration: np.ndarray | None = None,
    angular_velocity: np.ndarray | None = None,
    orientation: np.ndarray | None = None,
    joint_names: tuple[str, ...] = (),
    joint_angles: np.ndarray | None = None,
    imu: bool = True,
    device_id: str = "imu-0",
) -> SampleBatch:
    """Build a batch at 500 Hz; by default it has a resting IMU reading per sample."""
    seq = np.array(sequences, dtype=np.int64)
    n = len(seq)
    if imu and acceleration is None:
        acceleration = np.tile(GRAVITY, (n, 1))
    if imu and angular_velocity is None:
        angular_velocity = np.zeros((n, 3))
    return SampleBatch(
        device_id=device_id,
        timestamps=seq.astype(np.float64) * DT,
        sequences=seq,
        acceleration=acceleration,
        angular_velocity=angular_velocity,
        orientation=orientation,
        joint_names=joint_names,
        joint_angles=joint_angles,
    )


def statuses(frames: Sequence[PoseFrame]) -> list[FrameStatus]:
    return [frame.status for frame in frames]


def stacked(frames: Sequence[PoseFrame], name: str) -> npt.NDArray[np.float64]:
    """Stack one channel of every frame into an array; the channel must be present."""
    rows = [getattr(frame, name) for frame in frames]
    assert all(row is not None for row in rows)
    return np.array(rows, dtype=np.float64)


class TestCleanStream:
    def test_frames_pass_values_through_without_filters(self) -> None:
        processor = PipelineProcessor(make_config())
        accel = np.array([[0.1, 0.2, 9.8], [0.3, 0.4, 9.7], [0.5, 0.6, 9.6]])
        orientation = quat.from_axis_angle(Z_AXIS, np.array([0.1, 0.2, 0.3]))
        result = processor.process(
            make_batch([0, 1, 2], acceleration=accel, orientation=orientation)
        )
        frames = result.frames
        assert [f.sequence for f in frames] == [0, 1, 2]
        assert [f.timestamp for f in frames] == pytest.approx([0.0, DT, 2 * DT])
        assert statuses(frames) == [FrameStatus.OK] * 3
        assert_allclose(stacked(frames, "acceleration"), accel)
        assert_allclose(stacked(frames, "orientation"), orientation, atol=1e-15)
        assert frames[0].device_id == "imu-0"
        assert result.synthetic_count == 0
        assert result.validation.accepted_count == 3

    def test_frames_are_split_across_batches(self) -> None:
        processor = PipelineProcessor(make_config())
        first = processor.process(make_batch([0, 1]))
        second = processor.process(make_batch([2, 3]))
        assert [f.sequence for f in first.frames + second.frames] == [0, 1, 2, 3]
        assert statuses(second.frames) == [FrameStatus.OK] * 2

    def test_process_packet_converts_units(self) -> None:
        packet = TelemetryPacket.model_validate(
            {
                "device_id": "imu-7",
                "units": {"acceleration": "g", "angular_velocity": "deg/s"},
                "samples": [
                    {
                        "sequence": 0,
                        "timestamp": 0.0,
                        "acceleration": [0.0, 0.0, 1.0],
                        "angular_velocity": [0.0, 0.0, 180.0],
                    }
                ],
            }
        )
        frame = PipelineProcessor(make_config()).process_packet(packet).frames[0]
        assert frame.acceleration == pytest.approx((0.0, 0.0, STANDARD_GRAVITY))
        assert frame.angular_velocity == pytest.approx((0.0, 0.0, math.pi))
        assert frame.orientation is None


class TestFilters:
    def test_channel_chains_match_direct_filters(self) -> None:
        config = make_config(
            acceleration_filters=[{"kind": "low_pass", "cutoff_hz": 20}],
            angular_velocity_filters=[{"kind": "median", "window": 3}],
            joint_angle_filters=[{"kind": "slew_rate", "max_rate": 5.0}],
        )
        rng = np.random.default_rng(1)
        n = 40
        accel = GRAVITY + rng.normal(0.0, 0.5, (n, 3))
        gyro = rng.normal(0.0, 0.2, (n, 3))
        joints = np.linspace(0.0, 1.0, n)[:, None]
        batch = make_batch(
            range(n),
            acceleration=accel,
            angular_velocity=gyro,
            joint_names=("knee",),
            joint_angles=joints,
        )
        frames = PipelineProcessor(config).process(batch).frames
        times = batch.timestamps
        expected_accel = apply_to_batch(
            BiquadFilter(BiquadCoefficients.low_pass(20.0, RATE_HZ)), accel, times
        )
        assert_allclose(stacked(frames, "acceleration"), expected_accel)
        assert frames[2].angular_velocity == pytest.approx(tuple(np.median(gyro[:3], axis=0)))
        joint_steps = np.diff([f.joint_angles[0] for f in frames if f.joint_angles])
        assert np.all(joint_steps <= 5.0 * DT + 1e-12)
        assert frames[0].joint_names == ("knee",)

    def test_orientation_chain_limits_device_orientation(self) -> None:
        config = make_config(orientation_filters=[{"kind": "slew_rate", "max_rate": 1.0}])
        orientation = quat.from_axis_angle(Z_AXIS, np.array([0.0, 1.0]))
        frames = (
            PipelineProcessor(config).process(make_batch([0, 1], orientation=orientation)).frames
        )
        assert frames[1].orientation is not None
        assert quat.angle_between(frames[0].orientation, frames[1].orientation) == pytest.approx(
            1.0 * DT
        )


class TestFusion:
    def test_fusion_matches_complementary_filter(self) -> None:
        config = make_config(fusion={"enabled": True, "gain": 2.0})
        n = 50
        gyro = np.tile([0.0, 0.0, 1.0], (n, 1))
        tilted = np.tile([0.0, 0.5, 9.79], (n, 1))
        batch = make_batch(range(n), acceleration=tilted, angular_velocity=gyro)
        frames = PipelineProcessor(config).process(batch).frames
        reference = ComplementaryFilter(gain=2.0)
        expected = [
            reference.step(gyro[i], tilted[i], float(batch.timestamps[i])) for i in range(n)
        ]
        assert_allclose(stacked(frames, "orientation"), np.array(expected), atol=1e-12)

    def test_fusion_replaces_device_orientation(self) -> None:
        config = make_config(fusion={"enabled": True})
        device = quat.from_axis_angle(Z_AXIS, np.array([2.0, 2.0]))
        frames = PipelineProcessor(config).process(make_batch([0, 1], orientation=device)).frames
        assert frames[0].orientation == pytest.approx((1.0, 0.0, 0.0, 0.0))

    def test_device_orientation_is_used_without_gyroscope(self) -> None:
        config = make_config(fusion={"enabled": True})
        device = quat.from_axis_angle(Z_AXIS, np.array([0.5]))
        frames = (
            PipelineProcessor(config).process(make_batch([0], orientation=device, imu=False)).frames
        )
        assert_allclose(stacked(frames, "orientation")[0], device[0], atol=1e-15)

    def test_no_orientation_source_gives_none(self) -> None:
        config = make_config(fusion={"enabled": True})
        joints = np.array([[0.2]])
        frames = (
            PipelineProcessor(config)
            .process(make_batch([0], imu=False, joint_names=("knee",), joint_angles=joints))
            .frames
        )
        assert frames[0].orientation is None
        assert frames[0].euler is None
        assert frames[0].joint_angles == (0.2,)


class TestOrientationOutput:
    def test_output_stays_in_one_hemisphere(self) -> None:
        q = quat.from_axis_angle(Z_AXIS, np.array([0.1, 0.2, 0.3, 0.4]))
        q[1] *= -1.0
        q[3] *= -1.0
        frames = PipelineProcessor(make_config()).process(make_batch([0, 1, 2, 3], orientation=q))
        outputs = np.array([f.orientation for f in frames.frames])
        assert np.all(np.sum(outputs[1:] * outputs[:-1], axis=1) > 0.0)
        assert np.all(outputs[:, 0] > 0.0)

    def test_non_unit_orientation_is_normalized(self) -> None:
        q = np.array([[1.05, 0.0, 0.0, 0.0]])
        frame = PipelineProcessor(make_config()).process(make_batch([0], orientation=q)).frames[0]
        assert frame.orientation == pytest.approx((1.0, 0.0, 0.0, 0.0))


class TestGaps:
    def test_short_gap_is_filled_and_marks_next_frame_degraded(self) -> None:
        config = make_config(
            gap={"policy": "linear"},
            acceleration_filters=[{"kind": "low_pass", "cutoff_hz": 30}],
        )
        processor = PipelineProcessor(config)
        processor.process(make_batch([0, 1]))
        result = processor.process(make_batch([4, 5]))
        assert [f.sequence for f in result.frames] == [4, 5]
        assert statuses(result.frames) == [FrameStatus.DEGRADED, FrameStatus.OK]
        assert result.synthetic_count == 2
        stats = processor.stats()
        assert stats.synthetic_samples == 2
        assert stats.degraded_frames == 1
        assert stats.frames == 4
        assert stats.validation.missing_samples == 2

    def test_filters_see_the_filled_samples(self) -> None:
        config = make_config(
            gap={"policy": "linear"},
            acceleration_filters=[{"kind": "low_pass", "cutoff_hz": 30}],
        )
        accel = np.array([[0.0, 0.0, 9.0], [0.0, 0.0, 10.0]])
        frames = PipelineProcessor(config).process(make_batch([0, 3], acceleration=accel)).frames
        filled = np.array([[0, 0, 9.0], [0, 0, 9 + 1 / 3], [0, 0, 9 + 2 / 3], [0, 0, 10.0]])
        expected = apply_to_batch(
            BiquadFilter(BiquadCoefficients.low_pass(30.0, RATE_HZ)), filled, np.arange(4) * DT
        )
        assert frames[1].acceleration == pytest.approx(tuple(expected[3]))

    def test_long_gap_restarts_filters(self) -> None:
        config = make_config(
            gap={"max_fill_samples": 2},
            acceleration_filters=[{"kind": "slew_rate", "max_rate": 1.0}],
        )
        processor = PipelineProcessor(config)
        processor.process(make_batch([0, 1]))
        jumped = np.array([[5.0, 0.0, 9.8]])
        result = processor.process(make_batch([10], acceleration=jumped))
        frame = result.frames[0]
        # A fresh slew limiter passes the first value straight through.
        assert frame.acceleration == pytest.approx((5.0, 0.0, 9.8))
        assert frame.status is FrameStatus.DEGRADED
        assert result.synthetic_count == 0
        assert processor.stats().filter_restarts == 2

    def test_rejected_sample_is_filled(self) -> None:
        processor = PipelineProcessor(make_config(limits={"max_acceleration": 50.0}))
        accel = np.tile(GRAVITY, (3, 1))
        accel[1, 0] = 80.0
        result = processor.process(make_batch([0, 1, 2], acceleration=accel))
        assert result.validation.counts()[Verdict.ACCELERATION_OUT_OF_RANGE] == 1
        assert [f.sequence for f in result.frames] == [0, 2]
        assert statuses(result.frames) == [FrameStatus.OK, FrameStatus.DEGRADED]
        assert result.synthetic_count == 1

    def test_fully_rejected_batch_gives_no_frames(self) -> None:
        processor = PipelineProcessor(make_config())
        processor.process(make_batch([0, 1]))
        result = processor.process(make_batch([1]))
        assert result.frames == ()
        assert result.validation.counts()[Verdict.DUPLICATE] == 1
        assert result.synthetic_count == 0

    def test_sequence_settings_come_from_config(self) -> None:
        processor = PipelineProcessor(make_config(sequence={"max_forward_jump": 5}))
        processor.process(make_batch([0]))
        result = processor.process(make_batch([20]))
        # A jump larger than max_forward_jump is a device restart, not a gap to fill.
        assert result.synthetic_count == 0
        assert result.validation.missing_before.tolist() == [0]
        assert result.validation.restarts.tolist() == [True]
        assert result.frames[0].status is FrameStatus.DEGRADED


class TestLayoutChanges:
    def test_new_channel_restarts_filters_and_history(self) -> None:
        config = make_config(joint_angle_filters=[{"kind": "slew_rate", "max_rate": 1.0}])
        processor = PipelineProcessor(config)
        processor.process(make_batch([0, 1]))
        assert processor.history.layout is not None
        assert processor.history.layout.joint_names == ()
        joints = np.array([[1.0], [1.5]])
        result = processor.process(make_batch([2, 3], joint_names=("knee",), joint_angles=joints))
        assert statuses(result.frames) == [FrameStatus.DEGRADED, FrameStatus.OK]
        assert result.frames[0].joint_angles == (1.0,)
        window = processor.history.window()
        assert window.timestamps.tolist() == pytest.approx([2 * DT, 3 * DT])
        assert window.joint_names == ("knee",)

    def test_empty_joint_list_is_treated_as_no_joints(self) -> None:
        batch = make_batch([0], joint_names=(), joint_angles=np.zeros((1, 0)))
        frame = PipelineProcessor(make_config()).process(batch).frames[0]
        assert frame.joint_angles is None
        assert frame.joint_names == ()


class TestStateAndHistory:
    def test_history_matches_frames(self) -> None:
        processor = PipelineProcessor(make_config(history_size=3))
        orientation = quat.from_axis_angle(Z_AXIS, np.linspace(0.0, 0.4, 5))
        frames = processor.process(make_batch(range(5), orientation=orientation)).frames
        window = processor.history.window()
        assert window.timestamps.tolist() == [f.timestamp for f in frames[-3:]]
        assert window.orientation is not None
        assert_allclose(window.orientation, stacked(frames[-3:], "orientation"))
        assert window.acceleration is not None
        assert_allclose(window.acceleration, stacked(frames[-3:], "acceleration"))

    def test_rejects_other_device(self) -> None:
        processor = PipelineProcessor(make_config())
        processor.process(make_batch([0]))
        with pytest.raises(TelemetryError, match="bound to device"):
            processor.process(make_batch([1], device_id="imu-1"))

    def test_reset_starts_fresh_but_keeps_totals(self) -> None:
        processor = PipelineProcessor(make_config())
        processor.process(make_batch([0, 1]))
        processor.reset()
        assert len(processor.history) == 0
        result = processor.process(make_batch([100], device_id="imu-1"))
        assert result.frames[0].status is FrameStatus.OK
        assert processor.stats().frames == 3

    def test_config_property(self) -> None:
        config = make_config()
        assert PipelineProcessor(config).config is config


def noisy_stream(seed: int, n: int, drops: Sequence[bool]) -> list[SampleBatch]:
    """Build a noisy IMU stream with some samples dropped, split into batches of 5."""
    rng = np.random.default_rng(seed)
    kept = [i for i in range(n) if not drops[i % len(drops)]]
    batches = []
    for start in range(0, len(kept), 5):
        seq = kept[start : start + 5]
        m = len(seq)
        batches.append(
            make_batch(
                seq,
                acceleration=GRAVITY + rng.normal(0.0, 1.0, (m, 3)),
                angular_velocity=rng.normal(0.0, 2.0, (m, 3)),
                joint_names=("knee",),
                joint_angles=rng.uniform(-1.0, 1.0, (m, 1)),
            )
        )
    return batches


FULL_CONFIG: dict[str, Any] = {
    "acceleration_filters": [
        {"kind": "median", "window": 3},
        {"kind": "low_pass", "cutoff_hz": 25},
    ],
    "angular_velocity_filters": [{"kind": "notch", "center_hz": 50, "q": 4}, {"kind": "one_euro"}],
    "joint_angle_filters": [
        {"kind": "one_euro", "beta": 0.1},
        {"kind": "slew_rate", "max_rate": 4},
    ],
    "orientation_filters": [
        {"kind": "one_euro", "beta": 0.05},
        {"kind": "slew_rate", "max_rate": 8},
    ],
    "fusion": {"enabled": True, "gain": 1.5},
    "gap": {"policy": "linear", "max_fill_samples": 2},
}


def test_memory_does_not_grow_during_streaming() -> None:
    config = PipelineConfig.model_validate({"sample_rate_hz": RATE_HZ, **FULL_CONFIG})
    processor = PipelineProcessor(config)
    batches = noisy_stream(seed=3, n=3000, drops=[False] * 9 + [True])
    for batch in batches[:100]:
        processor.process(batch)
    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        for batch in batches[100:]:
            processor.process(batch)
        growth = tracemalloc.get_traced_memory()[0] - before
    finally:
        tracemalloc.stop()
    assert growth < 64 * 1024


@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    drops=st.lists(st.booleans(), min_size=1, max_size=12),
)
def test_full_pipeline_is_deterministic_and_always_valid(seed: int, drops: list[bool]) -> None:
    config = PipelineConfig.model_validate({"sample_rate_hz": RATE_HZ, **FULL_CONFIG})
    runs = []
    for _ in range(2):
        processor = PipelineProcessor(config)
        frames = [f for b in noisy_stream(seed, 40, drops) for f in processor.process(b).frames]
        runs.append([f.model_dump() for f in frames])
        for frame in frames:
            assert frame.orientation is not None
            assert quat.is_unit(np.array(frame.orientation))
            assert frame.angular_velocity is not None
            assert all(math.isfinite(v) for v in frame.angular_velocity)
    assert runs[0] == runs[1]
