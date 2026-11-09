"""Tests for biokinex.validation.validator."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_allclose

from biokinex.core.errors import TelemetryError
from biokinex.schemas.batch import SampleBatch
from biokinex.schemas.common import MAX_SEQUENCE
from biokinex.schemas.limits import SensorLimits
from biokinex.validation.validator import TelemetryValidator, ValidationResult, Verdict

DT = 0.002
LIMITS = SensorLimits(
    max_acceleration=100.0, max_angular_velocity=10.0, min_joint_angle=0.0, max_joint_angle=2.0
)


def make_batch(
    sequences: Sequence[int],
    timestamps: Sequence[float] | None = None,
    *,
    acceleration: Sequence[Sequence[float]] | None = None,
    angular_velocity: Sequence[Sequence[float]] | None = None,
    joint_angles: Sequence[Sequence[float]] | None = None,
    device_id: str = "imu-0",
) -> SampleBatch:
    """Build a batch; timestamps default to ``sequence * DT`` and readings to safe values."""
    n = len(sequences)
    if timestamps is None:
        timestamps = [s * DT for s in sequences]
    return SampleBatch(
        device_id=device_id,
        timestamps=np.array(timestamps, dtype=np.float64),
        sequences=np.array(sequences, dtype=np.int64),
        acceleration=np.array(acceleration or [[0.0, 0.0, 9.8]] * n, dtype=np.float64),
        angular_velocity=np.array(angular_velocity or [[0.0] * 3] * n, dtype=np.float64),
        joint_names=("knee",),
        joint_angles=np.array(joint_angles or [[1.0]] * n, dtype=np.float64),
    )


def accepted_sequences(result: ValidationResult) -> list[int]:
    """Return the sequence numbers of the accepted samples."""
    return [] if result.batch is None else result.batch.sequences.tolist()


@pytest.fixture
def validator() -> TelemetryValidator:
    return TelemetryValidator(LIMITS)


class TestCleanStream:
    def test_first_batch_marks_a_restart(self, validator: TelemetryValidator) -> None:
        result = validator.validate(make_batch([0, 1, 2]))
        assert accepted_sequences(result) == [0, 1, 2]
        assert result.restarts.tolist() == [True, False, False]
        assert result.missing_before.tolist() == [0, 0, 0]
        assert validator.device_id == "imu-0"

    def test_in_order_batch_is_returned_without_copy(self, validator: TelemetryValidator) -> None:
        validator.validate(make_batch([0, 1]))
        batch = make_batch([2, 3, 4])
        result = validator.validate(batch)
        assert result.batch is batch
        assert result.accepted_count == 3
        assert result.rejected_count == 0
        assert not result.restarts.any()

    def test_stream_continues_across_wrap(self, validator: TelemetryValidator) -> None:
        validator.validate(make_batch([MAX_SEQUENCE - 1], [1.0]))
        batch = make_batch([MAX_SEQUENCE, 0, 1], [1.002, 1.004, 1.006])
        result = validator.validate(batch)
        assert result.batch is batch
        assert result.missing_before.tolist() == [0, 0, 0]

    def test_result_arrays_are_read_only(self, validator: TelemetryValidator) -> None:
        result = validator.validate(make_batch([0, 1]))
        for array in (result.verdicts, result.missing_before, result.restarts):
            with pytest.raises(ValueError, match="read-only"):
                array[0] = 0

    def test_batch_without_optional_channels(self, validator: TelemetryValidator) -> None:
        batch = SampleBatch(
            device_id="imu-0",
            timestamps=np.array([0.0, DT]),
            sequences=np.array([0, 1], dtype=np.int64),
        )
        assert validator.validate(batch).accepted_count == 2


class TestOrder:
    def test_gap_across_batches(self, validator: TelemetryValidator) -> None:
        validator.validate(make_batch([0, 1]))
        result = validator.validate(make_batch([5, 6]))
        assert accepted_sequences(result) == [5, 6]
        assert result.missing_before.tolist() == [3, 0]
        assert result.restarts.tolist() == [False, False]

    def test_duplicates_are_rejected(self, validator: TelemetryValidator) -> None:
        validator.validate(make_batch([0, 1]))
        result = validator.validate(make_batch([1, 2, 2, 3], [0.002, 0.004, 0.0041, 0.006]))
        assert accepted_sequences(result) == [2, 3]
        assert result.verdicts.tolist() == [Verdict.DUPLICATE, 0, Verdict.DUPLICATE, 0]

    def test_late_samples_are_rejected(self, validator: TelemetryValidator) -> None:
        validator.validate(make_batch([10, 11, 12]))
        result = validator.validate(make_batch([9, 13]))
        assert accepted_sequences(result) == [13]
        assert result.verdicts.tolist() == [Verdict.LATE, Verdict.ACCEPTED]

    def test_device_restart_resyncs_and_allows_time_reset(
        self, validator: TelemetryValidator
    ) -> None:
        validator.validate(make_batch([5_000, 5_001]))
        result = validator.validate(make_batch([0, 1], [0.0, DT]))
        assert accepted_sequences(result) == [0, 1]
        assert result.restarts.tolist() == [True, False]
        assert result.missing_before.tolist() == [0, 0]

    def test_non_monotonic_time_is_rejected(self, validator: TelemetryValidator) -> None:
        validator.validate(make_batch([0, 1], [0.0, 0.002]))
        result = validator.validate(make_batch([2, 3], [0.002, 0.006]))
        assert result.verdicts.tolist() == [Verdict.NON_MONOTONIC_TIME, Verdict.ACCEPTED]
        assert result.missing_before.tolist() == [1]

    def test_unsorted_timestamps_inside_batch(self, validator: TelemetryValidator) -> None:
        validator.validate(make_batch([0]))
        result = validator.validate(make_batch([1, 2, 3], [0.004, 0.003, 0.006]))
        assert result.verdicts.tolist() == [0, Verdict.NON_MONOTONIC_TIME, 0]

    def test_recovers_from_timestamp_far_in_the_future(self) -> None:
        validator = TelemetryValidator(LIMITS, max_consecutive_time_errors=3)
        validator.validate(make_batch([0, 1]))
        validator.validate(make_batch([2], [1e6]))
        result = validator.validate(make_batch([3, 4, 5, 6, 7]))
        assert result.verdicts.tolist() == [Verdict.NON_MONOTONIC_TIME] * 3 + [0, 0]
        assert result.restarts.tolist() == [True, False]
        assert result.missing_before.tolist() == [0, 0]

    def test_time_error_count_resets_after_good_sample(self) -> None:
        validator = TelemetryValidator(LIMITS, max_consecutive_time_errors=2)
        validator.validate(make_batch([0, 1, 2], [1.0, 2.0, 3.0]))
        result = validator.validate(make_batch([3, 4, 5, 6, 7], [2.5, 3.5, 3.1, 3.2, 3.3]))
        assert result.verdicts.tolist() == [Verdict.NON_MONOTONIC_TIME, 0] + [
            Verdict.NON_MONOTONIC_TIME
        ] * 2 + [0]
        assert result.restarts.tolist() == [False, True]


class TestPhysicalLimits:
    @pytest.mark.parametrize(
        ("readings", "verdict"),
        [
            ({"acceleration": [[0.0, -100.5, 0.0]]}, Verdict.ACCELERATION_OUT_OF_RANGE),
            ({"angular_velocity": [[0.0, 0.0, 10.1]]}, Verdict.ANGULAR_VELOCITY_OUT_OF_RANGE),
            ({"joint_angles": [[-0.1]]}, Verdict.JOINT_ANGLE_OUT_OF_RANGE),
            ({"joint_angles": [[2.1]]}, Verdict.JOINT_ANGLE_OUT_OF_RANGE),
        ],
    )
    def test_out_of_range_readings_are_rejected(
        self,
        validator: TelemetryValidator,
        readings: dict[str, list[list[float]]],
        verdict: Verdict,
    ) -> None:
        result = validator.validate(make_batch([0], **readings))  # type: ignore[arg-type]
        assert result.batch is None
        assert result.verdicts.tolist() == [verdict]

    def test_values_at_the_limit_are_accepted(self, validator: TelemetryValidator) -> None:
        batch = make_batch(
            [0],
            acceleration=[[100.0, -100.0, 0.0]],
            angular_velocity=[[-10.0, 10.0, 0.0]],
            joint_angles=[[2.0]],
        )
        assert validator.validate(batch).accepted_count == 1

    def test_first_failing_check_is_reported(self, validator: TelemetryValidator) -> None:
        batch = make_batch([0], acceleration=[[500.0, 0.0, 0.0]], angular_velocity=[[50.0] * 3])
        assert validator.validate(batch).verdicts.tolist() == [Verdict.ACCELERATION_OUT_OF_RANGE]

    def test_rejected_sample_counts_as_missing(self, validator: TelemetryValidator) -> None:
        validator.validate(make_batch([0]))
        batch = make_batch(
            [1, 2, 3], acceleration=[[0.0, 0.0, 9.8], [0.0, 0.0, 999.0], [0.0, 0.0, 9.8]]
        )
        result = validator.validate(batch)
        assert accepted_sequences(result) == [1, 3]
        assert result.missing_before.tolist() == [0, 1]
        assert result.batch is not None
        assert result.batch.acceleration is not None
        assert_allclose(result.batch.acceleration[:, 2], [9.8, 9.8])

    def test_uses_default_limits(self) -> None:
        validator = TelemetryValidator()
        assert validator.limits == SensorLimits()
        assert validator.validate(make_batch([0], joint_angles=[[3.0]])).accepted_count == 1


class TestBookkeeping:
    def test_counts(self, validator: TelemetryValidator) -> None:
        validator.validate(make_batch([0]))
        result = validator.validate(
            make_batch([0, 1, 2], acceleration=[[0, 0, 1]] * 2 + [[999] * 3])
        )
        counts = result.counts()
        assert counts[Verdict.DUPLICATE] == 1
        assert counts[Verdict.ACCEPTED] == 1
        assert counts[Verdict.ACCELERATION_OUT_OF_RANGE] == 1
        assert sum(counts.values()) == 3

    def test_stats_accumulate(self, validator: TelemetryValidator) -> None:
        validator.validate(make_batch([0, 1]))
        validator.validate(make_batch([1, 4]))
        stats = validator.stats()
        assert stats.verdicts[Verdict.ACCEPTED] == 3
        assert stats.verdicts[Verdict.DUPLICATE] == 1
        assert stats.missing_samples == 2
        assert stats.restarts == 1

    def test_rejects_other_device(self, validator: TelemetryValidator) -> None:
        validator.validate(make_batch([0]))
        with pytest.raises(TelemetryError, match="bound to device 'imu-0'"):
            validator.validate(make_batch([1], device_id="imu-1"))

    def test_reset_forgets_stream_but_keeps_stats(self, validator: TelemetryValidator) -> None:
        validator.validate(make_batch([10, 11]))
        validator.reset()
        assert validator.device_id is None
        result = validator.validate(make_batch([0], [0.0], device_id="imu-1"))
        assert result.restarts.tolist() == [True]
        assert validator.stats().verdicts[Verdict.ACCEPTED] == 3

    def test_rejects_invalid_time_error_limit(self) -> None:
        with pytest.raises(ValueError, match="max_consecutive_time_errors"):
            TelemetryValidator(max_consecutive_time_errors=0)


@given(st.lists(st.integers(0, 40), min_size=1, max_size=60))
def test_accepted_stream_is_always_strictly_ordered(order: list[int]) -> None:
    """Whatever arrives, the accepted output never goes back in sequence or time."""
    validator = TelemetryValidator(LIMITS)
    accepted: list[tuple[int, float]] = []
    for start in range(0, len(order), 7):
        chunk = order[start : start + 7]
        result = validator.validate(make_batch(chunk, [1.0 + s * DT for s in chunk]))
        assert result.accepted_count + result.rejected_count == len(chunk)
        assert len(result.missing_before) == result.accepted_count
        if result.batch is not None:
            accepted.extend(
                zip(result.batch.sequences.tolist(), result.batch.timestamps.tolist(), strict=True)
            )
    sequences = [s for s, _ in accepted]
    timestamps = [t for _, t in accepted]
    assert sequences == sorted(set(sequences))
    assert timestamps == sorted(set(timestamps))
