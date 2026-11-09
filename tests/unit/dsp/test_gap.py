"""Tests for biokinex.dsp.gap."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_allclose, assert_array_equal

from biokinex.core import quaternion as quat
from biokinex.dsp.gap import GapFiller
from biokinex.schemas.batch import SampleBatch
from biokinex.schemas.common import MAX_SEQUENCE
from biokinex.schemas.config import GapConfig, GapPolicy
from biokinex.validation.limits import SensorLimits
from biokinex.validation.validator import TelemetryValidator

DT = 0.002
Z_AXIS = np.array([0.0, 0.0, 1.0])


def make_batch(
    sequences: Sequence[int],
    *,
    values: Sequence[float] | None = None,
    with_joints: bool = True,
    with_orientation: bool = True,
) -> SampleBatch:
    """Build a batch where every channel is a simple function of ``values``."""
    seq = np.array(sequences, dtype=np.int64)
    v = np.array(values if values is not None else seq, dtype=np.float64)
    angles = 0.1 * v
    return SampleBatch(
        device_id="imu-0",
        timestamps=seq.astype(np.float64) * DT,
        sequences=seq,
        acceleration=np.stack([v, -v, np.full_like(v, 9.8)], axis=1),
        angular_velocity=np.zeros((len(v), 3)),
        orientation=quat.from_axis_angle(Z_AXIS, angles) if with_orientation else None,
        joint_names=("knee",) if with_joints else (),
        joint_angles=v[:, None] if with_joints else None,
    )


def no_restarts(n: int) -> list[bool]:
    return [False] * n


class TestNoGaps:
    def test_batch_is_returned_without_copy(self) -> None:
        filler = GapFiller()
        batch = make_batch([0, 1, 2])
        result = filler.process(batch, [0, 0, 0], [True, False, False])
        assert result.batch is batch
        assert result.synthetic.tolist() == [False, False, False]
        assert result.restarts.tolist() == [True, False, False]
        assert not result.synthetic.flags.writeable
        assert not result.restarts.flags.writeable

    def test_input_restarts_are_not_modified(self) -> None:
        restarts = np.array([False, False])
        GapFiller(max_fill_samples=0).process(make_batch([0, 2]), [0, 1], restarts)
        assert restarts.tolist() == [False, False]


class TestHoldPolicy:
    def test_fills_gap_inside_batch(self) -> None:
        result = GapFiller(GapPolicy.HOLD).process(make_batch([0, 3]), [0, 2], no_restarts(2))
        batch = result.batch
        assert batch.sequences.tolist() == [0, 1, 2, 3]
        assert_allclose(batch.timestamps, [0.0, DT, 2 * DT, 3 * DT])
        assert result.synthetic.tolist() == [False, True, True, False]
        assert result.restarts.tolist() == [False, False, False, False]
        assert batch.acceleration is not None
        assert_allclose(batch.acceleration[:, 0], [0.0, 0.0, 0.0, 3.0])
        assert batch.orientation is not None
        assert_allclose(batch.orientation[1], batch.orientation[0])
        assert batch.joint_angles is not None
        assert_allclose(batch.joint_angles[:, 0], [0.0, 0.0, 0.0, 3.0])

    def test_output_arrays_are_read_only(self) -> None:
        result = GapFiller().process(make_batch([0, 2]), [0, 1], no_restarts(2))
        assert result.batch.acceleration is not None
        for array in (
            result.batch.timestamps,
            result.batch.sequences,
            result.batch.acceleration,
            result.synthetic,
            result.restarts,
        ):
            assert not array.flags.writeable

    def test_absent_channels_stay_absent(self) -> None:
        batch = make_batch([0, 2], with_joints=False, with_orientation=False)
        result = GapFiller().process(batch, [0, 1], no_restarts(2))
        assert result.batch.orientation is None
        assert result.batch.joint_angles is None
        assert result.batch.joint_names == ()


class TestLinearPolicy:
    def test_interpolates_vectors_and_orientation(self) -> None:
        filler = GapFiller(GapPolicy.LINEAR)
        result = filler.process(make_batch([0, 4]), [0, 3], no_restarts(2))
        batch = result.batch
        assert batch.acceleration is not None
        assert_allclose(batch.acceleration[:, 0], [0.0, 1.0, 2.0, 3.0, 4.0])
        assert batch.joint_angles is not None
        assert_allclose(batch.joint_angles[:, 0], [0.0, 1.0, 2.0, 3.0, 4.0])
        assert batch.orientation is not None
        expected = quat.from_axis_angle(Z_AXIS, 0.1 * np.arange(5.0))
        assert_allclose(batch.orientation, expected, atol=1e-12)

    def test_orientation_takes_the_short_way(self) -> None:
        batch = make_batch([0, 2])
        assert batch.orientation is not None
        flipped = SampleBatch(
            device_id=batch.device_id,
            timestamps=batch.timestamps,
            sequences=batch.sequences,
            orientation=batch.orientation * np.array([[1.0], [-1.0]]),
        )
        result = GapFiller(GapPolicy.LINEAR).process(flipped, [0, 1], no_restarts(2))
        assert result.batch.orientation is not None
        middle = result.batch.orientation[1]
        assert quat.angle_between(middle, quat.from_axis_angle(Z_AXIS, 0.1)) < 1e-12

    def test_several_gaps_in_one_batch(self) -> None:
        filler = GapFiller(GapPolicy.LINEAR)
        result = filler.process(make_batch([0, 2, 3, 6]), [0, 1, 0, 2], no_restarts(4))
        assert result.batch.sequences.tolist() == [0, 1, 2, 3, 4, 5, 6]
        assert result.synthetic.tolist() == [False, True, False, False, True, True, False]
        assert result.batch.acceleration is not None
        assert_allclose(result.batch.acceleration[:, 0], np.arange(7.0))


class TestAcrossBatches:
    def test_fills_gap_before_first_sample(self) -> None:
        filler = GapFiller(GapPolicy.LINEAR)
        filler.process(make_batch([0, 1]), [0, 0], [True, False])
        result = filler.process(make_batch([4, 5]), [2, 0], no_restarts(2))
        assert result.batch.sequences.tolist() == [2, 3, 4, 5]
        assert result.batch.acceleration is not None
        assert_allclose(result.batch.acceleration[:, 0], [2.0, 3.0, 4.0, 5.0])

    def test_gap_without_previous_batch_becomes_restart(self) -> None:
        result = GapFiller().process(make_batch([5, 6]), [3, 0], no_restarts(2))
        assert result.batch.sequences.tolist() == [5, 6]
        assert result.restarts.tolist() == [True, False]

    def test_reset_forgets_previous_batch(self) -> None:
        filler = GapFiller()
        filler.process(make_batch([0]), [0], [True])
        filler.reset()
        result = filler.process(make_batch([2]), [1], [False])
        assert result.restarts.tolist() == [True]

    @pytest.mark.parametrize(("with_joints", "with_orientation"), [(False, True), (True, False)])
    def test_layout_change_becomes_restart(self, with_joints: bool, with_orientation: bool) -> None:
        filler = GapFiller()
        filler.process(make_batch([0]), [0], [True])
        changed = make_batch([2], with_joints=with_joints, with_orientation=with_orientation)
        result = filler.process(changed, [1], [False])
        assert len(result.batch) == 1
        assert result.restarts.tolist() == [True]

    def test_sequences_wrap_around(self) -> None:
        filler = GapFiller()
        filler.process(make_batch([MAX_SEQUENCE - 1], values=[0.0]), [0], [True])
        result = filler.process(make_batch([1], values=[0.0]), [2], [False])
        assert result.batch.sequences.tolist() == [MAX_SEQUENCE, 0, 1]


class TestLongGaps:
    def test_gap_longer_than_limit_becomes_restart(self) -> None:
        result = GapFiller(max_fill_samples=2).process(
            make_batch([0, 4, 5]), [0, 3, 0], no_restarts(3)
        )
        assert result.batch.sequences.tolist() == [0, 4, 5]
        assert result.restarts.tolist() == [False, True, False]
        assert not result.synthetic.any()

    def test_gap_at_limit_is_filled(self) -> None:
        result = GapFiller(max_fill_samples=2).process(make_batch([0, 3]), [0, 2], no_restarts(2))
        assert result.synthetic.sum() == 2

    def test_restart_marked_sample_is_not_filled(self) -> None:
        result = GapFiller().process(make_batch([0, 3]), [0, 2], [False, True])
        assert len(result.batch) == 2
        assert result.restarts.tolist() == [False, True]

    def test_long_gap_and_short_gap_together(self) -> None:
        result = GapFiller(max_fill_samples=1).process(
            make_batch([0, 5, 7]), [0, 4, 1], no_restarts(3)
        )
        assert result.batch.sequences.tolist() == [0, 5, 6, 7]
        assert result.restarts.tolist() == [False, True, False, False]
        assert result.synthetic.tolist() == [False, False, True, False]


class TestInputChecks:
    def test_rejects_negative_limit(self) -> None:
        with pytest.raises(ValueError, match="max_fill_samples"):
            GapFiller(max_fill_samples=-1)

    @pytest.mark.parametrize(("missing", "restarts"), [([0], [False, False]), ([0, 0], [False])])
    def test_rejects_mismatched_lengths(self, missing: list[int], restarts: list[bool]) -> None:
        with pytest.raises(ValueError, match="must have shape"):
            GapFiller().process(make_batch([0, 1]), missing, restarts)

    def test_from_config(self) -> None:
        filler = GapFiller.from_config(GapConfig(policy=GapPolicy.LINEAR, max_fill_samples=0))
        result = filler.process(make_batch([0, 2]), [0, 1], no_restarts(2))
        assert result.restarts.tolist() == [False, True]


@given(
    st.lists(st.booleans(), min_size=2, max_size=60),
    st.integers(min_value=1, max_value=6),
    st.sampled_from(list(GapPolicy)),
)
def test_validator_output_becomes_contiguous(
    dropped: list[bool], batch_size: int, policy: GapPolicy
) -> None:
    """Short drops are filled, so the output has no sequence holes except at restarts."""
    kept = [seq for seq, drop in enumerate(dropped) if not drop]
    validator = TelemetryValidator(SensorLimits())
    filler = GapFiller(policy, max_fill_samples=3)
    outputs: list[int] = []
    restarts: list[bool] = []
    for start in range(0, len(kept), batch_size):
        chunk = kept[start : start + batch_size]
        validated = validator.validate(make_batch(chunk, values=[0.01 * s for s in chunk]))
        assert validated.batch is not None
        filled = filler.process(validated.batch, validated.missing_before, validated.restarts)
        outputs.extend(filled.batch.sequences.tolist())
        restarts.extend(filled.restarts.tolist())
        assert_array_equal(np.diff(filled.batch.timestamps) > 0, True)
    for index in range(1, len(outputs)):
        if not restarts[index]:
            assert outputs[index] == outputs[index - 1] + 1
    assert set(kept) <= set(outputs)
