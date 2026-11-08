"""Tests for biokinex.validation.sequence."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from biokinex.schemas.common import MAX_SEQUENCE
from biokinex.validation.sequence import (
    DEFAULT_MAX_FORWARD_JUMP,
    DEFAULT_REORDER_WINDOW,
    SEQUENCE_MODULUS,
    SequenceCheck,
    SequenceStatus,
    SequenceTracker,
)

sequences = st.integers(0, MAX_SEQUENCE)


def tracker_at(last: int, **kwargs: int) -> SequenceTracker:
    """Return a tracker whose last accepted number is ``last``."""
    tracker = SequenceTracker(**kwargs)
    tracker.advance(last)
    return tracker


def test_first_number_is_first() -> None:
    tracker = SequenceTracker()
    assert tracker.last is None
    assert tracker.classify(123) == SequenceCheck(SequenceStatus.FIRST, 0)


@pytest.mark.parametrize(
    ("last", "sequence", "expected"),
    [
        (10, 11, SequenceCheck(SequenceStatus.IN_ORDER, 0)),
        (10, 10, SequenceCheck(SequenceStatus.DUPLICATE, 0)),
        (10, 13, SequenceCheck(SequenceStatus.GAP, 2)),
        (10, 10 + DEFAULT_MAX_FORWARD_JUMP, SequenceCheck(SequenceStatus.GAP, 999)),
        (10, 11 + DEFAULT_MAX_FORWARD_JUMP, SequenceCheck(SequenceStatus.RESYNC, 0)),
        (100, 99, SequenceCheck(SequenceStatus.LATE, 0)),
        (100, 100 - DEFAULT_REORDER_WINDOW, SequenceCheck(SequenceStatus.LATE, 0)),
        (100, 99 - DEFAULT_REORDER_WINDOW, SequenceCheck(SequenceStatus.RESYNC, 0)),
        (5_000, 0, SequenceCheck(SequenceStatus.RESYNC, 0)),
    ],
    ids=[
        "in_order",
        "duplicate",
        "gap",
        "largest_gap",
        "jump_too_far",
        "late",
        "oldest_late",
        "too_old",
        "device_restart",
    ],
)
def test_classify(last: int, sequence: int, expected: SequenceCheck) -> None:
    assert tracker_at(last).classify(sequence) == expected


@pytest.mark.parametrize(
    ("last", "sequence", "expected"),
    [
        (MAX_SEQUENCE, 0, SequenceCheck(SequenceStatus.IN_ORDER, 0)),
        (MAX_SEQUENCE - 1, 1, SequenceCheck(SequenceStatus.GAP, 2)),
        (1, MAX_SEQUENCE - 1, SequenceCheck(SequenceStatus.LATE, 0)),
        (0, MAX_SEQUENCE, SequenceCheck(SequenceStatus.LATE, 0)),
    ],
    ids=["wrap_in_order", "wrap_gap", "late_across_wrap", "one_behind_zero"],
)
def test_classify_across_wrap(last: int, sequence: int, expected: SequenceCheck) -> None:
    assert tracker_at(last).classify(sequence) == expected


def test_classify_does_not_change_state() -> None:
    tracker = tracker_at(10)
    tracker.classify(11)
    tracker.classify(500)
    assert tracker.last == 10


def test_advance_and_reset() -> None:
    tracker = SequenceTracker()
    tracker.advance(7)
    assert tracker.last == 7
    tracker.reset()
    assert tracker.classify(8).status is SequenceStatus.FIRST


@pytest.mark.parametrize("sequence", [-1, MAX_SEQUENCE + 1])
def test_advance_rejects_out_of_range(sequence: int) -> None:
    with pytest.raises(ValueError, match="sequence must be in"):
        SequenceTracker().advance(sequence)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reorder_window": -1},
        {"reorder_window": SEQUENCE_MODULUS // 2},
        {"max_forward_jump": 0},
        {"max_forward_jump": SEQUENCE_MODULUS // 2},
    ],
)
def test_rejects_invalid_settings(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="must be in"):
        SequenceTracker(**kwargs)


def test_zero_reorder_window_treats_any_step_back_as_resync() -> None:
    assert tracker_at(10, reorder_window=0).classify(9).status is SequenceStatus.RESYNC


@given(sequences, st.integers(1, DEFAULT_MAX_FORWARD_JUMP))
def test_forward_steps_report_lost_samples(last: int, step: int) -> None:
    check = tracker_at(last).classify((last + step) % SEQUENCE_MODULUS)
    assert check.missing == step - 1
    assert check.status is (SequenceStatus.IN_ORDER if step == 1 else SequenceStatus.GAP)


@given(sequences, st.integers(1, DEFAULT_REORDER_WINDOW))
def test_small_backward_steps_are_late(last: int, step: int) -> None:
    assert tracker_at(last).classify((last - step) % SEQUENCE_MODULUS).status is (
        SequenceStatus.LATE
    )


@given(sequences, st.integers(DEFAULT_MAX_FORWARD_JUMP + 1, SEQUENCE_MODULUS - 65))
def test_everything_else_is_resync(last: int, step: int) -> None:
    check = tracker_at(last).classify((last + step) % SEQUENCE_MODULUS)
    assert check == SequenceCheck(SequenceStatus.RESYNC, 0)
