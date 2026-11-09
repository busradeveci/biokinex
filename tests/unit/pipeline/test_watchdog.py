"""Tests for biokinex.pipeline.watchdog."""

from __future__ import annotations

import math

import pytest

from biokinex.core.clock import ManualClock
from biokinex.pipeline.watchdog import Watchdog
from biokinex.schemas.pose import FrameStatus, PoseFrame

FRAME = PoseFrame(
    device_id="imu-0",
    sequence=7,
    timestamp=0.014,
    status=FrameStatus.OK,
    orientation=(1.0, 0.0, 0.0, 0.0),
    acceleration=(0.0, 0.0, 9.8),
)


@pytest.fixture
def clock() -> ManualClock:
    return ManualClock()


def test_silent_before_the_first_frame(clock: ManualClock) -> None:
    watchdog = Watchdog(0.1, clock)
    clock.advance(10.0)
    assert watchdog.check() is None
    assert not watchdog.is_stale


def test_reports_stale_copy_after_timeout(clock: ManualClock) -> None:
    watchdog = Watchdog(0.125, clock)
    watchdog.feed(FRAME)
    clock.advance(0.0625)
    assert watchdog.check() is None
    stale_before = watchdog.is_stale
    clock.advance(0.0625)
    stale = watchdog.check()
    stale_after = watchdog.is_stale
    assert not stale_before
    assert stale_after
    assert stale is not None
    assert stale.status is FrameStatus.STALE
    assert stale.model_dump(exclude={"status"}) == FRAME.model_dump(exclude={"status"})
    assert FRAME.status is FrameStatus.OK


def test_repeats_warning_once_per_timeout(clock: ManualClock) -> None:
    # Powers of two keep the clock sums exact, so the deadlines are hit precisely.
    watchdog = Watchdog(0.125, clock)
    watchdog.feed(FRAME)
    reports = 0
    for _ in range(40):
        clock.advance(0.03125)
        reports += watchdog.check() is not None
    assert reports == 10


def test_fresh_frame_clears_staleness(clock: ManualClock) -> None:
    watchdog = Watchdog(0.1, clock)
    watchdog.feed(FRAME)
    clock.advance(0.2)
    assert watchdog.check() is not None
    watchdog.feed(FRAME)
    assert not watchdog.is_stale
    assert watchdog.check() is None


def test_reset_forgets_the_device(clock: ManualClock) -> None:
    watchdog = Watchdog(0.1, clock)
    watchdog.feed(FRAME)
    watchdog.reset()
    clock.advance(1.0)
    assert watchdog.check() is None
    assert watchdog.timeout_s == 0.1


@pytest.mark.parametrize("timeout", [0.0, -1.0, math.inf, math.nan])
def test_rejects_invalid_timeout(clock: ManualClock, timeout: float) -> None:
    with pytest.raises(ValueError, match="timeout_s"):
        Watchdog(timeout, clock)
