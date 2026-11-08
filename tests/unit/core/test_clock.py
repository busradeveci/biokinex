"""Tests for biokinex.core.clock."""

from __future__ import annotations

import math

import pytest

from biokinex.core.clock import Clock, ManualClock, MonotonicClock


class TestMonotonicClock:
    def test_satisfies_clock_protocol(self) -> None:
        assert isinstance(MonotonicClock(), Clock)

    def test_never_goes_backwards(self) -> None:
        clock = MonotonicClock()
        readings = [clock.now() for _ in range(1_000)]
        assert readings == sorted(readings)


class TestManualClock:
    def test_satisfies_clock_protocol(self) -> None:
        assert isinstance(ManualClock(), Clock)

    def test_starts_at_given_time(self) -> None:
        assert ManualClock().now() == 0.0
        assert ManualClock(start=12.5).now() == 12.5

    def test_advance(self) -> None:
        clock = ManualClock(start=1.0)
        assert clock.advance(0.25) == 1.25
        assert clock.advance(0.0) == 1.25
        assert clock.now() == 1.25

    def test_set(self) -> None:
        clock = ManualClock(start=1.0)
        assert clock.set(3.0) == 3.0
        assert clock.set(3.0) == 3.0
        assert clock.now() == 3.0

    @pytest.mark.parametrize("start", [math.nan, math.inf, -math.inf])
    def test_rejects_non_finite_start(self, start: float) -> None:
        with pytest.raises(ValueError, match="start must be finite"):
            ManualClock(start=start)

    @pytest.mark.parametrize("seconds", [-0.001, math.nan, math.inf])
    def test_advance_rejects_invalid_step(self, seconds: float) -> None:
        clock = ManualClock()
        with pytest.raises(ValueError, match="seconds must be finite"):
            clock.advance(seconds)
        assert clock.now() == 0.0

    @pytest.mark.parametrize("timestamp", [4.999, math.nan, math.inf])
    def test_set_rejects_backwards_or_non_finite_time(self, timestamp: float) -> None:
        clock = ManualClock(start=5.0)
        with pytest.raises(ValueError, match="timestamp must be finite"):
            clock.set(timestamp)
        assert clock.now() == 5.0
