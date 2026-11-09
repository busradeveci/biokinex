"""Tests for biokinex.pipeline.factory."""

from __future__ import annotations

import pytest

from biokinex.core.errors import ConfigurationError
from biokinex.dsp.biquad import BiquadCoefficients, BiquadFilter
from biokinex.dsp.median import MedianFilter
from biokinex.dsp.one_euro import OneEuroFilter, OrientationOneEuroFilter
from biokinex.dsp.slew import OrientationSlewRateLimiter, SlewRateLimiter
from biokinex.pipeline.factory import (
    build_chain,
    build_filter,
    build_orientation_chain,
    build_orientation_filter,
)
from biokinex.schemas.config import (
    FilterConfig,
    LowPassConfig,
    MedianConfig,
    NotchConfig,
    OneEuroConfig,
    SlewRateConfig,
)

RATE_HZ = 500.0


@pytest.mark.parametrize(
    ("config", "expected_type"),
    [
        (OneEuroConfig(), OneEuroFilter),
        (LowPassConfig(cutoff_hz=20.0), BiquadFilter),
        (NotchConfig(center_hz=50.0, q=5.0), BiquadFilter),
        (MedianConfig(window=3), MedianFilter),
        (SlewRateConfig(max_rate=2.0), SlewRateLimiter),
    ],
)
def test_build_filter_maps_each_kind(config: FilterConfig, expected_type: type) -> None:
    assert type(build_filter(config, RATE_HZ)) is expected_type


def test_biquads_are_designed_for_the_sample_rate() -> None:
    built = build_filter(LowPassConfig(cutoff_hz=20.0, q=0.9), RATE_HZ)
    assert isinstance(built, BiquadFilter)
    assert built.coefficients == BiquadCoefficients.low_pass(20.0, RATE_HZ, 0.9)
    notch = build_filter(NotchConfig(center_hz=50.0, q=5.0), RATE_HZ)
    assert isinstance(notch, BiquadFilter)
    assert notch.coefficients == BiquadCoefficients.notch(50.0, RATE_HZ, 5.0)


def test_build_filter_passes_tuning_values() -> None:
    median = build_filter(MedianConfig(window=7), RATE_HZ)
    assert isinstance(median, MedianFilter)
    assert median.window == 7
    slew = build_filter(SlewRateConfig(max_rate=3.0), RATE_HZ)
    assert isinstance(slew, SlewRateLimiter)
    assert slew.max_rate == 3.0


@pytest.mark.parametrize(
    ("config", "expected_type"),
    [
        (OneEuroConfig(), OrientationOneEuroFilter),
        (SlewRateConfig(max_rate=2.0), OrientationSlewRateLimiter),
    ],
)
def test_build_orientation_filter_maps_each_kind(config: FilterConfig, expected_type: type) -> None:
    assert type(build_orientation_filter(config)) is expected_type


@pytest.mark.parametrize(
    "config",
    [LowPassConfig(cutoff_hz=20.0), NotchConfig(center_hz=50.0, q=5.0), MedianConfig()],
)
def test_build_orientation_filter_rejects_vector_only_kinds(config: FilterConfig) -> None:
    with pytest.raises(ConfigurationError, match="cannot be used on orientation"):
        build_orientation_filter(config)


def test_chains_keep_the_listed_order() -> None:
    chain = build_chain([MedianConfig(), LowPassConfig(cutoff_hz=10.0), OneEuroConfig()], RATE_HZ)
    assert [type(f) for f in chain.filters] == [MedianFilter, BiquadFilter, OneEuroFilter]
    orientation = build_orientation_chain([SlewRateConfig(max_rate=1.0), OneEuroConfig()])
    assert [type(f) for f in orientation.filters] == [
        OrientationSlewRateLimiter,
        OrientationOneEuroFilter,
    ]


def test_empty_chains() -> None:
    assert len(build_chain((), RATE_HZ)) == 0
    assert len(build_orientation_chain(())) == 0
