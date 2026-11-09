"""Builds filter objects from validated configuration.

The schema layer only describes filters; this module turns each description into a working filter.
Keeping the mapping in one place means a new filter kind needs only one change here, and mypy
checks that every kind is handled.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import assert_never

from biokinex.core.errors import ConfigurationError
from biokinex.dsp.base import Filter, FilterChain
from biokinex.dsp.biquad import BiquadFilter
from biokinex.dsp.median import MedianFilter
from biokinex.dsp.one_euro import OneEuroFilter, OrientationOneEuroFilter
from biokinex.dsp.slew import OrientationSlewRateLimiter, SlewRateLimiter
from biokinex.schemas.config import (
    FilterConfig,
    LowPassConfig,
    MedianConfig,
    NotchConfig,
    OneEuroConfig,
    SlewRateConfig,
)

__all__ = [
    "build_chain",
    "build_filter",
    "build_orientation_chain",
    "build_orientation_filter",
]


def build_filter(config: FilterConfig, sample_rate_hz: float) -> Filter:
    """Build one filter for a vector channel such as acceleration or joint angles.

    Args:
        config: The filter description.
        sample_rate_hz: Nominal sample rate; only fixed-rate biquad filters use it.
    """
    if isinstance(config, OneEuroConfig):
        return OneEuroFilter.from_config(config)
    if isinstance(config, LowPassConfig | NotchConfig):
        return BiquadFilter.from_config(config, sample_rate_hz)
    if isinstance(config, MedianConfig):
        return MedianFilter.from_config(config)
    if isinstance(config, SlewRateConfig):
        return SlewRateLimiter.from_config(config)
    assert_never(config)


def build_orientation_filter(config: FilterConfig) -> Filter:
    """Build one filter that works on unit quaternions.

    Raises:
        ConfigurationError: If the filter kind has no orientation version.
    """
    if isinstance(config, OneEuroConfig):
        return OrientationOneEuroFilter.from_config(config)
    if isinstance(config, SlewRateConfig):
        return OrientationSlewRateLimiter.from_config(config)
    # PipelineConfig already rejects these kinds; this guards direct calls with other configs.
    msg = f"filter kind {config.kind!r} cannot be used on orientation"
    raise ConfigurationError(msg)


def build_chain(configs: Iterable[FilterConfig], sample_rate_hz: float) -> FilterChain:
    """Build a chain of vector filters that run in the listed order."""
    return FilterChain(build_filter(config, sample_rate_hz) for config in configs)


def build_orientation_chain(configs: Iterable[FilterConfig]) -> FilterChain:
    """Build a chain of orientation filters that run in the listed order."""
    return FilterChain(build_orientation_filter(config) for config in configs)
