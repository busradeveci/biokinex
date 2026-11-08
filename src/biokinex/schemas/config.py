"""Pipeline configuration schema, loadable from TOML.

Example ``biokinex.toml``::

    sample_rate_hz = 500

    [[angular_velocity_filters]]
    kind = "low_pass"
    cutoff_hz = 40

    [[orientation_filters]]
    kind = "one_euro"
    min_cutoff_hz = 1.5
    beta = 0.05

    [fusion]
    gain = 2.0

Each channel has its own filter chain, applied in the listed order. All rates and cutoffs use SI
units, so a slew limit on angular velocity is in rad/s², and on orientation in rad/s.
"""

from __future__ import annotations

import tomllib
from enum import StrEnum
from os import PathLike
from typing import Annotated, Final, Literal, Self

from pydantic import Field, StrictBool, StrictInt, ValidationError, model_validator

from biokinex.core.errors import ConfigurationError
from biokinex.schemas.common import FrozenModel, NonNegativeFloat, PositiveFloat

__all__ = [
    "ComplementaryConfig",
    "FilterConfig",
    "GapConfig",
    "GapPolicy",
    "LowPassConfig",
    "MedianConfig",
    "NotchConfig",
    "OneEuroConfig",
    "PipelineConfig",
    "SlewRateConfig",
]

MAX_FILTERS_PER_CHAIN: Final = 16
MAX_SAMPLE_RATE_HZ: Final = 100_000.0


class OneEuroConfig(FrozenModel):
    """Adaptive low-pass filter that removes jitter at rest but keeps fast motion responsive."""

    kind: Literal["one_euro"] = "one_euro"
    min_cutoff_hz: PositiveFloat = 1.0
    beta: NonNegativeFloat = 0.0
    derivative_cutoff_hz: PositiveFloat = 1.0


class LowPassConfig(FrozenModel):
    """Second-order Butterworth-style low-pass biquad."""

    kind: Literal["low_pass"] = "low_pass"
    cutoff_hz: PositiveFloat
    # 1/sqrt(2) gives the maximally flat (Butterworth) response.
    q: PositiveFloat = 0.7071067811865476


class NotchConfig(FrozenModel):
    """Second-order notch biquad, for example to remove a known vibration frequency."""

    kind: Literal["notch"] = "notch"
    center_hz: PositiveFloat
    q: PositiveFloat


class MedianConfig(FrozenModel):
    """Sliding median that removes single-sample spikes without smearing edges."""

    kind: Literal["median"] = "median"
    window: StrictInt = Field(default=5, ge=3, le=63)

    @model_validator(mode="after")
    def _check_window_is_odd(self) -> Self:
        # An odd window has a single middle sample, so the median needs no averaging.
        if self.window % 2 == 0:
            msg = f"window must be odd, got {self.window}"
            raise ValueError(msg)
        return self


class SlewRateConfig(FrozenModel):
    """Caps how fast the output may change, in channel units per second."""

    kind: Literal["slew_rate"] = "slew_rate"
    max_rate: PositiveFloat


FilterConfig = Annotated[
    OneEuroConfig | LowPassConfig | NotchConfig | MedianConfig | SlewRateConfig,
    Field(discriminator="kind"),
]
"""Any filter configuration, selected by its ``kind`` field."""

FilterChain = Annotated[tuple[FilterConfig, ...], Field(max_length=MAX_FILTERS_PER_CHAIN)]

_ORIENTATION_FILTER_KINDS: Final = frozenset({"one_euro", "slew_rate"})


class ComplementaryConfig(FrozenModel):
    """Mahony-style attitude fusion of accelerometer and gyroscope data."""

    enabled: StrictBool = True
    """Set to ``false`` to use only device-reported orientation."""

    gain: NonNegativeFloat = 1.0
    """Proportional correction gain in 1/s; higher values trust the accelerometer more."""

    accel_tolerance: float = Field(default=0.1, gt=0.0, le=1.0, strict=True, allow_inf_nan=False)
    """Accelerometer correction is skipped when ``|a|`` differs from g by more than this fraction.

    During strong linear acceleration the accelerometer no longer points along gravity, so using it
    would tilt the estimate.
    """


class GapPolicy(StrEnum):
    """How missing samples between two valid samples are filled."""

    HOLD = "hold"
    LINEAR = "linear"


class GapConfig(FrozenModel):
    """Handling of dropped packets detected through sequence numbers."""

    policy: GapPolicy = GapPolicy.HOLD
    max_fill_samples: StrictInt = Field(default=10, ge=0, le=1_000)
    """Longest gap that is filled; longer gaps restart the filters instead of inventing data."""


class PipelineConfig(FrozenModel):
    """Top-level configuration for one processing pipeline."""

    sample_rate_hz: PositiveFloat = Field(le=MAX_SAMPLE_RATE_HZ)
    """Nominal device sample rate, used to design fixed-rate filters."""

    acceleration_filters: FilterChain = ()
    angular_velocity_filters: FilterChain = ()
    orientation_filters: FilterChain = ()
    joint_angle_filters: FilterChain = ()

    fusion: ComplementaryConfig = ComplementaryConfig()
    gap: GapConfig = GapConfig()
    watchdog_timeout_s: PositiveFloat = 0.1

    @model_validator(mode="after")
    def _check_against_sample_rate(self) -> Self:
        nyquist_hz = 0.5 * self.sample_rate_hz
        chains = {
            "acceleration_filters": self.acceleration_filters,
            "angular_velocity_filters": self.angular_velocity_filters,
            "orientation_filters": self.orientation_filters,
            "joint_angle_filters": self.joint_angle_filters,
        }
        for chain_name, chain in chains.items():
            for index, config in enumerate(chain):
                # Digital filters cannot represent frequencies at or above Nyquist.
                frequency = _design_frequency(config)
                if frequency is not None and frequency >= nyquist_hz:
                    msg = (
                        f"{chain_name}[{index}] frequency {frequency} Hz must be below "
                        f"the Nyquist frequency {nyquist_hz} Hz"
                    )
                    raise ValueError(msg)

        for index, config in enumerate(self.orientation_filters):
            # Orientation is filtered on rotation increments, where only these filters are defined.
            if config.kind not in _ORIENTATION_FILTER_KINDS:
                allowed = ", ".join(sorted(_ORIENTATION_FILTER_KINDS))
                msg = f"orientation_filters[{index}] kind {config.kind!r} is not one of: {allowed}"
                raise ValueError(msg)

        # A timeout shorter than one sample period would mark every frame as stale.
        if self.watchdog_timeout_s <= 1.0 / self.sample_rate_hz:
            msg = "watchdog_timeout_s must be longer than one sample period"
            raise ValueError(msg)
        return self

    @classmethod
    def from_toml(cls, text: str) -> PipelineConfig:
        """Parse a configuration from TOML text.

        Raises:
            ConfigurationError: If the text is not valid TOML or breaks any schema rule.
        """
        try:
            return cls.model_validate(tomllib.loads(text))
        except tomllib.TOMLDecodeError as exc:
            msg = f"invalid TOML: {exc}"
            raise ConfigurationError(msg) from exc
        except ValidationError as exc:
            msg = f"invalid pipeline configuration:\n{exc}"
            raise ConfigurationError(msg) from exc

    @classmethod
    def from_toml_file(cls, path: str | PathLike[str]) -> PipelineConfig:
        """Load a configuration from a UTF-8 TOML file.

        Raises:
            ConfigurationError: If the file content is not valid TOML or breaks any schema rule.
            OSError: If the file cannot be read.
        """
        with open(path, encoding="utf-8") as file:
            return cls.from_toml(file.read())


def _design_frequency(config: FilterConfig) -> float | None:
    """Return the fixed frequency a filter is designed around, if it has one."""
    if isinstance(config, LowPassConfig):
        return config.cutoff_hz
    if isinstance(config, NotchConfig):
        return config.center_hz
    return None
