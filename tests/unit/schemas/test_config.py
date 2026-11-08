"""Tests for biokinex.schemas.config."""

from __future__ import annotations

import textwrap
import tomllib
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from biokinex.core.errors import ConfigurationError
from biokinex.schemas import config as config_module
from biokinex.schemas.config import (
    ComplementaryConfig,
    GapConfig,
    GapPolicy,
    LowPassConfig,
    MedianConfig,
    NotchConfig,
    OneEuroConfig,
    PipelineConfig,
    SlewRateConfig,
)

FULL_TOML = """
sample_rate_hz = 500
watchdog_timeout_s = 0.05

[[acceleration_filters]]
kind = "median"
window = 3

[[acceleration_filters]]
kind = "low_pass"
cutoff_hz = 30

[[angular_velocity_filters]]
kind = "notch"
center_hz = 50
q = 10

[[orientation_filters]]
kind = "one_euro"
min_cutoff_hz = 1.5
beta = 0.05

[[orientation_filters]]
kind = "slew_rate"
max_rate = 6.0

[fusion]
gain = 2.0
accel_tolerance = 0.2

[gap]
policy = "linear"
max_fill_samples = 4
"""


def pipeline(**fields: Any) -> PipelineConfig:
    """Validate a pipeline config at 500 Hz with optional overrides."""
    return PipelineConfig.model_validate({"sample_rate_hz": 500, **fields})


class TestDefaults:
    def test_pipeline_defaults(self) -> None:
        config = pipeline()
        assert config.sample_rate_hz == 500.0
        assert config.acceleration_filters == ()
        assert config.orientation_filters == ()
        assert config.fusion == ComplementaryConfig()
        assert config.fusion.enabled is True
        assert config.gap == GapConfig(policy=GapPolicy.HOLD, max_fill_samples=10)
        assert config.watchdog_timeout_s == 0.1

    def test_filter_defaults(self) -> None:
        assert OneEuroConfig() == OneEuroConfig(
            min_cutoff_hz=1.0, beta=0.0, derivative_cutoff_hz=1.0
        )
        assert LowPassConfig(cutoff_hz=10.0).q == pytest.approx(2.0**-0.5)
        assert MedianConfig().window == 5


class TestToml:
    def test_full_example(self) -> None:
        config = PipelineConfig.from_toml(FULL_TOML)
        assert config.acceleration_filters == (
            MedianConfig(window=3),
            LowPassConfig(cutoff_hz=30.0),
        )
        assert config.angular_velocity_filters == (NotchConfig(center_hz=50.0, q=10.0),)
        assert config.orientation_filters == (
            OneEuroConfig(min_cutoff_hz=1.5, beta=0.05),
            SlewRateConfig(max_rate=6.0),
        )
        assert config.fusion == ComplementaryConfig(gain=2.0, accel_tolerance=0.2)
        assert config.gap == GapConfig(policy=GapPolicy.LINEAR, max_fill_samples=4)

    def test_module_docstring_example_is_valid(self) -> None:
        doc = config_module.__doc__ or ""
        example = doc.split("::", 1)[1].split("Each channel", 1)[0]
        config = PipelineConfig.from_toml(textwrap.dedent(example))
        assert config.angular_velocity_filters == (LowPassConfig(cutoff_hz=40.0),)
        assert config.fusion.gain == 2.0

    def test_fusion_can_be_disabled(self) -> None:
        config = PipelineConfig.from_toml("sample_rate_hz = 200\n[fusion]\nenabled = false\n")
        assert config.fusion.enabled is False

    def test_invalid_toml_raises_configuration_error(self) -> None:
        with pytest.raises(ConfigurationError, match="invalid TOML") as info:
            PipelineConfig.from_toml("sample_rate_hz = = 5")
        assert isinstance(info.value.__cause__, tomllib.TOMLDecodeError)

    def test_schema_violation_raises_configuration_error(self) -> None:
        with pytest.raises(ConfigurationError, match="invalid pipeline configuration") as info:
            PipelineConfig.from_toml("sample_rate = 500\n")
        assert isinstance(info.value.__cause__, ValidationError)

    def test_from_file(self, tmp_path: Path) -> None:
        path = tmp_path / "biokinex.toml"
        path.write_text(FULL_TOML, encoding="utf-8")
        assert PipelineConfig.from_toml_file(path) == PipelineConfig.from_toml(FULL_TOML)

    def test_missing_file_raises_os_error(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            PipelineConfig.from_toml_file(tmp_path / "missing.toml")


class TestCrossFieldRules:
    @pytest.mark.parametrize(
        "filter_config",
        [{"kind": "low_pass", "cutoff_hz": 250}, {"kind": "notch", "center_hz": 300, "q": 5}],
    )
    def test_rejects_frequency_at_or_above_nyquist(self, filter_config: dict[str, Any]) -> None:
        with pytest.raises(ValidationError, match="Nyquist"):
            pipeline(angular_velocity_filters=[filter_config])

    def test_accepts_frequency_below_nyquist(self) -> None:
        config = pipeline(joint_angle_filters=[{"kind": "low_pass", "cutoff_hz": 249.9}])
        assert config.joint_angle_filters == (LowPassConfig(cutoff_hz=249.9),)

    @pytest.mark.parametrize(
        "filter_config",
        [
            {"kind": "median"},
            {"kind": "low_pass", "cutoff_hz": 10},
            {"kind": "notch", "center_hz": 10, "q": 2},
        ],
    )
    def test_orientation_chain_rejects_unsupported_filters(
        self, filter_config: dict[str, Any]
    ) -> None:
        with pytest.raises(ValidationError, match="orientation_filters"):
            pipeline(orientation_filters=[filter_config])

    @pytest.mark.parametrize("timeout", [0.002, 0.001])
    def test_watchdog_must_exceed_one_sample_period(self, timeout: float) -> None:
        with pytest.raises(ValidationError, match="one sample period"):
            pipeline(watchdog_timeout_s=timeout)

    def test_rejects_too_many_filters(self) -> None:
        with pytest.raises(ValidationError):
            pipeline(acceleration_filters=[{"kind": "median"}] * 17)


class TestFieldRules:
    @pytest.mark.parametrize("rate", [0, -1, 100_001, True, "500"])
    def test_rejects_invalid_sample_rate(self, rate: Any) -> None:
        with pytest.raises(ValidationError):
            PipelineConfig.model_validate({"sample_rate_hz": rate})

    def test_sample_rate_is_required(self) -> None:
        with pytest.raises(ValidationError):
            PipelineConfig.model_validate({})

    def test_rejects_unknown_filter_kind(self) -> None:
        with pytest.raises(ValidationError):
            pipeline(acceleration_filters=[{"kind": "kalman"}])

    @pytest.mark.parametrize("window", [1, 4, 65, 3.0])
    def test_rejects_invalid_median_window(self, window: Any) -> None:
        with pytest.raises(ValidationError):
            MedianConfig.model_validate({"window": window})

    def test_rejects_even_median_window_with_clear_message(self) -> None:
        with pytest.raises(ValidationError, match="window must be odd"):
            MedianConfig(window=6)

    @pytest.mark.parametrize(
        "raw",
        [
            {"min_cutoff_hz": 0},
            {"beta": -0.1},
            {"derivative_cutoff_hz": float("inf")},
        ],
    )
    def test_rejects_invalid_one_euro(self, raw: dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            OneEuroConfig.model_validate(raw)

    @pytest.mark.parametrize(
        "raw", [{"accel_tolerance": 0.0}, {"accel_tolerance": 1.5}, {"gain": -1.0}]
    )
    def test_rejects_invalid_fusion(self, raw: dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            ComplementaryConfig.model_validate(raw)

    @pytest.mark.parametrize("raw", [{"policy": "spline"}, {"max_fill_samples": -1}])
    def test_rejects_invalid_gap(self, raw: dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            GapConfig.model_validate(raw)

    def test_notch_requires_q(self) -> None:
        with pytest.raises(ValidationError):
            NotchConfig.model_validate({"center_hz": 50})
