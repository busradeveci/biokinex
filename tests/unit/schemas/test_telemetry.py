"""Tests for biokinex.schemas.telemetry."""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pydantic import ValidationError

from biokinex.core import quaternion as quat
from biokinex.core.errors import TelemetryError
from biokinex.core.types import STANDARD_GRAVITY
from biokinex.schemas.telemetry import (
    MAX_SAMPLES_PER_PACKET,
    AccelerationUnit,
    AngleUnit,
    AngularVelocityUnit,
    TelemetryPacket,
    TelemetrySample,
    Units,
)


def sample(sequence: int = 0, timestamp: float = 0.0, **fields: Any) -> dict[str, Any]:
    """Build a raw sample dict with IMU readings unless fields are given explicitly."""
    if not fields:
        fields = {"acceleration": [0.0, 0.0, 9.81], "angular_velocity": [0.0, 0.0, 0.0]}
    return {"sequence": sequence, "timestamp": timestamp, **fields}


def packet(*samples: dict[str, Any], **fields: Any) -> dict[str, Any]:
    """Build a raw packet dict; defaults to two IMU samples at 500 Hz."""
    if not samples:
        samples = (sample(0, 0.0), sample(1, 0.002))
    return {"device_id": "imu-0", "samples": list(samples), **fields}


class TestUnits:
    def test_defaults_are_si(self) -> None:
        units = Units()
        assert units.acceleration is AccelerationUnit.METERS_PER_SECOND_SQUARED
        assert units.angular_velocity is AngularVelocityUnit.RADIANS_PER_SECOND
        assert units.angle is AngleUnit.RADIANS

    @pytest.mark.parametrize(
        ("unit", "factor"),
        [
            (AccelerationUnit.METERS_PER_SECOND_SQUARED, 1.0),
            (AccelerationUnit.STANDARD_GRAVITY, STANDARD_GRAVITY),
            (AngularVelocityUnit.RADIANS_PER_SECOND, 1.0),
            (AngularVelocityUnit.DEGREES_PER_SECOND, math.pi / 180.0),
            (AngleUnit.RADIANS, 1.0),
            (AngleUnit.DEGREES, math.pi / 180.0),
        ],
    )
    def test_si_factors(
        self, unit: AccelerationUnit | AngularVelocityUnit | AngleUnit, factor: float
    ) -> None:
        assert unit.si_factor == factor

    def test_rejects_unknown_unit(self) -> None:
        with pytest.raises(ValidationError):
            Units.model_validate({"acceleration": "mg"})


class TestTelemetrySample:
    def test_minimal_sample(self) -> None:
        parsed = TelemetrySample.model_validate(sample(7, 1.5, joint_angles=[0.3]))
        assert parsed.sequence == 7
        assert parsed.joint_angles == (0.3,)
        assert parsed.acceleration is None

    def test_layout_reports_present_fields(self) -> None:
        parsed = TelemetrySample.model_validate(sample())
        assert parsed.layout() == (True, True, False, False, False)

    @pytest.mark.parametrize(
        ("fields", "message"),
        [
            ({}, "at least one measurement"),
            (
                {"orientation_quaternion": [1, 0, 0, 0], "orientation_euler": [0, 0, 0]},
                "not both",
            ),
            ({"orientation_quaternion": [1.2, 0, 0, 0]}, "too far from 1"),
            ({"orientation_quaternion": [0, 0, 0, 0]}, "too far from 1"),
        ],
    )
    def test_rejects_invalid_contents(self, fields: dict[str, Any], message: str) -> None:
        raw = {"sequence": 0, "timestamp": 0.0, **fields}
        with pytest.raises(ValidationError, match=message):
            TelemetrySample.model_validate(raw)

    @pytest.mark.parametrize(
        "overrides",
        [
            {"acceleration": [0.0, math.nan, 0.0]},
            {"acceleration": [0.0, math.inf, 0.0]},
            {"acceleration": [True, 0.0, 0.0]},
            {"acceleration": ["1.0", 0.0, 0.0]},
            {"acceleration": [0.0, 0.0]},
            {"angular_velocity": [0.0, 0.0, 0.0, 0.0]},
            {"timestamp": -0.001},
            {"sequence": -1},
            {"sequence": 2**32},
            {"sequence": 1.0},
            {"joint_angles": [0.0] * 65},
            {"temperature": 36.6},
        ],
    )
    def test_rejects_malformed_fields(self, overrides: dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            TelemetrySample.model_validate({**sample(), **overrides})

    def test_is_immutable(self) -> None:
        parsed = TelemetrySample.model_validate(sample())
        with pytest.raises(ValidationError):
            parsed.sequence = 5  # type: ignore[misc]


class TestTelemetryPacket:
    def test_valid_packet(self) -> None:
        parsed = TelemetryPacket.model_validate(packet())
        assert parsed.schema_version == 1
        assert parsed.device_id == "imu-0"
        assert len(parsed.samples) == 2
        assert parsed.units == Units()

    def test_accepts_maximum_sample_count(self) -> None:
        samples = [sample(i, i * 0.002) for i in range(MAX_SAMPLES_PER_PACKET)]
        assert len(TelemetryPacket.model_validate(packet(*samples)).samples) == 256

    @pytest.mark.parametrize(
        ("raw", "message"),
        [
            (packet(sample(0, 0.0), sample(1, 0.0)), "not strictly increasing"),
            (packet(sample(0, 0.01), sample(1, 0.005)), "not strictly increasing"),
            (
                packet(sample(0, 0.0), sample(1, 0.002, acceleration=[0.0, 0.0, 9.81])),
                "different measurements",
            ),
            (
                packet(sample(0, 0.0, joint_angles=[0.1]), joint_names=["knee", "ankle"]),
                "1 joint angles but the packet declares 2",
            ),
            (
                packet(sample(0, 0.0, joint_angles=[0.1])),
                "1 joint angles but the packet declares 0",
            ),
            (packet(sample(), joint_names=["knee"]), "0 joint angles but the packet declares 1"),
            (
                packet(sample(0, 0.0, joint_angles=[0.1, 0.2]), joint_names=["knee", "knee"]),
                "unique",
            ),
        ],
    )
    def test_rejects_inconsistent_packets(self, raw: dict[str, Any], message: str) -> None:
        with pytest.raises(ValidationError, match=message):
            TelemetryPacket.model_validate(raw)

    @pytest.mark.parametrize(
        "raw",
        [
            packet(device_id="has space"),
            packet(device_id=""),
            packet(schema_version=2),
            packet(units={"angle": "grad"}),
            packet(patient_name="x"),
            packet(joint_names=["bad name"]),
            {"device_id": "imu-0", "samples": []},
            packet(*[sample(i, i * 0.002) for i in range(MAX_SAMPLES_PER_PACKET + 1)]),
        ],
        ids=[
            "device_id_space",
            "device_id_empty",
            "unknown_schema_version",
            "unknown_unit",
            "unknown_field",
            "bad_joint_name",
            "no_samples",
            "too_many_samples",
        ],
    )
    def test_rejects_malformed_packets(self, raw: dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            TelemetryPacket.model_validate(raw)


class TestFromJson:
    def test_parses_valid_payload(self) -> None:
        parsed = TelemetryPacket.from_json(json.dumps(packet()).encode())
        assert parsed.samples[1].timestamp == 0.002

    @pytest.mark.parametrize(
        "payload",
        [
            b"{not json",
            b"",
            json.dumps(packet(device_id="has space")).encode(),
            json.dumps(packet()).replace("9.81", "NaN").encode(),
        ],
        ids=["broken_json", "empty", "schema_violation", "nan_literal"],
    )
    def test_raises_telemetry_error(self, payload: bytes) -> None:
        with pytest.raises(TelemetryError, match="invalid telemetry packet") as info:
            TelemetryPacket.from_json(payload)
        assert isinstance(info.value.__cause__, ValidationError)


class TestToBatch:
    def test_si_values_pass_through(self) -> None:
        batch = TelemetryPacket.model_validate(packet()).to_batch()
        assert batch.device_id == "imu-0"
        assert len(batch) == 2
        assert_allclose(batch.timestamps, [0.0, 0.002])
        assert batch.sequences.tolist() == [0, 1]
        assert batch.acceleration is not None
        assert_allclose(batch.acceleration, [[0.0, 0.0, 9.81]] * 2)
        assert batch.orientation is None
        assert batch.joint_angles is None

    def test_converts_wire_units_to_si(self) -> None:
        raw = packet(
            sample(
                0,
                0.0,
                acceleration=[0.0, 0.0, 1.0],
                angular_velocity=[180.0, 0.0, -90.0],
                joint_angles=[90.0, 45.0],
            ),
            units={"acceleration": "g", "angular_velocity": "deg/s", "angle": "deg"},
            joint_names=["knee", "ankle"],
        )
        batch = TelemetryPacket.model_validate(raw).to_batch()
        assert batch.acceleration is not None
        assert batch.angular_velocity is not None
        assert batch.joint_angles is not None
        assert_allclose(batch.acceleration, [[0.0, 0.0, STANDARD_GRAVITY]])
        assert_allclose(batch.angular_velocity, [[math.pi, 0.0, -0.5 * math.pi]])
        assert_allclose(batch.joint_angles, [[0.5 * math.pi, 0.25 * math.pi]])
        assert batch.joint_names == ("knee", "ankle")

    def test_euler_orientation_becomes_quaternion(self) -> None:
        raw = packet(
            sample(0, 0.0, orientation_euler=[10.0, -20.0, 30.0]),
            units={"angle": "deg"},
        )
        batch = TelemetryPacket.model_validate(raw).to_batch()
        assert batch.orientation is not None
        expected = quat.from_euler_zyx(np.radians([10.0, -20.0, 30.0]))
        assert_allclose(batch.orientation[0], expected, atol=1e-12)

    def test_quaternion_orientation_is_normalized(self) -> None:
        raw = packet(
            sample(0, 0.0, orientation_quaternion=[1.05, 0.0, 0.0, 0.0]),
            sample(1, 0.002, orientation_quaternion=[0.0, 0.0, 0.0, 0.95]),
        )
        batch = TelemetryPacket.model_validate(raw).to_batch()
        assert batch.orientation is not None
        assert_allclose(batch.orientation, [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])

    def test_quaternion_ignores_angle_unit(self) -> None:
        raw = packet(sample(0, 0.0, orientation_quaternion=[1, 0, 0, 0]), units={"angle": "deg"})
        batch = TelemetryPacket.model_validate(raw).to_batch()
        assert batch.orientation is not None
        assert_allclose(batch.orientation, [[1.0, 0.0, 0.0, 0.0]])

    def test_arrays_are_read_only(self) -> None:
        batch = TelemetryPacket.model_validate(packet()).to_batch()
        assert batch.acceleration is not None
        for array in (batch.timestamps, batch.sequences, batch.acceleration):
            with pytest.raises(ValueError, match="read-only"):
                array[0] = 0
