"""Tests for the shared boundary field types in biokinex.schemas.common."""

from __future__ import annotations

import math
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from biokinex.schemas.common import (
    MAX_SEQUENCE,
    FiniteFloat,
    Identifier,
    PositiveFloat,
    SequenceNumber,
    Timestamp,
    Vector3Field,
)

finite = TypeAdapter(FiniteFloat)
positive = TypeAdapter(PositiveFloat)
timestamp = TypeAdapter(Timestamp)
sequence = TypeAdapter(SequenceNumber)
identifier = TypeAdapter(Identifier)
vector3 = TypeAdapter(Vector3Field)


class TestFiniteFloat:
    def test_accepts_int_and_float(self) -> None:
        assert finite.validate_python(3) == 3.0
        assert finite.validate_python(-2.5) == -2.5
        assert finite.validate_json("7") == 7.0

    @pytest.mark.parametrize("bad", [True, "1.5", math.nan, math.inf, -math.inf, None])
    def test_rejects_non_numbers_and_non_finite(self, bad: Any) -> None:
        with pytest.raises(ValidationError):
            finite.validate_python(bad)

    @pytest.mark.parametrize("payload", ["NaN", "Infinity", "-Infinity", '"2.0"'])
    def test_rejects_non_finite_json(self, payload: str) -> None:
        with pytest.raises(ValidationError):
            finite.validate_json(payload)


def test_positive_float_rejects_zero() -> None:
    assert positive.validate_python(1e-9) == 1e-9
    with pytest.raises(ValidationError):
        positive.validate_python(0.0)


def test_timestamp_must_be_non_negative() -> None:
    assert timestamp.validate_python(0.0) == 0.0
    with pytest.raises(ValidationError):
        timestamp.validate_python(-1e-6)


class TestSequenceNumber:
    @pytest.mark.parametrize("value", [0, 1, MAX_SEQUENCE])
    def test_accepts_uint32_range(self, value: int) -> None:
        assert sequence.validate_python(value) == value

    @pytest.mark.parametrize("bad", [-1, MAX_SEQUENCE + 1, 3.0, True, "4"])
    def test_rejects_out_of_range_or_non_int(self, bad: Any) -> None:
        with pytest.raises(ValidationError):
            sequence.validate_python(bad)


class TestIdentifier:
    @pytest.mark.parametrize("value", ["imu-1", "a", "left.knee:0", "IMU_7", "x" * 64])
    def test_accepts_machine_identifiers(self, value: str) -> None:
        assert identifier.validate_python(value) == value

    @pytest.mark.parametrize(
        "bad", ["", "-leading-dash", "has space", "x" * 65, "ü", "line\nbreak", 12]
    )
    def test_rejects_free_text(self, bad: Any) -> None:
        with pytest.raises(ValidationError):
            identifier.validate_python(bad)


class TestVector3Field:
    def test_accepts_list_and_tuple(self) -> None:
        assert vector3.validate_python([1, 2.5, -3]) == (1.0, 2.5, -3.0)
        assert vector3.validate_python((0.0, 0.0, 0.0)) == (0.0, 0.0, 0.0)

    @pytest.mark.parametrize("bad", [[1.0, 2.0], [1.0, 2.0, 3.0, 4.0], [1.0, math.nan, 0.0]])
    def test_rejects_wrong_length_or_values(self, bad: list[float]) -> None:
        with pytest.raises(ValidationError):
            vector3.validate_python(bad)
