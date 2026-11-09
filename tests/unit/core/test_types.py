"""Tests for biokinex.core.types and the public core namespace."""

from __future__ import annotations

import numpy as np

import biokinex
from biokinex import core
from biokinex.core.types import (
    QUATERNION_SIZE,
    STANDARD_GRAVITY,
    VECTOR3_SIZE,
    Axis,
    EulerAngle,
    QuaternionComponent,
)


def test_constants() -> None:
    assert QUATERNION_SIZE == 4
    assert VECTOR3_SIZE == 3
    assert STANDARD_GRAVITY == 9.80665


def test_index_enums_match_array_layout() -> None:
    vector = np.array([10.0, 20.0, 30.0])
    assert vector[Axis.Z] == 30.0
    assert [int(c) for c in QuaternionComponent] == [0, 1, 2, 3]
    assert [c.name for c in QuaternionComponent] == ["W", "X", "Y", "Z"]
    assert [a.name for a in EulerAngle] == ["ROLL", "PITCH", "YAW"]


def test_public_namespace_exports_resolve() -> None:
    for name in core.__all__:
        assert hasattr(core, name), name


def test_package_version() -> None:
    assert biokinex.__version__ == "1.0.0"
