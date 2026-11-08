"""Tests for biokinex.core.scalar_quaternion, checked against the batch implementation."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from numpy.testing import assert_allclose

from biokinex.core import quaternion as quat
from biokinex.core import scalar_quaternion as sq
from biokinex.core.errors import DegenerateQuaternionError
from biokinex.core.types import FloatArray
from biokinex.dsp import rotation

ATOL = 1e-12

unit_quaternions = (
    hnp.arrays(np.float64, 4, elements=st.floats(-1.0, 1.0))
    .filter(lambda q: bool(np.linalg.norm(q) > 0.1))
    .map(lambda q: q / np.linalg.norm(q))
)
raw_quaternions = hnp.arrays(np.float64, 4, elements=st.floats(-10.0, 10.0)).filter(
    lambda q: bool(np.linalg.norm(q) > 1e-3)
)
vectors = hnp.arrays(np.float64, 3, elements=st.floats(-3.0, 3.0))


def as_quat(q: FloatArray) -> sq.Quat:
    return sq.from_array(q)


def as_vec(v: FloatArray) -> sq.Vec3:
    return sq.vector_from_array(v, "v")


class TestConversions:
    def test_round_trip(self) -> None:
        q = np.array([0.5, -0.5, 0.5, 0.5])
        assert sq.from_array(q) == (0.5, -0.5, 0.5, 0.5)
        assert_allclose(sq.to_array(sq.from_array(q)), q)
        assert sq.vector_from_array([1, 2, 3], "v") == (1.0, 2.0, 3.0)

    def test_to_array_returns_float64(self) -> None:
        assert sq.to_array((1.0, 0.0, 0.0)).dtype == np.float64

    @pytest.mark.parametrize("bad", [[1.0, 0.0, 0.0], np.ones((2, 4)), 1.0])
    def test_from_array_rejects_wrong_shape(self, bad: object) -> None:
        with pytest.raises(ValueError, match="single quaternion of shape"):
            sq.from_array(bad)

    def test_vector_from_array_names_the_argument(self) -> None:
        with pytest.raises(ValueError, match="omega must have shape"):
            sq.vector_from_array([1.0, 2.0], "omega")


class TestMatchesBatchImplementation:
    @given(raw_quaternions)
    def test_normalize(self, q: FloatArray) -> None:
        assert_allclose(sq.normalize(as_quat(q)), quat.normalize(q), atol=ATOL)

    @given(raw_quaternions)
    def test_conjugate(self, q: FloatArray) -> None:
        assert_allclose(sq.conjugate(as_quat(q)), quat.conjugate(q))

    @given(unit_quaternions, unit_quaternions)
    def test_multiply(self, p: FloatArray, q: FloatArray) -> None:
        assert_allclose(sq.multiply(as_quat(p), as_quat(q)), quat.multiply(p, q), atol=ATOL)

    @given(vectors)
    def test_from_rotation_vector(self, v: FloatArray) -> None:
        assert_allclose(sq.from_rotation_vector(as_vec(v)), quat.from_rotation_vector(v), atol=ATOL)

    @given(unit_quaternions)
    def test_to_rotation_vector(self, q: FloatArray) -> None:
        assert_allclose(sq.to_rotation_vector(as_quat(q)), quat.to_rotation_vector(q), atol=1e-9)

    @given(unit_quaternions, unit_quaternions)
    def test_relative_rotation_vector(self, p: FloatArray, q: FloatArray) -> None:
        expected = rotation.relative_rotation_vector(p, q)
        assert_allclose(sq.relative_rotation_vector(as_quat(p), as_quat(q)), expected, atol=1e-9)

    @given(unit_quaternions, vectors)
    def test_apply_rotation_vector(self, q: FloatArray, v: FloatArray) -> None:
        expected = rotation.apply_rotation_vector(q, v)
        assert_allclose(sq.apply_rotation_vector(as_quat(q), as_vec(v)), expected, atol=ATOL)

    @given(unit_quaternions, vectors, st.floats(0.0, 0.1))
    def test_integrate(self, q: FloatArray, omega: FloatArray, dt: float) -> None:
        expected = quat.integrate(q, omega, dt)
        assert_allclose(sq.integrate(as_quat(q), as_vec(omega), dt), expected, atol=ATOL)

    @given(unit_quaternions)
    def test_body_up(self, q: FloatArray) -> None:
        expected = quat.rotate_vector(quat.conjugate(q), [0.0, 0.0, 1.0])
        assert_allclose(sq.body_up(as_quat(q)), expected, atol=ATOL)


class TestEdgeCases:
    @pytest.mark.parametrize(
        "bad", [(0.0, 0.0, 0.0, 0.0), (1e-15, 0.0, 0.0, 0.0), (math.nan, 0.0, 0.0, 0.0)]
    )
    def test_normalize_rejects_degenerate(self, bad: sq.Quat) -> None:
        with pytest.raises(DegenerateQuaternionError):
            sq.normalize(bad)

    def test_tiny_rotation_keeps_precision(self) -> None:
        v = (1e-10, -2e-10, 3e-10)
        assert_allclose(sq.to_rotation_vector(sq.from_rotation_vector(v)), v, rtol=1e-9)

    def test_negative_scalar_gives_shortest_rotation(self) -> None:
        q = sq.from_array(quat.from_axis_angle([0.0, 0.0, 1.0], 1.5 * math.pi))
        assert_allclose(sq.to_rotation_vector(q), [0.0, 0.0, -0.5 * math.pi], atol=1e-12)

    def test_zero_quaternion_rotation_vector_matches_batch(self) -> None:
        assert sq.to_rotation_vector((0.0, 0.0, 0.0, 0.0)) == (0.0, 0.0, 0.0)
        assert_allclose(quat.to_rotation_vector(np.zeros(4)), np.zeros(3))
