"""Tests for biokinex.dsp.rotation."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from numpy.testing import assert_allclose

from biokinex.core import quaternion as quat
from biokinex.core.types import FloatArray
from biokinex.dsp.rotation import (
    angular_velocity_between,
    apply_rotation_vector,
    make_continuous,
    relative_rotation_vector,
)

unit_quaternions = (
    hnp.arrays(np.float64, 4, elements=st.floats(-1.0, 1.0))
    .filter(lambda q: bool(np.linalg.norm(q) > 0.1))
    .map(lambda q: q / np.linalg.norm(q))
)
small_rotation_vectors = hnp.arrays(np.float64, 3, elements=st.floats(-1.5, 1.5))


@given(unit_quaternions, unit_quaternions)
def test_apply_undoes_relative(reference: FloatArray, target: FloatArray) -> None:
    rebuilt = apply_rotation_vector(reference, relative_rotation_vector(reference, target))
    assert float(quat.angle_between(rebuilt, target)) < 1e-9


@given(unit_quaternions, small_rotation_vectors)
def test_relative_undoes_apply(reference: FloatArray, rotation_vector: FloatArray) -> None:
    moved = apply_rotation_vector(reference, rotation_vector)
    assert_allclose(relative_rotation_vector(reference, moved), rotation_vector, atol=1e-9)


@given(unit_quaternions, unit_quaternions)
def test_relative_ignores_quaternion_sign(reference: FloatArray, target: FloatArray) -> None:
    # Rotations are compared, not vectors, because v and -v are the same rotation at 180°.
    expected = quat.from_rotation_vector(relative_rotation_vector(reference, target))
    for flipped in (
        relative_rotation_vector(-reference, target),
        relative_rotation_vector(reference, -target),
    ):
        assert float(quat.angle_between(quat.from_rotation_vector(flipped), expected)) < 1e-9


def test_relative_ignores_sign_below_half_turn() -> None:
    reference = quat.from_euler_zyx([0.4, -0.3, 2.0])
    target = quat.from_euler_zyx([-0.2, 0.5, -1.0])
    expected = relative_rotation_vector(reference, target)
    assert_allclose(relative_rotation_vector(-reference, target), expected, atol=1e-12)
    assert_allclose(relative_rotation_vector(reference, -target), expected, atol=1e-12)


def test_relative_is_expressed_in_body_frame() -> None:
    # After a 90° yaw, a world-frame turn about y is a body-frame turn about -x.
    start = quat.from_axis_angle([0.0, 0.0, 1.0], 0.5 * math.pi)
    end = quat.multiply(quat.from_axis_angle([0.0, 1.0, 0.0], 0.2), start)
    assert_allclose(relative_rotation_vector(start, end), [0.2, 0.0, 0.0], atol=1e-12)


def test_apply_returns_unit_quaternion() -> None:
    result = apply_rotation_vector([1.0, 0.0, 0.0, 0.0], [0.3, -0.2, 0.1])
    assert bool(quat.is_unit(result, atol=1e-15))


def test_angular_velocity_between_recovers_constant_rate() -> None:
    omega = np.array([0.4, -1.2, 2.0])
    start = quat.from_euler_zyx([0.1, 0.2, 0.3])
    end = quat.integrate(start, omega, 0.01)
    assert_allclose(angular_velocity_between(start, end, 0.01), omega, atol=1e-9)


@pytest.mark.parametrize("dt", [0.0, -0.01, math.nan])
def test_angular_velocity_between_rejects_invalid_dt(dt: float) -> None:
    with pytest.raises(ValueError, match="dt must be finite and positive"):
        angular_velocity_between(quat.identity(), quat.identity(), dt)


class TestMakeContinuous:
    def test_removes_sign_flips(self) -> None:
        base = quat.from_axis_angle(np.tile([0.0, 0.0, 1.0], (6, 1)), np.linspace(0.0, 1.0, 6))
        flipped = base * np.array([1, -1, -1, 1, -1, 1])[:, None]
        assert_allclose(make_continuous(flipped), base, atol=1e-15)

    def test_matches_reference_for_first_element(self) -> None:
        base = quat.from_axis_angle(np.tile([1.0, 0.0, 0.0], (3, 1)), [0.1, 0.2, 0.3])
        assert_allclose(make_continuous(-base, reference=base[0]), base, atol=1e-15)
        assert_allclose(make_continuous(base, reference=base[0]), base, atol=1e-15)

    def test_empty_sequence(self) -> None:
        assert make_continuous(np.zeros((0, 4))).shape == (0, 4)

    def test_does_not_modify_input(self) -> None:
        sequence = np.array([[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]])
        make_continuous(sequence)
        assert sequence[1, 0] == -1.0

    @pytest.mark.parametrize("shape", [(4,), (3, 3), (2, 4, 1)])
    def test_rejects_wrong_shape(self, shape: tuple[int, ...]) -> None:
        with pytest.raises(ValueError, match=r"shape \(N, 4\)"):
            make_continuous(np.ones(shape))

    @given(st.lists(unit_quaternions, min_size=1, max_size=20))
    def test_neighbours_share_hemisphere(self, items: list[FloatArray]) -> None:
        result = make_continuous(np.array(items))
        assert np.all(np.sum(result[1:] * result[:-1], axis=1) >= 0.0)
        assert_allclose(np.abs(result), np.abs(np.array(items)))
