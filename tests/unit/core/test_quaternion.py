"""Tests for biokinex.core.quaternion."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from numpy.testing import assert_allclose

from biokinex.core import quaternion as quat
from biokinex.core.errors import DegenerateQuaternionError
from biokinex.core.types import FloatArray

ATOL = 1e-12

ONE = np.array([1.0, 0.0, 0.0, 0.0])
I = np.array([0.0, 1.0, 0.0, 0.0])  # noqa: E741
J = np.array([0.0, 0.0, 1.0, 0.0])
K = np.array([0.0, 0.0, 0.0, 1.0])

X_AXIS = np.array([1.0, 0.0, 0.0])
Y_AXIS = np.array([0.0, 1.0, 0.0])
Z_AXIS = np.array([0.0, 0.0, 1.0])

HALF_PI = 0.5 * math.pi

unit_quaternions = (
    hnp.arrays(np.float64, 4, elements=st.floats(-1.0, 1.0))
    .filter(lambda q: bool(np.linalg.norm(q) > 0.1))
    .map(lambda q: q / np.linalg.norm(q))
)
vectors = hnp.arrays(np.float64, 3, elements=st.floats(-10.0, 10.0))
# Each component stays below pi / sqrt(3), so the full rotation angle is always below pi.
rotation_vectors = hnp.arrays(np.float64, 3, elements=st.floats(-1.8, 1.8))
fractions = st.floats(0.0, 1.0)
euler_angles = st.tuples(
    st.floats(-3.1, 3.1),
    st.floats(-1.5, 1.5),
    st.floats(-3.1, 3.1),
).map(np.array)


def assert_same_rotation(p: npt.ArrayLike, q: npt.ArrayLike, atol: float = 1e-9) -> None:
    """Assert that two quaternions describe the same rotation, ignoring their sign."""
    assert float(quat.angle_between(p, q)) <= atol


class TestBasics:
    def test_identity(self) -> None:
        assert_allclose(quat.identity(), ONE)

    def test_identity_returns_a_fresh_array(self) -> None:
        q = quat.identity()
        q[0] = 5.0
        assert quat.identity()[0] == 1.0

    @pytest.mark.parametrize(
        ("angle", "expected"),
        [
            (0.0, 0.0),
            (math.pi, -math.pi),
            (-math.pi, -math.pi),
            (1.5 * math.pi, -HALF_PI),
            (-1.5 * math.pi, HALF_PI),
            (2.0 * math.pi, 0.0),
            (7.0, 7.0 - 2.0 * math.pi),
        ],
    )
    def test_wrap_angle(self, angle: float, expected: float) -> None:
        assert_allclose(quat.wrap_angle(angle), expected, atol=ATOL)

    def test_wrap_angle_batch(self) -> None:
        assert_allclose(quat.wrap_angle([0.0, 4.0]), [0.0, 4.0 - 2.0 * math.pi], atol=ATOL)

    def test_is_unit(self) -> None:
        assert bool(quat.is_unit(ONE))
        assert not bool(quat.is_unit([1.0, 1.0, 0.0, 0.0]))
        assert quat.is_unit(np.stack([ONE, 2.0 * ONE])).tolist() == [True, False]


class TestNormalize:
    def test_scales_to_unit_length(self) -> None:
        assert_allclose(quat.normalize([2.0, 0.0, 0.0, 0.0]), ONE)
        assert_allclose(quat.normalize([1.0, 1.0, 1.0, 1.0]), [0.5, 0.5, 0.5, 0.5])

    def test_batch(self) -> None:
        result = quat.normalize([[3.0, 0.0, 0.0, 4.0], [0.0, 0.0, 2.0, 0.0]])
        assert_allclose(result, [[0.6, 0.0, 0.0, 0.8], [0.0, 0.0, 1.0, 0.0]])

    @pytest.mark.parametrize(
        "bad",
        [
            [0.0, 0.0, 0.0, 0.0],
            [1e-15, 0.0, 0.0, 0.0],
            [math.nan, 0.0, 0.0, 0.0],
            [math.inf, 0.0, 0.0, 0.0],
        ],
    )
    def test_rejects_degenerate_input(self, bad: list[float]) -> None:
        with pytest.raises(DegenerateQuaternionError):
            quat.normalize(bad)

    def test_rejects_batch_with_one_degenerate_row(self) -> None:
        with pytest.raises(DegenerateQuaternionError):
            quat.normalize([ONE, np.zeros(4)])


class TestShapeValidation:
    @pytest.mark.parametrize(
        "call",
        [
            lambda: quat.multiply([1.0, 0.0, 0.0], ONE),
            lambda: quat.multiply(ONE, 1.0),
            lambda: quat.rotate_vector(ONE, [1.0, 0.0]),
            lambda: quat.from_euler_zyx([0.0, 0.0, 0.0, 0.0]),
            lambda: quat.to_euler_zyx(np.zeros((2, 3))),
            lambda: quat.from_rotation_vector([0.0, 0.0]),
        ],
    )
    def test_wrong_last_axis_raises(self, call: Callable[[], FloatArray]) -> None:
        with pytest.raises(ValueError, match="must have shape"):
            call()


class TestAlgebra:
    @pytest.mark.parametrize(
        ("p", "q", "expected"),
        [
            (I, J, K),
            (J, K, I),
            (K, I, J),
            (J, I, -K),
            (K, J, -I),
            (I, K, -J),
            (I, I, -ONE),
            (J, J, -ONE),
            (K, K, -ONE),
        ],
    )
    def test_hamilton_basis_products(
        self, p: FloatArray, q: FloatArray, expected: FloatArray
    ) -> None:
        assert_allclose(quat.multiply(p, q), expected, atol=ATOL)

    def test_identity_is_neutral(self) -> None:
        q = quat.normalize([0.3, -0.1, 0.8, 0.2])
        assert_allclose(quat.multiply(ONE, q), q, atol=ATOL)
        assert_allclose(quat.multiply(q, ONE), q, atol=ATOL)

    def test_not_commutative(self) -> None:
        assert not np.allclose(quat.multiply(I, J), quat.multiply(J, I))

    def test_batch_broadcasts_against_single(self) -> None:
        result = quat.multiply(np.stack([I, J, K]), I)
        assert result.shape == (3, 4)
        assert_allclose(result, [-ONE, -K, J], atol=ATOL)

    def test_conjugate(self) -> None:
        assert_allclose(quat.conjugate([1.0, 2.0, 3.0, 4.0]), [1.0, -2.0, -3.0, -4.0])

    @given(unit_quaternions)
    def test_conjugate_is_inverse(self, q: FloatArray) -> None:
        assert_allclose(quat.multiply(q, quat.conjugate(q)), ONE, atol=ATOL)

    @given(unit_quaternions, unit_quaternions)
    def test_product_of_unit_quaternions_is_unit(self, p: FloatArray, q: FloatArray) -> None:
        assert bool(quat.is_unit(quat.multiply(p, q), atol=1e-12))

    @given(unit_quaternions, unit_quaternions, unit_quaternions)
    def test_associative(self, p: FloatArray, q: FloatArray, r: FloatArray) -> None:
        left = quat.multiply(quat.multiply(p, q), r)
        right = quat.multiply(p, quat.multiply(q, r))
        assert_allclose(left, right, atol=ATOL)


class TestRotateVector:
    def test_quarter_turn_about_z_maps_x_to_y(self) -> None:
        q = quat.from_axis_angle(Z_AXIS, HALF_PI)
        assert_allclose(quat.rotate_vector(q, X_AXIS), Y_AXIS, atol=ATOL)

    def test_batch(self) -> None:
        q = quat.from_axis_angle(Z_AXIS, HALF_PI)
        result = quat.rotate_vector(q, np.stack([X_AXIS, Y_AXIS, Z_AXIS]))
        assert_allclose(result, [Y_AXIS, -X_AXIS, Z_AXIS], atol=ATOL)

    @given(unit_quaternions, vectors)
    def test_preserves_length(self, q: FloatArray, v: FloatArray) -> None:
        rotated = quat.rotate_vector(q, v)
        assert_allclose(np.linalg.norm(rotated), np.linalg.norm(v), atol=1e-9)

    @given(unit_quaternions, vectors)
    def test_matches_sandwich_product(self, q: FloatArray, v: FloatArray) -> None:
        pure = np.concatenate([[0.0], v])
        expected = quat.multiply(quat.multiply(q, pure), quat.conjugate(q))[1:]
        assert_allclose(quat.rotate_vector(q, v), expected, atol=1e-9)

    @given(unit_quaternions, unit_quaternions, vectors)
    def test_product_applies_right_operand_first(
        self, p: FloatArray, q: FloatArray, v: FloatArray
    ) -> None:
        combined = quat.rotate_vector(quat.multiply(p, q), v)
        stepwise = quat.rotate_vector(p, quat.rotate_vector(q, v))
        assert_allclose(combined, stepwise, atol=1e-9)


class TestRotationVector:
    def test_zero_vector_is_identity(self) -> None:
        assert_allclose(quat.from_rotation_vector(np.zeros(3)), ONE, atol=ATOL)
        assert_allclose(quat.to_rotation_vector(ONE), np.zeros(3), atol=ATOL)

    def test_known_value(self) -> None:
        q = quat.from_rotation_vector([0.0, 0.0, HALF_PI])
        s = math.sqrt(0.5)
        assert_allclose(q, [s, 0.0, 0.0, s], atol=ATOL)

    @given(rotation_vectors)
    def test_round_trip(self, v: FloatArray) -> None:
        q = quat.from_rotation_vector(v)
        assert bool(quat.is_unit(q, atol=1e-12))
        assert_allclose(quat.to_rotation_vector(q), v, atol=1e-9)

    def test_tiny_angle_keeps_relative_precision(self) -> None:
        v = np.array([1e-10, -2e-10, 3e-10])
        assert_allclose(quat.to_rotation_vector(quat.from_rotation_vector(v)), v, rtol=1e-9)

    def test_half_turn_with_subnormal_scalar_does_not_overflow(self) -> None:
        s = 1.0 / math.sqrt(3.0)
        q = np.array([1e-311, s, s, s])
        assert_allclose(quat.to_rotation_vector(q), [math.pi * s] * 3, atol=1e-12)

    def test_negated_quaternion_gives_same_vector(self) -> None:
        q = quat.from_rotation_vector([0.4, -0.2, 1.1])
        assert_allclose(quat.to_rotation_vector(-q), quat.to_rotation_vector(q), atol=ATOL)

    def test_result_is_shortest_rotation(self) -> None:
        # A 270° turn about +z is the same as a 90° turn about -z.
        q = quat.from_axis_angle(Z_AXIS, 1.5 * math.pi)
        assert_allclose(quat.to_rotation_vector(q), [0.0, 0.0, -HALF_PI], atol=1e-9)

    def test_batch(self) -> None:
        vs = np.array([[0.0, 0.0, 0.0], [0.1, 0.2, 0.3], [1.0, 0.0, 0.0]])
        assert_allclose(quat.to_rotation_vector(quat.from_rotation_vector(vs)), vs, atol=1e-12)

    def test_from_axis_angle_accepts_non_unit_axis(self) -> None:
        assert_allclose(
            quat.from_axis_angle([0.0, 0.0, 5.0], HALF_PI),
            quat.from_rotation_vector([0.0, 0.0, HALF_PI]),
            atol=ATOL,
        )

    def test_from_axis_angle_batch(self) -> None:
        q = quat.from_axis_angle(np.stack([X_AXIS, Y_AXIS]), [HALF_PI, math.pi])
        assert q.shape == (2, 4)
        assert_allclose(q[1], J, atol=ATOL)

    @pytest.mark.parametrize("axis", [[0.0, 0.0, 0.0], [math.nan, 0.0, 1.0]])
    def test_from_axis_angle_rejects_invalid_axis(self, axis: list[float]) -> None:
        with pytest.raises(ValueError, match="axis must be finite and non-zero"):
            quat.from_axis_angle(axis, 1.0)


class TestEuler:
    @pytest.mark.parametrize(
        ("angles", "vector", "expected"),
        [
            ([0.0, 0.0, HALF_PI], X_AXIS, Y_AXIS),
            ([0.0, HALF_PI, 0.0], X_AXIS, -Z_AXIS),
            ([HALF_PI, 0.0, 0.0], Y_AXIS, Z_AXIS),
        ],
        ids=["yaw", "pitch", "roll"],
    )
    def test_single_axis_rotations_follow_right_hand_rule(
        self, angles: list[float], vector: FloatArray, expected: FloatArray
    ) -> None:
        q = quat.from_euler_zyx(angles)
        assert_allclose(quat.rotate_vector(q, vector), expected, atol=ATOL)

    @given(euler_angles)
    def test_composition_order_is_yaw_pitch_roll(self, angles: FloatArray) -> None:
        roll, pitch, yaw = angles
        expected = quat.multiply(
            quat.multiply(quat.from_axis_angle(Z_AXIS, yaw), quat.from_axis_angle(Y_AXIS, pitch)),
            quat.from_axis_angle(X_AXIS, roll),
        )
        assert_allclose(quat.from_euler_zyx(angles), expected, atol=ATOL)

    @given(euler_angles)
    def test_round_trip_away_from_gimbal_lock(self, angles: FloatArray) -> None:
        recovered = quat.to_euler_zyx(quat.from_euler_zyx(angles))
        assert_allclose(quat.wrap_angle(recovered - angles), 0.0, atol=1e-9)

    @given(unit_quaternions)
    def test_quaternion_round_trip(self, q: FloatArray) -> None:
        assert_same_rotation(quat.from_euler_zyx(quat.to_euler_zyx(q)), q, atol=1e-7)

    @pytest.mark.parametrize(
        ("q", "expected"),
        [(I, [-math.pi, 0.0, 0.0]), (K, [0.0, 0.0, -math.pi]), (J, [-math.pi, 0.0, -math.pi])],
        ids=["half_turn_roll", "half_turn_yaw", "half_turn_pitch"],
    )
    def test_half_turns_map_to_minus_pi(self, q: FloatArray, expected: list[float]) -> None:
        angles = quat.to_euler_zyx(q)
        assert_allclose(angles, expected, atol=ATOL)
        assert np.all(angles[[0, 2]] < math.pi)

    def test_output_ranges(self) -> None:
        angles = quat.to_euler_zyx(quat.from_euler_zyx([3.0, -1.4, -3.0]))
        assert -math.pi <= angles[0] <= math.pi
        assert -HALF_PI <= angles[1] <= HALF_PI
        assert -math.pi <= angles[2] <= math.pi

    @pytest.mark.parametrize(
        ("pitch", "expected_yaw"),
        [(HALF_PI, 0.5 - 0.3), (-HALF_PI, 0.5 + 0.3)],
        ids=["pitch_up", "pitch_down"],
    )
    def test_gimbal_lock_moves_roll_into_yaw(self, pitch: float, expected_yaw: float) -> None:
        q = quat.from_euler_zyx([0.3, pitch, 0.5])
        roll, recovered_pitch, yaw = quat.to_euler_zyx(q)
        assert roll == 0.0
        assert_allclose(recovered_pitch, pitch, atol=1e-12)
        assert_allclose(yaw, expected_yaw, atol=1e-9)
        assert_same_rotation(quat.from_euler_zyx([roll, recovered_pitch, yaw]), q)

    def test_keeps_precision_just_outside_gimbal_lock(self) -> None:
        angles = np.array([0.3, HALF_PI - 1e-6, 0.5])
        recovered = quat.to_euler_zyx(quat.from_euler_zyx(angles))
        assert_allclose(recovered, angles, atol=1e-9)

    def test_gimbal_lock_yaw_is_wrapped(self) -> None:
        q = quat.from_euler_zyx([-3.0, HALF_PI, 3.0])
        _, _, yaw = quat.to_euler_zyx(q)
        assert_allclose(yaw, 6.0 - 2.0 * math.pi, atol=1e-9)

    def test_batch(self) -> None:
        angles = np.array([[0.1, 0.2, 0.3], [-0.4, 0.5, -0.6]])
        q = quat.from_euler_zyx(angles)
        assert q.shape == (2, 4)
        assert_allclose(quat.to_euler_zyx(q), angles, atol=1e-12)

    @pytest.mark.parametrize(
        ("pitch_degrees", "expected"),
        [(0.0, False), (80.0, False), (87.9, False), (88.1, True), (90.0, True), (-89.0, True)],
    )
    def test_is_near_gimbal_lock(self, pitch_degrees: float, expected: bool) -> None:
        q = quat.from_euler_zyx([0.2, math.radians(pitch_degrees), -0.7])
        assert bool(quat.is_near_gimbal_lock(q)) is expected

    def test_is_near_gimbal_lock_custom_margin(self) -> None:
        q = quat.from_euler_zyx([0.0, math.radians(80.0), 0.0])
        assert bool(quat.is_near_gimbal_lock(q, margin=math.radians(15.0)))

    @pytest.mark.parametrize("margin", [0.0, -0.1, HALF_PI, math.nan])
    def test_is_near_gimbal_lock_rejects_invalid_margin(self, margin: float) -> None:
        with pytest.raises(ValueError, match="margin"):
            quat.is_near_gimbal_lock(ONE, margin=margin)


class TestHemisphereAndDistance:
    def test_canonicalize_makes_scalar_non_negative(self) -> None:
        result = quat.canonicalize([[-0.5, 0.5, -0.5, 0.5], [0.5, 0.5, 0.5, 0.5]])
        assert_allclose(result, [[0.5, -0.5, 0.5, -0.5], [0.5, 0.5, 0.5, 0.5]])

    def test_ensure_same_hemisphere_flips_opposite_quaternion(self) -> None:
        q = quat.normalize([0.9, 0.1, 0.0, 0.0])
        assert_allclose(quat.ensure_same_hemisphere(-q, q), q)
        assert_allclose(quat.ensure_same_hemisphere(q, q), q)

    def test_ensure_same_hemisphere_batch(self) -> None:
        result = quat.ensure_same_hemisphere(np.stack([ONE, -ONE]), ONE)
        assert_allclose(result, [ONE, ONE])

    def test_angle_between_known_value(self) -> None:
        q = quat.from_axis_angle(Y_AXIS, 0.75)
        assert_allclose(quat.angle_between(ONE, q), 0.75, atol=ATOL)

    def test_angle_between_ignores_sign(self) -> None:
        q = quat.from_axis_angle(Y_AXIS, 0.75)
        assert_allclose(quat.angle_between(q, -q), 0.0, atol=ATOL)

    @given(unit_quaternions, unit_quaternions)
    def test_angle_between_is_symmetric_and_bounded(self, p: FloatArray, q: FloatArray) -> None:
        forward = float(quat.angle_between(p, q))
        assert_allclose(forward, quat.angle_between(q, p), atol=1e-9)
        assert 0.0 <= forward <= math.pi + 1e-12


class TestInterpolation:
    @pytest.mark.parametrize("interpolate", [quat.slerp, quat.nlerp])
    def test_endpoints(self, interpolate: Callable[..., FloatArray]) -> None:
        p = quat.from_axis_angle(X_AXIS, 0.2)
        q = quat.from_axis_angle(Z_AXIS, 1.3)
        assert_same_rotation(interpolate(p, q, 0.0), p)
        assert_same_rotation(interpolate(p, q, 1.0), q)

    @pytest.mark.parametrize("interpolate", [quat.slerp, quat.nlerp])
    def test_takes_shortest_path(self, interpolate: Callable[..., FloatArray]) -> None:
        q = quat.from_axis_angle(Z_AXIS, 1.0)
        midpoint = interpolate(ONE, -q, 0.5)
        assert_same_rotation(midpoint, quat.from_axis_angle(Z_AXIS, 0.5))

    @pytest.mark.parametrize("interpolate", [quat.slerp, quat.nlerp])
    @pytest.mark.parametrize("t", [-0.01, 1.01, math.nan])
    def test_rejects_fraction_outside_unit_interval(
        self, interpolate: Callable[..., FloatArray], t: float
    ) -> None:
        with pytest.raises(ValueError, match="t must be in"):
            interpolate(ONE, ONE, t)

    @given(unit_quaternions, unit_quaternions, fractions)
    def test_slerp_has_constant_angular_speed(self, p: FloatArray, q: FloatArray, t: float) -> None:
        result = quat.slerp(p, q, t)
        assert bool(quat.is_unit(result, atol=1e-12))
        expected = t * float(quat.angle_between(p, q))
        assert_allclose(quat.angle_between(p, result), expected, atol=1e-6)

    def test_slerp_of_nearly_equal_quaternions_is_stable(self) -> None:
        p = quat.from_axis_angle(X_AXIS, 0.3)
        q = quat.from_axis_angle(X_AXIS, 0.3 + 1e-9)
        result = quat.slerp(p, q, 0.5)
        assert_same_rotation(result, quat.from_axis_angle(X_AXIS, 0.3 + 5e-10), atol=1e-12)

    def test_slerp_batch(self) -> None:
        ends = quat.from_axis_angle(np.stack([X_AXIS, Y_AXIS]), [1.0, 2.0])
        result = quat.slerp(ONE, ends, 0.5)
        assert_allclose(result, quat.from_axis_angle(np.stack([X_AXIS, Y_AXIS]), [0.5, 1.0]))


class TestIntegrate:
    def test_constant_rate_matches_closed_form(self) -> None:
        omega = np.array([0.0, 0.0, 2.0])
        q = quat.identity()
        for _ in range(500):
            q = quat.integrate(q, omega, 0.002)
        assert_same_rotation(q, quat.from_axis_angle(Z_AXIS, 2.0))

    def test_rate_is_applied_in_body_frame(self) -> None:
        # After a 90° yaw, a body-frame roll rate turns about the world y axis.
        start = quat.from_axis_angle(Z_AXIS, HALF_PI)
        result = quat.integrate(start, [HALF_PI, 0.0, 0.0], 1.0)
        expected = quat.multiply(quat.from_axis_angle(Y_AXIS, HALF_PI), start)
        assert_same_rotation(result, expected)

    def test_zero_dt_keeps_orientation(self) -> None:
        q = quat.from_axis_angle(X_AXIS, 0.4)
        assert_allclose(quat.integrate(q, [5.0, -3.0, 1.0], 0.0), q, atol=ATOL)

    def test_stays_unit_over_many_steps(self) -> None:
        rng = np.random.default_rng(seed=7)
        q = quat.identity()
        for omega in rng.normal(scale=3.0, size=(20_000, 3)):
            q = quat.integrate(q, omega, 0.002)
        assert bool(quat.is_unit(q, atol=1e-12))

    @pytest.mark.parametrize("dt", [-0.001, math.nan, math.inf])
    def test_rejects_invalid_dt(self, dt: float) -> None:
        with pytest.raises(ValueError, match="dt must be finite"):
            quat.integrate(ONE, [0.0, 0.0, 1.0], dt)
