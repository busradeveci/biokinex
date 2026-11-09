"""Tests for biokinex.transport.codecs."""

from __future__ import annotations

import json
import math
import struct
from typing import Any, cast

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.testing import assert_allclose, assert_array_equal

from biokinex.core import quaternion as quat
from biokinex.core.errors import TelemetryError
from biokinex.schemas.batch import SampleBatch
from biokinex.schemas.pose import FrameStatus, PoseFrame
from biokinex.schemas.telemetry import TelemetryPacket
from biokinex.transport.base import Codec
from biokinex.transport.codecs import BinaryCodec, ChannelFlags, JsonCodec

PACKET_HEADER = struct.Struct("<4sBBHBB")
FRAME_HEADER = struct.Struct("<4sBBBBB")
Z_AXIS = np.array([0.0, 0.0, 1.0])


def make_batch(n: int = 3, *, joints: tuple[str, ...] = ("knee", "hip")) -> SampleBatch:
    seq = np.arange(n, dtype=np.int64) + 10
    angles = np.linspace(0.0, 1.0, n)
    return SampleBatch(
        device_id="imu-0",
        timestamps=seq.astype(np.float64) * 0.002,
        sequences=seq,
        acceleration=np.column_stack([angles, -angles, np.full(n, 9.8)]),
        angular_velocity=np.column_stack([angles, angles, angles]) * 0.5,
        orientation=quat.from_axis_angle(Z_AXIS, angles),
        joint_names=joints,
        joint_angles=np.column_stack([angles] * len(joints)) if joints else None,
    )


def assert_same_batch(actual: SampleBatch, expected: SampleBatch) -> None:
    assert actual.device_id == expected.device_id
    assert actual.joint_names == expected.joint_names
    assert_array_equal(actual.sequences, expected.sequences)
    assert_allclose(actual.timestamps, expected.timestamps)
    for name in ("acceleration", "angular_velocity", "orientation", "joint_angles"):
        actual_values, expected_values = getattr(actual, name), getattr(expected, name)
        if expected_values is None:
            assert actual_values is None
        else:
            assert_allclose(actual_values, expected_values, atol=1e-15)


def raw_packet(
    *,
    magic: bytes = b"BKXP",
    version: int = 1,
    flags: int = ChannelFlags.ACCELERATION,
    count: int = 1,
    joint_names: tuple[bytes, ...] = (),
    device_id: bytes = b"imu-0",
    records: list[tuple[float, ...]] | None = None,
) -> bytes:
    """Build a binary packet by hand, so tests can break any single rule."""
    header = PACKET_HEADER.pack(magic, version, flags, count, len(joint_names), len(device_id))
    names = b"".join(bytes([len(name)]) + name for name in joint_names)
    if records is None:
        records = [(i, 0.002 * i, 0.0, 0.0, 9.8) for i in range(count)]
    body = b"".join(struct.pack(f"<I{len(r) - 1}d", int(r[0]), *r[1:]) for r in records)
    return header + device_id + names + body


def make_frame(**fields: Any) -> PoseFrame:
    raw: dict[str, Any] = {
        "device_id": "imu-0",
        "sequence": 42,
        "timestamp": 1.25,
        "status": FrameStatus.DEGRADED,
        "orientation": tuple(quat.from_axis_angle(Z_AXIS, 0.3).tolist()),
        "angular_velocity": (0.1, 0.2, 0.3),
        "acceleration": (0.0, 0.5, 9.8),
        "joint_names": ("knee",),
        "joint_angles": (0.7,),
    }
    raw.update(fields)
    return PoseFrame.model_validate(raw)


def test_codecs_follow_the_protocol() -> None:
    assert isinstance(JsonCodec(), Codec)
    assert isinstance(BinaryCodec(), Codec)


class TestJsonCodec:
    PAYLOAD = json.dumps(
        {
            "device_id": "imu-0",
            "units": {"acceleration": "g"},
            "samples": [
                {"sequence": 1, "timestamp": 0.0, "acceleration": [0.0, 0.0, 1.0]},
                {"sequence": 2, "timestamp": 0.002, "acceleration": [0.0, 0.0, 1.0]},
            ],
        }
    ).encode()

    def test_decode_packet_converts_to_si(self) -> None:
        batch = JsonCodec().decode_packet(self.PAYLOAD)
        assert batch.sequences.tolist() == [1, 2]
        assert batch.acceleration is not None
        assert batch.acceleration[0, 2] == pytest.approx(9.80665)

    def test_decode_packet_rejects_invalid_payload(self) -> None:
        with pytest.raises(TelemetryError):
            JsonCodec().decode_packet(b'{"device_id": "imu-0"}')

    def test_packet_round_trip(self) -> None:
        codec = JsonCodec()
        packet = TelemetryPacket.from_json(self.PAYLOAD)
        encoded = codec.encode_packet(packet)
        assert b"null" not in encoded
        assert_same_batch(codec.decode_packet(encoded), packet.to_batch())

    def test_frame_round_trip_includes_euler(self) -> None:
        codec = JsonCodec()
        frame = make_frame()
        encoded = codec.encode_frame(frame)
        assert b'"euler"' in encoded
        assert codec.decode_frame(encoded) == frame

    def test_decode_frame_rejects_invalid_payload(self) -> None:
        with pytest.raises(TelemetryError, match="invalid pose frame"):
            JsonCodec().decode_frame(b'{"device_id": "imu-0"}')


class TestBinaryPackets:
    def test_round_trip_with_every_channel(self) -> None:
        codec = BinaryCodec()
        batch = make_batch()
        assert_same_batch(codec.decode_packet(codec.encode_batch(batch)), batch)

    @pytest.mark.parametrize(
        "channels",
        [{"acceleration"}, {"angular_velocity"}, {"orientation"}, {"joint_angles"}],
    )
    def test_round_trip_with_one_channel(self, channels: set[str]) -> None:
        full = make_batch()
        batch = SampleBatch(
            device_id=full.device_id,
            timestamps=full.timestamps,
            sequences=full.sequences,
            joint_names=full.joint_names if "joint_angles" in channels else (),
            **{name: getattr(full, name) for name in channels},
        )
        codec = BinaryCodec()
        assert_same_batch(codec.decode_packet(codec.encode_batch(batch)), batch)

    def test_channel_arrays_are_read_only_views_of_the_payload(self) -> None:
        codec = BinaryCodec()
        payload = codec.encode_batch(make_batch())
        batch = codec.decode_packet(payload)
        raw = np.frombuffer(payload, dtype=np.uint8)
        assert batch.acceleration is not None
        assert np.shares_memory(batch.acceleration, raw)
        assert np.shares_memory(batch.timestamps, raw)
        assert not batch.acceleration.flags.writeable

    def test_bytearray_payload_still_gives_read_only_arrays(self) -> None:
        codec = BinaryCodec()
        payload = cast("bytes", bytearray(codec.encode_batch(make_batch())))
        batch = codec.decode_packet(payload)
        for array in (batch.timestamps, batch.acceleration, batch.joint_angles):
            assert array is not None
            assert not array.flags.writeable

    def test_orientation_is_normalized(self) -> None:
        payload = raw_packet(
            flags=ChannelFlags.ORIENTATION, records=[(0, 0.0, 1.05, 0.0, 0.0, 0.0)]
        )
        batch = BinaryCodec().decode_packet(payload)
        assert batch.orientation is not None
        assert_allclose(batch.orientation, [[1.0, 0.0, 0.0, 0.0]])

    def test_matches_json_decoding(self) -> None:
        packet = TelemetryPacket.model_validate(
            {
                "device_id": "imu-3",
                "joint_names": ["knee"],
                "samples": [
                    {
                        "sequence": 5,
                        "timestamp": 1.0,
                        "acceleration": [0.1, 0.2, 9.7],
                        "orientation_quaternion": [0.99, 0.0, 0.1, 0.0],
                        "joint_angles": [0.4],
                    }
                ],
            }
        )
        from_json = JsonCodec().decode_packet(JsonCodec().encode_packet(packet))
        codec = BinaryCodec()
        assert_same_batch(codec.decode_packet(codec.encode_batch(packet.to_batch())), from_json)

    def test_encode_rejects_too_many_samples(self) -> None:
        with pytest.raises(ValueError, match="at most 256"):
            BinaryCodec().encode_batch(make_batch(257))

    @pytest.mark.parametrize(
        ("payload", "match"),
        [
            (b"BKX", "truncated"),
            (raw_packet(magic=b"NOPE"), "does not start with"),
            (raw_packet(version=2), "unsupported binary version"),
            (raw_packet(flags=0x08), "unknown channel flags"),
            (raw_packet(count=0, records=[]), "sample count"),
            (raw_packet(count=300, records=[]), "sample count"),
            (raw_packet(flags=0, records=[(0, 0.0)]), "at least one measurement"),
            (raw_packet(device_id=b"bad id"), "device_id is not a valid identifier"),
            (raw_packet(device_id=b""), "device_id is not a valid identifier"),
            (raw_packet(device_id="imü".encode()), "device_id is not a valid identifier"),
            (raw_packet(device_id=b"imu\n"), "device_id is not a valid identifier"),
            (raw_packet() + b"\x00", "bytes of data"),
            (raw_packet()[:-1], "bytes of data"),
            (raw_packet(records=[(0, -1.0, 0.0, 0.0, 9.8)]), "non-negative"),
            (raw_packet(records=[(0, math.nan, 0.0, 0.0, 9.8)]), "finite"),
            (
                raw_packet(count=2, records=[(0, 0.5, 0, 0, 9.8), (1, 0.5, 0, 0, 9.8)]),
                "strictly increasing",
            ),
            (raw_packet(records=[(0, 0.0, math.inf, 0.0, 9.8)]), "acceleration values"),
            (
                raw_packet(flags=ChannelFlags.ORIENTATION, records=[(0, 0.0, 1.5, 0, 0, 0)]),
                "norm is too far",
            ),
        ],
    )
    def test_rejects_malformed_packets(self, payload: bytes, match: str) -> None:
        with pytest.raises(TelemetryError, match=match):
            BinaryCodec().decode_packet(payload)

    def test_rejects_duplicate_joint_names(self) -> None:
        payload = raw_packet(flags=0, joint_names=(b"knee", b"knee"), records=[(0, 0.0, 0.1, 0.2)])
        with pytest.raises(TelemetryError, match="unique"):
            BinaryCodec().decode_packet(payload)

    def test_rejects_too_many_joints(self) -> None:
        names = tuple(f"j{i}".encode() for i in range(65))
        with pytest.raises(TelemetryError, match="joint count"):
            BinaryCodec().decode_packet(raw_packet(joint_names=names))

    def test_rejects_invalid_joint_name(self) -> None:
        payload = raw_packet(flags=0, joint_names=(b"-knee",), records=[(0, 0.0, 0.1)])
        with pytest.raises(TelemetryError, match="joint name"):
            BinaryCodec().decode_packet(payload)

    @given(
        n=st.integers(min_value=1, max_value=20),
        joint_count=st.integers(min_value=0, max_value=4),
        seed=st.integers(min_value=0, max_value=2**32 - 1),
    )
    def test_round_trip_property(self, n: int, joint_count: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        names = tuple(f"joint{i}" for i in range(joint_count))
        q = rng.normal(size=(n, 4))
        batch = SampleBatch(
            device_id="dev:1",
            timestamps=np.cumsum(rng.uniform(1e-4, 0.1, n)),
            sequences=rng.integers(0, 2**32, n),
            acceleration=rng.normal(0.0, 50.0, (n, 3)),
            angular_velocity=rng.normal(0.0, 5.0, (n, 3)),
            orientation=q / np.linalg.norm(q, axis=1, keepdims=True),
            joint_names=names,
            joint_angles=rng.uniform(-3.0, 3.0, (n, joint_count)) if joint_count else None,
        )
        codec = BinaryCodec()
        assert_same_batch(codec.decode_packet(codec.encode_batch(batch)), batch)


class TestBinaryFrames:
    @pytest.mark.parametrize(
        "fields",
        [
            {},
            {"status": FrameStatus.OK, "joint_names": (), "joint_angles": None},
            {"status": FrameStatus.STALE, "orientation": None, "acceleration": None},
            {
                "orientation": None,
                "angular_velocity": None,
                "acceleration": None,
                "joint_names": ("a", "b"),
                "joint_angles": (0.1, 0.2),
            },
        ],
    )
    def test_round_trip(self, fields: dict[str, Any]) -> None:
        codec = BinaryCodec()
        frame = make_frame(**fields)
        assert codec.decode_frame(codec.encode_frame(frame)) == frame

    def test_frame_is_compact(self) -> None:
        payload = BinaryCodec().encode_frame(make_frame())
        assert len(payload) < len(JsonCodec().encode_frame(make_frame())) // 2

    @pytest.mark.parametrize(
        ("mutate", "match"),
        [
            (lambda p: b"NOPE" + p[4:], "does not start with"),
            (lambda p: p[:5] + b"\x09" + p[6:], "unknown frame status"),
            (lambda p: p[:6] + b"\x10" + p[7:], "unknown channel flags"),
            (lambda p: p[:-1], "truncated"),
            (lambda p: p + b"\x00", "bytes of data"),
        ],
    )
    def test_rejects_malformed_frames(self, mutate: Any, match: str) -> None:
        payload = mutate(BinaryCodec().encode_frame(make_frame()))
        with pytest.raises(TelemetryError, match=match):
            BinaryCodec().decode_frame(payload)

    def test_rejects_frame_that_breaks_the_schema(self) -> None:
        header = FRAME_HEADER.pack(b"BKXF", 1, 0, ChannelFlags.ORIENTATION, 0, 5)
        body = struct.pack("<Id4d", 1, 0.0, 2.0, 0.0, 0.0, 0.0)
        with pytest.raises(TelemetryError, match="invalid pose frame"):
            BinaryCodec().decode_frame(header + b"imu-0" + body)
