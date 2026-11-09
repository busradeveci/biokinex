"""Wire formats for telemetry packets and pose frames.

Two codecs are provided:

- :class:`JsonCodec` uses the Pydantic schemas directly. It is easy to read and to produce from
  any language, and it accepts every unit the schema supports.
- :class:`BinaryCodec` uses a fixed little-endian layout in SI units. Packets are decoded with
  ``numpy.frombuffer``, so the channel arrays are zero-copy views of the payload. It skips
  Pydantic for speed, but it applies the same safety rules: finite values, increasing timestamps,
  valid identifiers, and near-unit quaternions.

Binary packet layout (all little-endian)::

    header    magic "BKXP" | version u8 | flags u8 | sample_count u16 | joint_count u8
              | device_id_length u8
    names     device_id bytes, then for each joint: name_length u8 | name bytes
    samples   sample_count records of: sequence u32 | timestamp f64
              | acceleration 3 x f64 (flag 1) | angular_velocity 3 x f64 (flag 2)
              | orientation 4 x f64 (flag 4) | joint_angles joint_count x f64

Binary frame layout::

    header    magic "BKXF" | version u8 | status u8 | flags u8 | joint_count u8
              | device_id_length u8
    names     as in a packet
    body      sequence u32 | timestamp f64 | orientation 4 x f64 (flag 4)
              | angular_velocity 3 x f64 (flag 2) | acceleration 3 x f64 (flag 1)
              | joint_angles joint_count x f64
"""

from __future__ import annotations

import functools
import re
import struct
from collections.abc import Sequence
from enum import IntFlag
from typing import Any, Final

import numpy as np
from pydantic import ValidationError

from biokinex.core import quaternion
from biokinex.core.errors import TelemetryError
from biokinex.core.types import QUATERNION_SIZE, VECTOR3_SIZE, FloatArray
from biokinex.schemas.batch import SampleBatch, read_only
from biokinex.schemas.common import IDENTIFIER_PATTERN, MAX_JOINTS
from biokinex.schemas.pose import FrameStatus, PoseFrame
from biokinex.schemas.telemetry import (
    MAX_SAMPLES_PER_PACKET,
    QUATERNION_NORM_TOLERANCE,
    TelemetryPacket,
)

__all__ = ["BinaryCodec", "ChannelFlags", "JsonCodec"]

_PACKET_MAGIC: Final = b"BKXP"
_FRAME_MAGIC: Final = b"BKXF"
_VERSION: Final = 1
_PACKET_HEADER: Final = struct.Struct("<4sBBHBB")
_FRAME_HEADER: Final = struct.Struct("<4sBBBBB")
_NAME_LENGTH: Final = struct.Struct("<B")
_IDENTIFIER: Final = re.compile(IDENTIFIER_PATTERN)

_STATUS_CODES: Final = {FrameStatus.OK: 0, FrameStatus.DEGRADED: 1, FrameStatus.STALE: 2}
_STATUS_BY_CODE: Final = {code: status for status, code in _STATUS_CODES.items()}


class ChannelFlags(IntFlag):
    """Which optional channels a binary record carries; joints are given by ``joint_count``."""

    NONE = 0
    ACCELERATION = 1
    ANGULAR_VELOCITY = 2
    ORIENTATION = 4


_ALL_FLAGS: Final = (
    ChannelFlags.ACCELERATION | ChannelFlags.ANGULAR_VELOCITY | ChannelFlags.ORIENTATION
)


class JsonCodec:
    """JSON codec built on the Pydantic boundary schemas."""

    __slots__ = ()

    def decode_packet(self, data: bytes) -> SampleBatch:
        """Parse a JSON telemetry packet and convert it to SI arrays.

        Raises:
            TelemetryError: If the payload is not valid JSON or breaks any schema rule.
        """
        return TelemetryPacket.from_json(data).to_batch()

    def encode_packet(self, packet: TelemetryPacket) -> bytes:
        """Encode a packet as compact JSON, for device simulators and tests."""
        return packet.model_dump_json(exclude_none=True).encode("utf-8")

    def encode_frame(self, frame: PoseFrame) -> bytes:
        """Encode a frame as JSON, including the derived Euler angles."""
        return frame.model_dump_json().encode("utf-8")

    def decode_frame(self, data: bytes) -> PoseFrame:
        """Parse and validate a JSON pose frame, for consumers and tests.

        Raises:
            TelemetryError: If the payload is not a valid pose frame.
        """
        try:
            return PoseFrame.model_validate_json(data)
        except ValidationError as exc:
            msg = f"invalid pose frame ({exc.error_count()} error(s))"
            raise TelemetryError(msg) from exc


class BinaryCodec:
    """Compact binary codec with zero-copy packet decoding."""

    __slots__ = ()

    def decode_packet(self, data: bytes) -> SampleBatch:
        """Parse a binary telemetry packet.

        Raises:
            TelemetryError: If the payload is malformed or breaks any schema rule.
        """
        reader = _Reader(data)
        magic, version, raw_flags, count, joint_count, id_length = reader.unpack(_PACKET_HEADER)
        _check_header(magic, _PACKET_MAGIC, version)
        flags = _parse_flags(raw_flags)
        if not 1 <= count <= MAX_SAMPLES_PER_PACKET:
            msg = f"sample count must be in [1, {MAX_SAMPLES_PER_PACKET}], got {count}"
            raise TelemetryError(msg)
        if flags is ChannelFlags.NONE and joint_count == 0:
            msg = "a packet must carry at least one measurement"
            raise TelemetryError(msg)
        device_id, joint_names = _read_names(reader, id_length, joint_count)

        dtype = _sample_dtype(flags, joint_count)
        reader.expect_remaining(count * dtype.itemsize)
        # The records are a view of the payload, so no sample data is copied here.
        records = np.frombuffer(data, dtype=dtype, count=count, offset=reader.offset)

        # A bytearray payload gives writable views, so every view is locked before it is shared.
        timestamps = read_only(records["timestamp"])
        if not (np.isfinite(timestamps).all() and timestamps[0] >= 0.0):
            msg = "timestamps must be finite and non-negative"
            raise TelemetryError(msg)
        if count > 1 and (np.diff(timestamps) <= 0.0).any():
            msg = "timestamps must be strictly increasing"
            raise TelemetryError(msg)

        orientation = _optional_field(records, "orientation")
        if orientation is not None:
            # Like the JSON schema, small rounding errors are fixed but corrupted values are not.
            norms = np.linalg.norm(orientation, axis=1)
            if np.any(np.abs(norms - 1.0) > QUATERNION_NORM_TOLERANCE):
                msg = "orientation quaternion norm is too far from 1"
                raise TelemetryError(msg)
            orientation = read_only(quaternion.normalize(orientation))

        return SampleBatch(
            device_id=device_id,
            timestamps=timestamps,
            sequences=read_only(records["sequence"].astype(np.int64)),
            acceleration=_optional_field(records, "acceleration"),
            angular_velocity=_optional_field(records, "angular_velocity"),
            orientation=orientation,
            joint_names=joint_names,
            joint_angles=_optional_field(records, "joint_angles"),
        )

    def encode_batch(self, batch: SampleBatch) -> bytes:
        """Encode a batch as a binary packet, for device simulators and tests.

        Raises:
            ValueError: If the batch has more samples than one packet may carry.
        """
        count = len(batch)
        if count > MAX_SAMPLES_PER_PACKET:
            msg = f"a packet carries at most {MAX_SAMPLES_PER_PACKET} samples, got {count}"
            raise ValueError(msg)
        flags = ChannelFlags.NONE
        if batch.acceleration is not None:
            flags |= ChannelFlags.ACCELERATION
        if batch.angular_velocity is not None:
            flags |= ChannelFlags.ANGULAR_VELOCITY
        if batch.orientation is not None:
            flags |= ChannelFlags.ORIENTATION
        joint_count = 0 if batch.joint_angles is None else len(batch.joint_names)

        device_id = batch.device_id.encode("ascii")
        header = _PACKET_HEADER.pack(
            _PACKET_MAGIC, _VERSION, flags, count, joint_count, len(device_id)
        )
        records = np.empty(count, dtype=_sample_dtype(flags, joint_count))
        records["sequence"] = batch.sequences
        records["timestamp"] = batch.timestamps
        for name in records.dtype.names or ():
            if name not in {"sequence", "timestamp"}:
                records[name] = getattr(batch, name)
        names = batch.joint_names[:joint_count]
        return header + device_id + _encode_names(names) + records.tobytes()

    def encode_frame(self, frame: PoseFrame) -> bytes:
        """Encode a frame in the fixed binary layout."""
        flags = ChannelFlags.NONE
        body: list[float] = []
        if frame.orientation is not None:
            flags |= ChannelFlags.ORIENTATION
            body.extend(frame.orientation)
        if frame.angular_velocity is not None:
            flags |= ChannelFlags.ANGULAR_VELOCITY
            body.extend(frame.angular_velocity)
        if frame.acceleration is not None:
            flags |= ChannelFlags.ACCELERATION
            body.extend(frame.acceleration)
        joints = frame.joint_angles or ()
        body.extend(joints)

        device_id = frame.device_id.encode("ascii")
        header = _FRAME_HEADER.pack(
            _FRAME_MAGIC,
            _VERSION,
            _STATUS_CODES[frame.status],
            flags,
            len(joints),
            len(device_id),
        )
        values = struct.pack(f"<Id{len(body)}d", frame.sequence, frame.timestamp, *body)
        names = _encode_names(frame.joint_names if joints else ())
        return header + device_id + names + values

    def decode_frame(self, data: bytes) -> PoseFrame:
        """Parse and validate a binary pose frame, for consumers and tests.

        Raises:
            TelemetryError: If the payload is malformed or is not a valid pose frame.
        """
        reader = _Reader(data)
        magic, version, status_code, raw_flags, joint_count, id_length = reader.unpack(
            _FRAME_HEADER
        )
        _check_header(magic, _FRAME_MAGIC, version)
        status = _STATUS_BY_CODE.get(status_code)
        if status is None:
            msg = f"unknown frame status code {status_code}"
            raise TelemetryError(msg)
        flags = _parse_flags(raw_flags)
        device_id, joint_names = _read_names(reader, id_length, joint_count)

        sizes = [
            (ChannelFlags.ORIENTATION, "orientation", QUATERNION_SIZE),
            (ChannelFlags.ANGULAR_VELOCITY, "angular_velocity", VECTOR3_SIZE),
            (ChannelFlags.ACCELERATION, "acceleration", VECTOR3_SIZE),
        ]
        value_count = sum(size for flag, _, size in sizes if flag in flags) + joint_count
        sequence, timestamp, *values = reader.unpack(struct.Struct(f"<Id{value_count}d"))
        reader.expect_remaining(0)

        fields: dict[str, Any] = {
            "device_id": device_id,
            "sequence": sequence,
            "timestamp": timestamp,
            "status": status,
            "joint_names": joint_names,
        }
        start = 0
        for flag, name, size in sizes:
            if flag in flags:
                fields[name] = tuple(values[start : start + size])
                start += size
        if joint_count:
            fields["joint_angles"] = tuple(values[start:])
        try:
            return PoseFrame.model_validate(fields)
        except ValidationError as exc:
            msg = f"invalid pose frame ({exc.error_count()} error(s))"
            raise TelemetryError(msg) from exc


class _Reader:
    """Reads a payload from front to back and reports truncation as a telemetry error."""

    __slots__ = ("_data", "offset")

    def __init__(self, data: bytes) -> None:
        self._data = data
        self.offset = 0

    def take(self, size: int) -> bytes:
        end = self.offset + size
        if end > len(self._data):
            msg = "payload is truncated"
            raise TelemetryError(msg)
        chunk = self._data[self.offset : end]
        self.offset = end
        return chunk

    def unpack(self, layout: struct.Struct) -> tuple[Any, ...]:
        return layout.unpack(self.take(layout.size))

    def expect_remaining(self, size: int) -> None:
        """Check that exactly ``size`` bytes are left, so trailing garbage is not ignored."""
        remaining = len(self._data) - self.offset
        if remaining != size:
            msg = f"payload has {remaining} bytes of data where {size} were expected"
            raise TelemetryError(msg)


def _check_header(magic: bytes, expected_magic: bytes, version: int) -> None:
    if magic != expected_magic:
        msg = f"payload does not start with {expected_magic!r}"
        raise TelemetryError(msg)
    if version != _VERSION:
        msg = f"unsupported binary version {version}"
        raise TelemetryError(msg)


def _parse_flags(raw_flags: int) -> ChannelFlags:
    # Unknown bits may mean a newer format, which this decoder would read wrongly.
    # The mask is a plain int, because ~ on a flag enum keeps only the known bits.
    if raw_flags & ~int(_ALL_FLAGS):
        msg = f"unknown channel flags {raw_flags:#04x}"
        raise TelemetryError(msg)
    return ChannelFlags(raw_flags)


def _read_names(reader: _Reader, id_length: int, joint_count: int) -> tuple[str, tuple[str, ...]]:
    """Read and check the device identifier and the joint names."""
    if joint_count > MAX_JOINTS:
        msg = f"joint count must be at most {MAX_JOINTS}, got {joint_count}"
        raise TelemetryError(msg)
    device_id = _identifier(reader.take(id_length), "device_id")
    names = tuple(
        _identifier(reader.take(reader.unpack(_NAME_LENGTH)[0]), "joint name")
        for _ in range(joint_count)
    )
    if len(set(names)) != len(names):
        msg = "joint names must be unique"
        raise TelemetryError(msg)
    return device_id, names


def _identifier(raw: bytes, what: str) -> str:
    # fullmatch, unlike match, also refuses a trailing newline before the end anchor.
    text = raw.decode("ascii", errors="replace")
    if _IDENTIFIER.fullmatch(text) is None:
        msg = f"{what} is not a valid identifier"
        raise TelemetryError(msg)
    return text


def _encode_names(names: Sequence[str]) -> bytes:
    parts = []
    for name in names:
        encoded = name.encode("ascii")
        parts.append(_NAME_LENGTH.pack(len(encoded)) + encoded)
    return b"".join(parts)


@functools.cache
def _sample_dtype(flags: ChannelFlags, joint_count: int) -> np.dtype[np.void]:
    """Return the packed record layout of one binary sample.

    Building a dtype is slow compared to decoding a small packet, and there are only a few hundred
    possible layouts, so each one is built once and reused.
    """
    fields: list[tuple[str, str] | tuple[str, str, tuple[int]]] = [
        ("sequence", "<u4"),
        ("timestamp", "<f8"),
    ]
    if ChannelFlags.ACCELERATION in flags:
        fields.append(("acceleration", "<f8", (VECTOR3_SIZE,)))
    if ChannelFlags.ANGULAR_VELOCITY in flags:
        fields.append(("angular_velocity", "<f8", (VECTOR3_SIZE,)))
    if ChannelFlags.ORIENTATION in flags:
        fields.append(("orientation", "<f8", (QUATERNION_SIZE,)))
    if joint_count:
        fields.append(("joint_angles", "<f8", (joint_count,)))
    return np.dtype(fields)


def _optional_field(records: np.ndarray[Any, np.dtype[np.void]], name: str) -> FloatArray | None:
    """Return a finite, read-only view of one record field, or ``None`` if it is absent."""
    names = records.dtype.names or ()
    if name not in names:
        return None
    values: FloatArray = records[name]
    # The array method skips the dispatch layer of np.all, which matters on one-sample packets.
    if not np.isfinite(values).all():
        msg = f"{name} values must be finite"
        raise TelemetryError(msg)
    return read_only(values)
