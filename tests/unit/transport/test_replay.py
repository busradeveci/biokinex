"""Tests for biokinex.transport.replay."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from biokinex.core.clock import ManualClock
from biokinex.core.errors import TransportError
from biokinex.pipeline.processor import PipelineProcessor
from biokinex.schemas.config import PipelineConfig
from biokinex.schemas.pose import PoseFrame
from biokinex.transport.base import TelemetrySource
from biokinex.transport.codecs import JsonCodec
from biokinex.transport.memory import MemorySource
from biokinex.transport.replay import (
    RecordingSource,
    ReplayRecorder,
    ReplaySource,
    _packet_time,
)


def packet(sequence: int, timestamp: float) -> bytes:
    sample = {
        "sequence": sequence,
        "timestamp": timestamp,
        "acceleration": [0.0, 0.1 * sequence, 9.8],
        "angular_velocity": [0.0, 0.0, 0.5],
    }
    return json.dumps({"device_id": "imu-0", "samples": [sample]}).encode()


async def read_all(source: TelemetrySource) -> list[bytes]:
    payloads = []
    while (payload := await source.receive()) is not None:
        payloads.append(payload)
    return payloads


class FakeSleep:
    """Records requested delays and moves a manual clock instead of sleeping."""

    def __init__(self, clock: ManualClock) -> None:
        self.clock = clock
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)
        self.clock.advance(delay)


@pytest.fixture
def recording(tmp_path: Path) -> Path:
    path = tmp_path / "session.jsonl"
    with ReplayRecorder(path) as recorder:
        for index in range(3):
            recorder.write(packet(index, 0.5 * index))
        assert recorder.written == 3
    return path


class TestRecorderAndReplay:
    async def test_replay_returns_recorded_payloads(self, recording: Path) -> None:
        source = ReplaySource(recording)
        assert isinstance(source, TelemetrySource)
        assert await read_all(source) == [packet(i, 0.5 * i) for i in range(3)]
        assert await source.receive() is None

    async def test_blank_lines_and_crlf_are_ignored(self, tmp_path: Path) -> None:
        path = tmp_path / "windows.jsonl"
        path.write_bytes(b"\r\n" + packet(0, 0.0) + b"\r\n   \n" + packet(1, 0.1))
        assert await read_all(ReplaySource(path)) == [packet(0, 0.0), packet(1, 0.1)]

    async def test_close_stops_replay(self, recording: Path) -> None:
        source = ReplaySource(recording)
        await source.close()
        await source.close()
        assert await source.receive() is None

    async def test_rejects_too_long_line(self, tmp_path: Path) -> None:
        path = tmp_path / "long.jsonl"
        path.write_bytes(b"x" * 20 + b"\n")
        async with ReplaySource(path, max_line_bytes=10) as source:
            with pytest.raises(TransportError, match="longer than 10 bytes"):
                await source.receive()

    async def test_line_at_the_limit_is_accepted(self, tmp_path: Path) -> None:
        path = tmp_path / "exact.jsonl"
        path.write_bytes(b"y" * 10 + b"\n")
        async with ReplaySource(path, max_line_bytes=10) as source:
            assert await source.receive() == b"y" * 10

    @pytest.mark.parametrize("speed", [0.0, -1.0, float("inf"), float("nan")])
    def test_rejects_invalid_speed(self, recording: Path, speed: float) -> None:
        with pytest.raises(ValueError, match="speed"):
            ReplaySource(recording, speed=speed)

    def test_rejects_invalid_line_limit(self, recording: Path) -> None:
        with pytest.raises(ValueError, match="max_line_bytes"):
            ReplaySource(recording, max_line_bytes=0)

    def test_recorder_refuses_to_overwrite(self, recording: Path) -> None:
        with pytest.raises(FileExistsError):
            ReplayRecorder(recording)

    @pytest.mark.parametrize("payload", [b"a\nb", b"a\rb", b"", b"   "])
    def test_recorder_rejects_unrecordable_payloads(self, tmp_path: Path, payload: bytes) -> None:
        with (
            ReplayRecorder(tmp_path / "r.jsonl") as recorder,
            pytest.raises(ValueError, match="single-line"),
        ):
            recorder.write(payload)

    def test_recorder_rejects_writes_after_close(self, tmp_path: Path) -> None:
        recorder = ReplayRecorder(tmp_path / "r.jsonl")
        recorder.close()
        recorder.close()
        with pytest.raises(TransportError, match="closed"):
            recorder.write(b"{}")


class TestPacing:
    async def test_real_time_and_faster_replay(self, recording: Path) -> None:
        for speed, expected in [(1.0, [0.5, 0.5]), (2.0, [0.25, 0.25])]:
            clock = ManualClock()
            sleep = FakeSleep(clock)
            source = ReplaySource(recording, speed=speed, clock=clock, sleep=sleep)
            assert len(await read_all(source)) == 3
            assert sleep.delays == pytest.approx(expected)

    async def test_no_wait_when_already_late(self, recording: Path) -> None:
        clock = ManualClock()
        sleep = FakeSleep(clock)
        async with ReplaySource(recording, speed=1.0, clock=clock, sleep=sleep) as source:
            await source.receive()
            clock.advance(10.0)
            await source.receive()
        assert sleep.delays == []

    async def test_unreadable_lines_are_delivered_without_waiting(self, tmp_path: Path) -> None:
        path = tmp_path / "mixed.jsonl"
        path.write_bytes(packet(0, 0.0) + b"\nnot json\n" + packet(1, 1.0) + b"\n")
        clock = ManualClock()
        sleep = FakeSleep(clock)
        payloads = await read_all(ReplaySource(path, speed=1.0, clock=clock, sleep=sleep))
        assert payloads[1] == b"not json"
        assert sleep.delays == pytest.approx([1.0])

    @pytest.mark.parametrize(
        ("payload", "expected"),
        [
            (packet(0, 2.5), 2.5),
            (b'{"samples": [{"timestamp": 3}]}', 3.0),
            (b"[1, 2]", None),
            (b"\xff\xfe", None),
            (b'{"samples": []}', None),
            (b'{"samples": [5]}', None),
            (b'{"samples": "x"}', None),
            (b'{"samples": [{"timestamp": true}]}', None),
            (b'{"samples": [{"timestamp": "1.0"}]}', None),
            (b'{"samples": [{"timestamp": Infinity}]}', None),
            (b'{"samples": [{"timestamp": 1' + b"0" * 400 + b"}]}", None),
        ],
    )
    def test_packet_time(self, payload: bytes, expected: float | None) -> None:
        assert _packet_time(payload) == expected


class TestRecordingSource:
    async def test_records_while_passing_through(self, tmp_path: Path) -> None:
        inner = MemorySource()
        for payload in (packet(0, 0.0), b"two\nlines", packet(1, 0.1)):
            inner.push(payload)
        await inner.close()
        path = tmp_path / "live.jsonl"
        source = RecordingSource(inner, ReplayRecorder(path))
        delivered = await read_all(source)
        await source.close()
        assert delivered == [packet(0, 0.0), b"two\nlines", packet(1, 0.1)]
        assert source.skipped == 1
        assert await read_all(ReplaySource(path)) == [packet(0, 0.0), packet(1, 0.1)]


async def test_replay_through_pipeline_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "pipeline.jsonl"
    with ReplayRecorder(path) as recorder:
        for index in range(50):
            if index % 7 != 3:
                recorder.write(packet(index, 0.002 * index))
    config = PipelineConfig.model_validate(
        {
            "sample_rate_hz": 500,
            "acceleration_filters": [{"kind": "low_pass", "cutoff_hz": 20}],
            "orientation_filters": [{"kind": "one_euro", "beta": 0.1}],
        }
    )
    codec = JsonCodec()
    runs = []
    for _ in range(2):
        processor = PipelineProcessor(config)
        frames: list[PoseFrame] = []
        source = ReplaySource(path)
        while (payload := await source.receive()) is not None:
            frames.extend(processor.process(codec.decode_packet(payload)).frames)
        runs.append([codec.encode_frame(frame) for frame in frames])
    assert len(runs[0]) == 43
    assert runs[0] == runs[1]
