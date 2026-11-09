"""Tests for biokinex.pipeline.runtime, using memory transports and a manual clock."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

import pytest

from biokinex.core.clock import ManualClock
from biokinex.pipeline.processor import PipelineProcessor
from biokinex.pipeline.runtime import PipelineRuntime
from biokinex.schemas.batch import SampleBatch
from biokinex.schemas.config import PipelineConfig
from biokinex.schemas.pose import FrameStatus, PoseFrame
from biokinex.transport.codecs import BinaryCodec, JsonCodec
from biokinex.transport.memory import MemorySink, MemorySource

CONFIG = PipelineConfig.model_validate(
    {
        "sample_rate_hz": 500,
        "watchdog_timeout_s": 0.1,
        "acceleration_filters": [{"kind": "low_pass", "cutoff_hz": 30}],
    }
)
CODEC = JsonCodec()


def packet(sequence: int, device_id: str = "imu-0", samples: int = 1) -> bytes:
    rows = [
        {
            "sequence": sequence + i,
            "timestamp": 0.002 * (sequence + i),
            "acceleration": [0.0, 0.01 * (sequence + i), 9.8],
            "angular_velocity": [0.0, 0.0, 0.2],
        }
        for i in range(samples)
    ]
    return json.dumps({"device_id": device_id, "samples": rows}).encode()


def frames_in(sink: MemorySink) -> list[PoseFrame]:
    return [CODEC.decode_frame(payload) for payload in sink.drain()]


async def wait_until(condition: Callable[[], bool]) -> None:
    async with asyncio.timeout(2.0):
        # The runtime has no event for its counters, so the test polls them.
        while not condition():  # noqa: ASYNC110
            await asyncio.sleep(0)


async def run_to_end(runtime: PipelineRuntime, source: MemorySource, *payloads: bytes) -> Any:
    for payload in payloads:
        source.push(payload)
    await source.close()
    return await runtime.run()


class TestStreaming:
    async def test_frames_match_direct_processing(self) -> None:
        source, sink = MemorySource(), MemorySink()
        runtime = PipelineRuntime(CONFIG, source, sink, clock=ManualClock())
        payloads = [packet(i * 2, samples=2) for i in range(10)]
        stats = await run_to_end(runtime, source, *payloads)

        direct = PipelineProcessor(CONFIG)
        expected = [f for p in payloads for f in direct.process(CODEC.decode_packet(p)).frames]
        assert frames_in(sink) == expected
        assert stats.packets == 10
        assert stats.frames_sent == 20
        assert stats.devices == ("imu-0",)
        assert stats.latency.count == 10

    async def test_binary_codec(self) -> None:
        source, sink = MemorySource(), MemorySink()
        codec = BinaryCodec()
        runtime = PipelineRuntime(CONFIG, source, sink, codec=codec)
        batch = JsonCodec().decode_packet(packet(0, samples=3))
        await run_to_end(runtime, source, codec.encode_batch(batch))
        frames = [codec.decode_frame(p) for p in sink.drain()]
        assert [f.sequence for f in frames] == [0, 1, 2]

    async def test_bad_payloads_are_counted_and_skipped(self) -> None:
        source, sink = MemorySource(), MemorySink()
        runtime = PipelineRuntime(CONFIG, source, sink)
        stats = await run_to_end(runtime, source, b"not json", packet(0), b"{}", packet(1))
        assert stats.decode_errors == 2
        assert [f.sequence for f in frames_in(sink)] == [0, 1]

    async def test_each_device_has_its_own_processor(self) -> None:
        source, sink = MemorySource(), MemorySink()
        runtime = PipelineRuntime(CONFIG, source, sink, max_devices=2)
        stats = await run_to_end(
            runtime,
            source,
            packet(0, "imu-a"),
            packet(100, "imu-b"),
            packet(1, "imu-a"),
            packet(0, "imu-c"),
        )
        assert stats.devices == ("imu-a", "imu-b")
        assert stats.rejected_devices == 1
        assert set(runtime.processors) == {"imu-a", "imu-b"}
        statuses = {(f.device_id, f.sequence): f.status for f in frames_in(sink)}
        assert statuses == {
            ("imu-a", 0): FrameStatus.OK,
            ("imu-b", 100): FrameStatus.OK,
            ("imu-a", 1): FrameStatus.OK,
        }

    async def test_processing_error_resets_only_that_device(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source, sink = MemorySource(), MemorySink()
        runtime = PipelineRuntime(CONFIG, source, sink)
        await runtime.handle(packet(0))
        processor = runtime.processors["imu-0"]
        original = PipelineProcessor.process

        def failing(self: PipelineProcessor, batch: SampleBatch) -> Any:
            if int(batch.sequences[0]) == 1:
                msg = "simulated bug"
                raise ValueError(msg)
            return original(self, batch)

        monkeypatch.setattr(PipelineProcessor, "process", failing)
        assert await runtime.handle(packet(1)) == ()
        frames = await runtime.handle(packet(2))
        assert runtime.stats().processing_errors == 1
        # After the reset, the next packet starts the device fresh.
        assert frames[0].status is FrameStatus.OK
        assert len(processor.history) == 1

    async def test_sink_errors_do_not_stop_processing(self) -> None:
        source, sink = MemorySource(), MemorySink()
        await sink.close()
        runtime = PipelineRuntime(CONFIG, source, sink)
        stats = await run_to_end(runtime, source, packet(0), packet(1))
        assert stats.send_errors == 2
        assert stats.frames_sent == 0
        assert len(runtime.processors["imu-0"].history) == 2


class TestWatchdog:
    async def test_silent_device_gets_stale_frames(self) -> None:
        clock = ManualClock()
        sink = MemorySink()
        runtime = PipelineRuntime(CONFIG, MemorySource(), sink, clock=clock)
        await runtime.handle(packet(0))
        sink.drain()
        assert await runtime.check_watchdogs() == ()
        clock.advance(0.1)
        stale = await runtime.check_watchdogs()
        assert [f.status for f in stale] == [FrameStatus.STALE]
        assert [f.status for f in frames_in(sink)] == [FrameStatus.STALE]
        assert runtime.stats().stale_frames == 1
        await runtime.handle(packet(1))
        assert frames_in(sink)[0].status is FrameStatus.OK

    async def test_watchdog_loop_runs_while_streaming(self) -> None:
        clock = ManualClock()
        sleeps: list[float] = []

        async def fake_sleep(delay: float) -> None:
            sleeps.append(delay)
            clock.advance(delay)
            await asyncio.sleep(0)

        source, sink = MemorySource(), MemorySink(maxsize=8)
        runtime = PipelineRuntime(CONFIG, source, sink, clock=clock, sleep=fake_sleep)
        task = asyncio.create_task(runtime.run())
        source.push(packet(0))
        await wait_until(lambda: runtime.stats().stale_frames >= 2)
        await runtime.stop()
        stats = await task
        assert all(delay == pytest.approx(0.025) for delay in sleeps)
        assert stats.frames_sent >= 3


class TestLifecycle:
    async def test_cannot_run_twice_at_once(self) -> None:
        source = MemorySource()
        runtime = PipelineRuntime(CONFIG, source, MemorySink())
        task = asyncio.create_task(runtime.run())
        await asyncio.sleep(0)
        with pytest.raises(RuntimeError, match="already running"):
            await runtime.run()
        await runtime.stop()
        await task

    async def test_cancel_stops_both_loops(self) -> None:
        runtime = PipelineRuntime(CONFIG, MemorySource(), MemorySink())
        task = asyncio.create_task(runtime.run())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert {t.get_name() for t in asyncio.all_tasks()} & {"biokinex-watchdog"} == set()

    def test_rejects_invalid_device_limit(self) -> None:
        with pytest.raises(ValueError, match="max_devices"):
            PipelineRuntime(CONFIG, MemorySource(), MemorySink(), max_devices=0)

    def test_processors_view_is_read_only(self) -> None:
        runtime = PipelineRuntime(CONFIG, MemorySource(), MemorySink())
        with pytest.raises(TypeError):
            runtime.processors["x"] = PipelineProcessor(CONFIG)  # type: ignore[index]
