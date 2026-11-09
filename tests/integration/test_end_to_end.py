"""End-to-end tests: a simulated device, the full runtime, and a consumer, over real transports."""

from __future__ import annotations

import asyncio
import math
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from biokinex.core import quaternion as quat
from biokinex.pipeline.runtime import PipelineRuntime
from biokinex.schemas.batch import SampleBatch
from biokinex.schemas.config import PipelineConfig
from biokinex.schemas.pose import FrameStatus, PoseFrame
from biokinex.transport.codecs import BinaryCodec, JsonCodec
from biokinex.transport.memory import MemorySink
from biokinex.transport.replay import ReplayRecorder, ReplaySource
from biokinex.transport.udp import UdpSink, UdpSource

pytestmark = pytest.mark.integration

RATE_HZ = 500.0
DT = 1.0 / RATE_HZ
TIMEOUT_S = 5.0
CONFIG = PipelineConfig.model_validate(
    {
        "sample_rate_hz": RATE_HZ,
        "watchdog_timeout_s": 0.05,
        "acceleration_filters": [
            {"kind": "median", "window": 3},
            {"kind": "low_pass", "cutoff_hz": 30},
        ],
        "angular_velocity_filters": [{"kind": "one_euro", "beta": 0.05}],
        # A slew limiter has no lag below its limit, so the fused yaw can be checked exactly.
        "orientation_filters": [{"kind": "slew_rate", "max_rate": 5.0}],
        "fusion": {"enabled": True, "gain": 1.0},
        "gap": {"policy": "linear", "max_fill_samples": 5},
    }
)


def imu_batch(start: int, count: int, seed: int = 0) -> SampleBatch:
    """Simulate a device turning slowly about z, with sensor noise."""
    rng = np.random.default_rng(seed + start)
    seq = np.arange(start, start + count, dtype=np.int64)
    gyro = np.tile([0.0, 0.0, 0.5], (count, 1)) + rng.normal(0.0, 0.02, (count, 3))
    accel = np.tile([0.0, 0.0, 9.80665], (count, 1)) + rng.normal(0.0, 0.05, (count, 3))
    return SampleBatch(
        device_id="imu-0",
        timestamps=seq.astype(np.float64) * DT,
        sequences=seq,
        acceleration=accel,
        angular_velocity=gyro,
    )


@dataclass
class UdpRig:
    """A device socket, the runtime between two UDP ports, and a consumer socket."""

    device: UdpSink
    consumer: UdpSource
    runtime: PipelineRuntime
    task: asyncio.Task[object]


@pytest.fixture
async def udp_rig() -> AsyncIterator[UdpRig]:
    codec = BinaryCodec()
    consumer = await UdpSource.open()
    ingress = await UdpSource.open()
    egress = await UdpSink.open(*consumer.address)
    runtime = PipelineRuntime(CONFIG, ingress, egress, codec=codec)
    task: asyncio.Task[object] = asyncio.create_task(runtime.run())
    device = await UdpSink.open(*ingress.address)
    yield UdpRig(device, consumer, runtime, task)
    await runtime.stop()
    await asyncio.wait_for(task, TIMEOUT_S)
    for transport in (device, egress, consumer):
        await transport.close()


async def receive_frames(source: UdpSource, count: int) -> list[PoseFrame]:
    codec = BinaryCodec()
    frames = []
    for _ in range(count):
        payload = await asyncio.wait_for(source.receive(), TIMEOUT_S)
        assert payload is not None
        frames.append(codec.decode_frame(payload))
    return frames


async def test_udp_binary_stream_is_filtered_and_fused(udp_rig: UdpRig) -> None:
    codec = BinaryCodec()
    for start in range(0, 200, 10):
        await udp_rig.device.send(codec.encode_batch(imu_batch(start, 10)))
        # A short pause keeps the loopback socket buffers from overflowing.
        await asyncio.sleep(0.001)
    frames = await receive_frames(udp_rig.consumer, 200)
    assert [f.sequence for f in frames] == list(range(200))
    assert {f.status for f in frames} == {FrameStatus.OK}
    # The device turned at 0.5 rad/s for 0.4 s, so fused yaw should be close to 0.2 rad.
    last = frames[-1]
    assert last.orientation is not None
    yaw = quat.to_euler_zyx(np.array(last.orientation))[2]
    assert yaw == pytest.approx(0.2, abs=0.02)
    assert last.acceleration is not None
    assert last.acceleration[2] == pytest.approx(9.80665, abs=0.1)


async def test_silence_produces_stale_frames_then_recovers(udp_rig: UdpRig) -> None:
    codec = BinaryCodec()
    await udp_rig.device.send(codec.encode_batch(imu_batch(0, 5)))
    await receive_frames(udp_rig.consumer, 5)
    # Silence for more than the 50 ms timeout; the watchdog must report within 1.25 timeouts.
    stale = await receive_frames(udp_rig.consumer, 1)
    assert stale[0].status is FrameStatus.STALE
    assert stale[0].sequence == 4

    await udp_rig.device.send(codec.encode_batch(imu_batch(5, 5)))
    fresh: list[PoseFrame] = []
    while len(fresh) < 5:
        frame = (await receive_frames(udp_rig.consumer, 1))[0]
        if frame.status is not FrameStatus.STALE:
            fresh.append(frame)
    assert [f.sequence for f in fresh] == [5, 6, 7, 8, 9]
    assert fresh[0].status is FrameStatus.OK


async def test_dropped_packets_become_degraded_frames(udp_rig: UdpRig) -> None:
    codec = BinaryCodec()
    for start in (0, 10, 30):
        await udp_rig.device.send(codec.encode_batch(imu_batch(start, 10)))
        await asyncio.sleep(0.001)
    frames = [
        f for f in await receive_frames(udp_rig.consumer, 30) if f.status is not FrameStatus.STALE
    ]
    by_sequence = {f.sequence: f.status for f in frames}
    # The 10-sample gap is longer than max_fill_samples, so the filters restart.
    assert by_sequence[30] is FrameStatus.DEGRADED
    assert by_sequence[31] is FrameStatus.OK
    assert all(by_sequence[s] is FrameStatus.OK for s in range(1, 20))


async def test_garbage_datagrams_do_not_break_the_stream(udp_rig: UdpRig) -> None:
    codec = BinaryCodec()
    await udp_rig.device.send(b"\x00garbage")
    await udp_rig.device.send(codec.encode_batch(imu_batch(0, 3)))
    frames = await receive_frames(udp_rig.consumer, 3)
    assert [f.sequence for f in frames] == [0, 1, 2]
    assert udp_rig.runtime.stats().decode_errors == 1


async def test_replay_through_runtime_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    json_codec = JsonCodec()
    with ReplayRecorder(path) as recorder:
        for start in range(0, 300, 5):
            if start % 45 == 15:
                continue
            batch = imu_batch(start, 5, seed=7)
            recorder.write(_as_json_packet(batch))
    outputs = []
    for _ in range(2):
        sink = MemorySink(maxsize=1000)
        async with ReplaySource(path) as source:
            stats = await PipelineRuntime(CONFIG, source, sink).run()
        assert stats.decode_errors == 0
        outputs.append(sink.drain())
    assert len(outputs[0]) == stats.frames_sent
    assert outputs[0] == outputs[1]
    frames = [json_codec.decode_frame(p) for p in outputs[0]]
    assert all(math.isfinite(v) for f in frames for v in (f.orientation or ()))


def _as_json_packet(batch: SampleBatch) -> bytes:
    """Turn a simulated batch into the JSON wire format used by recordings."""
    import json  # noqa: PLC0415

    assert batch.acceleration is not None
    assert batch.angular_velocity is not None
    samples = [
        {
            "sequence": int(seq),
            "timestamp": float(t),
            "acceleration": a.tolist(),
            "angular_velocity": g.tolist(),
        }
        for seq, t, a, g in zip(
            batch.sequences,
            batch.timestamps,
            batch.acceleration,
            batch.angular_velocity,
            strict=True,
        )
    ]
    return json.dumps({"device_id": batch.device_id, "samples": samples}).encode()
