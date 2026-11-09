"""Run the full async runtime between two UDP sockets and watch for STALE frames.

This shows a production setup: a device sends binary packets to a UDP port, ``PipelineRuntime``
filters them, and a controller receives binary pose frames on another port. When the device goes
silent, the watchdog sends ``STALE`` frames so the controller knows to enter a safe state.

Everything runs on the loopback interface in one process, so no hardware is needed. In a real
deployment the device and the controller are separate programs that use the same two ports.

Run it with::

    python examples/udp_runtime.py
"""

from __future__ import annotations

import asyncio
import math
from collections import Counter

import numpy as np

from biokinex.pipeline import PipelineRuntime
from biokinex.schemas import FrameStatus, PipelineConfig, PoseFrame, SampleBatch
from biokinex.transport import BinaryCodec, UdpSink, UdpSource

RATE_HZ = 500.0
SAMPLES_PER_PACKET = 10
PACKET_COUNT = 25
SILENCE_S = 0.2
TURN_RATE = 0.5  # rad/s about the vertical axis.

CONFIG = PipelineConfig.model_validate(
    {
        "sample_rate_hz": RATE_HZ,
        "watchdog_timeout_s": 0.05,
        "acceleration_filters": [{"kind": "low_pass", "cutoff_hz": 25.0}],
        "angular_velocity_filters": [{"kind": "one_euro", "beta": 0.05}],
        "fusion": {"enabled": True},
    }
)


def simulated_batch(index: int, rng: np.random.Generator) -> SampleBatch:
    """Return one packet worth of IMU samples from a device turning slowly."""
    count = SAMPLES_PER_PACKET
    sequences = np.arange(index * count, (index + 1) * count, dtype=np.int64)
    return SampleBatch(
        device_id="hip-imu-01",
        timestamps=sequences / RATE_HZ,
        sequences=sequences,
        acceleration=np.array([0.0, 0.0, 9.80665]) + rng.normal(0.0, 0.05, (count, 3)),
        angular_velocity=np.array([0.0, 0.0, TURN_RATE]) + rng.normal(0.0, 0.02, (count, 3)),
    )


async def run_device(device: UdpSink, codec: BinaryCodec) -> None:
    """Send packets at the real sensor rate, then go silent."""
    rng = np.random.default_rng(seed=2)
    for index in range(PACKET_COUNT):
        await device.send(codec.encode_batch(simulated_batch(index, rng)))
        await asyncio.sleep(SAMPLES_PER_PACKET / RATE_HZ)
    # The device stops sending, as if its cable was unplugged.
    await asyncio.sleep(SILENCE_S)


async def run_controller(source: UdpSource, codec: BinaryCodec, frames: list[PoseFrame]) -> None:
    """Collect every pose frame until the socket is closed."""
    while (payload := await source.receive()) is not None:
        frame = codec.decode_frame(payload)
        if frame.status is FrameStatus.STALE and (
            not frames or frames[-1].status is not frame.status
        ):
            print(
                f"controller: device went STALE after sequence {frame.sequence}; holding safe state"
            )
        frames.append(frame)


async def main() -> None:
    codec = BinaryCodec()

    # Port 0 asks the OS for a free port; real deployments pass fixed ports instead.
    controller = await UdpSource.open()
    ingress = await UdpSource.open()
    egress = await UdpSink.open(*controller.address)
    runtime = PipelineRuntime(CONFIG, ingress, egress, codec=codec)
    device = await UdpSink.open(*ingress.address)

    frames: list[PoseFrame] = []
    runtime_task = asyncio.create_task(runtime.run())
    controller_task = asyncio.create_task(run_controller(controller, codec, frames))
    try:
        await run_device(device, codec)
    finally:
        # Stopping closes the ingress socket; run() then returns the final statistics.
        await runtime.stop()
        stats = await runtime_task
        for transport in (device, egress, controller):
            await transport.close()
        await controller_task

    counts = Counter(frame.status.value for frame in frames)
    last_ok = next(f for f in reversed(frames) if f.status is not FrameStatus.STALE)
    assert last_ok.euler is not None
    print()
    print(f"frames received:  {dict(counts)}")
    print(f"last yaw:         {math.degrees(last_ok.euler[2]):.2f} degrees")
    print(f"packets handled:  {stats.packets}")
    latency = stats.latency
    # Latency covers decode, processing, and encode of one whole packet of 10 samples.
    print(
        f"packet latency:   p50 {latency.p50 * 1e6:.0f} us, p99 {latency.p99 * 1e6:.0f} us "
        f"per {SAMPLES_PER_PACKET}-sample packet"
    )


if __name__ == "__main__":
    asyncio.run(main())
