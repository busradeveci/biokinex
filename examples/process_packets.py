"""Process JSON telemetry packets with the synchronous pipeline and read the pose frames.

This is the smallest complete use of BioKineX: load a configuration, feed packets to a
``PipelineProcessor``, and read the validated ``PoseFrame`` objects it returns. No sockets or
event loop are involved, so this is also the way to use BioKineX inside your own loop.

The script simulates a wrist IMU at 500 Hz that turns slowly about its vertical axis. To show the
safety features, the stream contains one lost packet and one gyroscope glitch.

Run it with::

    python examples/process_packets.py
"""

from __future__ import annotations

import math

import numpy as np

from biokinex.pipeline import PipelineProcessor
from biokinex.schemas import FrameStatus, PipelineConfig, TelemetryPacket
from biokinex.transport import JsonCodec

RATE_HZ = 500.0
SAMPLES_PER_PACKET = 5
PACKET_COUNT = 12
LOST_PACKET = 4
GLITCH_SEQUENCE = 42
TURN_RATE = 0.5  # rad/s about the vertical axis.

# Configuration is plain TOML, so it can live in a file next to your application.
CONFIG_TOML = """
sample_rate_hz = 500.0
watchdog_timeout_s = 0.05

[[acceleration_filters]]
kind = "median"     # Removes single-sample spikes.
window = 3

[[acceleration_filters]]
kind = "low_pass"   # Removes vibration above 25 Hz.
cutoff_hz = 25.0

[[angular_velocity_filters]]
kind = "one_euro"   # Smooths jitter at rest while keeping lag low during fast motion.
min_cutoff_hz = 1.0
beta = 0.05

[[orientation_filters]]
kind = "slew_rate"  # Caps how fast the output orientation may turn.
max_rate = 8.0

[fusion]
enabled = true      # Builds orientation from the gyroscope and the accelerometer.
gain = 1.0

[gap]
policy = "linear"   # Fills short gaps by interpolation and marks the result as degraded.
max_fill_samples = 10
"""


def simulated_packet(index: int, rng: np.random.Generator) -> bytes:
    """Return one JSON packet, as a device would send it over the network."""
    samples = []
    for offset in range(SAMPLES_PER_PACKET):
        sequence = index * SAMPLES_PER_PACKET + offset
        gyro = [*rng.normal(0.0, 0.02, 2), TURN_RATE + rng.normal(0.0, 0.02)]
        if sequence == GLITCH_SEQUENCE:
            # A value far above the sensor range, as a corrupted reading would produce.
            gyro = [0.0, 0.0, 500.0]
        samples.append(
            {
                "sequence": sequence,
                "timestamp": sequence / RATE_HZ,
                "acceleration": [*rng.normal(0.0, 0.05, 2), 9.80665 + rng.normal(0.0, 0.05)],
                "angular_velocity": gyro,
            }
        )
    packet = TelemetryPacket.model_validate({"device_id": "wrist-imu-01", "samples": samples})
    return JsonCodec().encode_packet(packet)


def main() -> None:
    config = PipelineConfig.from_toml(CONFIG_TOML)
    # One processor per device; it keeps the filter state between packets.
    processor = PipelineProcessor(config)
    codec = JsonCodec()
    rng = np.random.default_rng(seed=1)

    print(" seq  status     roll    pitch      yaw   (degrees)")
    for index in range(PACKET_COUNT):
        payload = simulated_packet(index, rng)
        if index == LOST_PACKET:
            continue  # This packet is lost on the network.

        # Decoding validates structure, types, and finite values before any processing.
        batch = codec.decode_packet(payload)
        result = processor.process(batch)

        for frame in result.frames:
            # Print every fifth frame, plus every frame that is not OK.
            if frame.sequence % 5 and frame.status is FrameStatus.OK:
                continue
            assert frame.euler is not None  # Fusion is on, so every frame has an orientation.
            roll, pitch, yaw = (math.degrees(angle) for angle in frame.euler)
            print(
                f"{frame.sequence:4d}  {frame.status.value:8s} "
                f"{roll:7.2f}  {pitch:7.2f}  {yaw:7.2f}"
            )

    stats = processor.stats()
    print()
    print(f"frames emitted:     {stats.frames}")
    print(f"degraded frames:    {stats.degraded_frames}")
    print(f"synthetic samples:  {stats.synthetic_samples}")
    rejected = {
        verdict.name: count
        for verdict, count in stats.validation.verdicts.items()
        if count and verdict.name != "ACCEPTED"
    }
    print(f"rejected samples:   {rejected}")

    # Frames serialize to JSON for downstream controllers; derived Euler angles are included.
    print()
    print("last frame as JSON:")
    print(codec.encode_frame(result.frames[-1]).decode())


if __name__ == "__main__":
    main()
