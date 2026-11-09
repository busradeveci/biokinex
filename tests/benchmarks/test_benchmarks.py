"""Performance benchmarks for the hot path.

Benchmarks are disabled by default, so the normal test run executes each one once as a smoke
test. Measure them with::

    pytest tests/benchmarks --benchmark-enable --no-cov

Each benchmark also has a generous upper bound on the mean time. The bounds catch large
regressions on slow CI machines without failing on normal timing noise.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from biokinex.core import quaternion as quat
from biokinex.dsp import GapFiller, OneEuroFilter, OrientationOneEuroFilter
from biokinex.observability import LatencyHistogram
from biokinex.pipeline.processor import PipelineProcessor
from biokinex.pipeline.runtime import PipelineRuntime
from biokinex.schemas.batch import SampleBatch
from biokinex.schemas.config import GapPolicy, PipelineConfig
from biokinex.transport.codecs import BinaryCodec, JsonCodec
from biokinex.transport.memory import MemorySink, MemorySource

RATE_HZ = 500.0
DT = 1.0 / RATE_HZ
ROUNDS = 2000
FULL_CONFIG = PipelineConfig.model_validate(
    {
        "sample_rate_hz": RATE_HZ,
        "acceleration_filters": [
            {"kind": "median", "window": 3},
            {"kind": "low_pass", "cutoff_hz": 25},
        ],
        "angular_velocity_filters": [
            {"kind": "notch", "center_hz": 50, "q": 4},
            {"kind": "one_euro"},
        ],
        "orientation_filters": [
            {"kind": "one_euro", "beta": 0.05},
            {"kind": "slew_rate", "max_rate": 8},
        ],
        "fusion": {"enabled": True},
        "gap": {"policy": "linear"},
    }
)


def imu_batch(start: int, count: int) -> SampleBatch:
    """Build a noisy IMU batch with gyro and accelerometer readings."""
    rng = np.random.default_rng(start)
    seq = np.arange(start, start + count, dtype=np.int64)
    return SampleBatch(
        device_id="imu-0",
        timestamps=seq.astype(np.float64) * DT,
        sequences=seq,
        acceleration=np.array([0.0, 0.0, 9.80665]) + rng.normal(0.0, 0.1, (count, 3)),
        angular_velocity=rng.normal(0.0, 0.5, (count, 3)),
    )


def json_packet(batch: SampleBatch) -> bytes:
    """Encode a batch in the JSON wire format."""
    assert batch.acceleration is not None
    assert batch.angular_velocity is not None
    samples = [
        {
            "sequence": int(s),
            "timestamp": float(t),
            "acceleration": a.tolist(),
            "angular_velocity": g.tolist(),
        }
        for s, t, a, g in zip(
            batch.sequences,
            batch.timestamps,
            batch.acceleration,
            batch.angular_velocity,
            strict=True,
        )
    ]
    return json.dumps({"device_id": batch.device_id, "samples": samples}).encode()


def check_mean(benchmark: BenchmarkFixture, limit_s: float) -> None:
    """Fail if the mean time is far above the expected range; skipped when timing is off.

    The p99 is saved in ``extra_info``, because the latency target is a tail value and
    pytest-benchmark does not report it. ``--benchmark-json`` writes it out.
    """
    if not benchmark.disabled:
        assert benchmark.stats is not None
        data = benchmark.stats.stats.data
        benchmark.extra_info["p99_us"] = round(float(np.percentile(data, 99)) * 1e6, 1)
        mean = benchmark.stats.stats.mean
        assert mean < limit_s, f"mean {mean * 1e6:.1f} us is above the {limit_s * 1e6:.0f} us bound"


def run_stream(benchmark: BenchmarkFixture, step: Callable[[], Any]) -> None:
    """Run a stateful step many times; each call must see new input, so rounds are explicit."""
    # pytest-benchmark does not annotate pedantic(), so it is typed here for strict mypy.
    pedantic: Callable[..., object] = benchmark.pedantic
    pedantic(step, rounds=ROUNDS, iterations=1, warmup_rounds=100)


def times() -> Iterator[float]:
    return (i * DT for i in itertools.count())


class TestFilters:
    def test_one_euro_vector(self, benchmark: BenchmarkFixture) -> None:
        one_euro = OneEuroFilter(beta=0.1)
        clock = times()
        sample = np.array([0.1, -0.2, 0.3])
        benchmark(lambda: one_euro.step(sample, next(clock)))
        check_mean(benchmark, 50e-6)

    def test_one_euro_orientation(self, benchmark: BenchmarkFixture) -> None:
        one_euro = OrientationOneEuroFilter(beta=0.1)
        clock = times()
        sample = quat.from_axis_angle(np.array([0.0, 0.0, 1.0]), np.array([0.3]))[0]
        benchmark(lambda: one_euro.step(sample, next(clock)))
        check_mean(benchmark, 50e-6)

    def test_gap_filler_one_missing_sample(self, benchmark: BenchmarkFixture) -> None:
        filler = GapFiller(GapPolicy.LINEAR)
        batch = imu_batch(0, 1)
        filler.process(batch, [0], [True])
        result = benchmark(filler.process, batch, [1], [False])
        assert len(result.batch) == 2
        check_mean(benchmark, 500e-6)

    def test_gap_filler_no_gap(self, benchmark: BenchmarkFixture) -> None:
        filler = GapFiller(GapPolicy.LINEAR)
        batch = imu_batch(0, 1)
        result = benchmark(filler.process, batch, [0], [False])
        assert result.batch is batch
        check_mean(benchmark, 100e-6)


class TestCodecs:
    @pytest.mark.parametrize("count", [1, 10])
    def test_binary_decode(self, benchmark: BenchmarkFixture, count: int) -> None:
        payload = BinaryCodec().encode_batch(imu_batch(0, count))
        batch = benchmark(BinaryCodec().decode_packet, payload)
        assert len(batch) == count
        check_mean(benchmark, 200e-6)

    @pytest.mark.parametrize("count", [1, 10])
    def test_json_decode(self, benchmark: BenchmarkFixture, count: int) -> None:
        payload = json_packet(imu_batch(0, count))
        batch = benchmark(JsonCodec().decode_packet, payload)
        assert len(batch) == count
        check_mean(benchmark, 500e-6)

    @pytest.mark.parametrize("codec", [BinaryCodec(), JsonCodec()], ids=["binary", "json"])
    def test_frame_encode(
        self, benchmark: BenchmarkFixture, codec: BinaryCodec | JsonCodec
    ) -> None:
        frame = PipelineProcessor(FULL_CONFIG).process(imu_batch(0, 1)).frames[0]
        payload = benchmark(codec.encode_frame, frame)
        assert codec.decode_frame(payload).sequence == 0
        check_mean(benchmark, 200e-6)


class TestPipeline:
    @pytest.mark.parametrize("count", [1, 10])
    def test_processor_full_config(self, benchmark: BenchmarkFixture, count: int) -> None:
        processor = PipelineProcessor(FULL_CONFIG)
        batches = iter([imu_batch(i * count, count) for i in range(ROUNDS + 101)])
        run_stream(benchmark, lambda: processor.process(next(batches)))
        assert processor.stats().frames > 0
        # The bound is per batch; the README reports the per-sample cost.
        check_mean(benchmark, count * 1e-3)

    def test_runtime_binary_one_sample(self, benchmark: BenchmarkFixture) -> None:
        """Measure one packet from bytes in to frame bytes out, as the runtime handles it."""
        codec = BinaryCodec()
        sink = MemorySink(maxsize=4)
        runtime = PipelineRuntime(FULL_CONFIG, MemorySource(), sink, codec=codec)
        payloads = iter([codec.encode_batch(imu_batch(i, 1)) for i in range(ROUNDS + 101)])

        async def handle() -> None:
            await runtime.handle(next(payloads))

        loop_step = _sync(handle)
        run_stream(benchmark, loop_step)
        assert runtime.stats().frames_sent > 0
        check_mean(benchmark, 2e-3)


def test_latency_histogram_record(benchmark: BenchmarkFixture) -> None:
    histogram = LatencyHistogram()
    benchmark(histogram.record, 150e-6)
    check_mean(benchmark, 20e-6)


def _sync(step: Callable[[], Any]) -> Callable[[], None]:
    """Drive a coroutine that never suspends, without the cost of an event loop per call."""

    def run() -> None:
        coroutine = step()
        try:
            coroutine.send(None)
        except StopIteration:
            return
        coroutine.close()
        msg = "the benchmarked coroutine suspended"
        raise AssertionError(msg)

    return run
