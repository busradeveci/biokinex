"""The synchronous processing core: telemetry batches in, stabilized pose frames out.

One processor serves one device. For each batch it runs these stages in order:

1. :class:`~biokinex.validation.TelemetryValidator` rejects implausible or out-of-order samples.
2. :class:`~biokinex.dsp.GapFiller` fills short gaps and turns long ones into restarts.
3. The filter chains smooth each channel, and the complementary filter can estimate orientation.
4. Each real sample becomes a validated :class:`~biokinex.schemas.PoseFrame`, and its values are
   stored in a fixed-size :class:`~biokinex.pipeline.history.FrameHistory`.

The processor does no I/O and never reads the wall clock, so the same input always gives the same
output. The async runtime and the watchdog are built around it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from biokinex.core import scalar_quaternion as sq
from biokinex.core.types import FloatArray, Quaternion
from biokinex.dsp.base import FilterChain
from biokinex.dsp.complementary import ComplementaryFilter
from biokinex.dsp.gap import GapFiller, GapFillResult
from biokinex.pipeline.factory import build_chain, build_orientation_chain
from biokinex.pipeline.history import FrameHistory, HistoryLayout
from biokinex.schemas.batch import SampleBatch
from biokinex.schemas.config import PipelineConfig
from biokinex.schemas.pose import FrameStatus, PoseFrame
from biokinex.schemas.telemetry import TelemetryPacket
from biokinex.validation.validator import TelemetryValidator, ValidationResult, ValidatorStats

__all__ = ["PipelineProcessor", "ProcessResult", "ProcessorStats"]


@dataclass(frozen=True, slots=True, eq=False)
class ProcessResult:
    """The outcome of processing one batch.

    Attributes:
        frames: One frame per accepted real sample, in time order. Synthetic samples are used to
            keep the filters at a steady rate, but they never become frames.
        validation: The validator's report on the input batch.
        synthetic_count: Number of samples invented to fill gaps.
    """

    frames: tuple[PoseFrame, ...]
    validation: ValidationResult
    synthetic_count: int


@dataclass(frozen=True, slots=True)
class ProcessorStats:
    """Totals since the processor was created."""

    validation: ValidatorStats
    frames: int
    degraded_frames: int
    synthetic_samples: int
    filter_restarts: int


@dataclass(frozen=True, slots=True, eq=False)
class _FilteredRow:
    """Filtered values of one sample; ``None`` marks a channel the sample does not have."""

    orientation: Quaternion | None
    angular_velocity: FloatArray | None
    acceleration: FloatArray | None
    joint_angles: FloatArray | None


class PipelineProcessor:
    """Turns telemetry from one device into stabilized pose frames.

    Args:
        config: Validated pipeline configuration.
    """

    __slots__ = (
        "_acceleration_chain",
        "_angular_velocity_chain",
        "_config",
        "_degraded_total",
        "_frame_total",
        "_fusion",
        "_gap_filler",
        "_has_output",
        "_history",
        "_joint_chain",
        "_layout",
        "_orientation_chain",
        "_previous_orientation",
        "_restart_total",
        "_synthetic_total",
        "_validator",
    )

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        sequence = config.sequence
        self._validator = TelemetryValidator(
            config.limits,
            reorder_window=sequence.reorder_window,
            max_forward_jump=sequence.max_forward_jump,
            max_consecutive_time_errors=sequence.max_consecutive_time_errors,
        )
        self._gap_filler = GapFiller.from_config(config.gap)
        rate_hz = config.sample_rate_hz
        self._acceleration_chain = build_chain(config.acceleration_filters, rate_hz)
        self._angular_velocity_chain = build_chain(config.angular_velocity_filters, rate_hz)
        self._joint_chain = build_chain(config.joint_angle_filters, rate_hz)
        self._orientation_chain = build_orientation_chain(config.orientation_filters)
        self._fusion = (
            ComplementaryFilter.from_config(config.fusion) if config.fusion.enabled else None
        )
        self._history = FrameHistory(config.history_size)
        self._layout: HistoryLayout | None = None
        self._previous_orientation: sq.Quat | None = None
        self._has_output = False
        self._frame_total = 0
        self._degraded_total = 0
        self._synthetic_total = 0
        self._restart_total = 0

    @property
    def config(self) -> PipelineConfig:
        """The configuration the processor was built from."""
        return self._config

    @property
    def history(self) -> FrameHistory:
        """Values of the newest output frames, for zero-copy access."""
        return self._history

    def stats(self) -> ProcessorStats:
        """Return totals since the processor was created."""
        return ProcessorStats(
            validation=self._validator.stats(),
            frames=self._frame_total,
            degraded_frames=self._degraded_total,
            synthetic_samples=self._synthetic_total,
            filter_restarts=self._restart_total,
        )

    def process_packet(self, packet: TelemetryPacket) -> ProcessResult:
        """Convert a parsed packet to SI arrays and process it."""
        return self.process(packet.to_batch())

    def process(self, batch: SampleBatch) -> ProcessResult:
        """Validate, fill, and filter one batch, and return its frames.

        Raises:
            TelemetryError: If ``batch`` comes from a different device than earlier batches.
        """
        validation = self._validator.validate(batch)
        if validation.batch is None:
            return ProcessResult(frames=(), validation=validation, synthetic_count=0)
        filled = self._gap_filler.process(
            validation.batch, validation.missing_before, validation.restarts
        )
        frames = self._run(filled)
        synthetic_count = int(np.count_nonzero(filled.synthetic))
        self._synthetic_total += synthetic_count
        return ProcessResult(frames=frames, validation=validation, synthetic_count=synthetic_count)

    def reset(self) -> None:
        """Forget the device and all filter state, but keep the totals."""
        self._validator.reset()
        self._gap_filler.reset()
        self._reset_filters()
        self._history.clear()
        self._layout = None
        self._previous_orientation = None
        self._has_output = False

    def _run(self, filled: GapFillResult) -> tuple[PoseFrame, ...]:
        """Filter every sample of ``filled`` and build frames for the real ones."""
        batch = filled.batch
        layout = self._layout_of(batch)
        # Filters fix their shape on the first sample, so new channels need fresh filters.
        layout_changed = layout != self._layout
        self._layout = layout

        restarts = filled.restarts.tolist()
        synthetic = filled.synthetic.tolist()
        timestamps = batch.timestamps.tolist()
        sequences = batch.sequences.tolist()
        joint_angles = batch.joint_angles if layout.joint_names else None
        frames: list[PoseFrame] = []
        degraded = False

        for index, timestamp in enumerate(timestamps):
            if restarts[index] or (index == 0 and layout_changed):
                self._reset_filters()
                self._restart_total += 1
                # A restart after earlier output means data was lost or the device restarted.
                degraded = degraded or self._has_output
            row = self._filter_row(batch, joint_angles, index, timestamp)
            if synthetic[index]:
                # The next real frame depends on invented data, so it is reported as degraded.
                degraded = True
                continue
            frames.append(
                self._emit(batch, layout, sequences[index], timestamp, row=row, degraded=degraded)
            )
            degraded = False
        return tuple(frames)

    def _layout_of(self, batch: SampleBatch) -> HistoryLayout:
        """Describe which channels the frames built from ``batch`` will have."""
        fused = self._fusion is not None and batch.angular_velocity is not None
        has_joints = batch.joint_angles is not None and batch.joint_angles.shape[1] > 0
        return HistoryLayout(
            orientation=fused or batch.orientation is not None,
            angular_velocity=batch.angular_velocity is not None,
            acceleration=batch.acceleration is not None,
            joint_names=batch.joint_names if has_joints else (),
        )

    def _filter_row(
        self,
        batch: SampleBatch,
        joint_angles: FloatArray | None,
        index: int,
        timestamp: float,
    ) -> _FilteredRow:
        """Run every channel of one sample through its filters."""
        raw_gyro = None if batch.angular_velocity is None else batch.angular_velocity[index]
        raw_accel = None if batch.acceleration is None else batch.acceleration[index]
        return _FilteredRow(
            orientation=self._orientation_for(batch, index, raw_gyro, raw_accel, timestamp),
            angular_velocity=_step(self._angular_velocity_chain, raw_gyro, timestamp),
            acceleration=_step(self._acceleration_chain, raw_accel, timestamp),
            joint_angles=_step(
                self._joint_chain, None if joint_angles is None else joint_angles[index], timestamp
            ),
        )

    def _orientation_for(
        self,
        batch: SampleBatch,
        index: int,
        raw_gyro: FloatArray | None,
        raw_accel: FloatArray | None,
        timestamp: float,
    ) -> Quaternion | None:
        """Return the filtered orientation of one sample, or ``None`` if it has none."""
        if self._fusion is not None and raw_gyro is not None:
            # Fusion uses unfiltered readings, because the complementary filter already smooths
            # the accelerometer and extra lag on the gyroscope would delay the orientation.
            source = self._fusion.step(raw_gyro, raw_accel, timestamp)
        elif batch.orientation is not None:
            source = batch.orientation[index]
        else:
            return None
        w, x, y, z = sq.normalize(sq.from_array(self._orientation_chain.step(source, timestamp)))
        previous = self._previous_orientation
        # Keeping each output in the same hemisphere as the last one avoids q / -q jumps.
        if previous is not None and (
            previous[0] * w + previous[1] * x + previous[2] * y + previous[3] * z < 0.0
        ):
            w, x, y, z = -w, -x, -y, -z
        self._previous_orientation = (w, x, y, z)
        return sq.to_array(self._previous_orientation)

    def _emit(
        self,
        batch: SampleBatch,
        layout: HistoryLayout,
        sequence: int,
        timestamp: float,
        *,
        row: _FilteredRow,
        degraded: bool,
    ) -> PoseFrame:
        """Build and record the frame for one real sample."""
        status = FrameStatus.DEGRADED if degraded else FrameStatus.OK
        # Building the frame runs full validation, so a NaN or a broken quaternion from a bug in a
        # filter stops here instead of reaching an actuator.
        frame = PoseFrame.model_validate(
            {
                "device_id": batch.device_id,
                "sequence": sequence,
                "timestamp": timestamp,
                "status": status,
                "orientation": _as_tuple(row.orientation),
                "angular_velocity": _as_tuple(row.angular_velocity),
                "acceleration": _as_tuple(row.acceleration),
                "joint_names": layout.joint_names,
                "joint_angles": _as_tuple(row.joint_angles),
            }
        )
        self._history.append(
            layout,
            timestamp,
            orientation=row.orientation,
            angular_velocity=row.angular_velocity,
            acceleration=row.acceleration,
            joint_angles=row.joint_angles,
        )
        self._has_output = True
        self._frame_total += 1
        self._degraded_total += int(degraded)
        return frame

    def _reset_filters(self) -> None:
        """Reset every filter, so the next sample starts them fresh."""
        self._acceleration_chain.reset()
        self._angular_velocity_chain.reset()
        self._joint_chain.reset()
        self._orientation_chain.reset()
        if self._fusion is not None:
            self._fusion.reset()


def _step(chain: FilterChain, value: FloatArray | None, timestamp: float) -> FloatArray | None:
    """Run ``value`` through ``chain``, or return ``None`` for a missing channel."""
    return None if value is None else chain.step(value, timestamp)


def _as_tuple(values: FloatArray | None) -> tuple[float, ...] | None:
    """Convert an array to the plain tuple form used by the pose schema."""
    return None if values is None else tuple(values.tolist())
