"""Filling short gaps left by dropped or rejected samples.

Filters assume a continuous stream. When the validator reports a few missing samples, this stage
inserts synthetic samples so that filters see a steady rate. Long gaps are not filled, because
invented data over a long time could hide a real problem; they become restarts instead, and the
filters begin again from the next real sample.

Synthetic samples are always marked, so the pipeline can report frames built from them as
degraded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from biokinex.core.types import BoolArray, FloatArray, IntArray
from biokinex.dsp.rotation import apply_rotation_vector, relative_rotation_vector
from biokinex.schemas.batch import SampleBatch, read_only
from biokinex.schemas.config import GapConfig, GapPolicy
from biokinex.validation.sequence import SEQUENCE_MODULUS

__all__ = ["GapFillResult", "GapFiller"]

_CHANNELS = ("acceleration", "angular_velocity", "orientation", "joint_angles")


@dataclass(frozen=True, slots=True, eq=False)
class GapFillResult:
    """A batch with short gaps filled in.

    Attributes:
        batch: Real and synthetic samples in time order. When nothing was filled, this is the
            input batch itself, so no data is copied.
        synthetic: For each output sample, whether it was invented to fill a gap.
        restarts: For each output sample, whether filters must start fresh there.
    """

    batch: SampleBatch
    synthetic: BoolArray
    restarts: BoolArray


class GapFiller:
    """Fills gaps of up to ``max_fill_samples`` and turns longer gaps into restarts.

    It remembers the last real sample of the previous batch, so gaps between batches are filled
    as well as gaps inside one batch.

    Args:
        policy: ``HOLD`` repeats the last real value; ``LINEAR`` moves in a straight line to the
            next real value, using slerp for orientation.
        max_fill_samples: Longest gap that is filled. Zero turns every gap into a restart.
    """

    __slots__ = ("_last", "_max_fill_samples", "_policy")

    def __init__(self, policy: GapPolicy = GapPolicy.HOLD, max_fill_samples: int = 10) -> None:
        if max_fill_samples < 0:
            msg = f"max_fill_samples must be >= 0, got {max_fill_samples}"
            raise ValueError(msg)
        self._policy = policy
        self._max_fill_samples = max_fill_samples
        self._last: SampleBatch | None = None

    @classmethod
    def from_config(cls, config: GapConfig) -> GapFiller:
        """Build a gap filler from validated configuration."""
        return cls(config.policy, config.max_fill_samples)

    def process(
        self, batch: SampleBatch, missing_before: npt.ArrayLike, restarts: npt.ArrayLike
    ) -> GapFillResult:
        """Fill the gaps reported by the validator for ``batch``.

        Args:
            batch: Accepted samples, for example ``ValidationResult.batch``.
            missing_before: Samples missing before each accepted sample.
            restarts: Whether the stream restarts at each accepted sample.
        """
        missing = np.asarray(missing_before, dtype=np.int64)
        restart = np.array(restarts, dtype=np.bool_)
        n = len(batch)
        if missing.shape != (n,) or restart.shape != (n,):
            msg = f"missing_before and restarts must have shape ({n},)"
            raise ValueError(msg)

        fill_counts = np.zeros(n, dtype=np.int64)
        for index in np.flatnonzero((missing > 0) & ~restart).tolist():
            if self._can_fill(batch, index, int(missing[index])):
                fill_counts[index] = missing[index]
            else:
                restart[index] = True

        if not fill_counts.any():
            result = GapFillResult(
                batch=batch,
                synthetic=read_only(np.zeros(n, dtype=np.bool_)),
                restarts=read_only(restart),
            )
        else:
            result = self._build(batch, fill_counts, restart)
        self._last = batch.select(np.array([n - 1], dtype=np.int64))
        return result

    def reset(self) -> None:
        """Forget the previous batch, so a gap at the start of the next batch becomes a restart."""
        self._last = None

    def _can_fill(self, batch: SampleBatch, index: int, count: int) -> bool:
        """Return whether the gap before ``batch[index]`` is short and has a usable start."""
        if count > self._max_fill_samples:
            return False
        if index > 0:
            return True
        # A gap at the start of the batch needs the previous batch, with the same channels.
        previous = self._last
        if previous is None or previous.joint_names != batch.joint_names:
            return False
        return all(
            _shape_tail(getattr(previous, name)) == _shape_tail(getattr(batch, name))
            for name in _CHANNELS
        )

    def _build(
        self, batch: SampleBatch, fill_counts: IntArray, restarts: BoolArray
    ) -> GapFillResult:
        """Return a new batch with synthetic samples inserted before each gap end."""
        n = len(batch)
        total = n + int(fill_counts.sum())
        # Every real sample moves right by the number of samples inserted before it.
        real_positions = np.arange(n) + np.cumsum(fill_counts)
        synthetic = np.ones(total, dtype=np.bool_)
        synthetic[real_positions] = False
        out_restarts = np.zeros(total, dtype=np.bool_)
        out_restarts[real_positions] = restarts

        gaps = [
            (int(index), int(fill_counts[index]), self._row(batch, int(index) - 1))
            for index in np.flatnonzero(fill_counts)
        ]
        timestamps = self._fill_channel(
            batch.timestamps, real_positions, total, gaps, "timestamps", linear=True
        )
        sequences = np.empty(total, dtype=np.int64)
        sequences[real_positions] = batch.sequences
        for index, count, previous in gaps:
            start = real_positions[index] - count
            steps = np.arange(1, count + 1, dtype=np.int64)
            sequences[start : start + count] = (previous.sequences[0] + steps) % SEQUENCE_MODULUS

        linear = self._policy is GapPolicy.LINEAR
        channels: dict[str, FloatArray | None] = {}
        for name in _CHANNELS:
            values = getattr(batch, name)
            channels[name] = (
                None
                if values is None
                else self._fill_channel(values, real_positions, total, gaps, name, linear=linear)
            )

        filled = SampleBatch(
            device_id=batch.device_id,
            timestamps=timestamps,
            sequences=read_only(sequences),
            acceleration=channels["acceleration"],
            angular_velocity=channels["angular_velocity"],
            orientation=channels["orientation"],
            joint_names=batch.joint_names,
            joint_angles=channels["joint_angles"],
        )
        return GapFillResult(
            batch=filled, synthetic=read_only(synthetic), restarts=read_only(out_restarts)
        )

    def _row(self, batch: SampleBatch, index: int) -> SampleBatch:
        """Return the real sample before a gap; index ``-1`` means the previous batch."""
        if index >= 0:
            return batch.select(np.array([index], dtype=np.int64))
        assert self._last is not None  # Guaranteed by _can_fill.
        return self._last

    @staticmethod
    def _fill_channel(
        values: FloatArray,
        real_positions: IntArray,
        total: int,
        gaps: list[tuple[int, int, SampleBatch]],
        name: str,
        *,
        linear: bool,
    ) -> FloatArray:
        """Copy one channel into the output and fill every gap in it."""
        output = np.empty((total, *values.shape[1:]), dtype=np.float64)
        output[real_positions] = values
        for index, count, previous in gaps:
            start_value = getattr(previous, name)[0]
            end_value = values[index]
            fractions = np.arange(1, count + 1, dtype=np.float64) / (count + 1)
            start = real_positions[index] - count
            if not linear:
                output[start : start + count] = start_value
            elif name == "orientation":
                # Quaternion components cannot be averaged, so orientation moves along the arc.
                delta = relative_rotation_vector(start_value, end_value)
                output[start : start + count] = apply_rotation_vector(
                    start_value, fractions[:, None] * delta
                )
            else:
                step = end_value - start_value
                shaped = fractions.reshape((count,) + (1,) * step.ndim)
                output[start : start + count] = start_value + shaped * step
        return read_only(output)


def _shape_tail(values: FloatArray | None) -> tuple[int, ...] | None:
    """Return the per-sample shape of a channel, or ``None`` if the channel is absent."""
    return None if values is None else values.shape[1:]
