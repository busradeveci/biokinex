"""Semantic validation of one device stream.

The schema layer already guarantees structure, types, and finite values. This layer checks what
the schema cannot see: physical plausibility, and order across packets. Rejected samples are
reported with a reason and counted, but never raise, because one bad sample must not stop a
real-time stream.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum
from typing import Final

import numpy as np
import numpy.typing as npt

from biokinex.core.errors import TelemetryError
from biokinex.core.types import BoolArray, IntArray
from biokinex.schemas.batch import SampleBatch, read_only
from biokinex.schemas.limits import SensorLimits
from biokinex.validation.sequence import (
    DEFAULT_MAX_FORWARD_JUMP,
    DEFAULT_REORDER_WINDOW,
    SEQUENCE_MODULUS,
    SequenceStatus,
    SequenceTracker,
)

__all__ = [
    "DEFAULT_MAX_CONSECUTIVE_TIME_ERRORS",
    "TelemetryValidator",
    "ValidationResult",
    "ValidatorStats",
    "Verdict",
]

DEFAULT_MAX_CONSECUTIVE_TIME_ERRORS: Final = 10
"""Time errors in a row after which the stream restarts from the next sample."""

_RESTART_STATUSES: Final = frozenset({SequenceStatus.FIRST, SequenceStatus.RESYNC})


class Verdict(IntEnum):
    """The outcome for one input sample. Only the first failing check is reported."""

    ACCEPTED = 0
    ACCELERATION_OUT_OF_RANGE = 1
    ANGULAR_VELOCITY_OUT_OF_RANGE = 2
    JOINT_ANGLE_OUT_OF_RANGE = 3
    DUPLICATE = 4
    LATE = 5
    NON_MONOTONIC_TIME = 6


@dataclass(frozen=True, slots=True, eq=False)
class ValidationResult:
    """The outcome of validating one batch.

    Attributes:
        batch: The accepted samples in their original order, or ``None`` if none passed. When every
            sample passes, this is the input batch itself, so no data is copied.
        verdicts: One :class:`Verdict` per input sample, shape ``(N,)``.
        missing_before: For each accepted sample, how many sequence numbers were skipped since the
            previous accepted sample. Rejected samples count as skipped.
        restarts: For each accepted sample, whether the stream starts or restarts there. Filters
            must reinitialize at these samples instead of bridging the gap.
    """

    batch: SampleBatch | None
    verdicts: npt.NDArray[np.uint8]
    missing_before: IntArray
    restarts: BoolArray

    @property
    def accepted_count(self) -> int:
        """Number of accepted samples."""
        return int(np.count_nonzero(self.verdicts == Verdict.ACCEPTED))

    @property
    def rejected_count(self) -> int:
        """Number of rejected samples."""
        return int(self.verdicts.shape[0]) - self.accepted_count

    def counts(self) -> dict[Verdict, int]:
        """Return how many samples received each verdict."""
        return _count_verdicts(self.verdicts)


@dataclass(frozen=True, slots=True)
class ValidatorStats:
    """Totals since the validator was created."""

    verdicts: Mapping[Verdict, int]
    missing_samples: int
    restarts: int


class TelemetryValidator:
    """Checks physical limits and stream order for a single device.

    The first batch binds the validator to its device. A separate validator is needed per device,
    because order checks only make sense within one stream.
    """

    __slots__ = (
        "_device_id",
        "_last_timestamp",
        "_limits",
        "_max_time_errors",
        "_missing_total",
        "_restart_total",
        "_time_errors",
        "_tracker",
        "_verdict_totals",
    )

    def __init__(
        self,
        limits: SensorLimits | None = None,
        *,
        reorder_window: int = DEFAULT_REORDER_WINDOW,
        max_forward_jump: int = DEFAULT_MAX_FORWARD_JUMP,
        max_consecutive_time_errors: int = DEFAULT_MAX_CONSECUTIVE_TIME_ERRORS,
    ) -> None:
        if max_consecutive_time_errors < 1:
            msg = f"max_consecutive_time_errors must be >= 1, got {max_consecutive_time_errors}"
            raise ValueError(msg)
        self._limits = limits if limits is not None else SensorLimits()
        self._tracker = SequenceTracker(
            reorder_window=reorder_window, max_forward_jump=max_forward_jump
        )
        self._max_time_errors = max_consecutive_time_errors
        self._device_id: str | None = None
        self._last_timestamp = -math.inf
        self._time_errors = 0
        self._verdict_totals = np.zeros(len(Verdict), dtype=np.int64)
        self._missing_total = 0
        self._restart_total = 0

    @property
    def device_id(self) -> str | None:
        """The device this validator is bound to, or ``None`` before the first batch."""
        return self._device_id

    @property
    def limits(self) -> SensorLimits:
        """The physical limits in use."""
        return self._limits

    def validate(self, batch: SampleBatch) -> ValidationResult:
        """Validate ``batch`` and update the stream state with the accepted samples.

        Raises:
            TelemetryError: If ``batch`` comes from a different device than earlier batches.
        """
        self._bind_device(batch.device_id)
        verdicts = self._check_limits(batch)
        result = self._try_fast_path(batch, verdicts)
        if result is None:
            result = self._validate_each_sample(batch, verdicts)
        self._record(result)
        return result

    def stats(self) -> ValidatorStats:
        """Return totals since the validator was created."""
        return ValidatorStats(
            verdicts={verdict: int(self._verdict_totals[verdict]) for verdict in Verdict},
            missing_samples=self._missing_total,
            restarts=self._restart_total,
        )

    def reset(self) -> None:
        """Forget the device and stream history, but keep the totals."""
        self._device_id = None
        self._tracker.reset()
        self._last_timestamp = -math.inf
        self._time_errors = 0

    def _bind_device(self, device_id: str) -> None:
        if self._device_id is None:
            self._device_id = device_id
        elif device_id != self._device_id:
            msg = f"validator is bound to device {self._device_id!r}, got {device_id!r}"
            raise TelemetryError(msg)

    def _check_limits(self, batch: SampleBatch) -> npt.NDArray[np.uint8]:
        """Return a verdict per sample for the physical limit checks, using whole-array math."""
        limits = self._limits
        passed = np.zeros(len(batch), dtype=np.bool_)
        accel_bad = passed
        if batch.acceleration is not None:
            accel_bad = np.any(np.abs(batch.acceleration) > limits.max_acceleration, axis=1)
        gyro_bad = passed
        if batch.angular_velocity is not None:
            gyro_bad = np.any(np.abs(batch.angular_velocity) > limits.max_angular_velocity, axis=1)
        joint_bad = passed
        if batch.joint_angles is not None:
            angles = batch.joint_angles
            outside = (angles < limits.min_joint_angle) | (angles > limits.max_joint_angle)
            joint_bad = np.any(outside, axis=1)
        # np.select keeps the first matching condition, so this order sets the reported reason.
        verdicts = np.select(
            [accel_bad, gyro_bad, joint_bad],
            [
                Verdict.ACCELERATION_OUT_OF_RANGE,
                Verdict.ANGULAR_VELOCITY_OUT_OF_RANGE,
                Verdict.JOINT_ANGLE_OUT_OF_RANGE,
            ],
            default=Verdict.ACCEPTED,
        )
        return verdicts.astype(np.uint8)

    def _try_fast_path(
        self, batch: SampleBatch, verdicts: npt.NDArray[np.uint8]
    ) -> ValidationResult | None:
        """Accept a clean, in-order batch with array checks only, or return ``None``.

        Almost every batch in a healthy stream takes this path, so it avoids Python-level loops.
        """
        last = self._tracker.last
        if last is None or verdicts.any():
            return None
        n = len(batch)
        expected = (last + 1 + np.arange(n, dtype=np.int64)) % SEQUENCE_MODULUS
        timestamps = batch.timestamps
        if (
            not np.array_equal(batch.sequences, expected)
            or timestamps[0] <= self._last_timestamp
            or np.any(np.diff(timestamps) <= 0.0)
        ):
            return None
        self._tracker.advance(int(batch.sequences[-1]))
        self._last_timestamp = float(timestamps[-1])
        self._time_errors = 0
        return ValidationResult(
            batch=batch,
            verdicts=read_only(verdicts),
            missing_before=read_only(np.zeros(n, dtype=np.int64)),
            restarts=read_only(np.zeros(n, dtype=np.bool_)),
        )

    def _validate_each_sample(
        self, batch: SampleBatch, verdicts: npt.NDArray[np.uint8]
    ) -> ValidationResult:
        """Check order sample by sample; used when the batch has losses, repeats, or bad data."""
        sequences = batch.sequences.tolist()
        timestamps = batch.timestamps.tolist()
        accepted: list[int] = []
        missing: list[int] = []
        restarts: list[bool] = []

        for index in range(len(batch)):
            if verdicts[index] != Verdict.ACCEPTED:
                continue
            status, lost = self._tracker.classify(sequences[index])
            if status is SequenceStatus.DUPLICATE:
                verdicts[index] = Verdict.DUPLICATE
                continue
            if status is SequenceStatus.LATE:
                verdicts[index] = Verdict.LATE
                continue

            restart = status in _RESTART_STATUSES
            # After a restart the device clock may also have been reset, so the old time is no
            # longer a valid reference.
            if not restart and timestamps[index] <= self._last_timestamp:
                self._time_errors += 1
                if self._time_errors <= self._max_time_errors:
                    verdicts[index] = Verdict.NON_MONOTONIC_TIME
                    continue
                # So many errors in a row mean the reference time itself is wrong, for example
                # after one glitched timestamp far in the future, so we restart from here.
                restart = True
                lost = 0

            self._time_errors = 0
            self._tracker.advance(sequences[index])
            self._last_timestamp = timestamps[index]
            accepted.append(index)
            missing.append(lost)
            restarts.append(restart)

        selected: SampleBatch | None = None
        if len(accepted) == len(batch):
            selected = batch
        elif accepted:
            selected = batch.select(np.array(accepted, dtype=np.int64))
        return ValidationResult(
            batch=selected,
            verdicts=read_only(verdicts),
            missing_before=read_only(np.array(missing, dtype=np.int64)),
            restarts=read_only(np.array(restarts, dtype=np.bool_)),
        )

    def _record(self, result: ValidationResult) -> None:
        """Add one result to the running totals."""
        self._verdict_totals += np.bincount(result.verdicts, minlength=len(Verdict))
        self._missing_total += int(result.missing_before.sum())
        self._restart_total += int(np.count_nonzero(result.restarts))


def _count_verdicts(verdicts: npt.NDArray[np.uint8]) -> dict[Verdict, int]:
    """Count how often each verdict appears."""
    totals = np.bincount(verdicts, minlength=len(Verdict))
    return {verdict: int(totals[verdict]) for verdict in Verdict}
