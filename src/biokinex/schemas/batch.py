"""The internal, array-based form of validated telemetry.

Pydantic models are ideal at the trust boundary but slow to iterate in the hot path. After a packet
is validated it becomes a :class:`SampleBatch`: plain NumPy arrays in SI units, ready for DSP.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from biokinex.core.types import QUATERNION_SIZE, VECTOR3_SIZE, FloatArray, IntArray


@dataclass(frozen=True, slots=True, eq=False)
class SampleBatch:
    """A run of ``N`` consecutive samples from one device, in SI units.

    Attributes:
        device_id: Opaque identifier of the source device.
        timestamps: Sensor time in seconds, shape ``(N,)``, strictly increasing.
        sequences: Device sequence numbers, shape ``(N,)``.
        acceleration: Specific force in m/s², shape ``(N, 3)``, or ``None``.
        angular_velocity: Body-frame angular rate in rad/s, shape ``(N, 3)``, or ``None``.
        orientation: Unit quaternions ``(w, x, y, z)``, shape ``(N, 4)``, or ``None``.
        joint_names: Names of the joint channels, in column order.
        joint_angles: Joint angles in radians, shape ``(N, len(joint_names))``, or ``None``.
    """

    device_id: str
    timestamps: FloatArray
    sequences: IntArray
    acceleration: FloatArray | None = None
    angular_velocity: FloatArray | None = None
    orientation: FloatArray | None = None
    joint_names: tuple[str, ...] = ()
    joint_angles: FloatArray | None = None

    def __post_init__(self) -> None:
        if self.timestamps.ndim != 1 or self.timestamps.shape[0] == 0:
            msg = f"timestamps must be a non-empty 1-D array, got shape {self.timestamps.shape}"
            raise ValueError(msg)
        n = self.timestamps.shape[0]
        _check_array("timestamps", self.timestamps, (n,), np.float64)
        _check_array("sequences", self.sequences, (n,), np.int64)
        _check_optional("acceleration", self.acceleration, (n, VECTOR3_SIZE))
        _check_optional("angular_velocity", self.angular_velocity, (n, VECTOR3_SIZE))
        _check_optional("orientation", self.orientation, (n, QUATERNION_SIZE))
        _check_optional("joint_angles", self.joint_angles, (n, len(self.joint_names)))

    def __len__(self) -> int:
        return int(self.timestamps.shape[0])


def _check_array(
    name: str, array: npt.NDArray[np.generic], shape: tuple[int, ...], dtype: type[np.generic]
) -> None:
    """Raise ``ValueError`` unless ``array`` has exactly the expected shape and dtype."""
    if array.shape != shape or array.dtype != dtype:
        msg = (
            f"{name} must have shape {shape} and dtype {np.dtype(dtype)}, "
            f"got shape {array.shape} and dtype {array.dtype}"
        )
        raise ValueError(msg)


def _check_optional(name: str, array: FloatArray | None, shape: tuple[int, ...]) -> None:
    """Check an optional float64 channel when it is present."""
    if array is not None:
        _check_array(name, array, shape, np.float64)
