"""Reusable Pydantic field types and the base model for every boundary schema.

Numbers use per-field strict mode: an ``int`` is accepted where a ``float`` is expected, but
booleans, numeric strings, NaN, and infinity are rejected. Containers stay lax, so JSON arrays and
Python lists both validate into tuples.
"""

from __future__ import annotations

from typing import Annotated, Final, TypeAlias

from pydantic import AllowInfNan, BaseModel, ConfigDict, Field, Strict, StringConstraints

MAX_SEQUENCE: Final = 2**32 - 1
"""Largest sequence number; devices use an unsigned 32-bit counter that wraps to zero."""

MAX_JOINTS: Final = 64
"""Upper bound on joint channels per device, which caps memory use per packet."""

FiniteFloat: TypeAlias = Annotated[float, Strict(), AllowInfNan(False)]
PositiveFloat: TypeAlias = Annotated[float, Strict(), AllowInfNan(False), Field(gt=0.0)]
NonNegativeFloat: TypeAlias = Annotated[float, Strict(), AllowInfNan(False), Field(ge=0.0)]

Timestamp: TypeAlias = Annotated[float, Strict(), AllowInfNan(False), Field(ge=0.0)]
"""Sensor time in seconds; only differences matter, so the origin is device-defined."""

SequenceNumber: TypeAlias = Annotated[int, Strict(), Field(ge=0, le=MAX_SEQUENCE)]

IDENTIFIER_PATTERN: Final = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$"
"""Allowed form of device and joint identifiers; also used by codecs that skip Pydantic."""

Identifier: TypeAlias = Annotated[str, StringConstraints(pattern=IDENTIFIER_PATTERN)]
"""Opaque machine identifier for a device or joint.

The narrow character set keeps identifiers technical and stops free text from entering the system.
"""

Vector3Field: TypeAlias = tuple[FiniteFloat, FiniteFloat, FiniteFloat]
QuaternionField: TypeAlias = tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat]


class FrozenModel(BaseModel):
    """Immutable base model that rejects unknown fields."""

    # Forbidding extra fields stops typos from being ignored and blocks unexpected data at the edge.
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
