"""Interfaces shared by every transport and codec.

Transports only move bytes, and codecs only convert bytes. Keeping the two apart means any codec
works over any transport, and the runtime can be tested with in-memory transports.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from biokinex.schemas.batch import SampleBatch
from biokinex.schemas.pose import PoseFrame

__all__ = ["Codec", "FrameSink", "TelemetrySource"]


@runtime_checkable
class TelemetrySource(Protocol):
    """Delivers raw telemetry payloads, one packet per call."""

    async def receive(self) -> bytes | None:
        """Wait for the next payload; return ``None`` once the source is closed and empty."""
        ...

    async def close(self) -> None:
        """Stop receiving and release resources; calling it twice is safe."""
        ...


@runtime_checkable
class FrameSink(Protocol):
    """Delivers encoded pose frames to a consumer."""

    async def send(self, payload: bytes) -> None:
        """Send one encoded frame without waiting for slow consumers."""
        ...

    async def close(self) -> None:
        """Stop sending and release resources; calling it twice is safe."""
        ...


@runtime_checkable
class Codec(Protocol):
    """Converts wire payloads to batches, and pose frames to wire payloads."""

    def decode_packet(self, data: bytes) -> SampleBatch:
        """Parse and validate one telemetry packet.

        Raises:
            TelemetryError: If the payload is malformed or breaks any schema rule.
        """
        ...

    def encode_frame(self, frame: PoseFrame) -> bytes:
        """Encode one pose frame."""
        ...
