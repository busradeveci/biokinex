"""In-process transports for tests and for embedding BioKineX inside another application.

Both ends use a :class:`~biokinex.transport.queue.DropOldestQueue`, so a slow consumer can never
make memory grow or block the pipeline.
"""

from __future__ import annotations

from biokinex.transport.queue import DEFAULT_QUEUE_SIZE, DropOldestQueue

__all__ = ["MemorySink", "MemorySource"]


class MemorySource:
    """A telemetry source that the application fills by calling :meth:`push`."""

    __slots__ = ("_queue",)

    def __init__(self, maxsize: int = DEFAULT_QUEUE_SIZE) -> None:
        self._queue: DropOldestQueue[bytes] = DropOldestQueue(maxsize)

    @property
    def dropped(self) -> int:
        """Number of payloads dropped because the queue was full."""
        return self._queue.dropped

    def push(self, payload: bytes) -> None:
        """Queue one payload without waiting.

        Raises:
            TransportError: If the source is closed.
        """
        self._queue.put_nowait(payload)

    async def receive(self) -> bytes | None:
        """Wait for the next payload; return ``None`` once closed and empty."""
        return await self._queue.get()

    async def close(self) -> None:
        """Refuse new payloads; queued payloads can still be received."""
        self._queue.close()


class MemorySink:
    """A frame sink that the application reads with :meth:`receive` or :meth:`drain`."""

    __slots__ = ("_queue",)

    def __init__(self, maxsize: int = DEFAULT_QUEUE_SIZE) -> None:
        self._queue: DropOldestQueue[bytes] = DropOldestQueue(maxsize)

    @property
    def dropped(self) -> int:
        """Number of frames dropped because the reader was too slow."""
        return self._queue.dropped

    async def send(self, payload: bytes) -> None:
        """Queue one encoded frame; the oldest frame is dropped if the queue is full.

        Raises:
            TransportError: If the sink is closed.
        """
        self._queue.put_nowait(payload)

    async def receive(self) -> bytes | None:
        """Wait for the next frame; return ``None`` once closed and empty."""
        return await self._queue.get()

    def drain(self) -> list[bytes]:
        """Remove and return every queued frame, oldest first."""
        frames: list[bytes] = []
        while (payload := self._queue.get_nowait()) is not None:
            frames.append(payload)
        return frames

    async def close(self) -> None:
        """Refuse new frames; queued frames can still be read."""
        self._queue.close()
