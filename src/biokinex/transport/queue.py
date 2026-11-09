"""A bounded async queue that drops the oldest item when it is full.

For real-time control a fresh sample is worth more than a complete history. A normal bounded queue
would either block the producer or drop the newest item, and both make the consumer fall further
behind. Dropping the oldest item keeps memory fixed and the data fresh.
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Final, Generic, TypeVar

from biokinex.core.errors import TransportError

__all__ = ["DEFAULT_QUEUE_SIZE", "DropOldestQueue"]

DEFAULT_QUEUE_SIZE: Final = 1024
"""About two seconds of packets at 500 Hz, which covers short pauses in the consumer."""

_T = TypeVar("_T")


class DropOldestQueue(Generic[_T]):
    """Bounded queue for one event loop; ``put_nowait`` never blocks.

    Args:
        maxsize: Largest number of queued items.
    """

    __slots__ = ("_closed", "_dropped", "_items", "_ready")

    def __init__(self, maxsize: int) -> None:
        if maxsize < 1:
            msg = f"maxsize must be >= 1, got {maxsize}"
            raise ValueError(msg)
        self._items: deque[_T] = deque(maxlen=maxsize)
        self._ready = asyncio.Event()
        self._closed = False
        self._dropped = 0

    @property
    def maxsize(self) -> int:
        """Largest number of queued items."""
        return self._items.maxlen or 0

    @property
    def dropped(self) -> int:
        """Number of items dropped because the queue was full."""
        return self._dropped

    @property
    def closed(self) -> bool:
        """Whether :meth:`close` has been called."""
        return self._closed

    def __len__(self) -> int:
        return len(self._items)

    def put_nowait(self, item: _T) -> None:
        """Add ``item``, dropping the oldest item if the queue is full.

        Raises:
            TransportError: If the queue is closed.
        """
        if self._closed:
            msg = "queue is closed"
            raise TransportError(msg)
        if len(self._items) == self._items.maxlen:
            self._dropped += 1
        # A deque with maxlen removes the oldest item by itself on append.
        self._items.append(item)
        self._ready.set()

    async def get(self) -> _T | None:
        """Wait for the next item; return ``None`` once the queue is closed and empty."""
        while not self._items:
            if self._closed:
                return None
            self._ready.clear()
            await self._ready.wait()
        return self._items.popleft()

    def get_nowait(self) -> _T | None:
        """Return the next item, or ``None`` if the queue is empty."""
        return self._items.popleft() if self._items else None

    def close(self) -> None:
        """Refuse new items; waiting consumers still receive the queued ones."""
        self._closed = True
        # Waking every waiter lets each one see the closed state and return.
        self._ready.set()
