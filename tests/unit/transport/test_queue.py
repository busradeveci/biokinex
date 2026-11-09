"""Tests for biokinex.transport.queue."""

from __future__ import annotations

import asyncio

import pytest

from biokinex.core.errors import TransportError
from biokinex.transport.queue import DropOldestQueue


async def test_items_come_out_in_order() -> None:
    queue: DropOldestQueue[int] = DropOldestQueue(4)
    for item in (1, 2, 3):
        queue.put_nowait(item)
    assert len(queue) == 3
    assert [await queue.get() for _ in range(3)] == [1, 2, 3]
    assert queue.maxsize == 4


async def test_full_queue_drops_oldest() -> None:
    queue: DropOldestQueue[int] = DropOldestQueue(2)
    for item in range(5):
        queue.put_nowait(item)
    assert queue.dropped == 3
    assert [queue.get_nowait(), queue.get_nowait(), queue.get_nowait()] == [3, 4, None]


async def test_get_waits_for_a_new_item() -> None:
    queue: DropOldestQueue[str] = DropOldestQueue(2)
    waiter = asyncio.create_task(queue.get())
    await asyncio.sleep(0)
    assert not waiter.done()
    queue.put_nowait("a")
    assert await asyncio.wait_for(waiter, 1.0) == "a"


async def test_close_wakes_every_waiter() -> None:
    queue: DropOldestQueue[int] = DropOldestQueue(2)
    waiters = [asyncio.create_task(queue.get()) for _ in range(2)]
    await asyncio.sleep(0)
    queue.close()
    assert await asyncio.wait_for(asyncio.gather(*waiters), 1.0) == [None, None]


async def test_queued_items_survive_close() -> None:
    queue: DropOldestQueue[int] = DropOldestQueue(2)
    queue.put_nowait(7)
    queue.close()
    assert queue.closed
    assert await queue.get() == 7
    assert await queue.get() is None


def test_put_after_close_raises() -> None:
    queue: DropOldestQueue[int] = DropOldestQueue(1)
    queue.close()
    with pytest.raises(TransportError, match="closed"):
        queue.put_nowait(1)


@pytest.mark.parametrize("maxsize", [0, -1])
def test_rejects_invalid_size(maxsize: int) -> None:
    with pytest.raises(ValueError, match="maxsize"):
        DropOldestQueue[int](maxsize)
