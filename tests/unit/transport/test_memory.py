"""Tests for biokinex.transport.memory."""

from __future__ import annotations

import asyncio

import pytest

from biokinex.core.errors import TransportError
from biokinex.transport.base import FrameSink, TelemetrySource
from biokinex.transport.memory import MemorySink, MemorySource


def test_memory_transports_follow_the_protocols() -> None:
    assert isinstance(MemorySource(), TelemetrySource)
    assert isinstance(MemorySink(), FrameSink)


class TestMemorySource:
    async def test_push_and_receive_in_order(self) -> None:
        source = MemorySource()
        source.push(b"a")
        source.push(b"b")
        assert [await source.receive(), await source.receive()] == [b"a", b"b"]

    async def test_full_source_drops_oldest(self) -> None:
        source = MemorySource(maxsize=2)
        for payload in (b"1", b"2", b"3"):
            source.push(payload)
        assert source.dropped == 1
        assert await source.receive() == b"2"

    async def test_close_ends_the_stream_after_queued_payloads(self) -> None:
        source = MemorySource()
        source.push(b"last")
        await source.close()
        assert await source.receive() == b"last"
        assert await source.receive() is None
        with pytest.raises(TransportError):
            source.push(b"late")

    async def test_receive_waits_for_push(self) -> None:
        source = MemorySource()
        waiter = asyncio.create_task(source.receive())
        await asyncio.sleep(0)
        source.push(b"x")
        assert await asyncio.wait_for(waiter, 1.0) == b"x"


class TestMemorySink:
    async def test_send_and_drain(self) -> None:
        sink = MemorySink()
        await sink.send(b"f1")
        await sink.send(b"f2")
        assert sink.drain() == [b"f1", b"f2"]
        assert sink.drain() == []

    async def test_slow_reader_loses_oldest_frames(self) -> None:
        sink = MemorySink(maxsize=2)
        for payload in (b"1", b"2", b"3", b"4"):
            await sink.send(payload)
        assert sink.dropped == 2
        assert await sink.receive() == b"3"

    async def test_close(self) -> None:
        sink = MemorySink()
        await sink.send(b"f")
        await sink.close()
        assert await sink.receive() == b"f"
        assert await sink.receive() is None
        with pytest.raises(TransportError):
            await sink.send(b"late")
