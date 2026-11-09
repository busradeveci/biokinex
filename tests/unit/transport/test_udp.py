"""Tests for biokinex.transport.udp, using real sockets on the loopback interface."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable

import pytest

from biokinex.core.errors import TransportError
from biokinex.transport.base import FrameSink, TelemetrySource
from biokinex.transport.udp import MAX_DATAGRAM_BYTES, UdpSink, UdpSource

TIMEOUT_S = 2.0


async def wait_until(condition: Callable[[], bool]) -> None:
    """Poll ``condition``; datagrams on loopback arrive quickly, but not instantly."""
    async with asyncio.timeout(TIMEOUT_S):
        # The counters belong to the socket protocol and have no event to wait on.
        while not condition():  # noqa: ASYNC110
            await asyncio.sleep(0.001)


@pytest.fixture
async def source() -> AsyncIterator[UdpSource]:
    udp_source = await UdpSource.open()
    yield udp_source
    await udp_source.close()


@pytest.fixture
async def sink(source: UdpSource) -> AsyncIterator[UdpSink]:
    udp_sink = await UdpSink.open(*source.address)
    yield udp_sink
    await udp_sink.close()


async def test_transports_follow_the_protocols(source: UdpSource, sink: UdpSink) -> None:
    assert isinstance(source, TelemetrySource)
    assert isinstance(sink, FrameSink)


async def test_source_binds_to_loopback_by_default(source: UdpSource) -> None:
    host, port = source.address
    assert host == "127.0.0.1"
    assert port > 0


async def test_datagrams_arrive_in_order(source: UdpSource, sink: UdpSink) -> None:
    payloads = [f"packet-{i}".encode() for i in range(5)]
    for payload in payloads:
        await sink.send(payload)
    received = [await asyncio.wait_for(source.receive(), TIMEOUT_S) for _ in payloads]
    assert received == payloads
    assert sink.sent == 5


async def test_full_queue_drops_oldest() -> None:
    small = await UdpSource.open(queue_size=2)
    sink = await UdpSink.open(*small.address)
    try:
        for index in range(5):
            await sink.send(bytes([index]))
        await wait_until(lambda: small.dropped == 3)
        assert await small.receive() == bytes([3])
    finally:
        await sink.close()
        await small.close()


async def test_oversized_datagrams_are_ignored() -> None:
    limited = await UdpSource.open(max_datagram_bytes=8)
    sink = await UdpSink.open(*limited.address)
    try:
        await sink.send(b"x" * 20)
        await sink.send(b"ok")
        assert await asyncio.wait_for(limited.receive(), TIMEOUT_S) == b"ok"
        assert limited.oversized == 1
    finally:
        await sink.close()
        await limited.close()


@pytest.mark.parametrize("size", [0, MAX_DATAGRAM_BYTES + 1])
async def test_rejects_invalid_datagram_limit(size: int) -> None:
    with pytest.raises(ValueError, match="max_datagram_bytes"):
        await UdpSource.open(max_datagram_bytes=size)


async def test_close_ends_the_stream(source: UdpSource) -> None:
    await source.close()
    assert await asyncio.wait_for(source.receive(), TIMEOUT_S) is None
    # A datagram that races with closing must not raise inside the event loop.
    source._protocol.datagram_received(b"late", ("127.0.0.1", 1))
    assert await source.receive() is None


async def test_socket_errors_are_counted(source: UdpSource, sink: UdpSink) -> None:
    source._protocol.error_received(OSError("network down"))
    sink._protocol.error_received(ConnectionRefusedError())
    assert source.errors == 1
    assert sink.errors == 1


async def test_sink_rejects_payload_larger_than_a_datagram(sink: UdpSink) -> None:
    with pytest.raises(TransportError, match="does not fit"):
        await sink.send(b"x" * (MAX_DATAGRAM_BYTES + 1))


async def test_sink_rejects_send_after_close(sink: UdpSink) -> None:
    await sink.close()
    await sink.close()
    with pytest.raises(TransportError, match="closed"):
        await sink.send(b"late")
