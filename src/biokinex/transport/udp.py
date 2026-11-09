"""UDP transports built on asyncio datagram endpoints.

UDP suits real-time telemetry: a lost datagram is simply a gap, which the validator and gap filler
already handle, while TCP would delay every later packet until the lost one is sent again.

Each datagram carries exactly one packet or one frame. Received datagrams go into a bounded
drop-oldest queue, so a burst of traffic can never make memory grow.
"""

from __future__ import annotations

import asyncio
from typing import Final

from biokinex.core.errors import TransportError
from biokinex.transport.queue import DEFAULT_QUEUE_SIZE, DropOldestQueue

__all__ = ["DEFAULT_HOST", "MAX_DATAGRAM_BYTES", "UdpSink", "UdpSource"]

MAX_DATAGRAM_BYTES: Final = 65_507
"""Largest UDP payload over IPv4."""

DEFAULT_HOST: Final = "127.0.0.1"
"""Sources listen on the local machine only, unless the caller chooses a wider address."""


class _ReceiverProtocol(asyncio.DatagramProtocol):
    """Moves received datagrams into a queue; asyncio calls these methods on the event loop."""

    def __init__(self, queue: DropOldestQueue[bytes], max_datagram_bytes: int) -> None:
        self.queue = queue
        self.max_datagram_bytes = max_datagram_bytes
        self.oversized = 0
        self.errors = 0

    def datagram_received(self, data: bytes, addr: tuple[str | object, ...]) -> None:
        if len(data) > self.max_datagram_bytes:
            self.oversized += 1
            return
        # Datagrams can still arrive while the source is closing; they are ignored.
        if not self.queue.closed:
            self.queue.put_nowait(data)

    def error_received(self, exc: Exception) -> None:
        # Network errors on UDP are reported per datagram and do not close the socket.
        self.errors += 1

    def connection_lost(self, exc: Exception | None) -> None:
        self.queue.close()


class _SenderProtocol(asyncio.DatagramProtocol):
    """Counts send errors, such as "port unreachable" replies when no consumer listens."""

    def __init__(self) -> None:
        self.errors = 0

    def error_received(self, exc: Exception) -> None:
        self.errors += 1


class UdpSource:
    """Receives telemetry datagrams on a local UDP port.

    Create it with :meth:`open`, which binds the socket on the running event loop.
    """

    __slots__ = ("_protocol", "_transport")

    def __init__(self, transport: asyncio.DatagramTransport, protocol: _ReceiverProtocol) -> None:
        self._transport = transport
        self._protocol = protocol

    @classmethod
    async def open(
        cls,
        host: str = DEFAULT_HOST,
        port: int = 0,
        *,
        queue_size: int = DEFAULT_QUEUE_SIZE,
        max_datagram_bytes: int = MAX_DATAGRAM_BYTES,
    ) -> UdpSource:
        """Bind a UDP socket; port ``0`` lets the system pick a free port.

        Raises:
            OSError: If the address cannot be bound.
        """
        if not 1 <= max_datagram_bytes <= MAX_DATAGRAM_BYTES:
            msg = f"max_datagram_bytes must be in [1, {MAX_DATAGRAM_BYTES}]"
            raise ValueError(msg)
        queue: DropOldestQueue[bytes] = DropOldestQueue(queue_size)
        loop = asyncio.get_running_loop()
        transport, protocol = await loop.create_datagram_endpoint(
            lambda: _ReceiverProtocol(queue, max_datagram_bytes), local_addr=(host, port)
        )
        return cls(transport, protocol)

    @property
    def address(self) -> tuple[str, int]:
        """The bound host and port."""
        host, port = self._transport.get_extra_info("sockname")[:2]
        return str(host), int(port)

    @property
    def dropped(self) -> int:
        """Datagrams dropped because the queue was full."""
        return self._protocol.queue.dropped

    @property
    def oversized(self) -> int:
        """Datagrams ignored because they were larger than ``max_datagram_bytes``."""
        return self._protocol.oversized

    @property
    def errors(self) -> int:
        """Network errors reported by the socket."""
        return self._protocol.errors

    async def receive(self) -> bytes | None:
        """Wait for the next datagram; return ``None`` once closed and empty."""
        return await self._protocol.queue.get()

    async def close(self) -> None:
        """Close the socket; queued datagrams can still be received."""
        self._transport.close()
        self._protocol.queue.close()


class UdpSink:
    """Sends encoded frames as datagrams to one remote address.

    Create it with :meth:`open`. Sending never waits: the datagram is handed to the operating
    system, and a slow or absent consumer only causes lost datagrams.
    """

    __slots__ = ("_protocol", "_sent", "_transport")

    def __init__(self, transport: asyncio.DatagramTransport, protocol: _SenderProtocol) -> None:
        self._transport = transport
        self._protocol = protocol
        self._sent = 0

    @classmethod
    async def open(cls, host: str, port: int) -> UdpSink:
        """Create a UDP socket connected to ``(host, port)``.

        Raises:
            OSError: If the address cannot be resolved or used.
        """
        loop = asyncio.get_running_loop()
        transport, protocol = await loop.create_datagram_endpoint(
            _SenderProtocol, remote_addr=(host, port)
        )
        return cls(transport, protocol)

    @property
    def sent(self) -> int:
        """Datagrams handed to the operating system."""
        return self._sent

    @property
    def errors(self) -> int:
        """Send errors reported by the socket."""
        return self._protocol.errors

    async def send(self, payload: bytes) -> None:
        """Send one encoded frame as a single datagram.

        Raises:
            TransportError: If the sink is closed or the payload does not fit in one datagram.
        """
        if self._transport.is_closing():
            msg = "sink is closed"
            raise TransportError(msg)
        if len(payload) > MAX_DATAGRAM_BYTES:
            msg = f"payload of {len(payload)} bytes does not fit in one datagram"
            raise TransportError(msg)
        self._transport.sendto(payload)
        self._sent += 1

    async def close(self) -> None:
        """Close the socket; calling it twice is safe."""
        self._transport.close()
