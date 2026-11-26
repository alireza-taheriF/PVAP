"""Minimal RFC 6455 WebSocket + HTTP/1.1 on asyncio streams (no dependencies).

Enough for a browser call: the opening handshake, masked client frames,
binary and text messages, ping/pong and close. Fragmented messages are
reassembled; extensions (compression) are not negotiated.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import struct
from dataclasses import dataclass, field
from urllib.parse import parse_qs, urlsplit

__all__ = ["HttpRequest", "read_request", "WebSocket", "accept_key", "http_response"]

_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
MAX_MESSAGE = 1 << 20


@dataclass
class HttpRequest:
    method: str
    target: str
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def path(self) -> str:
        return urlsplit(self.target).path

    @property
    def query(self) -> dict[str, str]:
        return {k: v[0] for k, v in parse_qs(urlsplit(self.target).query).items()}

    @property
    def is_websocket(self) -> bool:
        return (self.headers.get("upgrade", "").lower() == "websocket"
                and "upgrade" in self.headers.get("connection", "").lower())


async def read_request(reader: asyncio.StreamReader) -> HttpRequest | None:
    try:
        head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=15)
    except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, asyncio.TimeoutError, ConnectionError):
        return None
    lines = head.decode("latin-1").split("\r\n")
    parts = lines[0].split(" ")
    if len(parts) != 3:
        return None
    headers = {}
    for ln in lines[1:]:
        if ":" in ln:
            k, v = ln.split(":", 1)
            headers[k.strip().lower()] = v.strip()
    return HttpRequest(parts[0], parts[1], headers)


def accept_key(key: str) -> str:
    return base64.b64encode(hashlib.sha1((key + _GUID).encode()).digest()).decode()


def http_response(status: str, body: bytes = b"", ctype: str = "text/plain; charset=utf-8",
                  extra: dict[str, str] | None = None) -> bytes:
    h = [f"HTTP/1.1 {status}", f"Content-Type: {ctype}", f"Content-Length: {len(body)}",
         "Cache-Control: no-store", "Connection: close"]
    h += [f"{k}: {v}" for k, v in (extra or {}).items()]
    return ("\r\n".join(h) + "\r\n\r\n").encode() + body


class WebSocket:
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.reader, self.writer = reader, writer
        self.closed = False
        self._wlock = asyncio.Lock()

    @classmethod
    async def accept(cls, req: HttpRequest, reader, writer) -> "WebSocket | None":
        key = req.headers.get("sec-websocket-key")
        if not key or req.headers.get("sec-websocket-version") != "13":
            writer.write(http_response("400 Bad Request", b"websocket required"))
            await writer.drain()
            return None
        writer.write(("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                      f"Sec-WebSocket-Accept: {accept_key(key)}\r\n\r\n").encode())
        await writer.drain()
        return cls(reader, writer)

    async def _send(self, opcode: int, payload: bytes) -> None:
        if self.closed and opcode != 0x8:
            raise ConnectionError("websocket closed")
        n = len(payload)
        if n < 126:
            hdr = struct.pack("!BB", 0x80 | opcode, n)
        elif n < 1 << 16:
            hdr = struct.pack("!BBH", 0x80 | opcode, 126, n)
        else:
            hdr = struct.pack("!BBQ", 0x80 | opcode, 127, n)
        async with self._wlock:
            self.writer.write(hdr + payload)
            await self.writer.drain()

    async def send_bytes(self, data: bytes) -> None:
        await self._send(0x2, data)

    async def send_text(self, text: str) -> None:
        await self._send(0x1, text.encode())

    async def close(self, code: int = 1000) -> None:
        if not self.closed:
            try:
                await self._send(0x8, struct.pack("!H", code))
            except (ConnectionError, RuntimeError):
                pass
            self.closed = True
        self.writer.close()

    async def recv(self) -> bytes | str | None:
        """Next message (bytes for binary, str for text) or None when closed."""
        buf, kind = b"", None
        while True:
            try:
                b0, b1 = await self.reader.readexactly(2)
                n = b1 & 0x7F
                if n == 126:
                    (n,) = struct.unpack("!H", await self.reader.readexactly(2))
                elif n == 127:
                    (n,) = struct.unpack("!Q", await self.reader.readexactly(8))
                if n > MAX_MESSAGE:
                    await self.close(1009)
                    return None
                mask = await self.reader.readexactly(4) if b1 & 0x80 else b""
                data = await self.reader.readexactly(n) if n else b""
            except (asyncio.IncompleteReadError, ConnectionError):
                self.closed = True
                return None
            if mask:
                data = bytes(c ^ mask[i % 4] for i, c in enumerate(data)) if n < 64 else _unmask(data, mask)
            op, fin = b0 & 0x0F, b0 & 0x80
            if op == 0x8:
                await self.close()
                return None
            if op == 0x9:
                await self._send(0xA, data)
                continue
            if op == 0xA:
                continue
            if op in (0x1, 0x2):
                kind, buf = op, data
            elif op == 0x0:
                buf += data
            if len(buf) > MAX_MESSAGE:
                await self.close(1009)
                return None
            if fin and kind is not None:
                return buf.decode("utf-8", "replace") if kind == 0x1 else buf


def _unmask(data: bytes, mask: bytes) -> bytes:
    import numpy as np

    a = np.frombuffer(data, np.uint8)
    m = np.resize(np.frombuffer(mask, np.uint8), len(a))
    return (a ^ m).tobytes()
