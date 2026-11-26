"""Wire protocol between the Go telephony gateway and the Python dialog engine.

One TCP connection per call. Every message is::

    +--------+-------------------+-------------+
    | type:1 | length:4 (BE u32) | payload     |
    +--------+-------------------+-------------+

==========  =====  =========================  ===================================
type        value  direction                  payload
==========  =====  =========================  ===================================
START       0x01   gateway → engine           JSON: call_id, from, to, codec, sample_rate
AUDIO       0x02   both                       PCM16 little-endian, 8 kHz, 20 ms (320 B)
CLEAR       0x03   engine → gateway           empty — drop queued playout (barge-in)
HANGUP      0x04   both                       empty
==========  =====  =========================  ===================================

The gateway does G.711 ↔ PCM and RTP; the engine never sees RTP. This is a
deliberately small interim contract — the spec's gRPC streaming API can
replace it without touching the dialog logic.
"""

from __future__ import annotations

import asyncio
import json
import struct
from enum import IntEnum

__all__ = ["MsgType", "encode", "read_message", "MAX_PAYLOAD"]

MAX_PAYLOAD = 64 * 1024
_HDR = struct.Struct(">BI")


class MsgType(IntEnum):
    START = 0x01
    AUDIO = 0x02
    CLEAR = 0x03
    HANGUP = 0x04


def encode(kind: MsgType, payload: bytes | dict = b"") -> bytes:
    if isinstance(payload, dict):
        payload = json.dumps(payload).encode()
    if len(payload) > MAX_PAYLOAD:
        raise ValueError("payload too large")
    return _HDR.pack(int(kind), len(payload)) + payload


async def read_message(reader: asyncio.StreamReader) -> tuple[MsgType, bytes] | None:
    """Return the next message, or ``None`` on clean EOF."""
    try:
        hdr = await reader.readexactly(_HDR.size)
    except asyncio.IncompleteReadError:
        return None
    kind, length = _HDR.unpack(hdr)
    if length > MAX_PAYLOAD:
        raise ValueError(f"payload too large: {length}")
    payload = await reader.readexactly(length) if length else b""
    return MsgType(kind), payload
