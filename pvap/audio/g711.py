"""ITU-T G.711 μ-law (PCMU) and A-law (PCMA) codecs, vectorized with NumPy.

Telephony carries 8 kHz audio as one byte per sample. These are the exact
segment-encoding algorithms from the G.711 reference (Sun/CCITT g711.c), so
output is bit-compatible with FreeSWITCH, Asterisk and carrier gear.
"""

from __future__ import annotations

import numpy as np

__all__ = ["ulaw_encode", "ulaw_decode", "alaw_encode", "alaw_decode"]

_BIAS = 0x84
_CLIP = 32635
_SEG_END = np.array([0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF, 0x3FFF, 0x7FFF])


def _segment(mag: np.ndarray) -> np.ndarray:
    return np.searchsorted(_SEG_END, mag, side="left")


def ulaw_encode(pcm: np.ndarray) -> np.ndarray:
    """int16 PCM → μ-law bytes (uint8)."""
    x = pcm.astype(np.int32)
    mask = np.where(x < 0, 0x7F, 0xFF)
    mag = np.minimum(np.abs(x), _CLIP) + _BIAS
    seg = _segment(mag)
    over = seg >= 8
    seg = np.minimum(seg, 7)
    mant = (mag >> (seg + 3)) & 0x0F
    code = np.where(over, 0x7F, (seg << 4) | mant)
    return (code ^ mask).astype(np.uint8)


def _build_ulaw_table() -> np.ndarray:
    u = (~np.arange(256, dtype=np.int32)) & 0xFF
    t = ((u & 0x0F) << 3) + _BIAS
    t = t << ((u & 0x70) >> 4)
    return np.where(u & 0x80, _BIAS - t, t - _BIAS).astype(np.int16)


_ULAW_TABLE = _build_ulaw_table()


def ulaw_decode(data: np.ndarray | bytes) -> np.ndarray:
    """μ-law bytes → int16 PCM."""
    b = np.frombuffer(data, dtype=np.uint8) if isinstance(data, (bytes, bytearray)) else data
    return _ULAW_TABLE[b.astype(np.intp)]


_A_SEG_END = np.array([0x1F, 0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF])


def alaw_encode(pcm: np.ndarray) -> np.ndarray:
    """int16 PCM → A-law bytes (uint8)."""
    x = pcm.astype(np.int32) >> 3  # 13-bit magnitude domain
    mask = np.where(x >= 0, 0xD5, 0x55)
    mag = np.where(x >= 0, x, -x - 1)
    seg = np.searchsorted(_A_SEG_END, mag, side="left")
    over = seg >= 8
    seg_c = np.minimum(seg, 7)
    aval = seg_c << 4
    aval = np.where(seg_c < 2, aval | ((mag >> 1) & 0x0F), aval | ((mag >> seg_c) & 0x0F))
    out = np.where(over, 0x7F ^ mask, aval ^ mask)
    return out.astype(np.uint8)


def _build_alaw_table() -> np.ndarray:
    a = np.arange(256, dtype=np.int32) ^ 0x55
    t = (a & 0x0F) << 4
    seg = (a & 0x70) >> 4
    t = np.where(seg == 0, t + 8, np.where(seg == 1, t + 0x108, (t + 0x108) << (seg - 1)))
    return np.where(a & 0x80, t, -t).astype(np.int16)


_ALAW_TABLE = _build_alaw_table()


def alaw_decode(data: np.ndarray | bytes) -> np.ndarray:
    """A-law bytes → int16 PCM."""
    b = np.frombuffer(data, dtype=np.uint8) if isinstance(data, (bytes, bytearray)) else data
    return _ALAW_TABLE[b.astype(np.intp)]
