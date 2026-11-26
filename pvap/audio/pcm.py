"""PCM helpers: framing, resampling, levels and WAV I/O.

Conventions used across PVAP:

* the **call leg** is 8 kHz mono int16 (narrowband telephony, G.711);
* **frames** are 20 ms (160 samples at 8 kHz), the RTP packetization interval;
* the **ASR** side may want 16 kHz; :func:`resample` converts.
"""

from __future__ import annotations

import io
import math
import wave
from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly

__all__ = [
    "TELEPHONY_RATE",
    "FRAME_MS",
    "frame_samples",
    "bytes_to_pcm",
    "pcm_to_bytes",
    "float_to_pcm",
    "resample",
    "split_frames",
    "dbfs",
    "read_wav",
    "write_wav",
]

TELEPHONY_RATE = 8_000
FRAME_MS = 20


def frame_samples(sample_rate: int = TELEPHONY_RATE, frame_ms: int = FRAME_MS) -> int:
    return sample_rate * frame_ms // 1000


def bytes_to_pcm(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype="<i2").astype(np.int16)


def pcm_to_bytes(pcm: np.ndarray) -> bytes:
    return np.asarray(pcm, dtype="<i2").tobytes()


def float_to_pcm(x: np.ndarray) -> np.ndarray:
    return (np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0) * 32767.0).astype(np.int16)


def resample(pcm: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Polyphase resampling (anti-aliased) between integer rates."""
    if src_rate == dst_rate:
        return np.asarray(pcm, dtype=np.int16)
    ratio = Fraction(dst_rate, src_rate)
    y = resample_poly(np.asarray(pcm, dtype=np.float64), ratio.numerator, ratio.denominator)
    return np.clip(np.round(y), -32768, 32767).astype(np.int16)


def split_frames(pcm: np.ndarray, n: int, pad: bool = True) -> list[np.ndarray]:
    """Cut ``pcm`` into frames of ``n`` samples; zero-pad the last if ``pad``."""
    pcm = np.asarray(pcm, dtype=np.int16)
    frames = [pcm[i : i + n] for i in range(0, len(pcm), n)]
    if frames and len(frames[-1]) < n:
        if pad:
            frames[-1] = np.concatenate([frames[-1], np.zeros(n - len(frames[-1]), np.int16)])
        else:
            frames.pop()
    return frames


def dbfs(pcm: np.ndarray) -> float:
    """RMS level in dB relative to int16 full scale (−inf-safe: floor −120)."""
    if len(pcm) == 0:
        return -120.0
    rms = math.sqrt(float(np.mean(np.asarray(pcm, dtype=np.float64) ** 2)))
    return 20.0 * math.log10(rms / 32768.0) if rms > 0 else -120.0


def read_wav(path_or_bytes: str | bytes) -> tuple[np.ndarray, int]:
    """Read a mono/stereo 16-bit WAV → (mono int16, sample_rate)."""
    src = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, bytes) else path_or_bytes
    with wave.open(src, "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError("only 16-bit PCM WAV is supported")
        sr = w.getframerate()
        ch = w.getnchannels()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1).astype(np.int16)
    return data.astype(np.int16), sr


def write_wav(path: str, pcm: np.ndarray, sample_rate: int) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm_to_bytes(pcm))
