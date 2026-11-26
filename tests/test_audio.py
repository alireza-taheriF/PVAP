import numpy as np
import pytest

from pvap.audio.g711 import alaw_decode, alaw_encode, ulaw_decode, ulaw_encode
from pvap.audio.pcm import dbfs, read_wav, resample, split_frames, write_wav

FULL = np.arange(-32768, 32768, dtype=np.int32).astype(np.int16)


def test_ulaw_reference_values() -> None:
    # Known G.711 μ-law codewords (ITU-T G.711 / Sun g711.c).
    pcm = np.array([0, 1000, -1000, 32767, -32768], np.int16)
    assert ulaw_encode(pcm).tolist() == [0xFF, 0xCE, 0x4E, 0x80, 0x00]
    assert ulaw_decode(np.array([0xFF, 0x80, 0x00], np.uint8)).tolist() == [0, 32124, -32124]


def test_alaw_reference_values() -> None:
    pcm = np.array([0, 32767, -32768], np.int16)
    assert alaw_encode(pcm).tolist() == [0xD5, 0xAA, 0x2A]
    assert alaw_decode(np.array([0xD5, 0xAA, 0x2A], np.uint8)).tolist() == [8, 32256, -32256]


@pytest.mark.parametrize("enc,dec", [(ulaw_encode, ulaw_decode), (alaw_encode, alaw_decode)])
def test_companding_is_monotonic_and_relative_error_small(enc, dec) -> None:
    y = dec(enc(FULL)).astype(np.int64)
    assert np.all(np.diff(y) >= 0)
    big = np.abs(FULL) > 1000
    rel = np.abs(y[big] - FULL[big]) / np.abs(FULL[big].astype(np.int64))
    assert rel.max() < 0.035


@pytest.mark.parametrize("enc,dec", [(ulaw_encode, ulaw_decode), (alaw_encode, alaw_decode)])
def test_codeword_roundtrip(enc, dec) -> None:
    codes = np.arange(256, dtype=np.uint8)
    same = enc(dec(codes)) == codes
    # μ-law has two zeros (0x7F and 0xFF); everything else is exact.
    assert same.sum() >= 255


def test_resample_preserves_tone_frequency() -> None:
    sr = 8000
    t = np.arange(sr) / sr
    x = (0.5 * 32767 * np.sin(2 * np.pi * 440 * t)).astype(np.int16)
    y = resample(x, 8000, 16000)
    assert len(y) == 16000
    spec = np.abs(np.fft.rfft(y.astype(float)))
    assert abs(np.argmax(spec) * 16000 / len(y) - 440) < 2


def test_split_frames_pads_last() -> None:
    frames = split_frames(np.ones(330, np.int16), 160)
    assert [len(f) for f in frames] == [160, 160, 160]
    assert frames[-1][10:].sum() == 0


def test_dbfs() -> None:
    assert dbfs(np.zeros(160, np.int16)) == -120.0
    full_sine = (32767 * np.sin(np.linspace(0, 20 * np.pi, 1600))).astype(np.int16)
    assert abs(dbfs(full_sine) - (-3.01)) < 0.1


def test_wav_roundtrip(tmp_path) -> None:
    x = (np.random.default_rng(0).standard_normal(800) * 1000).astype(np.int16)
    p = tmp_path / "a.wav"
    write_wav(str(p), x, 8000)
    y, sr = read_wav(str(p))
    assert sr == 8000 and np.array_equal(x, y)
