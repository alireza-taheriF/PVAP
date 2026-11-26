"""Integration with Ava Voice (skipped when ava_voice is not installed)."""

import numpy as np
import pytest

from pvap.audio.pcm import dbfs

pytest.importorskip("ava_voice.tts.formant_vocoder")

from pvap.backends.tts import AvaTTS  # noqa: E402


def test_ava_tts_produces_8k_persian_audio_with_numbers() -> None:
    tts = AvaTTS()
    pcm = tts.synthesize("هزینه‌ی ویزیت ۴۵۰,۰۰۰ تومان است.")
    assert tts.sample_rate == 8000
    assert pcm.dtype == np.int16
    # number is verbalized, so the utterance is long: well over a second
    assert len(pcm) > 8000 * 1.5
    assert dbfs(pcm) > -35
    assert len(tts.synthesize("   ")) == 0
