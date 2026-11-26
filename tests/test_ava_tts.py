"""Integration with Ava Voice (skipped when ava_voice is not installed)."""

import os

import numpy as np
import pytest

from pvap.audio.pcm import dbfs

pytest.importorskip("ava_voice.tts.formant_vocoder")

from pvap.backends.tts import AvaTTS  # noqa: E402


def test_ava_tts_auto_produces_8k_persian_audio_with_numbers() -> None:
    tts = AvaTTS()  # neural if Ava's models are downloaded, formant fallback otherwise
    pcm = tts.synthesize("هزینه‌ی ویزیت ۴۵۰,۰۰۰ تومان است.")
    assert tts.sample_rate == 8000 and pcm.dtype == np.int16
    assert len(pcm) > 8000 * 1.5  # the number was verbalized, so the utterance is long
    assert dbfs(pcm) > -35
    assert len(tts.synthesize("   ")) == 0


def test_formant_engine_is_labelled_robotic() -> None:
    assert "robotic" in AvaTTS(engine="formant").name


@pytest.mark.skipif(not os.environ.get("AVA_TTS_VOICE_DIR"), reason="set AVA_TTS_VOICE_DIR to test the neural voice")
def test_neural_engine() -> None:
    tts = AvaTTS(engine="neural")
    assert tts.name.startswith("ava-voice neural")
    pcm = tts.synthesize("سلام، خوش آمدید.")
    assert 0.6 < len(pcm) / 8000 < 4 and dbfs(pcm) > -35
