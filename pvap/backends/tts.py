"""TTS backends.

* :class:`AvaTTS` — Persian TTS through `Ava Voice
  <https://github.com/alireza-taheriF/Ava-Voice>`_: Ava's normalizer
  (including number/date verbalization) → speech plan → formant vocoder
  baseline, resampled to the 8 kHz call leg. The voice is robotic today; when
  Ava gets a neural acoustic model, PVAP picks it up with no change here.
* :class:`ToneTTS` — **test double**: a deterministic tone burst whose length
  scales with the text. It lets tests check timing and barge-in without any
  TTS installed.
"""

from __future__ import annotations

import numpy as np

from pvap.audio.pcm import TELEPHONY_RATE, float_to_pcm, resample

__all__ = ["AvaTTS", "ToneTTS"]


class AvaTTS:
    def __init__(self, sample_rate: int = TELEPHONY_RATE) -> None:
        from ava_voice.core.speech_plan import plan_speech  # optional dependency
        from ava_voice.tts.formant_vocoder import VocoderConfig, synthesize_plan

        self.name = "ava-voice formant baseline"
        self.sample_rate = sample_rate
        self._plan = plan_speech
        self._synth = synthesize_plan
        self._cfg = VocoderConfig(sample_rate=16_000)

    def synthesize(self, text: str) -> np.ndarray:
        text = text.strip()
        if not text:
            return np.zeros(0, np.int16)
        samples = np.asarray(self._synth(self._plan(text), self._cfg), dtype=np.float64)
        return resample(float_to_pcm(samples), 16_000, self.sample_rate)


class ToneTTS:
    name = "tone (test double)"

    def __init__(self, sample_rate: int = TELEPHONY_RATE, ms_per_char: float = 25.0, hz: float = 440.0) -> None:
        self.sample_rate = sample_rate
        self._ms_per_char = ms_per_char
        self._hz = hz

    def synthesize(self, text: str) -> np.ndarray:
        n = int(len(text.strip()) * self._ms_per_char * self.sample_rate / 1000)
        t = np.arange(n) / self.sample_rate
        return float_to_pcm(0.3 * np.sin(2 * np.pi * self._hz * t))
