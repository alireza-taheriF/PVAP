"""TTS backends.

* :class:`AvaTTS` — Persian TTS through `Ava Voice
  <https://github.com/alireza-taheriF/Ava-Voice>`_. By default it uses Ava's
  **neural voice**: Ava's normalizer, which verbalizes numbers and dates, then
  Homo-GE2PE G2P, then a Persian VITS model on onnxruntime, resampled to the
  8 kHz call leg. If the models are not downloaded, it falls back to Ava's
  rule-based formant vocoder. That fallback is robotic; the name of the engine
  in use is reported in :attr:`name`.
* :class:`ToneTTS` — **test double**: a deterministic tone burst whose length
  scales with the text. It lets tests check timing and barge-in without any
  TTS installed.
"""

from __future__ import annotations

import logging

import numpy as np

from pvap.audio.pcm import TELEPHONY_RATE, float_to_pcm, resample

__all__ = ["AvaTTS", "ToneTTS"]

log = logging.getLogger("pvap.tts")


class AvaTTS:
    def __init__(
        self,
        sample_rate: int = TELEPHONY_RATE,
        engine: str = "auto",
        voice_dir: str | None = None,
        g2p_dir: str | None = None,
        speed: float = 1.0,
    ) -> None:
        if engine not in ("auto", "neural", "formant"):
            raise ValueError(f"unknown engine {engine!r}")
        self.sample_rate = sample_rate
        self._speed = speed
        self._neural = None
        if engine in ("auto", "neural"):
            try:
                from ava_voice.tts.vits import PersianVITS

                self._neural = PersianVITS(voice_dir, g2p_dir)
            except (ImportError, FileNotFoundError) as exc:
                if engine == "neural":
                    raise
                log.warning("Ava neural voice unavailable (%s); using the formant fallback", exc)
        if self._neural is not None:
            self.name = f"ava-voice neural ({self._neural.name})"
        else:
            from ava_voice.core.speech_plan import plan_speech
            from ava_voice.tts.formant_vocoder import VocoderConfig, synthesize_plan

            self.name = "ava-voice formant fallback (robotic)"
            self._plan, self._synth = plan_speech, synthesize_plan
            self._cfg = VocoderConfig(sample_rate=16_000)

    def synthesize(self, text: str) -> np.ndarray:
        text = text.strip()
        if not text:
            return np.zeros(0, np.int16)
        if self._neural is not None:
            wav = self._neural.synthesize(text, speed=self._speed)
            return resample(float_to_pcm(wav), self._neural.sample_rate, self.sample_rate)
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
