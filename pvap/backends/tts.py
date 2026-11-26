"""TTS backends.

* :class:`AvaTTS` — Persian TTS through `Ava Voice
  <https://github.com/alireza-taheriF/Ava-Voice>`_. By default it uses Ava's
  **expressive neural voice**. Ava's normalizer verbalizes numbers and dates,
  Homo-GE2PE produces the phonemes, and a Persian VITS model speaks them. Ava's
  emotion engine then sets tempo, pitch, intonation contour, loudness and
  breathing for each sentence. The audio is resampled to the 8 kHz call leg.
  It also exposes :meth:`AvaTTS.breath` and :meth:`AvaTTS.backchannel`, which
  the call session uses to react like a person. If the models are not
  downloaded, it falls back to Ava's rule-based formant vocoder. That fallback
  is robotic; the engine in use is reported in :attr:`name`.
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
    backchannel_keys: tuple[str, ...] = ()

    def __init__(
        self,
        sample_rate: int = TELEPHONY_RATE,
        engine: str = "auto",
        voice_dir: str | None = None,
        g2p_dir: str | None = None,
        speed: float = 1.0,
        expressive: bool = True,
        intensity: float = 1.0,
        seed: int | None = None,
        pitch_semitones: float = 0.0,
    ) -> None:
        if engine not in ("auto", "neural", "formant"):
            raise ValueError(f"unknown engine {engine!r}")
        self.sample_rate = sample_rate
        self._speed = speed
        self._phrase_cache: dict[str, np.ndarray] = {}
        self._neural = None
        self._expr = None
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
            if expressive:
                from ava_voice.tts.expressive import BACKCHANNELS, ExpressiveVoice

                self._expr = ExpressiveVoice(self._neural, intensity=intensity, seed=seed,
                                             speed=speed, pitch_semitones=pitch_semitones)
                self.name += f", expressive ×{intensity:g}, speed {speed:g}, pitch {pitch_semitones:+g} st"
                self.backchannel_keys = tuple(BACKCHANNELS)
                self._bc_cache: dict[str, np.ndarray] = {}
                for key in self.backchannel_keys:  # warm: reactions must be instant
                    self.backchannel(key)
        else:
            from ava_voice.core.speech_plan import plan_speech
            from ava_voice.tts.formant_vocoder import VocoderConfig, synthesize_plan

            self.name = "ava-voice formant fallback (robotic)"
            self._plan, self._synth = plan_speech, synthesize_plan
            self._cfg = VocoderConfig(sample_rate=16_000)

    def _to_call(self, wav: np.ndarray) -> np.ndarray:
        return resample(float_to_pcm(wav), self._neural.sample_rate, self.sample_rate)

    def synthesize(self, text: str) -> np.ndarray:
        text = text.strip()
        if not text:
            return np.zeros(0, np.int16)
        hit = self._phrase_cache.get(text)
        if hit is not None:
            return hit
        pcm = self._synthesize(text)
        if len(self._phrase_cache) < 256:  # greetings and stock phrases repeat
            self._phrase_cache[text] = pcm
        return pcm

    def warm(self, *texts: str) -> None:
        """Pre-render phrases (e.g. the greeting) so they play instantly."""
        for t in texts:
            if t:
                self.synthesize(t)

    def _synthesize(self, text: str) -> np.ndarray:
        if self._expr is not None:
            # the call session decides about the breath before a reply
            return self._to_call(self._expr.render(text, leading_breath=False))
        if self._neural is not None:
            return self._to_call(self._neural.synthesize(text, speed=self._speed))
        samples = np.asarray(self._synth(self._plan(text), self._cfg), dtype=np.float64)
        return resample(float_to_pcm(samples), 16_000, self.sample_rate)

    def breath(self, kind: str = "inhale") -> np.ndarray:
        """A fresh, quiet human breath (expressive neural voice only)."""
        if self._expr is None:
            return np.zeros(0, np.int16)
        return self._to_call(self._expr.breath(kind))

    def backchannel(self, key: str) -> np.ndarray:
        """Cached "بله،" / "آهان،" / "اوهوم،" / "خب،" / "باشه،"."""
        if self._expr is None:
            return np.zeros(0, np.int16)
        hit = self._bc_cache.get(key)
        if hit is None:
            hit = self._bc_cache[key] = self._to_call(self._expr.backchannel(key))
        return hit


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
