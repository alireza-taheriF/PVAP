"""ASR backends.

* :class:`FasterWhisperASR` — the real path: Whisper (CTranslate2) on CPU or
  GPU, Persian forced. Needs ``pip install faster-whisper`` and a model
  download; it is **not exercised by this repository's test suite** (the
  build sandbox has no model weights).
* :class:`ScriptedASR` — **simulation only**. It does no recognition: it
  returns pre-written transcripts in order, one per detected utterance, so the
  telephony, VAD, barge-in and latency machinery can be tested end to end
  without a speech model. Its outputs are labelled as scripted in call logs.
"""

from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np

from pvap.audio.pcm import resample

__all__ = ["ScriptedASR", "FasterWhisperASR"]


class ScriptedASR:
    name = "scripted (simulation — not speech recognition)"

    def __init__(self, transcripts: Iterable[str]) -> None:
        self._queue: deque[str] = deque(transcripts)

    def transcribe(self, pcm: np.ndarray, sample_rate: int) -> str:
        return self._queue.popleft() if self._queue else ""


class FasterWhisperASR:
    """Whisper via ``faster-whisper``. Model choice is left to the deployer
    (e.g. ``small``/``medium``, or a Persian fine-tune converted to CT2)."""

    def __init__(
        self,
        model: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "fa",
        beam_size: int = 1,
    ) -> None:
        from faster_whisper import WhisperModel  # lazy: optional dependency

        self.name = f"faster-whisper:{model}"
        self._model = WhisperModel(model, device=device, compute_type=compute_type)
        self._language = language
        self._beam = beam_size

    def transcribe(self, pcm: np.ndarray, sample_rate: int) -> str:
        audio = resample(pcm, sample_rate, 16_000).astype(np.float32) / 32768.0
        segments, _ = self._model.transcribe(
            audio, language=self._language, beam_size=self._beam, vad_filter=False
        )
        return " ".join(s.text.strip() for s in segments).strip()
