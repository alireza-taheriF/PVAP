"""Whisper ASR on ONNX Runtime — provided by Ava Voice (``ava_voice.asr``).

The implementation (NumPy log-mel + onnxruntime greedy decoding of the
sherpa-onnx Whisper export) lives in Ava Voice, shared with Ava's voice
intelligibility tests. This adapter adds telephony-quality resampling:
8 kHz call audio is upsampled with a polyphase anti-imaging filter.
"""

from __future__ import annotations

import numpy as np

from pvap.audio.pcm import resample

try:
    from ava_voice.asr.whisper_onnx import WhisperOnnxASR as _AvaWhisper
    from ava_voice.asr.whisper_onnx import log_mel_spectrogram  # noqa: F401  (re-export)
except ImportError as exc:  # pragma: no cover - exercised only without Ava installed
    raise ImportError("WhisperOnnxASR needs Ava Voice: pip install -e '.[ava,asr]'") from exc

__all__ = ["WhisperOnnxASR", "log_mel_spectrogram"]


class WhisperOnnxASR(_AvaWhisper):
    def transcribe(self, pcm: np.ndarray, sample_rate: int) -> str:
        pcm16k = resample(np.asarray(pcm, dtype=np.int16), sample_rate, 16_000)
        return self.transcribe_16k(pcm16k.astype(np.float32) / 32768.0)
