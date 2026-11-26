"""Backend contracts for the three model stages of a call.

Every backend is *synchronous*: the dialog loop runs it in a worker thread,
so blocking model calls (Whisper, a vLLM HTTP stream, a vocoder) never stall
the 20 ms audio clock. Swapping a model = writing one small class.
"""

from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable

import numpy as np

__all__ = ["ASRBackend", "LLMBackend", "TTSBackend", "Message"]

Message = dict[str, str]  # {"role": "system"|"user"|"assistant", "content": ...}


@runtime_checkable
class ASRBackend(Protocol):
    name: str

    def transcribe(self, pcm: np.ndarray, sample_rate: int) -> str:
        """Return the transcript of one complete caller utterance."""
        ...


@runtime_checkable
class LLMBackend(Protocol):
    name: str

    def stream(self, messages: list[Message]) -> Iterator[str]:
        """Yield the reply incrementally (token or word deltas)."""
        ...


@runtime_checkable
class TTSBackend(Protocol):
    name: str
    sample_rate: int

    def synthesize(self, text: str) -> np.ndarray:
        """Return int16 PCM at :attr:`sample_rate` for one text chunk."""
        ...
