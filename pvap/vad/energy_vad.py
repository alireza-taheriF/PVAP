"""Streaming energy VAD with an adaptive noise floor and endpointing.

Frame-by-frame (20 ms) decisions, designed for the two jobs the dialog loop
needs:

* **barge-in** — detect the caller starting to talk *quickly* (default 100 ms
  of speech) so agent playback can be cut;
* **endpointing** — decide the caller has *finished* (default 600 ms of
  silence) so the turn can be handed to ASR.

Speech is a frame whose level exceeds both an absolute gate and the tracked
noise floor by ``margin_db``. The noise floor is a low percentile of the last
~2 s of frame levels (minimum-statistics style), so it rises to a new, louder
line noise within seconds instead of mistaking it for endless speech, and it
is seeded during a short warm-up at the start of the call. This is a
deliberately simple, deterministic baseline; a neural VAD (e.g. Silero) can
replace :meth:`EnergyVAD.is_speech_frame` without changing the event logic.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from pvap.audio.pcm import dbfs

__all__ = ["VADConfig", "VADEvent", "EnergyVAD"]


class VADEvent(str, Enum):
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"


@dataclass(frozen=True)
class VADConfig:
    frame_ms: int = 20
    margin_db: float = 12.0
    absolute_gate_dbfs: float = -48.0
    start_ms: int = 100  # consecutive speech needed to declare start
    end_ms: int = 600  # consecutive non-speech needed to declare end
    initial_noise_dbfs: float = -65.0
    noise_window_ms: int = 2000  # history used for the noise-floor estimate
    noise_percentile: float = 10.0
    warmup_ms: int = 200  # no decisions until the floor is seeded
    # Speech frames are kept out of the noise history (so talking can't raise
    # the floor) unless "speech" has lasted this long — then it is presumed to
    # be a noise step and the floor is allowed to catch up.
    max_speech_ms: int = 6000
    min_utterance_ms: int = 200  # shorter utterances are flagged via last_was_click


@dataclass
class EnergyVAD:
    config: VADConfig = field(default_factory=VADConfig)

    def __post_init__(self) -> None:
        c = self.config
        self.noise_dbfs = c.initial_noise_dbfs
        self._levels: deque[float] = deque(maxlen=max(1, c.noise_window_ms // c.frame_ms))
        self._warmup_left = c.warmup_ms // c.frame_ms
        self.in_speech = False
        self._run_speech = 0
        self._run_silence = 0
        self._speech_frames = 0
        self.last_speech_ms = 0

    @property
    def trailing_silence_ms(self) -> int:
        """Silence since the last speech frame while an utterance is open (else 0)."""
        return self._run_silence * self.config.frame_ms if self.in_speech else 0

    @property
    def last_was_click(self) -> bool:
        """True if the utterance that just ended is too short to be speech."""
        return self.last_speech_ms < self.config.min_utterance_ms

    def reset(self) -> None:
        self.__post_init__()

    def is_speech_frame(self, level_dbfs: float) -> bool:
        c = self.config
        return level_dbfs > c.absolute_gate_dbfs and level_dbfs > self.noise_dbfs + c.margin_db

    def process(self, frame: np.ndarray) -> VADEvent | None:
        """Feed one frame; return an event when the state flips."""
        c = self.config
        level = max(dbfs(frame), -90.0)
        if self._warmup_left > 0:
            self._warmup_left -= 1
            self._levels.append(level)
            self.noise_dbfs = float(np.percentile(self._levels, c.noise_percentile))
            return None
        speech = self.is_speech_frame(level)
        stuck = self.in_speech and self._speech_frames * c.frame_ms >= c.max_speech_ms
        if not speech or stuck:
            self._levels.append(level)
            if len(self._levels) >= 5:
                self.noise_dbfs = float(np.percentile(self._levels, c.noise_percentile))
        if not speech:
            self._run_silence += 1
            self._run_speech = 0
        else:
            self._run_speech += 1
            self._run_silence = 0

        if not self.in_speech:
            if self._run_speech * c.frame_ms >= c.start_ms:
                self.in_speech = True
                self._speech_frames = self._run_speech
                return VADEvent.SPEECH_START
            return None

        if speech:
            self._speech_frames += 1
        if self._run_silence * c.frame_ms >= c.end_ms:
            self.in_speech = False
            self.last_speech_ms = self._speech_frames * c.frame_ms
            self._speech_frames = 0
            return VADEvent.SPEECH_END
        return None
