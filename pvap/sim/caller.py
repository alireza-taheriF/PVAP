"""Simulated caller: a paced inbound audio stream that follows a script.

The simulator plays the *caller's* side of a call into a
:class:`~pvap.dialog.session.CallSession` in real time (one 20 ms frame per
tick), with low-level line noise between utterances. Each scripted step
either waits for the agent to finish and then talks (polite turn-taking), or
starts talking a fixed time after the agent starts — a deliberate
**barge-in**.

Caller audio is whatever PCM you give it — a recorded WAV, or synthetic
speech from :class:`~pvap.backends.tts.AvaTTS`.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import AsyncIterator

import numpy as np

from pvap.audio.pcm import split_frames
from pvap.dialog.session import CallSession

__all__ = ["CallerStep", "simulated_caller"]


@dataclass(frozen=True)
class CallerStep:
    audio: np.ndarray  # int16 at the call rate
    pause_ms: int = 400  # wait this long after the agent goes quiet …
    barge_in_after_ms: int | None = None  # … or interrupt this long after agent audio starts


async def simulated_caller(
    session: CallSession,
    script: list[CallerStep],
    *,
    noise_dbfs: float = -62.0,
    tail_ms: int = 1500,
    max_wait_s: float = 30.0,
    seed: int = 0,
) -> AsyncIterator[np.ndarray]:
    n, sr, period = session.n, session.sr, session.frame_ms / 1000.0
    rng = np.random.default_rng(seed)
    noise_amp = 32768.0 * 10 ** (noise_dbfs / 20.0)
    loop = asyncio.get_running_loop()
    next_t = loop.time()

    def noise() -> np.ndarray:
        return (rng.standard_normal(n) * noise_amp).astype(np.int16)

    async def tick(frame: np.ndarray):
        nonlocal next_t
        next_t += period
        await asyncio.sleep(max(0.0, next_t - loop.time()))
        return frame

    for idx, step in enumerate(script):
        waited = 0.0
        # politeness: after we speak, wait until the agent has replied
        expect_agent = idx > 0 or session.greeting is not None
        if step.barge_in_after_ms is not None:
            # wait for agent audio to start, then let it play for a while
            while not session._out and waited < max_wait_s:
                yield await tick(noise())
                waited += period
            for _ in range(step.barge_in_after_ms // session.frame_ms):
                yield await tick(noise())
        else:
            quiet = 0
            seen_agent = not expect_agent
            while (not seen_agent or quiet * session.frame_ms < step.pause_ms) and waited < max_wait_s:
                if session.agent_active:
                    seen_agent, quiet = True, 0
                else:
                    quiet += 1
                yield await tick(noise())
                waited += period
        for frame in split_frames(step.audio, n):
            yield await tick((frame.astype(np.int32) + noise()).clip(-32768, 32767).astype(np.int16))

    # stay on the line until the agent has answered the last utterance, then a short tail
    waited, quiet, seen_agent = 0.0, 0, False
    while (not seen_agent or quiet * session.frame_ms < 300) and waited < max_wait_s:
        if session.agent_active:
            seen_agent, quiet = True, 0
        else:
            quiet += 1
        yield await tick(noise())
        waited += period
    for _ in range(tail_ms // session.frame_ms):
        yield await tick(noise())
