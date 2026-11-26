"""End-to-end call-session tests with fast test doubles (real time, ~10 s total)."""

import asyncio
import time

import numpy as np

from pvap.backends.asr import ScriptedASR
from pvap.backends.llm import RuleBasedLLM
from pvap.backends.tts import ToneTTS
from pvap.dialog.session import CallSession
from pvap.sim.caller import CallerStep, simulated_caller

SR = 8000


def voice(ms: int) -> np.ndarray:
    t = np.arange(SR * ms // 1000) / SR
    sig = sum(np.sin(2 * np.pi * f * t) / k for k, f in enumerate((140, 280, 420, 840), 1))
    return (0.2 * 32767 * sig / 2.1 * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t))).astype(np.int16)


class SlowLLM(RuleBasedLLM):
    """Rule LLM with a fixed time-to-first-token, to check latency accounting."""

    def __init__(self, delay_s: float) -> None:
        self.delay_s = delay_s

    def stream(self, messages):
        time.sleep(self.delay_s)
        yield from super().stream(messages)


def run_call(session, script):
    sent: list[np.ndarray] = []
    controls: list[str] = []

    async def outbound(frame):
        sent.append(frame)

    async def control(cmd):
        controls.append(cmd)

    async def main():
        return await session.run(simulated_caller(session, script), outbound, control)

    report = asyncio.run(main())
    return report, sent, controls


def test_turn_taking_and_latency_accounting() -> None:
    session = CallSession(
        ScriptedASR(["سلام، یک نوبت می‌خواستم", "فردا لطفاً"]),
        SlowLLM(0.15),
        ToneTTS(ms_per_char=8),
        greeting="سلام، بفرمایید.",
    )
    report, sent, controls = run_call(
        session, [CallerStep(voice(900)), CallerStep(voice(700))]
    )
    turns = report.turns
    assert [t.user_text for t in turns] == ["", "سلام، یک نوبت می‌خواستم", "فردا لطفاً"]
    assert "نوبت" in turns[1].agent_text and "ثبت شد" in turns[2].agent_text
    assert report.barge_in_ms == [] and controls == []
    for t in turns[1:]:
        # ≥ the 150 ms model delay, and well under a second of overhead
        assert 150 <= t.response_latency_ms < 1000, t.to_dict()
        assert t.llm_first_token_ms >= 140
    # history alternates user/assistant
    assert [m["role"] for m in session.history] == ["user", "assistant", "user", "assistant"]
    assert len(sent) > 30


def test_barge_in_cuts_agent_audio_fast() -> None:
    long_reply = RuleBasedLLM()
    long_reply._FALLBACK = "این یک پاسخ بسیار طولانی است که باید وسط آن حرف زده شود. " * 4
    session = CallSession(
        ScriptedASR(["چیزی نامفهوم", "ممنون"]),
        long_reply,
        ToneTTS(ms_per_char=15),
    )
    report, sent, controls = run_call(
        session,
        [CallerStep(voice(800)), CallerStep(voice(800), barge_in_after_ms=700)],
    )
    assert controls == ["clear"]
    assert report.turns[0].interrupted
    assert len(report.barge_in_ms) == 1
    # VAD start window (100 ms) + at most ~2 frames of pump latency
    assert report.barge_in_ms[0] <= 160, report.barge_in_ms
    # the interrupted reply was far from finished
    full_reply_frames = len(ToneTTS(ms_per_char=15).synthesize(long_reply._FALLBACK)) // 160
    spoken_before_barge = 700 // 20 + 10
    assert full_reply_frames > 3 * spoken_before_barge
    assert report.turns[1].user_text == "ممنون"
