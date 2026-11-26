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


class LengthASR:
    """Reports how much audio it was given (ms), so tests can see what ASR heard."""

    name = "length (test double)"

    def __init__(self, delay_s: float = 0.0) -> None:
        self.calls: list[int] = []
        self.delay_s = delay_s

    def transcribe(self, pcm, sample_rate):
        time.sleep(self.delay_s)
        self.calls.append(len(pcm) * 1000 // sample_rate)
        return f"صدا {self.calls[-1]}"


class HumanTTS(ToneTTS):
    """Tone TTS plus the optional breath/backchannel hooks AvaTTS provides."""

    backchannel_keys = ("bale", "ahan", "ohom", "khob")

    def breath(self):
        return (np.random.default_rng(0).standard_normal(8 * 300) * 200).astype(np.int16)

    def backchannel(self, key):
        return self.synthesize("بله")


def test_speculative_asr_is_used_and_discarded_when_caller_continues() -> None:
    from pvap.sim.caller import CallerStep

    asr = LengthASR(delay_s=0.3)
    session = CallSession(asr, RuleBasedLLM(), ToneTTS(ms_per_char=5))
    # one utterance with a 300 ms pause inside: longer than the speculative
    # trigger (200 ms), shorter than the endpoint (500 ms)
    gap = np.zeros(8 * 300, np.int16)
    utter = np.concatenate([voice(700), gap, voice(700)])
    report, _, _ = run_call(session, [CallerStep(utter)])
    assert len(report.turns) == 1
    t = report.turns[0]
    assert t.asr_speculative
    discarded = [e for e in report.events if e[1] == "asr_speculative_discarded"]
    assert len(discarded) == 1
    # the transcript used covers the whole utterance (both halves)
    assert int(t.user_text.split()[1]) >= 1700
    # speculation started 300 ms before the endpoint, so ASR "finished" fast
    assert t.asr_ms < 150, t.to_dict()


def test_reaction_fills_the_wait_before_a_slow_reply() -> None:
    session = CallSession(ScriptedASR(["سلام"]), SlowLLM(1.0), HumanTTS(ms_per_char=8), seed=1)
    report, _, _ = run_call(session, [CallerStep(voice(800))])
    t = report.turns[0]
    assert t.reaction in HumanTTS.backchannel_keys
    assert t.response_latency_ms < 400 < 1000 < t.reply_latency_ms, t.to_dict()


def test_no_reaction_when_reply_is_fast() -> None:
    session = CallSession(ScriptedASR(["سلام"]), RuleBasedLLM(), HumanTTS(ms_per_char=8))
    session._expected_reply_ms = 200.0
    report, _, _ = run_call(session, [CallerStep(voice(800))])
    t = report.turns[0]
    assert t.reaction == ""
    # a breath precedes the reply words
    assert t.first_audio_t < t.first_speech_t


def test_words_added_while_thinking_continue_the_turn() -> None:
    asr = LengthASR()
    session = CallSession(asr, SlowLLM(1.5), ToneTTS(ms_per_char=5))
    # second utterance starts ~0.7 s after the first ends: the agent is still
    # thinking (1.5 s LLM), so this must merge, not count as a barge-in
    first = CallerStep(voice(800))

    async def main():
        async def caller():
            from pvap.sim.caller import simulated_caller as sc
            async for f in sc(session, [first], tail_ms=0, max_wait_s=0.7):
                yield f
            for f in [np.zeros(160, np.int16)] * 5:  # ~0.1 s more silence, then talk again
                await asyncio.sleep(0.02)
                yield f
            async for f in sc(session, [CallerStep(voice(800), pause_ms=0)], max_wait_s=0.0):
                yield f
            for _ in range(150):
                await asyncio.sleep(0.02)
                yield np.zeros(160, np.int16)

        async def out(frame):
            pass

        return await session.run(caller(), out)

    report = asyncio.run(main())
    kinds = [k for _, k, _ in report.events]
    assert "continuation" in kinds and "barge_in" not in kinds
    assert report.barge_in_ms == []
    final = report.turns[-1]
    assert not final.interrupted
    assert int(final.user_text.split()[1]) >= 1600  # ASR heard both parts
