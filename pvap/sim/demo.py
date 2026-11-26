"""Offline end-to-end call demo.

Runs a scripted phone call through the full PVAP loop in real time and writes:

* ``<out>/call.wav`` — stereo recording, **left = caller, right = agent**;
* ``<out>/report.json`` — per-turn transcripts, latencies, barge-in reaction.

Both sides speak Persian synthesized by Ava Voice (formant baseline); the
caller's voice is pitch/tempo-shifted so the two are distinguishable. The
caller's transcripts come from :class:`~pvap.backends.asr.ScriptedASR`
unless ``--asr whisper`` is given — say so when you show this demo.

    python -m pvap.sim.demo --out demo_out
    python -m pvap.sim.demo --llm openai --base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os

import numpy as np

from pvap.audio.pcm import TELEPHONY_RATE, resample, write_wav
from pvap.backends.asr import FasterWhisperASR, ScriptedASR
from pvap.backends.llm import OpenAICompatLLM, RuleBasedLLM
from pvap.backends.tts import AvaTTS
from pvap.dialog.session import CallSession
from pvap.sim.caller import CallerStep, simulated_caller

# (caller line, barge-in after N ms of agent audio, or None for polite turn)
SCRIPT: list[tuple[str, int | None]] = [
    ("سلام، می‌خواستم یک نوبت بگیرم.", None),
    ("فردا اگر می‌شود.", None),
    ("هزینه‌ی ویزیت چقدر است؟", None),
    ("ممنون، خداحافظ.", 900),  # interrupts the price answer
]


def caller_voice(tts: AvaTTS, text: str) -> np.ndarray:
    """Agent TTS, shifted ~+15 % in pitch and tempo, so the caller sounds different."""
    pcm = tts.synthesize(text)
    return resample(pcm, 23, 20)


async def _run(args: argparse.Namespace) -> dict:
    tts = AvaTTS(TELEPHONY_RATE)
    if args.asr == "whisper":
        asr = FasterWhisperASR(model=args.whisper_model)
    else:
        asr = ScriptedASR([line for line, _ in SCRIPT])
    if args.llm == "openai":
        llm = OpenAICompatLLM(args.base_url, args.model, api_key=os.environ.get("PVAP_LLM_API_KEY"))
    else:
        llm = RuleBasedLLM()

    session = CallSession(asr, llm, tts, greeting=RuleBasedLLM.GREETING)
    steps = [
        CallerStep(caller_voice(tts, line), pause_ms=500, barge_in_after_ms=barge)
        for line, barge in SCRIPT
    ]
    inbound_log: list[np.ndarray] = []

    async def inbound():
        async for frame in simulated_caller(session, steps):
            inbound_log.append(frame)
            yield frame

    async def outbound(frame: np.ndarray) -> None:
        pass  # a real deployment hands this to the telephony gateway

    report = await session.run(inbound(), outbound)

    os.makedirs(args.out, exist_ok=True)
    caller = np.concatenate(inbound_log)
    agent = session.recording_pcm()
    n = max(len(caller), len(agent))
    stereo = np.zeros((n, 2), np.int16)
    stereo[: len(caller), 0] = caller
    stereo[: len(agent), 1] = agent
    import wave

    with wave.open(os.path.join(args.out, "call.wav"), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(TELEPHONY_RATE)
        w.writeframes(stereo.astype("<i2").tobytes())
    write_wav(os.path.join(args.out, "agent.wav"), agent, TELEPHONY_RATE)
    data = report.to_dict()
    with open(os.path.join(args.out, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    return data


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="demo_out")
    ap.add_argument("--asr", choices=["scripted", "whisper"], default="scripted")
    ap.add_argument("--whisper-model", default="small")
    ap.add_argument("--llm", choices=["rule", "openai"], default="rule")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    args = ap.parse_args(argv)
    data = asyncio.run(_run(args))
    for t in data["turns"]:
        print(
            f"turn {t['turn']}: caller={t['user_text']!r}\n"
            f"         agent={t['agent_text']!r}\n"
            f"         response={t['response_latency_ms']} ms  perceived={t['perceived_latency_ms']} ms"
            f"{'  [interrupted]' if t['interrupted'] else ''}"
        )
    print("barge-in reaction:", data["barge_in_reaction_ms"], "ms")
    print("backends:", data["backends"])
    print(f"wrote {args.out}/call.wav (L=caller, R=agent) and report.json")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
