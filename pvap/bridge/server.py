"""Dialog-engine server: accepts calls from the Go gateway over TCP.

    python -m pvap.bridge.server --port 9300 --llm rule --asr scripted

Each connection becomes one :class:`~pvap.dialog.session.CallSession`.
Backends are created per call by a factory, so stateful backends (the
scripted ASR) never leak between calls.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from typing import Callable

import numpy as np

from pvap.audio.pcm import bytes_to_pcm, pcm_to_bytes
from pvap.bridge.protocol import MsgType, encode, read_message
from pvap.dialog.session import CallReport, CallSession

__all__ = ["serve", "handle_call"]

log = logging.getLogger("pvap.bridge")
SessionFactory = Callable[[dict], CallSession]


async def handle_call(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    factory: SessionFactory,
    on_report: Callable[[dict, CallReport], None] | None = None,
) -> None:
    first = await read_message(reader)
    if first is None or first[0] is not MsgType.START:
        writer.close()
        return
    meta = json.loads(first[1] or b"{}")
    if int(meta.get("sample_rate", 8000)) != 8000:
        log.warning("unsupported sample rate %s", meta.get("sample_rate"))
        writer.close()
        return
    session = factory(meta)
    log.info("call %s from %s", meta.get("call_id"), meta.get("from"))
    lock = asyncio.Lock()

    async def send(raw: bytes) -> None:
        async with lock:
            writer.write(raw)
            await writer.drain()

    async def inbound():
        while (msg := await read_message(reader)) is not None:
            kind, payload = msg
            if kind is MsgType.AUDIO:
                yield bytes_to_pcm(payload)
            elif kind is MsgType.HANGUP:
                return

    async def outbound(frame: np.ndarray) -> None:
        await send(encode(MsgType.AUDIO, pcm_to_bytes(frame)))

    async def control(cmd: str) -> None:
        if cmd == "clear":
            await send(encode(MsgType.CLEAR))

    try:
        report = await session.run(inbound(), outbound, control)
        if on_report:
            on_report(meta, report)
        try:
            await send(encode(MsgType.HANGUP))
        except (ConnectionError, RuntimeError):
            pass
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except ConnectionError:
            pass


def default_factory(args: argparse.Namespace) -> SessionFactory:
    from pvap.backends.asr import FasterWhisperASR, ScriptedASR
    from pvap.backends.llm import OpenAICompatLLM, RuleBasedLLM
    from pvap.backends.tts import AvaTTS, ToneTTS

    if args.tts == "tone":
        tts = ToneTTS()
    else:
        tts = AvaTTS(engine="formant" if args.tts == "ava-formant" else "auto",
                     voice_dir=args.voice_dir, g2p_dir=args.g2p_dir)
    log.info("TTS: %s", tts.name)
    whisper = None
    if args.asr == "whisper-onnx":
        from pvap.backends.whisper_onnx import WhisperOnnxASR

        whisper = WhisperOnnxASR(args.whisper_dir, threads=args.asr_threads)
    elif args.asr == "faster-whisper":
        whisper = FasterWhisperASR(model=args.whisper_model)
    script = [s for s in (args.script or "").split("|") if s]

    def make(meta: dict) -> CallSession:
        asr = whisper or ScriptedASR(script)
        llm = (
            OpenAICompatLLM(args.base_url, args.model, api_key=os.environ.get("PVAP_LLM_API_KEY"))
            if args.llm == "openai"
            else RuleBasedLLM()
        )
        return CallSession(asr, llm, tts, greeting=args.greeting)

    return make


async def serve(host: str, port: int, factory: SessionFactory, on_report=None) -> asyncio.base_events.Server:
    return await asyncio.start_server(lambda r, w: handle_call(r, w, factory, on_report), host, port)


def main(argv: list[str] | None = None) -> int:
    from pvap.backends.llm import RuleBasedLLM

    ap = argparse.ArgumentParser(description="PVAP dialog engine (TCP bridge server)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9300)
    ap.add_argument("--asr", choices=["scripted", "whisper-onnx", "faster-whisper"], default="scripted")
    ap.add_argument("--whisper-dir", help="sherpa-onnx Whisper export (for --asr whisper-onnx)")
    ap.add_argument("--asr-threads", type=int, default=4)
    ap.add_argument("--script", help="scripted ASR transcripts, '|'-separated")
    ap.add_argument("--whisper-model", default="small")
    ap.add_argument("--llm", choices=["rule", "openai"], default="rule")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--tts", choices=["ava", "ava-formant", "tone"], default="ava")
    ap.add_argument("--voice-dir", help="Ava VITS voice directory (default: Ava's model cache)")
    ap.add_argument("--g2p-dir", help="Homo-GE2PE directory (default: Ava's model cache)")
    ap.add_argument("--greeting", default=RuleBasedLLM.GREETING)
    ap.add_argument("--reports", help="directory to write per-call report JSON")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    def on_report(meta: dict, report: CallReport) -> None:
        data = report.to_dict()
        for t in data["turns"]:
            log.info("turn %s user=%r agent=%r latency=%s ms", t["turn"], t["user_text"], t["agent_text"], t["response_latency_ms"])
        if args.reports:
            os.makedirs(args.reports, exist_ok=True)
            path = os.path.join(args.reports, f"{meta.get('call_id', 'call')}.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"meta": meta, **data}, fh, ensure_ascii=False, indent=2)

    async def run() -> None:
        server = await serve(args.host, args.port, default_factory(args), on_report)
        log.info("listening on %s:%s", args.host, args.port)
        async with server:
            await server.serve_forever()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
