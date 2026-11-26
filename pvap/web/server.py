"""Browser calls: talk to the PVAP agent from any phone or laptop browser.

    python -m pvap.web --config pvap.toml            # http://<mac-ip>:8443/?who=<persona>
    python -m pvap.web --config pvap.toml --cert cert.pem --key key.pem   # HTTPS

The page asks for the microphone, streams 16 kHz PCM over a WebSocket, and
plays the agent's audio back, so the call leg is **wideband** (better ASR and a
clearer voice than 8 kHz telephony). The browser cancels echo, and the session's
echo guard keeps the agent's own voice from triggering barge-in.

Browsers only allow the microphone on HTTPS (or ``localhost``). To reach a
phone outside your network, run a tunnel such as
``cloudflared tunnel --url http://localhost:8443``, or use ``--cert/--key`` on
your LAN (see ``tools/make_cert.sh``).

After each call, in the background:

* durable facts about the caller are extracted by the LLM and merged into the
  caller's memory (used in the next call);
* the transcript is saved under ``~/.pvap/memory/calls/<caller>/``;
* the agent's sentences are re-spoken and re-heard, and mispronounced words are
  fixed in Ava's learned lexicon (:mod:`ava_voice.tts.self_check`).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import socket
import ssl
import threading
from pathlib import Path

import numpy as np

from pvap.audio.pcm import bytes_to_pcm, pcm_to_bytes
from pvap.config import Config, build_llm, load_config
from pvap.dialog.session import CallSession, TurnTakingConfig
from pvap.persona import Memory, Persona, PersonaLLM, extract_facts, load_persona
from pvap.web.ws import WebSocket, http_response, read_request

__all__ = ["CallServer", "main"]

log = logging.getLogger("pvap.web")
STATIC = Path(__file__).parent / "static"
RATE = 16_000
FRAME = RATE * 20 // 1000


class CallServer:
    def __init__(self, cfg: Config, asr, tts, llm, personas: dict[str, Persona],
                 learner=None, on_call_end=None) -> None:
        self.cfg, self.asr, self.tts, self.llm = cfg, asr, tts, llm
        self.personas = personas
        self.learner = learner
        self.on_call_end = on_call_end
        self._learn_lock = threading.Lock()

    def persona_for(self, who: str | None) -> Persona:
        if who and who in self.personas:
            return self.personas[who]
        return next(iter(self.personas.values())) if self.personas else Persona()

    # ---------------------------------------------------------------- http
    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            req = await read_request(reader)
            if req is None:
                return
            if req.path == "/ws" and req.is_websocket:
                ws = await WebSocket.accept(req, reader, writer)
                if ws:
                    await self.call(ws, req.query)
                return
            if req.path in ("/", "/index.html", "/call"):
                body = (STATIC / "call.html").read_bytes()
                writer.write(http_response("200 OK", body, "text/html; charset=utf-8"))
            elif req.path == "/api/info":
                p = self.persona_for(req.query.get("who"))
                info = {"agent": p.agent_name, "caller": p.caller_name, "rate": RATE}
                writer.write(http_response("200 OK", json.dumps(info, ensure_ascii=False).encode(),
                                           "application/json; charset=utf-8"))
            else:
                writer.write(http_response("404 Not Found", b"not found"))
            await writer.drain()
        except (ConnectionError, ssl.SSLError):
            pass
        finally:
            try:
                writer.close()
            except Exception:
                pass

    # ---------------------------------------------------------------- call
    async def call(self, ws: WebSocket, query: dict[str, str]) -> None:
        persona = self.persona_for(query.get("who"))
        memory = Memory(persona.caller_id)
        llm = PersonaLLM(self.llm, persona, memory)
        session = CallSession(
            self.asr, llm, self.tts, greeting=llm.greeting(), sample_rate=RATE,
            turn_taking=TurnTakingConfig(endpoint_ms=self.cfg.endpoint_ms), echo_guard=True,
        )
        debug = query.get("debug") == "1"
        await ws.send_text(json.dumps({"type": "hello", "rate": RATE, "agent": persona.agent_name},
                                      ensure_ascii=False))
        log.info("call from browser as caller=%s (%s)", persona.caller_id, persona.caller_name)

        async def inbound():
            pending = np.zeros(0, np.int16)
            while (msg := await ws.recv()) is not None:
                if isinstance(msg, str):
                    if '"hangup"' in msg:
                        return
                    continue
                pending = np.concatenate([pending, bytes_to_pcm(msg)])
                while len(pending) >= FRAME:
                    yield pending[:FRAME]
                    pending = pending[FRAME:]

        async def outbound(frame: np.ndarray) -> None:
            if not ws.closed:
                await ws.send_bytes(pcm_to_bytes(frame))

        async def control(cmd: str) -> None:
            if cmd == "clear" and not ws.closed:
                await ws.send_text('{"type":"clear"}')

        async def captions() -> None:  # debug view: live transcript
            sent: dict[int, tuple[str, str]] = {}
            while not ws.closed:
                await asyncio.sleep(0.3)
                for t in list(session.report.turns):
                    cur = (t.user_text, t.agent_text)
                    if any(cur) and sent.get(t.turn) != cur:
                        sent[t.turn] = cur
                        await ws.send_text(json.dumps({"type": "turn", "id": t.turn, "user": t.user_text,
                                                       "agent": t.agent_text}, ensure_ascii=False))

        cap = asyncio.create_task(captions()) if debug else None
        try:
            report = await session.run(inbound(), outbound, control)
        finally:
            if cap:
                cap.cancel()
            await ws.close()
        turns = [t.to_dict() for t in report.turns]
        log.info("call ended: %d turns", len(turns))
        threading.Thread(target=self._after_call, args=(persona, memory, turns), daemon=True).start()
        if self.on_call_end:
            self.on_call_end(persona, report)

    def _after_call(self, persona: Persona, memory: Memory, turns: list[dict]) -> None:
        """Learn from the call: memory facts + pronunciation (background)."""
        try:
            path = memory.save_transcript(turns)
            log.info("transcript saved: %s", path)
            if self.cfg.learn_memory:
                facts = extract_facts(self.llm, turns, persona)
                new = memory.add(facts)
                memory.save()
                log.info("memory: +%d facts for %s", len(new), persona.caller_id)
            if self.cfg.learn_pronunciation and self.learner is not None:
                sents = [t["agent_text"] for t in turns if t.get("agent_text")]
                with self._learn_lock:
                    rep = self.learner.learn(sents, max_words=6)
                log.info("pronunciation: %s", rep.summary().replace("\n", " | "))
        except Exception as exc:  # learning must never break the server
            log.warning("post-call learning failed: %s: %s", type(exc).__name__, exc)


def _local_ips() -> list[str]:
    ips = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    return sorted(ips) or ["127.0.0.1"]


def load_personas(cfg: Config, extra: list[str]) -> dict[str, Persona]:
    files = [f for f in [cfg.persona, *extra] if f]
    if not files and Path("personas").is_dir():
        files = sorted(str(p) for p in Path("personas").glob("*.toml"))
    out: dict[str, Persona] = {}
    for f in files:
        p = load_persona(f)
        out[p.caller_id] = p
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="PVAP browser-call server")
    ap.add_argument("--config", default=os.environ.get("PVAP_CONFIG", "pvap.toml"))
    ap.add_argument("--persona", action="append", default=[], help="persona TOML (repeatable)")
    ap.add_argument("--host")
    ap.add_argument("--port", type=int)
    ap.add_argument("--cert")
    ap.add_argument("--key")
    ap.add_argument("--rule-llm", action="store_true", help="offline demo without any LLM server")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = load_config(args.config)
    cfg.host = args.host or cfg.host
    cfg.port = args.port or cfg.port
    cfg.cert = args.cert or cfg.cert
    cfg.key = args.key or cfg.key

    from pvap.backends.tts import AvaTTS
    from pvap.backends.whisper_onnx import WhisperOnnxASR

    if not cfg.whisper_dir:
        raise SystemExit("set [asr] whisper_dir in pvap.toml (sherpa-onnx Whisper export)")
    asr = WhisperOnnxASR(cfg.whisper_dir)
    tts = AvaTTS(RATE, voice_dir=cfg.voice_dir or None, g2p_dir=cfg.g2p_dir or None,
                 intensity=cfg.intensity, speed=cfg.speed, pitch_semitones=cfg.pitch_semitones)
    if args.rule_llm:
        from pvap.backends.llm import RuleBasedLLM

        llm = RuleBasedLLM()
    else:
        llm = build_llm(cfg)
    personas = load_personas(cfg, args.persona)
    for p in personas.values():
        tts.warm(PersonaLLM(llm, p).greeting())
    learner = None
    if cfg.learn_pronunciation and getattr(tts, "_neural", None) is not None:
        from ava_voice.asr.whisper_onnx import WhisperOnnxASR as AvaWhisper
        from ava_voice.tts.self_check import PronunciationLearner

        learner = PronunciationLearner(tts._neural, AvaWhisper(cfg.whisper_dir, threads=1))
    server = CallServer(cfg, asr, tts, llm, personas, learner)
    ctx = None
    if cfg.cert and cfg.key:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(cfg.cert, cfg.key)

    async def run() -> None:
        srv = await asyncio.start_server(server.handle, cfg.host, cfg.port, ssl=ctx)
        scheme = "https" if ctx else "http"
        log.info("TTS: %s | LLM: %s", tts.name, llm.name)
        for ip in _local_ips():
            for who in personas or {"": None}:
                log.info("call link: %s://%s:%d/?who=%s", scheme, ip, cfg.port, who)
        if not ctx:
            log.info("note: phones need HTTPS for the microphone — use a tunnel or --cert/--key")
        async with srv:
            await srv.serve_forever()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass
    return 0
