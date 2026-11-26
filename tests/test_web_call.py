"""Browser-call server: WebSocket protocol, persona prompt, memory, reveal."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import struct
import time

import numpy as np

from pvap.backends.tts import ToneTTS
from pvap.config import Config, load_config
from pvap.persona import Memory, Persona, PersonaLLM, extract_facts, load_persona
from pvap.web.server import CallServer
from pvap.web.ws import accept_key

RATE = 16000


def voice(ms: int) -> np.ndarray:
    t = np.arange(RATE * ms // 1000) / RATE
    sig = sum(np.sin(2 * np.pi * f * t) / k for k, f in enumerate((140, 280, 420, 840), 1))
    return (0.2 * 32767 * sig / 2.1 * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t))).astype(np.int16)


class EchoASR:
    name = "test"

    def __init__(self) -> None:
        self.n = 0

    def transcribe(self, pcm, sr):
        self.n += 1
        return ["سلام علی جان، فردا امتحان دارم", "باشه، خداحافظ"][min(self.n - 1, 1)]


class FakeLLM:
    """Records prompts; replies, or returns JSON facts for the extraction prompt."""

    name = "fake"

    def __init__(self) -> None:
        self.prompts: list[list[dict]] = []

    def stream_raw(self, messages):
        self.prompts.append(messages)
        if "JSON" in messages[-1]["content"]:
            yield '["فردا امتحان دارد"]'
            return
        yield "قربونت برم، "
        yield "موفق باشی!"

    stream = stream_raw


def test_persona_file_and_prompt(tmp_path) -> None:
    p = load_persona(os.path.join(os.path.dirname(__file__), "..", "personas", "example.toml"))
    assert p.agent_name == "علی" and p.caller_id == "example" and p.reveal == "end"
    prompt = p.system_prompt(["فردا امتحان دارد"])
    assert "علی" in prompt and "فردا امتحان دارد" in prompt and "همسر" in prompt


def test_reveal_policies() -> None:
    llm = FakeLLM()
    end = PersonaLLM(llm, Persona(agent_name="علی", reveal="end"))
    out = "".join(end.stream([{"role": "user", "content": "سلام"}]))
    assert "هوش مصنوعی" not in out
    out = "".join(end.stream([{"role": "user", "content": "باشه، خداحافظ"}]))
    assert "هوش مصنوعی" in out and "علی" in out
    out = "".join(end.stream([{"role": "user", "content": "خداحافظ"}]))
    assert "هوش مصنوعی" not in out  # only once
    turns = PersonaLLM(llm, Persona(reveal="turns:2"))
    assert "هوش مصنوعی" not in "".join(turns.stream([{"role": "user", "content": "a"}]))
    assert "هوش مصنوعی" in "".join(turns.stream([{"role": "user", "content": "b"}]))
    start = PersonaLLM(llm, Persona(agent_name="علی", reveal="start", greeting="سلام!"))
    assert "هوش مصنوعی" in start.greeting()
    never = PersonaLLM(llm, Persona(reveal="never"))
    assert "هوش مصنوعی" not in "".join(never.stream([{"role": "user", "content": "خداحافظ"}]))


def test_memory_roundtrip_and_fact_extraction(tmp_path) -> None:
    m = Memory("wife", root=tmp_path)
    assert m.add(["فردا امتحان دارد", "فردا امتحان دارد", " قهوه دوست دارد "]) == ["فردا امتحان دارد", "قهوه دوست دارد"]
    m.save()
    assert Memory("wife", root=tmp_path).facts == ["فردا امتحان دارد", "قهوه دوست دارد"]
    facts = extract_facts(FakeLLM(), [{"user_text": "فردا امتحان دارم", "agent_text": "موفق باشی"}], Persona())
    assert facts == ["فردا امتحان دارد"]


def test_config_switch(tmp_path, monkeypatch) -> None:
    f = tmp_path / "pvap.toml"
    f.write_text('[llm]\nprovider = "cloud"\n[llm.cloud]\nmodel = "m1"\napi_key_env = "K"\n', encoding="utf-8")
    monkeypatch.setenv("K", "secret")
    cfg = load_config(f)
    assert cfg.llm_profile().model == "m1" and cfg.llm_profile().api_key == "secret"
    monkeypatch.setenv("PVAP_LLM_PROVIDER", "local")
    assert load_config(f).llm_profile().base_url.startswith("http://localhost:11434")


# --------------------------------------------------------------------------- #
# WebSocket client (minimal, for the test)
# --------------------------------------------------------------------------- #
async def ws_connect(port: int, path: str):
    r, w = await asyncio.open_connection("127.0.0.1", port)
    key = base64.b64encode(os.urandom(16)).decode()
    w.write((f"GET {path} HTTP/1.1\r\nHost: x\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
             f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
    await w.drain()
    head = await r.readuntil(b"\r\n\r\n")
    assert b"101" in head.split(b"\r\n")[0] and accept_key(key).encode() in head
    return r, w


def frame(op: int, payload: bytes) -> bytes:
    mask = os.urandom(4)
    n = len(payload)
    hdr = bytes([0x80 | op]) + (bytes([0x80 | n]) if n < 126 else bytes([0x80 | 126]) + struct.pack("!H", n))
    return hdr + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(payload))


async def read_frame(r):
    b0, b1 = await r.readexactly(2)
    n = b1 & 0x7F
    if n == 126:
        (n,) = struct.unpack("!H", await r.readexactly(2))
    elif n == 127:
        (n,) = struct.unpack("!Q", await r.readexactly(8))
    return b0 & 0x0F, await r.readexactly(n)


def test_browser_call_end_to_end(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PVAP_HOME", str(tmp_path))
    llm = FakeLLM()
    persona = Persona(agent_name="علی", caller_id="wife", caller_name="سایه", reveal="end", greeting="سلام عزیزم!")
    cfg = Config(learn_pronunciation=False)
    server = CallServer(cfg, EchoASR(), ToneTTS(sample_rate=RATE, ms_per_char=10), llm, {"wife": persona})

    async def main():
        srv = await asyncio.start_server(server.handle, "127.0.0.1", 0)
        port = srv.sockets[0].getsockname()[1]
        r, w = await ws_connect(port, "/ws?who=wife&debug=1")
        op, hello = await read_frame(r)
        assert op == 1 and json.loads(hello)["rate"] == RATE
        audio_bytes = 0
        texts = []

        async def reader():
            nonlocal audio_bytes
            try:
                while True:
                    op, data = await read_frame(r)
                    if op == 2:
                        audio_bytes += len(data)
                    elif op == 1:
                        texts.append(json.loads(data))
                    elif op == 8:
                        return
            except asyncio.IncompleteReadError:
                return

        rt = asyncio.create_task(reader())
        silence = np.zeros(320, np.int16)

        async def send(pcm):
            for i in range(0, len(pcm), 320):
                w.write(frame(2, pcm[i:i + 320].tobytes()))
                await w.drain()
                await asyncio.sleep(0.02)

        await send(np.tile(silence, 40))  # listen to the greeting
        await send(voice(900))
        await send(np.tile(silence, 100))  # 2 s: agent answers
        await send(voice(700))
        await send(np.tile(silence, 120))
        w.write(frame(1, b'{"type":"hangup"}'))
        await w.drain()
        await asyncio.wait_for(rt, 5)
        srv.close()
        await srv.wait_closed()
        return audio_bytes, texts

    audio_bytes, texts = asyncio.run(main())
    assert audio_bytes > RATE * 2 * 1.0  # >1 s of agent audio came back
    turns = [t for t in texts if t.get("type") == "turn"]
    assert any("امتحان" in t.get("user", "") for t in turns)
    assert any("هوش مصنوعی" in t.get("agent", "") for t in turns)  # revealed at goodbye
    # the persona system prompt reached the model
    system = llm.prompts[0][0]
    assert system["role"] == "system" and "سایه" in system["content"]
    # after-call learning wrote memory in the background
    for _ in range(50):
        if (tmp_path / "memory" / "wife.json").exists():
            break
        time.sleep(0.1)
    assert Memory("wife", root=tmp_path / "memory").facts == ["فردا امتحان دارد"]
