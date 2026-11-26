import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from pvap.backends.asr import ScriptedASR
from pvap.backends.llm import OpenAICompatLLM, RuleBasedLLM
from pvap.backends.tts import ToneTTS
from pvap.dialog.session import split_sentences


def test_split_sentences_hard_and_soft_breaks() -> None:
    ready, rest = split_sentences("سلام. حال شما چطور است؟ من")
    assert ready == ["سلام.", "حال شما چطور است؟"] and rest == " من" or rest == "من"
    # a short clause before a comma is not cut …
    ready, rest = split_sentences("بله، ")
    assert ready == []
    # … a long one is
    ready, _ = split_sentences("نوبت شما برای فردا صبح، ساعت ده")
    assert ready == ["نوبت شما برای فردا صبح،"]


def test_scripted_asr_returns_in_order() -> None:
    asr = ScriptedASR(["الف", "ب"])
    z = np.zeros(10, np.int16)
    assert [asr.transcribe(z, 8000) for _ in range(3)] == ["الف", "ب", ""]


def test_rule_llm_streams_words() -> None:
    llm = RuleBasedLLM()
    out = "".join(llm.stream([{"role": "user", "content": "یک نوبت می‌خواستم"}]))
    assert "نوبت" in out


def test_tone_tts_length_scales_with_text() -> None:
    tts = ToneTTS(ms_per_char=10)
    assert len(tts.synthesize("abcd")) == 4 * 80


class _FakeSSE(BaseHTTPRequestHandler):
    received: dict = {}

    def do_POST(self):  # noqa: N802
        n = int(self.headers["Content-Length"])
        _FakeSSE.received = {
            "body": json.loads(self.rfile.read(n)),
            "auth": self.headers.get("Authorization"),
            "path": self.path,
        }
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for piece in ["سلام", "، ", "خوش آمدید."]:
            chunk = {"choices": [{"delta": {"content": piece}}]}
            self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
            self.wfile.flush()
        self.wfile.write(b": keep-alive comment\n\n")
        self.wfile.write(b'data: {"choices":[{"delta":{}}]}\n\n')
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *a):  # silence
        pass


@pytest.fixture()
def sse_server():
    srv = HTTPServer(("127.0.0.1", 0), _FakeSSE)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()


def test_openai_compat_llm_streams_sse(sse_server) -> None:
    llm = OpenAICompatLLM(sse_server, model="qwen", api_key="k")
    deltas = list(llm.stream([{"role": "user", "content": "سلام"}]))
    assert "".join(deltas) == "سلام، خوش آمدید."
    req = _FakeSSE.received
    assert req["path"] == "/v1/chat/completions"
    assert req["auth"] == "Bearer k"
    assert req["body"]["stream"] is True
    assert req["body"]["messages"][0]["role"] == "system"
    assert req["body"]["messages"][-1] == {"role": "user", "content": "سلام"}
