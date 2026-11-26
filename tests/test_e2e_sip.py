"""Full-stack test: real SIP/RTP call → Go gateway → TCP bridge → dialog engine.

Builds ``gateway/`` with the local Go toolchain (skipped if Go is absent),
runs it as a subprocess, serves the Python engine in-process, and places a
call with the test UAC. Uses fast test doubles (tone TTS, scripted ASR).
"""

from __future__ import annotations

import asyncio
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

from pvap.audio.pcm import dbfs
from pvap.backends.asr import ScriptedASR
from pvap.backends.llm import RuleBasedLLM
from pvap.backends.tts import ToneTTS
from pvap.bridge.server import serve
from pvap.dialog.session import CallSession
from pvap.sim.uac import UACStep, place_call

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(shutil.which("go") is None, reason="Go toolchain not installed")


def free_port(kind=socket.SOCK_DGRAM) -> int:
    s = socket.socket(socket.AF_INET, kind)
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def voice(ms: int) -> np.ndarray:
    t = np.arange(8 * ms) / 8000
    sig = sum(np.sin(2 * np.pi * f * t) / k for k, f in enumerate((140, 280, 420, 840), 1))
    return (0.2 * 32767 * sig / 2.1 * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t))).astype(np.int16)


@pytest.fixture(scope="module")
def gateway_bin(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("bin") / "pvap-gateway"
    subprocess.run(["go", "build", "-o", str(out), "./cmd/pvap-gateway"], cwd=ROOT / "gateway", check=True)
    return out


def test_sip_call_end_to_end(gateway_bin: Path) -> None:
    sip_port, engine_port = free_port(), free_port(socket.SOCK_STREAM)
    reports = []

    def factory(meta: dict) -> CallSession:
        llm = RuleBasedLLM()
        llm._FALLBACK = "این یک پاسخ طولانی است که تماس‌گیرنده وسط آن صحبت می‌کند. " * 3
        return CallSession(
            ScriptedASR(["یک نوبت می‌خواستم", "یک سؤال دیگر", "ممنون"]),
            llm,
            ToneTTS(ms_per_char=12),
            greeting="سلام، بفرمایید.",
        )

    async def main():
        server = await serve("127.0.0.1", engine_port, factory, lambda m, r: reports.append((m, r)))
        proc = subprocess.Popen(
            [str(gateway_bin), "-sip", f"127.0.0.1:{sip_port}", "-public-ip", "127.0.0.1",
             "-engine", f"127.0.0.1:{engine_port}", "-rtp-min", "32000", "-rtp-max", "32999"],
            stderr=subprocess.PIPE,
        )
        try:
            await asyncio.sleep(0.4)
            res = await place_call(
                ("127.0.0.1", sip_port),
                [
                    UACStep(voice(800)),
                    UACStep(voice(800)),
                    UACStep(voice(700), barge_in_after_ms=600),
                ],
            )
            for _ in range(50):
                if reports:
                    break
                await asyncio.sleep(0.1)
        finally:
            proc.terminate()
            proc.wait(timeout=5)
            server.close()
            await server.wait_closed()
        return res, proc.stderr.read().decode()

    res, gw_log = asyncio.run(main())
    assert res.status == 200, gw_log
    assert res.rtp_packets > 100, gw_log
    agent = np.concatenate(res.received)
    assert dbfs(agent) > -40  # the caller actually heard the agent
    assert reports, gw_log
    meta, report = reports[0]
    assert meta["codec"] == "PCMU" and meta["sample_rate"] == 8000
    users = [t.user_text for t in report.turns]
    assert users == ["", "یک نوبت می‌خواستم", "یک سؤال دیگر", "ممنون"], report.to_dict()
    assert "نوبت" in report.turns[1].agent_text
    assert report.turns[2].interrupted
    assert len(report.barge_in_ms) == 1 and report.barge_in_ms[0] < 200
    for t in report.turns[1:]:
        assert t.response_latency_ms is not None and t.response_latency_ms < 500
