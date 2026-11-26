"""Test SIP caller (UAC): places a real SIP/RTP call to the PVAP gateway.

It sends INVITE with a PCMU offer, ACKs the 200, streams the caller's
utterances as paced RTP, records everything the agent sends back, and hangs
up with BYE. Turn-taking is driven by listening to the agent's RTP energy:
a polite step waits until the agent has spoken and then gone quiet; a
barge-in step starts talking a fixed time after the agent starts.

    python -m pvap.sim.uac --gateway 127.0.0.1:5060 --out call.wav \
        "سلام، یک نوبت می‌خواستم" "فردا لطفاً" "ممنون"

Utterances are synthesized with Ava Voice (pitch-shifted) unless WAV paths
are given. This is a test tool, not a softphone.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import socket
import time
from dataclasses import dataclass, field

import numpy as np

from pvap.audio.g711 import ulaw_decode, ulaw_encode
from pvap.audio.pcm import dbfs, read_wav, resample, split_frames

__all__ = ["UACStep", "UACResult", "place_call"]


@dataclass(frozen=True)
class UACStep:
    audio: np.ndarray  # int16, 8 kHz
    pause_ms: int = 500
    barge_in_after_ms: int | None = None


@dataclass
class UACResult:
    status: int
    sent: list[np.ndarray] = field(default_factory=list)  # caller audio, one frame per tick
    received: list[np.ndarray] = field(default_factory=list)  # agent audio, one frame per RTP packet
    remote_bye: bool = False
    rtp_packets: int = 0

    def stereo(self) -> np.ndarray:
        a = np.concatenate(self.sent) if self.sent else np.zeros(0, np.int16)
        b = np.concatenate(self.received) if self.received else np.zeros(0, np.int16)
        n = max(len(a), len(b))
        out = np.zeros((n, 2), np.int16)
        out[: len(a), 0], out[: len(b), 1] = a, b
        return out


def _parse(data: bytes) -> tuple[str, dict[str, str], bytes]:
    head, _, body = data.partition(b"\r\n\r\n")
    lines = head.decode("utf-8", "replace").split("\r\n")
    headers: dict[str, str] = {}
    for ln in lines[1:]:
        k, _, v = ln.partition(":")
        headers.setdefault(k.strip().lower(), v.strip())
    return lines[0], headers, body


async def place_call(
    gateway: tuple[str, int],
    steps: list[UACStep],
    *,
    local_ip: str = "127.0.0.1",
    speech_dbfs: float = -45.0,
    tail_ms: int = 1500,
    max_wait_s: float = 30.0,
    min_reply_ms: int = 1200,
) -> UACResult:
    loop = asyncio.get_running_loop()
    sip = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sip.bind((local_ip, 0))
    sip.setblocking(False)
    rtp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rtp.bind((local_ip, 0))
    rtp.setblocking(False)
    sip_port, rtp_port = sip.getsockname()[1], rtp.getsockname()[1]
    call_id = f"{random.getrandbits(48):x}@pvap-uac"
    tag = f"{random.getrandbits(32):x}"
    sdp = (
        f"v=0\r\no=uac 1 1 IN IP4 {local_ip}\r\ns=-\r\nc=IN IP4 {local_ip}\r\nt=0 0\r\n"
        f"m=audio {rtp_port} RTP/AVP 0\r\na=ptime:20\r\n"
    )

    def request(method: str, cseq: int, to_tag: str = "", body: str = "") -> bytes:
        to = f"<sip:agent@{gateway[0]}>" + (f";tag={to_tag}" if to_tag else "")
        ctype = "Content-Type: application/sdp\r\n" if body else ""
        return (
            f"{method} sip:agent@{gateway[0]}:{gateway[1]} SIP/2.0\r\n"
            f"Via: SIP/2.0/UDP {local_ip}:{sip_port};branch=z9hG4bK{random.getrandbits(40):x}\r\n"
            f"Max-Forwards: 70\r\nFrom: <sip:caller@{local_ip}>;tag={tag}\r\nTo: {to}\r\n"
            f"Call-ID: {call_id}\r\nCSeq: {cseq} {method}\r\n"
            f"Contact: <sip:caller@{local_ip}:{sip_port}>\r\n{ctype}"
            f"Content-Length: {len(body.encode())}\r\n\r\n{body}"
        ).encode()

    await loop.sock_sendto(sip, request("INVITE", 1, body=sdp), gateway)
    to_tag, remote_rtp, status = "", None, 0
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        data = await asyncio.wait_for(loop.sock_recv(sip, 65535), timeout=5)
        line, hdrs, body = _parse(data)
        status = int(line.split(" ")[1])
        if status >= 200:
            break
    result = UACResult(status=status)
    if status != 200:
        return result
    to_tag = hdrs["to"].split("tag=")[-1].split(";")[0]
    for ln in body.decode().split("\r\n"):
        if ln.startswith("c=IN IP4 "):
            ip = ln.split()[-1]
        if ln.startswith("m=audio "):
            remote_rtp = (ip, int(ln.split()[1]))
    await loop.sock_sendto(sip, request("ACK", 1, to_tag), gateway)

    # ---- media: one tick = 20 ms; receive whatever arrived, send one frame
    agent_active_ms = 0  # how long agent has been continuously audible
    agent_quiet_ms = 0
    agent_spoke = False
    agent_heard_ms = 0
    seq, ts, ssrc = random.getrandbits(16), random.getrandbits(32), random.getrandbits(32)
    silence = np.zeros(160, np.int16)
    t_next = loop.time()

    async def tick(frame: np.ndarray) -> None:
        nonlocal t_next, seq, ts, agent_active_ms, agent_quiet_ms, agent_spoke, agent_heard_ms
        t_next += 0.02
        await asyncio.sleep(max(0.0, t_next - loop.time()))
        loud = False
        while True:
            try:
                pkt = rtp.recv(2048)
            except BlockingIOError:
                break
            result.rtp_packets += 1
            got = ulaw_decode(np.frombuffer(pkt[12:], np.uint8))
            result.received.append(got)  # keep every packet, in arrival order
            loud = loud or dbfs(got) > speech_dbfs
        # packets and ticks drift against each other, so bridge gaps < 100 ms
        if loud:
            agent_active_ms += 20
            agent_quiet_ms = 0
            agent_heard_ms += 20
            # a short backchannel ("آهان") is not the reply; wait for real speech
            agent_spoke = agent_heard_ms >= min_reply_ms
        else:
            agent_quiet_ms += 20
            if agent_quiet_ms >= 100:
                agent_active_ms = 0
            elif agent_active_ms:
                agent_active_ms += 20
        hdr = bytes([0x80, 0]) + seq.to_bytes(2, "big") + ts.to_bytes(4, "big") + ssrc.to_bytes(4, "big")
        rtp.sendto(hdr + ulaw_encode(frame).tobytes(), remote_rtp)
        result.sent.append(frame)
        seq, ts = (seq + 1) & 0xFFFF, (ts + 160) & 0xFFFFFFFF

    for step in steps:
        waited = 0
        if step.barge_in_after_ms is not None:
            while agent_active_ms < step.barge_in_after_ms and waited < max_wait_s * 1000:
                await tick(silence)
                waited += 20
        else:
            agent_spoke, agent_heard_ms = False, 0
            while not (agent_spoke and agent_quiet_ms >= step.pause_ms) and waited < max_wait_s * 1000:
                await tick(silence)
                waited += 20
        for frame in split_frames(step.audio, 160):
            await tick(frame)
    agent_spoke, agent_heard_ms = False, 0
    waited = 0
    while not (agent_spoke and agent_quiet_ms >= 700) and waited < 8000:
        await tick(silence)
        waited += 20
    for _ in range(tail_ms // 20):
        await tick(silence)

    # ---- hang up (unless the gateway already sent BYE)
    try:
        while True:
            data = sip.recv(65535)
            if data.startswith(b"BYE "):
                result.remote_bye = True
    except BlockingIOError:
        pass
    if not result.remote_bye:
        await loop.sock_sendto(sip, request("BYE", 2, to_tag), gateway)
        try:
            await asyncio.wait_for(loop.sock_recv(sip, 65535), timeout=2)
        except asyncio.TimeoutError:
            pass
    sip.close()
    rtp.close()
    return result


_CALLER_TTS = None


def _voice_for(text: str, voice_dir: str | None = None) -> np.ndarray:
    global _CALLER_TTS
    if os.path.exists(text):
        pcm, sr = read_wav(text)
        return resample(pcm, sr, 8000)
    from pvap.backends.tts import AvaTTS

    if _CALLER_TTS is None:
        _CALLER_TTS = AvaTTS(voice_dir=voice_dir)
    pcm = _CALLER_TTS.synthesize(text)
    # with no separate caller voice, shift pitch/tempo so caller ≠ agent
    return pcm if voice_dir else resample(pcm, 23, 20)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="PVAP test caller (SIP UAC)")
    ap.add_argument("utterances", nargs="+", help="Persian text (synthesized) or WAV paths")
    ap.add_argument("--gateway", default="127.0.0.1:5060")
    ap.add_argument("--barge-last", type=int, metavar="MS", help="last utterance interrupts after MS of agent audio")
    ap.add_argument("--out", default="uac_call.wav")
    ap.add_argument("--voice-dir", help="Ava voice for the caller (use a different voice than the agent)")
    args = ap.parse_args(argv)
    host, port = args.gateway.rsplit(":", 1)
    steps = [UACStep(_voice_for(u, args.voice_dir)) for u in args.utterances]
    if args.barge_last:
        steps[-1] = UACStep(steps[-1].audio, barge_in_after_ms=args.barge_last)
    res = asyncio.run(place_call((host, int(port)), steps))
    print(f"final status {res.status}, {res.rtp_packets} RTP packets received, remote BYE={res.remote_bye}")
    if res.status == 200:
        st = res.stereo()
        import wave

        with wave.open(args.out, "wb") as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(8000)
            w.writeframes(st.astype("<i2").tobytes())
        print(f"wrote {args.out} (L=caller, R=agent)")
    return 0 if res.status == 200 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
