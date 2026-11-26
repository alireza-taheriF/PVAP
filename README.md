# PVAP — Persian Voice AI Platform

A real-time Persian voice agent that answers **phone calls**: SIP/RTP in,
streaming speech pipeline in the middle, barge-in, Persian speech out.

This repository is the **MVP slice** of the PVAP engineering design: the call
path is implemented and tested end to end over real SIP/RTP. The model stages
are behind small interfaces, and only some of them are real today. The table
below gives the exact status.

```
 PSTN / PBX / softphone
        │  SIP (UDP) + RTP G.711
        ▼
┌──────────────────────┐   TCP framed PCM (20 ms)   ┌─────────────────────────────────────────┐
│ gateway/  (Go)       │ ◀────────────────────────▶ │ pvap/  (Python dialog engine)           │
│  SIP UAS, SDP        │   START · AUDIO · CLEAR ·  │  VAD ─▶ ASR ─▶ LLM (stream) ─▶ TTS      │
│  RTP, PCMU/PCMA      │   HANGUP                   │   ▲                          │          │
│  paced playout       │                            │   └── barge-in: cancel + flush ◀┘       │
└──────────────────────┘                            └─────────────────────────────────────────┘
```

## Status: what is real

| Stage | Implementation | Status |
| --- | --- | --- |
| SIP signalling | `gateway/internal/sip`, `gw`: INVITE/ACK/BYE/CANCEL/OPTIONS, 2xx retransmission, gateway-initiated BYE | ✅ real, tested against a SIP UA |
| Media | RTP, G.711 μ-law/A-law (bit-exact vs reference), 20 ms paced playout, symmetric RTP | ✅ real, tested |
| VAD / endpointing | `pvap/vad`: energy VAD, adaptive noise floor (min-statistics) | ✅ real, simple baseline |
| Barge-in | caller speech during agent audio → cancel reply + `CLEAR` to gateway | ✅ real, ~100 ms reaction measured |
| Turn loop | streaming LLM → sentence chunking → TTS → paced audio, latency metrics | ✅ real, tested |
| **ASR** | `FasterWhisperASR` (Whisper, Persian) | ⚠️ written, **not run here** (no model weights in the build sandbox) |
|  | `ScriptedASR` | 🧪 simulation only: returns pre-written transcripts, does no recognition |
| **LLM** | `OpenAICompatLLM` → vLLM / Ollama / llama.cpp (SSE streaming) | ⚠️ tested against a fake SSE server, not a real model |
|  | `RuleBasedLLM` | 🧪 keyword demo receptionist, not a language model |
| **TTS** | `AvaTTS` → [Ava Voice](https://github.com/alireza-taheriF/Ava-Voice) | ✅ real Persian audio, but a **robotic formant baseline** |

**Measured on the loopback demo** (`tools/live_demo.sh`, real SIP/RTP,
Ava TTS, scripted ASR, rule LLM, 2-core CPU): agent audio starts 45–200 ms
after the VAD end-of-turn decision. That is 650–800 ms after the caller's last
word, including the 600 ms endpointing wait. Barge-in reaction was 100 ms.
These numbers **exclude real ASR and LLM inference**, which will dominate
once they are plugged in.

## Run it

Requirements: Python ≥ 3.11, Go ≥ 1.22.

```bash
pip install -e ".[dev]"            # numpy, scipy, pytest
pip install -e ".[ava]"            # Persian TTS (Ava Voice)
pytest                             # Python unit + full SIP end-to-end test
(cd gateway && go test -race ./...)
```

**Offline demo** (no network, no telephony): a scripted call with a barge-in
writes `demo_out/call.wav` (left = caller, right = agent) and `report.json`.

```bash
python -m pvap.sim.demo --out demo_out
```

**Live loopback over SIP** (engine + gateway + test caller on localhost):

```bash
./tools/live_demo.sh               # → demo_out/sip_call.wav, demo_out/reports/*.json
```

**With real models**:

```bash
pip install -e ".[whisper]"
vllm serve Qwen/Qwen2.5-7B-Instruct        # or: ollama serve  (base-url http://localhost:11434/v1)
python -m pvap.bridge.server --asr whisper --whisper-model small \
    --llm openai --base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-7B-Instruct
./gateway/pvap-gateway -sip 0.0.0.0:5060 -public-ip <LAN-IP> -engine 127.0.0.1:9300
```

Then point a softphone (Linphone, MicroSIP, Zoiper) or a PBX trunk at
`sip:agent@<LAN-IP>:5060`.

## Layout

```
pvap/
  audio/      g711.py (PCMU/PCMA), pcm.py (framing, resampling, WAV)
  vad/        energy_vad.py
  backends/   base.py (ASR/LLM/TTS protocols), asr.py, llm.py, tts.py
  dialog/     session.py (full-duplex call loop, barge-in, metrics)
  bridge/     protocol.py, server.py (engine side of the gateway link)
  sim/        caller.py (simulated caller), demo.py, uac.py (SIP test caller)
gateway/      Go: cmd/pvap-gateway, internal/{sip,rtp,g711,bridge,gw}
docs/         ARCHITECTURE.md, adr/
tools/        live_demo.sh
```

## Known limits (MVP, by design)

- **Security:** no SIP auth, no TLS/SRTP, no REGISTER. Run it only on a
  trusted network or behind an SBC.
- **Media:** no jitter buffer, no packet-loss concealment, no DTMF, and no echo
  cancellation. Barge-in assumes the far end cancels echo; most carriers do.
- **VAD** is energy-based, so loud background talk can trigger a false barge-in.
  Silero-VAD can replace it behind the same events.
- **Scale:** one Python process handles calls concurrently with asyncio. Model
  workers are threads, so throughput is bounded by the model servers.
- **Voice quality:** depends on Ava Voice's acoustic model, which is a formant
  baseline today.

## Roadmap

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), which maps this MVP to
the full design. Next steps in order:

1. Run Whisper on real Persian calls and measure WER.
2. Run a served LLM and measure time-to-first-token.
3. Train a neural TTS for Ava.
4. Add a jitter buffer and Silero VAD.
5. Connect an Iranian SIP trunk.
