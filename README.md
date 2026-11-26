# PVAP — Persian Voice AI Platform

A real-time Persian voice agent that answers **phone calls**: SIP/RTP in,
streaming speech pipeline in the middle, barge-in, Persian speech out.

This repository is the **MVP slice** of the PVAP engineering design. The
call path is implemented and tested end to end over real SIP/RTP. The agent
**hears with Whisper and speaks with a neural Persian voice** (Ava Voice), and
neither needs PyTorch. The language model is still a rule-based stand-in until
you point PVAP at a served LLM.

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
| **ASR** | `WhisperOnnxASR`: Whisper-turbo on onnxruntime + NumPy log-mel | ✅ real, Persian, used in the demos |
|  | `FasterWhisperASR` (CTranslate2) | ⚠️ alternative, not run here |
|  | `ScriptedASR` | 🧪 simulation only: returns pre-written transcripts, does no recognition |
| **LLM** | `OpenAICompatLLM` → vLLM / Ollama / llama.cpp (SSE streaming) | ⚠️ tested against a fake SSE server, not a real model |
|  | `RuleBasedLLM` | 🧪 keyword demo receptionist, not a language model |
| **TTS** | `AvaTTS` → [Ava Voice](https://github.com/alireza-taheriF/Ava-Voice) neural voice: verbalizer → Homo-GE2PE (NumPy) → VITS (onnxruntime) | ✅ real, natural Persian. Whisper recovers it with ≈5–6 % CER. Falls back to Ava's robotic formant vocoder only if the models are missing. |

**Measured on a live loopback SIP call** (`tools/live_demo.sh`). The setup:
real SIP/RTP/G.711, Whisper-turbo ASR, the Ava neural voice for the agent, and
a different Ava voice playing the caller. It ran on 2 CPU cores. All five
caller turns were recognized well enough for the receptionist to answer
correctly. Barge-in reaction was 120 ms.

| Stage per turn | Measured |
| --- | --- |
| Endpointing wait (VAD) | 600 ms |
| ASR (Whisper-turbo, int8, 2 cores) | 3.9–5.6 s ← the bottleneck |
| LLM (rule stand-in) + first TTS sentence + first frame | 0.1–0.4 s |
| TTS real-time factor | ≈ 0.1 |

ASR dominates on this tiny CPU. Expect roughly 3–5× faster on an 8–10-core
laptop (M-series Mac), and well under 500 ms on a GPU. Streaming ASR is the
next latency step.

### Persian voice quality

The voice comes from Ava Voice. Text is verbalized (numbers, dates, times,
phone numbers). **Homo-GE2PE** then restores the short vowels, ezafe and
homograph readings Persian script omits, and a Persian **VITS** model speaks
the result. Intelligibility is measured with `tools/eval_tts.py`: synthesize →
Whisper → character error rate. On 20 phone-agent sentences the error is
about 5–6 %, and that number includes Whisper's own mistakes.

## Run it

Requirements: Python ≥ 3.11, Go ≥ 1.22.

```bash
pip install -e ".[dev,ava,asr]"    # numpy, scipy, onnxruntime, Ava Voice, pytest
python -m ava_voice.tts.models fetch   # Persian G2P + voice → ~/.cache/ava-voice
python -m ava_voice.tts.models fetch --voice ganji_adabi   # optional 2nd voice for the test caller
curl -LO https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-turbo.tar.bz2
tar xjf sherpa-onnx-whisper-turbo.tar.bz2 -C models/

pytest                             # Python unit + full SIP end-to-end test
(cd gateway && go test -race ./...)
python tools/eval_tts.py --whisper-dir models/sherpa-onnx-whisper-turbo --verbose   # voice CER
```

**Offline demo** (no telephony): a call with a barge-in. It writes
`demo_out/call.wav` (left = caller, right = agent) and `report.json`. Pass
`--whisper-dir` for real recognition; without it, transcripts are scripted.

```bash
python -m pvap.sim.demo --out demo_out --whisper-dir models/sherpa-onnx-whisper-turbo \
    --caller-voice-dir ~/.cache/ava-voice/vits-piper-fa_IR-ganji_adabi-medium
```

**Live loopback over SIP** (engine + gateway + test caller on localhost):

```bash
WHISPER_DIR=models/sherpa-onnx-whisper-turbo \
CALLER_VOICE_DIR=~/.cache/ava-voice/vits-piper-fa_IR-ganji_adabi-medium \
./tools/live_demo.sh               # → demo_out/sip_call.wav, demo_out/reports/*.json
```

**With real models**:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct        # or: ollama serve  (base-url http://localhost:11434/v1)
python -m pvap.bridge.server --asr whisper-onnx --whisper-dir models/sherpa-onnx-whisper-turbo \
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
  backends/   base.py (ASR/LLM/TTS protocols), asr.py, whisper_onnx.py, llm.py, tts.py
  dialog/     session.py (full-duplex call loop, barge-in, metrics)
  bridge/     protocol.py, server.py (engine side of the gateway link)
  sim/        caller.py (simulated caller), demo.py, uac.py (SIP test caller)
gateway/      Go: cmd/pvap-gateway, internal/{sip,rtp,g711,bridge,gw}
docs/         ARCHITECTURE.md, adr/
tools/        live_demo.sh, eval_tts.py (voice intelligibility), data/
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
- **Voice:** available Persian VITS voices are male, plus one low-quality
  female voice. The voice does not yet follow Ava's per-phoneme emotional
  prosody, only its tempo.
- **ASR latency** is too high on a small CPU (above). Fix it with a GPU, a
  bigger CPU, or streaming ASR.

## Roadmap

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), which maps this MVP to
the full design. Next steps in order:

1. Measure WER on *recorded human* Persian calls, not only synthetic speech.
2. Run a served LLM and measure time-to-first-token.
3. Make ASR fast: GPU, or streaming / partial decoding.
4. Add a jitter buffer and Silero VAD.
5. Connect an Iranian SIP trunk.
