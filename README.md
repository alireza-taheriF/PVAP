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
| VAD / endpointing | `pvap/vad`: energy VAD, adaptive noise floor (min-statistics), 500 ms endpoint | ✅ real, simple baseline |
| Turn-taking | speculative ASR, instant breath + backchannel, continuation merge | ✅ real, tested |
| Barge-in | caller speech during agent audio → cancel reply + `CLEAR` to gateway | ✅ real, ~100 ms reaction measured |
| Turn loop | streaming LLM → sentence chunking → TTS → paced audio, latency metrics | ✅ real, tested |
| **ASR** | `WhisperOnnxASR`: Whisper-turbo on onnxruntime + NumPy log-mel (shared with Ava Voice) | ✅ real, Persian, used in the demos |
|  | `FasterWhisperASR` (CTranslate2) | ⚠️ alternative, not run here |
|  | `ScriptedASR` | 🧪 simulation only: returns pre-written transcripts, does no recognition |
| **LLM** | `OpenAICompatLLM` → vLLM / Ollama / llama.cpp (SSE streaming) | ⚠️ tested against a fake SSE server, not a real model |
|  | `RuleBasedLLM` | 🧪 keyword demo receptionist, not a language model |
| **TTS** | `AvaTTS` → [Ava Voice](https://github.com/alireza-taheriF/Ava-Voice) expressive neural voice: verbalizer → Homo-GE2PE (NumPy) → VITS (onnxruntime) → emotion-driven PSOLA, gain, pauses and breaths | ✅ real, natural, emotional Persian. Whisper recovers it with ≈5 % CER, the same as without emotion. Falls back to Ava's robotic formant vocoder only if the models are missing. |

### Turn-taking like a person

When the caller stops talking, PVAP does what a person on the phone does. It
reacts at once and then answers.

1. **500 ms endpoint.** The default was 600 ms.
2. **Speculative ASR.** After 200 ms of silence, Whisper already starts on
   the audio so far. If the caller keeps talking, the early pass is discarded.
   If not, the transcript is ~300 ms ahead.
3. **Instant reaction.** If the reply is expected to take longer than 0.7 s,
   the agent plays a breath and a backchannel ("اوهوم،"، "آهان،"، "بله،"،
   "خب،"، "باشه،") 120 ms after the endpoint. The backchannels are
   pre-rendered and never repeat back to back. Long caller turns get
   "آهان/اوهوم", short ones "بله/خب".
4. **Inhale before speaking** when no reaction was needed.
5. **Continuation, not interruption.** If the caller adds words while the
   agent is still thinking, the two parts are merged and recognized together.
   Barge-in applies only once the agent is actually talking.

The voice is expressive. Ava's emotion engine sets tempo, pitch, intonation
contour, loudness and breathing per sentence. The rule receptionist and the
LLM prompt use warm, emotional wording, such as «عالیه!»، «متأسفانه»،
«نگران نباشید».

**Measured on a live loopback SIP call** (`tools/live_demo.sh`). The setup:
real SIP/RTP/G.711, Whisper-turbo, the expressive Ava voice, and a different
Ava voice as the caller, on 2 CPU cores shared by caller, gateway and engine.

| Per turn | Before | Now |
| --- | --- | --- |
| Caller's last word → first agent sound | 4.6–6.2 s | **≈ 0.64 s** (500 ms endpoint + 140 ms reaction) |
| Endpoint → reply words | 3.8–5.7 s | 2.6–4.0 s |
| Whisper per utterance (2 cores) | 3.9–5.6 s | ~1.9 s alone, ~2.7 s under call load |
| Barge-in reaction | 120 ms | 100–120 ms |

The biggest ASR win was not oversubscribing threads: 4 threads on 2 cores ran
1.6× slower. The second was 600 instead of 1500 frames of encoder padding,
which cost about 1 % CER. The reply itself is still ASR-bound on this CPU.
Expect roughly 3–5× faster on an M-series Mac and well under 0.5 s on a GPU.

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
python tools/eval_tts.py --whisper-dir models/sherpa-onnx-whisper-turbo --expressive --verbose   # voice CER
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
tools/        live_demo.sh, eval_tts.py (runs Ava's voice-intelligibility evaluation)
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
  female voice. Emotion is applied per sentence: tempo, pitch, contour,
  loudness and breath. Word-level emphasis is not applied yet.
- **Backchannels** fire on timing, before the transcript is known. They are
  neutral by design ("اوهوم"، "بله"), but cannot yet react to content, for
  example sympathy for bad news.
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
