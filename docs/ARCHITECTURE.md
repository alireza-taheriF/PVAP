# Architecture: MVP vs. the full PVAP design

The original Engineering Design Specification targets a production platform:
Kamailio + FreeSWITCH, Go/Python microservices over gRPC, vLLM/TensorRT-LLM,
Whisper-FA, VITS2, Kafka, Patroni, and a service mesh. This repository builds
the **smallest slice that carries a real call end to end**. Each component
sits behind an interface the full design can grow into.

| Full design | MVP here | Path to the full design |
| --- | --- | --- |
| Kamailio SBC + FreeSWITCH media | `gateway/` Go SIP UAS + RTP | Put Kamailio in front for auth, NAT and load-balancing. Or replace the gateway with FreeSWITCH + `mod_audio_stream` speaking the same framed protocol. |
| gRPC bidirectional streaming | framed TCP (`START/AUDIO/CLEAR/HANGUP`) | Same four messages map 1:1 onto a gRPC stream. Only `bridge/` changes. |
| Whisper-FA (fine-tuned) | `WhisperOnnxASR` (Whisper-turbo, onnxruntime) | Export the fine-tune with sherpa-onnx's `export-onnx.py` and pass the directory. |
| vLLM / TensorRT-LLM | `OpenAICompatLLM` (SSE) | No change; vLLM already serves this API. |
| VITS2 Persian TTS | `AvaTTS` → Ava Voice: Homo-GE2PE (NumPy) + Persian VITS (onnxruntime) | Fine-tune VITS2 on in-house voices and condition it on Ava's prosody plan. PVAP is unchanged. |
| Kafka, Patroni, Redis, MinIO | per-call JSON reports | Add when there is multi-tenant traffic to justify them. |
| Cilium + Linkerd | — | Out of scope until there are several services to mesh. |

## Call loop invariants

1. **The audio clock never waits on a model.** ASR, LLM and TTS run in worker
   threads. The outbound pump emits one 20 ms frame per tick.
2. **Speak while generating.** LLM output is cut into sentences, or long
   clauses at `،`, and each chunk goes to TTS as soon as it is complete.
   First-audio latency is therefore set by the first sentence, not the whole
   reply.
3. **Barge-in drops everything.** When the caller starts speaking during an
   agent turn, the reply task is cancelled, the LLM stream is told to stop,
   queued audio is cleared, and `CLEAR` flushes the gateway's playout. The
   interrupted turn is marked `interrupted` in the report.
4. **Every number in a report is measured on the wall clock** at the point the
   caller would notice it. "First audio" means the first frame actually handed
   to the gateway.

## Latency budget (target vs. measured)

| Segment | Design target | Measured here |
| --- | --- | --- |
| Endpointing (caller stops → turn ends) | 300–600 ms | 600 ms (configurable `VADConfig.end_ms`) |
| ASR | ≤ 150 ms | 3.9–5.6 s (Whisper-turbo, 2 CPU cores) ← bottleneck |
| LLM first token | ≤ 200 ms | not measured (rule-based) |
| TTS first chunk + pump | ≤ 150 ms | 100–400 ms (Ava neural voice, RTF ≈ 0.1) |
| Barge-in reaction | ≤ 200 ms | 100–120 ms |
