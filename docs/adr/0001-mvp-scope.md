# ADR 0001: Build a thin end-to-end slice before the platform

**Status:** accepted, 2026-11-26

## Context

The design specification describes a production-grade platform with about 15
infrastructure components. None of it was implemented. The riskiest unknowns
are not the infrastructure. They are the *call loop* (latency, turn-taking,
barge-in over real telephony) and *Persian speech quality* (ASR WER, TTS
naturalness).

## Decision

- Implement the call path first: SIP/RTP gateway (Go), dialog engine
  (Python), and a minimal framed TCP link between them. Test it with a real
  SIP caller.
- Put every model stage behind a three-method interface. Ship clearly
  labelled stand-ins (`ScriptedASR`, `RuleBasedLLM`) so the loop can be tested
  without GPUs. Never present their output as model output.
- Reuse Ava Voice for Persian TTS rather than starting a second TTS codebase.
  Improvements to Persian text handling, such as number verbalization, belong
  in Ava and reach PVAP for free.
- Defer Kafka, Patroni, the service mesh and gRPC until there is load or
  multiple teams to justify them.

## Consequences

- A working demo exists with real SIP/RTP, but its "intelligence" is scripted
  until Whisper and an LLM server are connected.
- Voice quality is capped by Ava's formant baseline until Ava gets a neural
  acoustic model.
- Swapping in FreeSWITCH or gRPC later touches only `gateway/` or `bridge/`.
