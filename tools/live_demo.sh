#!/usr/bin/env bash
# Live loopback demo: Python engine + Go gateway + test caller, all on localhost.
# Real SIP/RTP/G.711 on the wire, Persian TTS from Ava Voice (neural voice if
# its models are downloaded). Speech recognition is real Whisper when
# WHISPER_DIR is set, scripted otherwise.
#
#   WHISPER_DIR=models/sherpa-onnx-whisper-turbo \
#   CALLER_VOICE_DIR=~/.cache/ava-voice/vits-piper-fa_IR-ganji_adabi-medium \
#   ./tools/live_demo.sh
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-demo_out}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
LINES=("سلام، می‌خواستم یک نوبت بگیرم." "فردا اگر می‌شود." "ساعت کاری کلینیک تا کی است؟" "ممنون، خداحافظ.")
SCRIPT=$(IFS='|'; echo "${LINES[*]}")

ENGINE_ARGS=(--port 9300 --tts ava --reports "$OUT/reports")
[[ -n "${VOICE_DIR:-}" ]] && ENGINE_ARGS+=(--voice-dir "$VOICE_DIR")
[[ -n "${G2P_DIR:-}" ]] && ENGINE_ARGS+=(--g2p-dir "$G2P_DIR")
if [[ -n "${WHISPER_DIR:-}" ]]; then
  ENGINE_ARGS+=(--asr whisper-onnx --whisper-dir "$WHISPER_DIR")
else
  ENGINE_ARGS+=(--asr scripted --script "$SCRIPT")
fi
UAC_ARGS=(--gateway 127.0.0.1:5060 --out "$OUT/sip_call.wav" --barge-last 900)
[[ -n "${CALLER_VOICE_DIR:-}" ]] && UAC_ARGS+=(--voice-dir "$CALLER_VOICE_DIR")

(cd gateway && go build -o "$OUT/pvap-gateway" ./cmd/pvap-gateway)
python -m pvap.bridge.server "${ENGINE_ARGS[@]}" &
ENGINE=$!
"$OUT/pvap-gateway" -sip 127.0.0.1:5060 -public-ip 127.0.0.1 -engine 127.0.0.1:9300 &
GW=$!
trap 'kill $ENGINE $GW 2>/dev/null || true' EXIT
sleep "${ENGINE_WARMUP_S:-4}"
python -m pvap.sim.uac "${UAC_ARGS[@]}" "${LINES[@]}"
sleep 1
echo "per-call report:"; ls "$OUT/reports"
