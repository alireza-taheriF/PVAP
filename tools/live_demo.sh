#!/usr/bin/env bash
# Live loopback demo: Python engine + Go gateway + test caller, all on localhost.
# Real SIP/RTP/G.711 on the wire; Persian TTS from Ava Voice; scripted ASR.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-demo_out}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
LINES=("سلام، می‌خواستم یک نوبت بگیرم." "فردا اگر می‌شود." "هزینه‌ی ویزیت چقدر است؟" "ممنون، خداحافظ.")
SCRIPT=$(IFS='|'; echo "${LINES[*]}")

(cd gateway && go build -o "$OUT/pvap-gateway" ./cmd/pvap-gateway)
python -m pvap.bridge.server --port 9300 --tts ava --asr scripted --script "$SCRIPT" --reports "$OUT/reports" &
ENGINE=$!
"$OUT/pvap-gateway" -sip 127.0.0.1:5060 -public-ip 127.0.0.1 -engine 127.0.0.1:9300 &
GW=$!
trap 'kill $ENGINE $GW 2>/dev/null || true' EXIT
sleep 1.5
python -m pvap.sim.uac --gateway 127.0.0.1:5060 --out "$OUT/sip_call.wav" --barge-last 900 "${LINES[@]}"
sleep 1
echo "per-call report:"; ls "$OUT/reports"
