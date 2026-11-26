"""Voice intelligibility (synthesize → Whisper → CER) — runs Ava Voice's script.

    python tools/eval_tts.py --whisper-dir models/sherpa-onnx-whisper-turbo [--expressive] [--verbose]

The evaluation lives in Ava Voice (``scripts/eval_intelligibility.py``) next
to the voice it measures; this wrapper locates and runs it.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import ava_voice

script = Path(ava_voice.__file__).resolve().parent / "scripts" / "eval_intelligibility.py"
if not script.exists():
    sys.exit(f"Ava Voice evaluation script not found at {script} (install Ava from source)")
sys.argv[0] = str(script)
runpy.run_path(str(script), run_name="__main__")
