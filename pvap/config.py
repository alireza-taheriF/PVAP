"""PVAP configuration (``pvap.toml``) and backend construction.

Everything that differs between "local on my Mac" and "cloud" lives in one
file, so switching is one line: ``[llm] provider = "local"`` or ``"cloud"``.
See ``pvap.example.toml``. Environment variables override the file:
``PVAP_LLM_PROVIDER``, ``PVAP_LLM_MODEL``, ``PVAP_LLM_BASE_URL``.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["Config", "load_config", "build_llm"]


@dataclass
class LLMProfile:
    base_url: str
    model: str
    api_key_env: str = ""
    temperature: float = 0.7
    max_tokens: int = 160
    timeout_s: float = 60.0

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env) if self.api_key_env else None


@dataclass
class Config:
    provider: str = "local"
    local: LLMProfile = field(default_factory=lambda: LLMProfile("http://localhost:11434/v1", "aya-expanse:8b"))
    cloud: LLMProfile = field(default_factory=lambda: LLMProfile("https://api.openai.com/v1", "gpt-4o-mini", "OPENAI_API_KEY"))
    persona: str = ""
    whisper_dir: str = ""
    voice_dir: str = ""
    g2p_dir: str = ""
    speed: float = 1.2
    pitch_semitones: float = 2.0
    intensity: float = 1.0
    endpoint_ms: int = 500
    learn_pronunciation: bool = True
    learn_memory: bool = True
    host: str = "0.0.0.0"
    port: int = 8443
    cert: str = ""
    key: str = ""

    def llm_profile(self) -> LLMProfile:
        return self.cloud if self.provider == "cloud" else self.local


def _profile(d: dict, base: LLMProfile) -> LLMProfile:
    return LLMProfile(
        base_url=d.get("base_url", base.base_url),
        model=d.get("model", base.model),
        api_key_env=d.get("api_key_env", base.api_key_env),
        temperature=float(d.get("temperature", base.temperature)),
        max_tokens=int(d.get("max_tokens", base.max_tokens)),
        timeout_s=float(d.get("timeout_s", base.timeout_s)),
    )


def load_config(path: str | os.PathLike | None = None) -> Config:
    cfg = Config()
    p = Path(path) if path else Path(os.environ.get("PVAP_CONFIG", "pvap.toml"))
    if p.exists():
        d = tomllib.loads(p.read_text(encoding="utf-8"))
        llm = d.get("llm", {})
        cfg.provider = llm.get("provider", cfg.provider)
        cfg.local = _profile(llm.get("local", {}), cfg.local)
        cfg.cloud = _profile(llm.get("cloud", {}), cfg.cloud)
        cfg.persona = d.get("persona", {}).get("file", cfg.persona)
        asr = d.get("asr", {})
        cfg.whisper_dir = os.path.expanduser(asr.get("whisper_dir", cfg.whisper_dir))
        v = d.get("voice", {})
        cfg.voice_dir = os.path.expanduser(v.get("voice_dir", cfg.voice_dir))
        cfg.g2p_dir = os.path.expanduser(v.get("g2p_dir", cfg.g2p_dir))
        cfg.speed = float(v.get("speed", cfg.speed))
        cfg.pitch_semitones = float(v.get("pitch_semitones", cfg.pitch_semitones))
        cfg.intensity = float(v.get("intensity", cfg.intensity))
        c = d.get("call", {})
        cfg.endpoint_ms = int(c.get("endpoint_ms", cfg.endpoint_ms))
        learn = d.get("learning", {})
        cfg.learn_pronunciation = bool(learn.get("pronunciation", cfg.learn_pronunciation))
        cfg.learn_memory = bool(learn.get("memory", cfg.learn_memory))
        w = d.get("web", {})
        cfg.host = w.get("host", cfg.host)
        cfg.port = int(w.get("port", cfg.port))
        cfg.cert = os.path.expanduser(w.get("cert", cfg.cert))
        cfg.key = os.path.expanduser(w.get("key", cfg.key))
    cfg.provider = os.environ.get("PVAP_LLM_PROVIDER", cfg.provider)
    prof = cfg.llm_profile()
    prof.model = os.environ.get("PVAP_LLM_MODEL", prof.model)
    prof.base_url = os.environ.get("PVAP_LLM_BASE_URL", prof.base_url)
    return cfg


def build_llm(cfg: Config):
    from pvap.backends.llm import OpenAICompatLLM

    p = cfg.llm_profile()
    return OpenAICompatLLM(p.base_url, p.model, api_key=p.api_key, temperature=p.temperature,
                           max_tokens=p.max_tokens, timeout_s=p.timeout_s)
