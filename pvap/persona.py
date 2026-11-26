"""Personas, per-person memory, and the surprise reveal.

A **persona** is who the agent is on a call and who it is talking to. It is a
TOML file with the agent's name and speaking style, the caller's name and the
relationship, facts the agent may use, and boundaries. See
``personas/example.toml``.

**Memory** makes every call teach the next one. After a call, the language
model reads the transcript and extracts durable facts about the caller and the
relationship, such as plans, preferences, names and things to follow up. They
are merged into ``~/.pvap/memory/<caller>.json`` and given to the model on the
next call.

:class:`PersonaLLM` wraps any :class:`~pvap.backends.base.LLMBackend`. It
builds the system prompt from persona and memory, and it can **reveal** that
the caller is talking to an AI: at the start, after N turns, when the caller
says goodbye, or never. Use "never" only when the other person has agreed in
advance to talk to an AI without being told each time.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from pvap.backends.base import LLMBackend, Message

__all__ = ["Persona", "Memory", "PersonaLLM", "load_persona", "extract_facts"]

_GOODBYE = re.compile(r"خداحافظ|خدانگهدار|فعلاً|فعلا|بای\b|می‌بینمت|میبینمت|شب بخیر")


@dataclass
class Persona:
    agent_name: str = "دستیار"
    caller_id: str = "caller"
    caller_name: str = ""
    relationship: str = ""
    style: str = "محاوره‌ای، گرم و صمیمی، جمله‌های کوتاه"
    facts: list[str] = field(default_factory=list)
    boundaries: list[str] = field(default_factory=list)
    greeting: str = "سلام! خوبی؟"
    reveal: str = "end"  # "start" | "end" | "turns:N" | "never"
    reveal_text: str = "راستش رو بخوای، من خودِ {agent} نیستم؛ دستیار هوش مصنوعیِ {agent}‌ام که با صدای خودش حرف می‌زنه. خواست یه کم سورپرایزت کنه!"

    def system_prompt(self, memory: list[str]) -> str:
        who = f"اسم تو «{self.agent_name}» است." if self.agent_name else ""
        rel = ""
        if self.caller_name or self.relationship:
            rel = f"داری تلفنی با {self.caller_name or 'طرف مقابل'}" + (f" ({self.relationship})" if self.relationship else "") + " حرف می‌زنی."
        parts = [
            f"تو در یک تماس صوتی زنده هستی. {who} {rel}".strip(),
            f"لحن: {self.style}. مثل آدم واقعی پشت تلفن حرف بزن: کوتاه (معمولاً یک یا دو جمله)، طبیعی، "
            "با احساس مناسب و فارسی محاوره‌ای. فهرست، مارک‌داون، ایموجی و کلمه‌ی انگلیسی ممنوع.",
            "اگر چیزی را نمی‌دانی، حدس نزن و صادقانه بگو بعداً خبر می‌دهی. "
            "قول مالی، قرار قطعی یا تصمیم مهم از طرف کسی نده.",
        ]
        if self.facts:
            parts.append("چیزهایی که می‌دانی:\n- " + "\n- ".join(self.facts))
        if memory:
            parts.append("از تماس‌های قبلی یادت هست:\n- " + "\n- ".join(memory[-40:]))
        if self.boundaries:
            parts.append("مرزها:\n- " + "\n- ".join(self.boundaries))
        return "\n\n".join(parts)

    def reveal_sentence(self) -> str:
        return self.reveal_text.format(agent=self.agent_name or "صاحب این تلفن")


def load_persona(path: str | os.PathLike) -> Persona:
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    p = Persona()
    agent = data.get("agent", {})
    caller = data.get("caller", {})
    call = data.get("call", {})
    p.agent_name = agent.get("name", p.agent_name)
    p.style = agent.get("style", p.style)
    p.facts = list(agent.get("facts", []))
    p.boundaries = list(agent.get("boundaries", []))
    p.caller_id = caller.get("id", p.caller_id)
    p.caller_name = caller.get("name", p.caller_name)
    p.relationship = caller.get("relationship", p.relationship)
    p.greeting = call.get("greeting", p.greeting)
    p.reveal = str(call.get("reveal", p.reveal))
    p.reveal_text = call.get("reveal_text", p.reveal_text)
    return p


# --------------------------------------------------------------------------- #
# Memory
# --------------------------------------------------------------------------- #
def memory_dir() -> Path:
    return Path(os.environ.get("PVAP_HOME", Path.home() / ".pvap")).expanduser() / "memory"


class Memory:
    """Durable facts about one caller, plus call transcripts."""

    def __init__(self, caller_id: str, root: Path | None = None) -> None:
        safe = re.sub(r"[^\w.-]", "_", caller_id) or "caller"
        self.root = root or memory_dir()
        self.path = self.root / f"{safe}.json"
        self.calls_dir = self.root / "calls" / safe
        self.facts: list[str] = []
        if self.path.exists():
            self.facts = list(json.loads(self.path.read_text(encoding="utf-8")).get("facts", []))

    def add(self, facts: list[str]) -> list[str]:
        new = []
        have = {re.sub(r"\s+", " ", f).strip() for f in self.facts}
        for f in facts:
            f = re.sub(r"\s+", " ", f).strip(" -•\t")
            if f and f not in have:
                self.facts.append(f)
                have.add(f)
                new.append(f)
        self.facts = self.facts[-200:]
        return new

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"facts": self.facts}, ensure_ascii=False, indent=2), encoding="utf-8")

    def save_transcript(self, turns: list[dict]) -> Path:
        self.calls_dir.mkdir(parents=True, exist_ok=True)
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        path = self.calls_dir / f"{stamp}.json"
        path.write_text(json.dumps(turns, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


_EXTRACT_PROMPT = (
    "متن زیر گفت‌وگوی یک تماس تلفنی است. فقط واقعیت‌های ماندگار و مفید برای تماس‌های بعدی را درباره‌ی "
    "تماس‌گیرنده و رابطه استخراج کن: برنامه‌ها، علاقه‌ها، اسم‌ها، قرارها، چیزهایی که باید پیگیری شود. "
    "حرف‌های گذرا، احوال‌پرسی و حدس‌های خودت را ننویس. خروجی فقط یک آرایه‌ی JSON از رشته‌های فارسی کوتاه باشد، "
    "یا [] اگر چیزی نیست."
)


def extract_facts(llm: LLMBackend, transcript: list[dict], persona: Persona) -> list[str]:
    """Ask the model for durable facts from a call transcript."""
    lines = []
    for t in transcript:
        if t.get("user_text"):
            lines.append(f"{persona.caller_name or 'تماس‌گیرنده'}: {t['user_text']}")
        if t.get("agent_text"):
            lines.append(f"{persona.agent_name or 'دستیار'}: {t['agent_text']}")
    if not lines:
        return []
    msgs: list[Message] = [{"role": "user", "content": _EXTRACT_PROMPT + "\n\n" + "\n".join(lines)}]
    raw = "".join(_raw_stream(llm, msgs))
    m = re.search(r"\[.*\]", raw, re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return [str(x) for x in data if isinstance(x, (str, int, float)) and str(x).strip()]


def _raw_stream(llm: LLMBackend, messages: list[Message]) -> Iterator[str]:
    """Stream without the persona system prompt (for utility prompts)."""
    inner = getattr(llm, "inner", llm)
    if hasattr(inner, "stream_raw"):
        return inner.stream_raw(messages)
    return inner.stream(messages)


# --------------------------------------------------------------------------- #
# LLM wrapper
# --------------------------------------------------------------------------- #
class PersonaLLM:
    """Adds persona + memory system prompt and the reveal policy to an LLM."""

    def __init__(self, inner: LLMBackend, persona: Persona, memory: Memory | None = None) -> None:
        self.inner = inner
        self.persona = persona
        self.memory = memory
        self.name = f"{inner.name} as «{persona.agent_name}»"
        self.turns = 0
        self.revealed = persona.reveal == "start"

    def _should_reveal(self, user_text: str) -> bool:
        if self.revealed or self.persona.reveal == "never":
            return False
        r = self.persona.reveal
        if r.startswith("turns:"):
            try:
                return self.turns >= int(r.split(":", 1)[1])
            except ValueError:
                return False
        if r == "end":
            return bool(_GOODBYE.search(user_text))
        return False

    def greeting(self) -> str:
        g = self.persona.greeting
        if self.persona.reveal == "start":
            g = f"{g} {self.persona.reveal_sentence()}"
        return g

    def stream(self, messages: list[Message]) -> Iterator[str]:
        self.turns += 1
        user_text = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        system = self.persona.system_prompt(self.memory.facts if self.memory else [])
        reveal = self._should_reveal(user_text)
        if reveal:
            system += ("\n\nدر این پاسخ، کوتاه و با لحن شوخ و مهربان جواب بده؛ "
                       "جمله‌ی افشای هوش مصنوعی را سیستم خودش اضافه می‌کند.")
        msgs: list[Message] = [{"role": "system", "content": system}, *[m for m in messages if m["role"] != "system"]]
        stream_raw = getattr(self.inner, "stream_raw", None)
        yield from (stream_raw(msgs) if stream_raw else self.inner.stream(msgs))
        if reveal:
            self.revealed = True
            yield " " + self.persona.reveal_sentence()
