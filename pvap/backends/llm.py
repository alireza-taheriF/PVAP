"""LLM backends.

* :class:`OpenAICompatLLM` — streams chat completions from any server that
  speaks the OpenAI ``/v1/chat/completions`` SSE protocol: **vLLM**, Ollama,
  llama.cpp server, LM Studio. Standard library only. Tested here against a
  local fake SSE server, not against a real model.
* :class:`RuleBasedLLM` — a deterministic keyword-intent responder for a
  sample clinic reception line. It is a stand-in for demos and tests, not a
  language model; it lets the full call loop run with no GPU and no network.

``SPOKEN_STYLE_PROMPT`` is the system prompt PVAP uses to keep a real LLM's
output speakable: short sentences, no markdown, no lists, no emoji.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Iterator

from pvap.backends.base import Message

__all__ = ["OpenAICompatLLM", "RuleBasedLLM", "SPOKEN_STYLE_PROMPT"]

SPOKEN_STYLE_PROMPT = (
    "تو دستیار تلفنی فارسی‌زبان هستی و مثل یک انسان گرم و باحوصله حرف می‌زنی. "
    "پاسخ‌ها کوتاه باشند: حداکثر دو جمله‌ی کوتاه، به زبان رسمیِ مؤدبانه ولی طبیعی. "
    "احساس مناسب را نشان بده: برای خبر خوب از «عالیه!» یا «خیلی خوشحالم» و علامت تعجب، "
    "برای خبر بد از «متأسفانه» یا «ببخشید» استفاده کن، و اگر طرف نگران است آرامش بده. "
    "جمله را با «بله» شروع نکن (سیستم خودش تأیید کوتاه پخش می‌کند). "
    "از فهرست، مارک‌داون، ایموجی و کلمات انگلیسی استفاده نکن. اعداد را می‌توانی با رقم بنویسی."
)


class OpenAICompatLLM:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        system_prompt: str = SPOKEN_STYLE_PROMPT,
        temperature: float = 0.4,
        max_tokens: int = 160,
        timeout_s: float = 30.0,
    ) -> None:
        self.name = f"openai-compat:{model}"
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._model = model
        self._key = api_key
        self._system = system_prompt
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout_s

    def stream(self, messages: list[Message]) -> Iterator[str]:
        body = {
            "model": self._model,
            "stream": True,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "messages": [{"role": "system", "content": self._system}, *messages],
        }
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if self._key:
            headers["Authorization"] = f"Bearer {self._key}"
        req = urllib.request.Request(
            self._url, data=json.dumps(body).encode(), headers=headers, method="POST"
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    return
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                for choice in chunk.get("choices", []):
                    delta = (choice.get("delta") or {}).get("content")
                    if delta:
                        yield delta


class RuleBasedLLM:
    """Keyword intents for a sample clinic reception line (demo stand-in)."""

    name = "rule-based clinic receptionist (demo stand-in — not an LLM)"

    GREETING = "سلام! وقتتون بخیر. با کلینیک آوا تماس گرفتید. چطور می‌توانم کمکتان کنم؟"

    _INTENTS: tuple[tuple[tuple[str, ...], str], ...] = (
        (("نوبت", "وقت"), "حتماً، با کمال میل! برای چه روزی نوبت می‌خواهید؟"),
        (("فردا", "شنبه", "یکشنبه", "دوشنبه", "سه‌شنبه", "چهارشنبه", "پنجشنبه"),
         "عالیه! نوبت شما برای ساعت ۱۰:۳۰ ثبت شد. پیامک تأیید هم برایتان ارسال می‌شود."),
        (("ساعت کاری", "تا کی", "باز"), "کلینیک هر روز از ساعت ۸ صبح تا ۸ شب باز است."),
        (("قیمت", "هزینه", "چقدر"), "هزینه‌ی ویزیت ۴۵۰,۰۰۰ تومان است."),
        (("آدرس", "کجا"), "کلینیک در مشهد، بلوار وکیل‌آباد است."),
        (("درد", "نگران", "اورژانس"), "نگران نباشید، حتماً کمکتان می‌کنیم. اولین نوبت خالی امروز ساعت ۵ است."),
        (("ممنون", "مرسی", "خداحافظ"), "خواهش می‌کنم! خیلی خوشحال شدم. روز خوبی داشته باشید."),
    )
    _FALLBACK = "ببخشید، درست متوجه نشدم. می‌شود کمی بیشتر توضیح بدهید؟"

    def reply(self, text: str) -> str:
        for keys, answer in self._INTENTS:
            if any(k in text for k in keys):
                return answer
        return self._FALLBACK

    def stream(self, messages: list[Message]) -> Iterator[str]:
        last = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        for i, word in enumerate(self.reply(last).split(" ")):
            yield word if i == 0 else " " + word
