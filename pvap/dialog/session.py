"""Full-duplex call session: VAD → ASR → streaming LLM → sentence TTS, with barge-in.

One :class:`CallSession` drives one phone call. It consumes inbound 20 ms
PCM frames, and owns a paced outbound pump that emits one 20 ms frame per
tick while the agent is talking.

Turn flow::

    caller speech ─200 ms silence─▶ speculative ASR ┐
                 ─500 ms silence─▶ endpoint ─┬─────┴─▶ LLM stream ─sentence─▶ TTS ─▶ out queue ─20 ms pump─▶ caller
                                             └─120 ms─▶ breath + backchannel ("اوهوم،") ────────▲
          ▲                                                                                  │
          └─ VAD start while agent talks = BARGE-IN (cancel + flush); while thinking = CONTINUATION ┘

Latency is measured on the wall clock at the points that matter on a phone
line: end of caller speech → ASR done → first LLM token → **first agent audio
frame actually sent** (the caller-perceived response latency), and caller
speech onset → agent audio stopped (barge-in reaction).

Model calls run in worker threads (``asyncio.to_thread``) so a slow model
never stalls the audio clock. Text is spoken sentence by sentence as the LLM
streams, which is what keeps first-audio latency independent of reply length.
"""

from __future__ import annotations

import asyncio
import random
import re
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import AsyncIterator, Awaitable, Callable

import numpy as np

from pvap.audio.pcm import TELEPHONY_RATE, frame_samples, split_frames
from pvap.audio.pcm import dbfs as _dbfs
from pvap.backends.base import ASRBackend, LLMBackend, Message, TTSBackend
from pvap.vad.energy_vad import EnergyVAD, VADConfig, VADEvent

__all__ = ["CallSession", "TurnMetrics", "CallReport", "TurnTakingConfig", "split_sentences"]

_SENTENCE_END = re.compile(r"([.!?؟؛\n]|…)\s*")
_SOFT_BREAK = re.compile(r"(،)\s*")


def split_sentences(buffer: str, min_chars: int = 12) -> tuple[list[str], str]:
    """Cut complete sentences off the front of a streaming text buffer.

    Hard breaks (``. ! ? ؟ ؛ …``) always cut; a Persian comma cuts only once
    the pending chunk is long enough to be worth synthesizing on its own.
    Returns ``(ready_chunks, remainder)``.
    """
    out: list[str] = []
    while True:
        m = _SENTENCE_END.search(buffer)
        soft = _SOFT_BREAK.search(buffer)
        if soft and (not m or soft.start() < m.start()) and soft.start() >= min_chars:
            m = soft
        if not m:
            return out, buffer
        chunk = buffer[: m.end()].strip()
        buffer = buffer[m.end() :]
        if chunk:
            out.append(chunk)


@dataclass
class TurnMetrics:
    turn: int
    user_text: str = ""
    agent_text: str = ""
    speech_end_t: float | None = None
    asr_done_t: float | None = None
    llm_first_t: float | None = None
    first_audio_t: float | None = None  # any agent sound (breath, backchannel, words)
    first_speech_t: float | None = None  # first frame of the actual reply
    interrupted: bool = False
    endpoint_ms: int = 0  # VAD silence needed to decide the caller finished
    asr_speculative: bool = False  # transcript came from the early (speculative) pass
    reaction: str = ""  # backchannel played while thinking, if any

    def _ms(self, a: float | None, b: float | None) -> float | None:
        return None if a is None or b is None else round((b - a) * 1000.0, 1)

    @property
    def asr_ms(self) -> float | None:
        return self._ms(self.speech_end_t, self.asr_done_t)

    @property
    def llm_first_token_ms(self) -> float | None:
        return self._ms(self.asr_done_t, self.llm_first_t)

    @property
    def response_latency_ms(self) -> float | None:
        """VAD end-of-turn decision → first agent sound (reaction or reply)."""
        return self._ms(self.speech_end_t, self.first_audio_t)

    @property
    def reply_latency_ms(self) -> float | None:
        """VAD end-of-turn decision → first frame of the reply itself."""
        return self._ms(self.speech_end_t, self.first_speech_t)

    @property
    def perceived_latency_ms(self) -> float | None:
        """Caller's last speech frame → first agent sound (adds the endpoint wait)."""
        r = self.response_latency_ms
        return None if r is None else round(r + self.endpoint_ms, 1)

    def to_dict(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if not k.endswith("_t")}
        d.update(
            asr_ms=self.asr_ms,
            llm_first_token_ms=self.llm_first_token_ms,
            response_latency_ms=self.response_latency_ms,
            reply_latency_ms=self.reply_latency_ms,
            perceived_latency_ms=self.perceived_latency_ms,
        )
        return d


@dataclass
class CallReport:
    backends: dict[str, str]
    turns: list[TurnMetrics] = field(default_factory=list)
    barge_in_ms: list[float] = field(default_factory=list)
    events: list[tuple[float, str, str]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "backends": self.backends,
            "turns": [t.to_dict() for t in self.turns],
            "barge_in_reaction_ms": self.barge_in_ms,
            "events": [{"t_ms": round(t * 1000, 1), "kind": k, "detail": d} for t, k, d in self.events],
        }


@dataclass(frozen=True)
class TurnTakingConfig:
    """How quickly and how humanly the agent takes its turn.

    * ``endpoint_ms`` — silence that ends the caller's turn.
    * ``speculative_ms`` — silence after which ASR starts *early* on the audio
      so far. If the caller keeps talking the result is discarded; if not, the
      transcript is ready (or nearly) when the endpoint fires.
    * ``react`` — when the reply is expected to take longer than
      ``react_if_expected_ms``, play a breath and a short backchannel
      ("بله،" / "آهان،" / "اوهوم،" / "خب،") ``react_after_ms`` after the
      endpoint, like a person who has heard you and is about to answer.
    * ``breath_before_reply`` — an inhale before the first words of a reply.
    """

    endpoint_ms: int = 500
    speculative_ms: int = 200
    react: bool = True
    react_after_ms: int = 120
    react_if_expected_ms: int = 700
    breath_before_reply: bool = True


class CallSession:
    def __init__(
        self,
        asr: ASRBackend,
        llm: LLMBackend,
        tts: TTSBackend,
        *,
        greeting: str | None = None,
        sample_rate: int = TELEPHONY_RATE,
        frame_ms: int = 20,
        vad_config: VADConfig | None = None,
        turn_taking: TurnTakingConfig | None = None,
        preroll_ms: int = 300,
        max_history: int = 12,
        clock: Callable[[], float] = time.monotonic,
        seed: int | None = None,
        echo_guard: bool = False,
        echo_margin_db: float = 8.0,
    ) -> None:
        if tts.sample_rate != sample_rate:
            raise ValueError(f"TTS rate {tts.sample_rate} != call rate {sample_rate}")
        self.asr, self.llm, self.tts = asr, llm, tts
        self.greeting = greeting
        self.sr = sample_rate
        self.frame_ms = frame_ms
        self.n = frame_samples(sample_rate, frame_ms)
        self.tt = turn_taking or TurnTakingConfig()
        self.vad = EnergyVAD(vad_config or VADConfig(frame_ms=frame_ms, end_ms=self.tt.endpoint_ms))
        self._preroll: deque[np.ndarray] = deque(maxlen=max(1, preroll_ms // frame_ms))
        self._max_history = max_history
        self.clock = clock
        self._rng = random.Random(seed)

        self.history: list[Message] = []
        self.report = CallReport(backends={"asr": asr.name, "llm": llm.name, "tts": tts.name})
        self._out: deque[tuple[int, np.ndarray, str]] = deque()  # (turn, frame, kind)
        self._respond: asyncio.Task | None = None
        self._cancel = threading.Event()
        self._turn_id = 0
        self._t0 = clock()
        self._pending_barge_t: float | None = None
        self._expected_reply_ms = 1500.0  # running estimate, endpoint → first reply word
        self._last_reaction = ""
        self._turn_reacted: dict[int, float] = {}
        self._turn_spoke: set[int] = set()
        self.recording: list[np.ndarray] = []  # outbound audio per tick (silence when idle)
        self._can_react = bool(getattr(tts, "backchannel", None)) and bool(getattr(tts, "backchannel_keys", ()))
        self._can_breathe = bool(getattr(tts, "breath", None))
        # Echo guard (for loudspeaker endpoints such as a phone browser): the
        # agent's own voice leaking back must not count as a barge-in. We
        # learn the echo return level (mic dB − played dB) while the caller is
        # quiet, and accept speech during playback only if it is clearly louder.
        self.echo_guard = echo_guard
        self._echo_margin = echo_margin_db
        self._out_levels: deque[float] = deque(maxlen=max(1, 400 // frame_ms))
        self._erl_db = -30.0  # learned: typical mic level relative to playback while caller is silent

    # ------------------------------------------------------------------ utils
    def _log(self, kind: str, detail: str = "") -> None:
        self.report.events.append((self.clock() - self._t0, kind, detail))

    @property
    def agent_active(self) -> bool:
        return bool(self._out) or (self._respond is not None and not self._respond.done())

    def _current(self) -> TurnMetrics:
        return self.report.turns[-1]

    def _enqueue(self, turn: int, pcm: np.ndarray, kind: str) -> None:
        for frame in split_frames(pcm, self.n):
            self._out.append((turn, frame, kind))

    # --------------------------------------------------------------- turn work
    async def _react(self, turn: int, metrics: TurnMetrics, utter_ms: int) -> None:
        """Backchannel while thinking, if the reply is expected to be slow."""
        tt = self.tt
        if not (tt.react and self._can_react) or self._expected_reply_ms < tt.react_if_expected_ms:
            return
        await asyncio.sleep(tt.react_after_ms / 1000.0)
        if turn != self._turn_id or self._cancel.is_set() or turn in self._turn_spoke:
            return
        keys = list(self.tts.backchannel_keys)
        # long turns invite "آهان/اوهوم" (I follow you); short ones "بله/خب"
        prefer = [k for k in keys if (k in ("ahan", "ohom")) == (utter_ms > 2200)] or keys
        choices = [k for k in prefer if k != self._last_reaction] or prefer
        key = self._rng.choice(choices)
        self._last_reaction = key
        parts = []
        if self._can_breathe and self._rng.random() < 0.6:
            parts.append(await asyncio.to_thread(self.tts.breath))
        parts.append(await asyncio.to_thread(self.tts.backchannel, key))
        if turn != self._turn_id or self._cancel.is_set() or turn in self._turn_spoke:
            return
        self._enqueue(turn, np.concatenate(parts), "reaction")
        self._turn_reacted[turn] = self.clock()
        metrics.reaction = key
        self._log("reaction", key)

    async def _speak(self, turn: int, text: str, metrics: TurnMetrics | None = None) -> None:
        pcm = await asyncio.to_thread(self.tts.synthesize, text)
        if self._cancel.is_set() or turn != self._turn_id:
            return
        if turn not in self._turn_spoke:
            self._turn_spoke.add(turn)
            if self.tt.breath_before_reply and self._can_breathe and turn not in self._turn_reacted:
                self._enqueue(turn, await asyncio.to_thread(self.tts.breath), "breath")
            elif turn in self._turn_reacted:
                self._enqueue(turn, np.zeros(int(self.sr * 0.12), np.int16), "breath")
            if metrics is not None and metrics.speech_end_t is not None and metrics.endpoint_ms:
                took = (self.clock() - metrics.speech_end_t) * 1000.0
                self._expected_reply_ms = 0.5 * self._expected_reply_ms + 0.5 * took
        self._enqueue(turn, pcm, "speech")

    async def _run_llm(self, turn: int, metrics: TurnMetrics) -> None:
        loop = asyncio.get_running_loop()
        q: asyncio.Queue[str | None] = asyncio.Queue()
        cancel = self._cancel
        history = list(self.history)

        def produce() -> None:
            try:
                for delta in self.llm.stream(history):
                    if cancel.is_set():
                        break
                    loop.call_soon_threadsafe(q.put_nowait, delta)
            except Exception as exc:  # surfaced in the call log, call continues
                loop.call_soon_threadsafe(q.put_nowait, f"\x00{type(exc).__name__}: {exc}")
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)

        threading.Thread(target=produce, daemon=True).start()
        buffer = ""
        while (delta := await q.get()) is not None:
            if delta.startswith("\x00"):
                self._log("llm_error", delta[1:])
                break
            if metrics.llm_first_t is None:
                metrics.llm_first_t = self.clock()
            buffer += delta
            ready, buffer = split_sentences(buffer)
            for chunk in ready:
                metrics.agent_text += (" " if metrics.agent_text else "") + chunk
                await self._speak(turn, chunk, metrics)
        if buffer.strip():
            metrics.agent_text += (" " if metrics.agent_text else "") + buffer.strip()
            await self._speak(turn, buffer.strip(), metrics)

    async def _respond_to(self, turn: int, transcript: "asyncio.Future[str]", metrics: TurnMetrics,
                          utter_ms: int) -> None:
        react = asyncio.create_task(self._react(turn, metrics, utter_ms))
        try:
            text = (await transcript).strip()
            metrics.asr_done_t = self.clock()
            metrics.user_text = text
            self._log("asr", text)
            if not text:
                return
            self.history.append({"role": "user", "content": text})
            await self._run_llm(turn, metrics)
            if metrics.agent_text:
                self.history.append({"role": "assistant", "content": metrics.agent_text})
            del self.history[: -self._max_history]
        finally:
            if not react.done():
                react.cancel()

    def _start_turn(self, coro_factory: Callable[[int, TurnMetrics], Awaitable[None]], metrics: TurnMetrics) -> None:
        self._turn_id += 1
        metrics.turn = self._turn_id
        self.report.turns.append(metrics)
        self._cancel = threading.Event()
        self._respond = asyncio.create_task(coro_factory(self._turn_id, metrics))

    def _barge_in(self, onset_t: float) -> None:
        playing = bool(self._out)
        self._cancel.set()
        if self._respond and not self._respond.done():
            self._respond.cancel()
        self._out.clear()
        if self.report.turns:
            self._current().interrupted = True
        if playing:
            self._pending_barge_t = onset_t
        self._log("barge_in" if playing else "interrupt_thinking")

    # ----------------------------------------------------------------- loops
    async def _pump(self, outbound: Callable[[np.ndarray], Awaitable[None]]) -> None:
        period = self.frame_ms / 1000.0
        silence = np.zeros(self.n, np.int16)
        next_t = self.clock()
        while True:
            next_t += period
            if self._out:
                turn, frame, kind = self._out.popleft()
                if turn == self._turn_id and self.report.turns:
                    m = self._current()
                    if m.first_audio_t is None:
                        m.first_audio_t = self.clock()
                        self._log("first_audio", f"turn {turn} ({kind})")
                    if kind == "speech" and m.first_speech_t is None:
                        m.first_speech_t = self.clock()
                await outbound(frame)
                self.recording.append(frame)
                self._out_levels.append(_dbfs(frame))
            else:
                self._out_levels.append(-120.0)
                if self._pending_barge_t is not None:
                    self.report.barge_in_ms.append(round((self.clock() - self._pending_barge_t) * 1000, 1))
                    self._pending_barge_t = None
                self.recording.append(silence)
            await asyncio.sleep(max(0.0, next_t - self.clock()))

    def _asr_task(self, pcm: np.ndarray) -> asyncio.Task:
        return asyncio.create_task(asyncio.to_thread(self.asr.transcribe, pcm, self.sr))

    async def run(
        self,
        inbound: AsyncIterator[np.ndarray],
        outbound: Callable[[np.ndarray], Awaitable[None]],
        control: Callable[[str], Awaitable[None]] | None = None,
    ) -> CallReport:
        """Drive the call until ``inbound`` is exhausted (hang-up)."""
        self._t0 = self.clock()
        pump = asyncio.create_task(self._pump(outbound))
        if self.greeting:
            greet = self.greeting

            async def say_greeting(turn: int, m: TurnMetrics) -> None:
                m.agent_text = greet
                await self._speak(turn, greet)

            m0 = TurnMetrics(turn=0, speech_end_t=self.clock())
            self._start_turn(say_greeting, m0)
            self._log("greeting", greet)

        collecting = False
        utterance: list[np.ndarray] = []
        speech_onset_t = 0.0
        spec: tuple[int, asyncio.Task] | None = None  # (frames covered, ASR task)
        last_utterance: np.ndarray | None = None
        echo_utterance = False
        stale: list[asyncio.Task] = []
        try:
            async for frame in inbound:
                self._preroll.append(frame)
                event = self.vad.process(frame)
                playing_ref = max(self._out_levels) if self._out_levels else -120.0
                if self.echo_guard and playing_ref > -60.0 and not collecting:
                    mic = _dbfs(frame)
                    if mic > -70.0:  # learn the echo path level while the caller is quiet
                        self._erl_db = 0.95 * self._erl_db + 0.05 * (mic - playing_ref)
                if event is VADEvent.SPEECH_START and self.echo_guard and playing_ref > -60.0:
                    recent = np.concatenate(list(self._preroll)[-max(1, self.vad.config.start_ms // self.frame_ms):])
                    if _dbfs(recent) < playing_ref + self._erl_db + self._echo_margin:
                        echo_utterance = True
                        self._log("echo_ignored", f"{_dbfs(recent):.0f} dB vs echo {playing_ref + self._erl_db:.0f} dB")
                        continue
                if event is VADEvent.SPEECH_END and echo_utterance:
                    echo_utterance = False
                    continue
                if echo_utterance:
                    continue
                if event is VADEvent.SPEECH_START:
                    speech_onset_t = self.clock() - self.vad.config.start_ms / 1000.0
                    collecting = True
                    utterance = list(self._preroll)
                    spec = None
                    self._log("speech_start")
                    if self.agent_active:
                        thinking = self._turn_id not in self._turn_spoke and last_utterance is not None
                        self._barge_in(speech_onset_t)
                        if thinking:
                            # The caller is still talking (added words while we
                            # were thinking): this is a continuation, not an
                            # interruption. Keep what they already said.
                            utterance = [last_utterance] + utterance
                            self._current().interrupted = False
                            self._log("continuation")
                        if control:
                            await control("clear")
                elif collecting:
                    utterance.append(frame)
                if collecting and event is None:
                    silent = self.vad.trailing_silence_ms
                    stale = [t for t in stale if not t.done()]
                    if spec is not None and silent < self.tt.speculative_ms:
                        stale.append(spec[1])  # caller kept talking: discard the early pass
                        spec = None
                        self._log("asr_speculative_discarded")
                    elif (spec is None and self.tt.speculative_ms < self.vad.config.end_ms
                          and silent >= self.tt.speculative_ms and not stale):
                        spec = (len(utterance), self._asr_task(np.concatenate(utterance)))
                        self._log("asr_speculative_start")
                if event is VADEvent.SPEECH_END:
                    collecting = False
                    self._log("speech_end", f"{self.vad.last_speech_ms} ms")
                    if self.vad.last_was_click:
                        if spec is not None:
                            stale.append(spec[1])
                        spec = None
                        continue
                    m = TurnMetrics(turn=0, speech_end_t=self.clock(), endpoint_ms=self.vad.config.end_ms)
                    last_utterance = np.concatenate(utterance)
                    if spec is not None:
                        transcript, m.asr_speculative = spec[1], True
                    else:
                        transcript = self._asr_task(last_utterance)
                    spec = None
                    utter_ms = self.vad.last_speech_ms
                    self._start_turn(
                        lambda t, mm, tr=transcript, u=utter_ms: self._respond_to(t, tr, mm, u), m
                    )
            # inbound ended = caller hung up: stop everything in flight.
        finally:
            self._cancel.set()
            for task in (self._respond, pump):
                if task and not task.done():
                    task.cancel()
            await asyncio.gather(*(t for t in (self._respond, pump) if t), return_exceptions=True)
            self._log("hangup")
        return self.report

    def recording_pcm(self) -> np.ndarray:
        return np.concatenate(self.recording) if self.recording else np.zeros(0, np.int16)
