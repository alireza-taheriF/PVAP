import numpy as np

from pvap.audio.pcm import split_frames
from pvap.vad.energy_vad import EnergyVAD, VADConfig, VADEvent

SR, N = 8000, 160


def noise(ms: int, dbfs: float = -62.0, seed: int = 0) -> np.ndarray:
    amp = 32768 * 10 ** (dbfs / 20)
    return (np.random.default_rng(seed).standard_normal(SR * ms // 1000) * amp).astype(np.int16)


def voice(ms: int, level: float = 0.2) -> np.ndarray:
    t = np.arange(SR * ms // 1000) / SR
    # amplitude-modulated harmonic tone, roughly speech-like in level
    sig = sum(np.sin(2 * np.pi * f * t) / k for k, f in enumerate((150, 300, 450, 900), 1))
    return (level * 32767 * sig / 2.1 * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t))).astype(np.int16)


def run(vad: EnergyVAD, pcm: np.ndarray) -> list[tuple[int, VADEvent]]:
    events = []
    for i, f in enumerate(split_frames(pcm, N)):
        e = vad.process(f)
        if e:
            events.append((i * 20, e))
    return events


def test_detects_start_and_end_with_expected_timing() -> None:
    vad = EnergyVAD(VADConfig(start_ms=100, end_ms=600))
    pcm = np.concatenate([noise(1000), voice(1200), noise(1500, seed=1)])
    ev = run(vad, pcm)
    assert [e for _, e in ev] == [VADEvent.SPEECH_START, VADEvent.SPEECH_END]
    start_t, end_t = ev[0][0], ev[1][0]
    assert 1000 + 60 <= start_t <= 1000 + 120  # 100 ms of speech needed
    assert 2200 + 560 <= end_t <= 2200 + 640  # 600 ms of silence needed
    assert not vad.last_was_click


def test_noise_alone_never_triggers() -> None:
    vad = EnergyVAD()
    assert run(vad, noise(5000, dbfs=-50)) == []


def test_adapts_to_a_louder_noise_floor() -> None:
    # Line noise well above the absolute gate must become the floor, not "speech".
    vad = EnergyVAD()
    assert run(vad, noise(3000, dbfs=-40)) == []
    assert vad.noise_dbfs > -45


def test_noise_step_up_mid_call_settles() -> None:
    vad = EnergyVAD()
    ev = run(vad, np.concatenate([noise(2000, -62), noise(10000, -38, seed=3)]))
    # at most one spurious start/end pair while the floor re-adapts
    assert len(ev) <= 2
    assert not vad.in_speech


def test_short_click_is_flagged() -> None:
    vad = EnergyVAD(VADConfig(start_ms=40, min_utterance_ms=200))
    ev = run(vad, np.concatenate([noise(500), voice(100), noise(1000)]))
    assert [e for _, e in ev] == [VADEvent.SPEECH_START, VADEvent.SPEECH_END]
    assert vad.last_was_click


def test_long_utterance_is_not_cut_by_floor_adaptation() -> None:
    vad = EnergyVAD()
    ev = run(vad, np.concatenate([noise(1000), voice(4500), noise(1500, seed=2)]))
    assert [e for _, e in ev] == [VADEvent.SPEECH_START, VADEvent.SPEECH_END]
    assert ev[1][0] > 1000 + 4500
