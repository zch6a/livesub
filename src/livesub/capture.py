"""Live audio: what the computer is playing (loopback) or a microphone, as 16 kHz mono blocks.

Uses the `soundcard` package: WASAPI loopback on Windows and PulseAudio/PipeWire monitor sources on
Linux. macOS has no system loopback; install a virtual device such as BlackHole and pick it as a
microphone.
"""

from __future__ import annotations

from collections.abc import Iterator

from .audio import RATE, Resampler, to_mono


def _soundcard():
    try:
        import soundcard
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise RuntimeError("live capture needs the soundcard package: pip install \"livesub[capture]\"") from e
    return soundcard


def list_devices() -> dict[str, list[str]]:
    sc = _soundcard()
    return {
        "speakers (loopback sources)": [s.name for s in sc.all_speakers()],
        "microphones": [m.name for m in sc.all_microphones(include_loopback=False)],
    }


def blocks(source: str = "loopback", device: str | None = None, block: float = 0.1, rate: int = 48_000) -> Iterator:
    """Yield float32 blocks at 16 kHz forever. `source` is "loopback" or "mic"; `device` is part of
    a device name (default: the system default)."""
    sc = _soundcard()
    if source == "loopback":
        speaker = sc.get_speaker(device) if device else sc.default_speaker()
        mic = sc.get_microphone(str(speaker.name), include_loopback=True)
    elif source == "mic":
        mic = sc.get_microphone(device) if device else sc.default_microphone()
    else:
        raise ValueError('source must be "loopback" or "mic"')
    resample = Resampler(rate, RATE)
    with mic.recorder(samplerate=rate, blocksize=int(rate * block)) as recorder:
        while True:
            yield resample(to_mono(recorder.record(numframes=int(rate * block))))
