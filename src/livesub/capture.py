"""Live audio: what the computer is playing (loopback) or a microphone, as 16 kHz mono blocks.

Uses the `soundcard` package: WASAPI loopback on Windows and PulseAudio/PipeWire monitor sources on
Linux. macOS has no system loopback; install a virtual device such as BlackHole and pick it as a
microphone.
"""

from __future__ import annotations

import sys
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


def open_source(source: str = "loopback", device: str | None = None):
    """The recording device, found now so that a wrong name fails before the model loads. `source` is
    "loopback" (what the computer plays) or "mic"; `device` is part of a device name."""
    if source not in ("loopback", "mic"):
        raise ValueError('source must be "loopback" or "mic"')
    if source == "loopback" and sys.platform == "darwin":
        raise RuntimeError("macOS has no system loopback: install a virtual audio device such as BlackHole, send the "
                           "sound to it, and use --source mic --input-device BlackHole")
    sc = _soundcard()
    try:
        if source == "loopback":
            speaker = sc.get_speaker(device) if device else sc.default_speaker()
            return sc.get_microphone(str(speaker.name), include_loopback=True)
        return sc.get_microphone(device) if device else sc.default_microphone()
    except (IndexError, RuntimeError, ValueError) as e:
        raise RuntimeError(f"no {'speaker' if source == 'loopback' else 'microphone'} matching {device!r} "
                           "(see `livesub devices`)" if device else f"no default {source} device: {e}") from e


def blocks(source: str = "loopback", device: str | None = None, block: float = 0.1, rate: int = 48_000, mic=None) -> Iterator:
    """Yield float32 blocks at 16 kHz until the caller stops. Read them on a thread of their own: the
    device buffer is small, and audio arriving while nobody reads it is lost."""
    mic = mic or open_source(source, device)
    resample = Resampler(rate, RATE)
    # A one-second device buffer: room for the reading thread to be late now and then.
    with mic.recorder(samplerate=rate, blocksize=rate) as recorder:
        while True:
            yield resample(to_mono(recorder.record(numframes=int(rate * block))))
