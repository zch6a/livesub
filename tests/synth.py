"""Synthetic test material: speech-like audio for a script of words, and a recognizer that "hears"
those words the way Whisper does — including a wrong guess for a word cut off at the window's end."""

from __future__ import annotations

import numpy as np

from livesub.audio import RATE
from livesub.streaming import Word


def script(text: str, start: float, word: float = 0.32, gap: float = 0.08) -> list[Word]:
    """Words of `text` one after another from `start`; Latin words get Whisper's leading space."""
    out, t = [], start
    for w in text.split():
        out.append(Word(round(t, 3), round(t + word, 3), " " + w))
        t += word + gap
    return out


def cjk_script(text: str, start: float, char: float = 0.22) -> list[Word]:
    return [Word(round(start + i * char, 3), round(start + (i + 1) * char - 0.02, 3), ch) for i, ch in enumerate(text)]


def render(words: list[Word], duration: float, *, noise_db: float = -65.0, speech_db: float = -20.0, seed: int = 0) -> np.ndarray:
    """Background noise everywhere, and a syllable-rate-modulated noise burst for every word."""
    rng = np.random.default_rng(seed)
    n = int(duration * RATE)
    x = rng.standard_normal(n).astype(np.float32) * 10 ** (noise_db / 20)
    amp = 10 ** (speech_db / 20)
    for w in words:
        a, b = int(w.start * RATE), min(n, int(w.end * RATE))
        if b <= a:
            continue
        t = np.arange(b - a) / RATE
        envelope = np.sin(np.pi * t / (w.end - w.start)) ** 0.5 * (0.7 + 0.3 * np.sin(2 * np.pi * 5 * t))
        x[a:b] += (rng.standard_normal(b - a) * amp * envelope).astype(np.float32)
    return x


class ScriptedTranscriber:
    """Returns the scripted words inside the audio window. A word still being spoken when the window
    ends comes back as a wrong, truncated guess ("hel-"), as a real recognizer's would."""

    def __init__(self, timeline: list[Word], unstable_tail: bool = True, whisper_times: bool = True):
        self.timeline = timeline
        self.unstable_tail = unstable_tail
        self.whisper_times = whisper_times
        self.calls: list[tuple[float, float, str]] = []

    def transcribe(self, audio, *, offset=0.0, prompt="", language=None):
        end = offset + len(audio) / RATE
        self.calls.append((offset, end, prompt))
        out = [w for w in self.timeline if w.start >= offset - 0.05 and w.end <= end]
        if self.unstable_tail and len(self.calls) % 2:
            # A word still being spoken at the window's end: a wrong guess on odd readings; left out on
            # even ones, where (below) the word before it is stretched over it, as Whisper often does.
            cut = [w for w in self.timeline if w.start < end - 0.05 < w.end and w.start >= offset - 0.05]
            for w in cut:
                stem = w.text.strip()
                out.append(Word(w.start, end, (" " if w.text.startswith(" ") else "") + stem[: max(1, len(stem) // 2)] + "-"))
        if self.whisper_times and out:
            # Like Whisper's word timestamps: each reading shifts them by up to a tenth of a second, and
            # the last word of the window is stretched to the window's end.
            j = [0.1, -0.1, 0.08, -0.06, 0.0][len(self.calls) % 5]
            out = [Word(round(w.start + j, 3), round(w.end + j, 3), w.text) for w in out]
            out[-1] = Word(out[-1].start, max(out[-1].end, end), out[-1].text)
        return out
