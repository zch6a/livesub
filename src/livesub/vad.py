"""Voice activity detection: is someone talking right now?

An energy detector with an adaptive estimate of the background. Each 30 ms frame's level is compared
with that estimate: it falls quickly (so it finds the quiet between words) and rises slowly, and only
while nobody is speaking (so speech itself does not drag it up). A frame is "voice" when it is clearly
above the background and above an absolute minimum. Hysteresis turns frames into speech: a few voice
frames in a row to start (a click is not speech), half a second of non-voice to stop (the pause between
words is not the end of speech). Once speech has started, a frame needs to be only half as far above
the background to keep it going.

It is deliberately simple — no model, no download — and it only decides *when* to run the recognizer.
Music and steady noise above the background count as voice for a while; Whisper hears no words in them,
and after 15 s without a pause the background is re-estimated.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .audio import RATE


@dataclass(frozen=True)
class VadConfig:
    frame_ms: int = 30
    margin_db: float = 9.0  # how far above the background a frame must be to start speech
    stay_db: float = 4.5  # and to continue it
    min_db: float = -50.0  # and never quieter than this (dBFS)
    start_ms: int = 90  # voice this long in a row starts speech
    hangover_ms: int = 500  # non-voice this long in a row ends it
    initial_db: float = -45.0  # the background is assumed no louder than this until it is heard
    recheck_s: float = 15.0  # "speech" without a pause for this long: re-estimate the background
    silence_db: float = -120.0  # quieter than this is digital silence (muted output), not a background


class EnergyVAD:
    def __init__(self, rate: int = RATE, config: VadConfig | None = None):
        self.config = c = config or VadConfig()
        self.frame = int(rate * c.frame_ms / 1000)
        self.start_frames = max(1, round(c.start_ms / c.frame_ms))
        self.hang_frames = max(1, round(c.hangover_ms / c.frame_ms))
        self.recheck_frames = max(1, round(c.recheck_s * 1000 / c.frame_ms))
        self.noise_db: float | None = None
        self.speaking = False
        self.level_db = -120.0
        self._run = 0
        self._speech_frames = 0
        self._recent: deque[float] = deque(maxlen=max(1, round(5000 / c.frame_ms)))  # the last 5 s
        self._rest = np.zeros(0, np.float32)

    def process(self, block: np.ndarray) -> list[bool]:
        """Feed audio; returns the speaking state after each complete frame (partial frames carry over)."""
        x = np.concatenate([self._rest, np.asarray(block, dtype=np.float32)])
        n = len(x) // self.frame
        self._rest = x[n * self.frame:]
        states = []
        for i in range(n):
            frame = x[i * self.frame:(i + 1) * self.frame]
            states.append(self._frame(float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))))
        return states

    def _frame(self, rms: float) -> bool:
        c = self.config
        level = 20 * np.log10(rms + 1e-10)
        self.level_db = level
        if level < c.silence_db:
            # Exact zeros (nothing playing) say nothing about the background. Learning a floor of
            # -200 dB from them once made a quiet hum count as speech for ever after.
            voice = False
        else:
            self._background(level)
            # Two thresholds: clearly above the background to start, a little above it to go on. Quiet
            # syllables in a noisy room are only a few dB above the noise; with one threshold, speech at
            # 10 dB SNR kept stopping mid-sentence and Whisper got 3-second scraps (20% word errors).
            margin = c.stay_db if self.speaking else c.margin_db
            voice = level > max(self.noise_db + margin, c.min_db)
        if voice != self.speaking:
            self._run += 1
            if self._run >= (self.start_frames if voice else self.hang_frames):
                self.speaking = voice
                self._run = 0
        else:
            self._run = 0
        self._speech_frames = self._speech_frames + 1 if self.speaking else 0
        return self.speaking

    def _background(self, level: float) -> None:
        c = self.config
        self._recent.append(level)
        if self.noise_db is None:
            # Capped, so that a recording that starts mid-word still has its first word detected.
            self.noise_db = min(level, c.initial_db)
        elif level < self.noise_db:
            self.noise_db += 0.3 * (level - self.noise_db)
        elif level < self.noise_db + c.margin_db and not self.speaking:
            self.noise_db += 0.05 * (level - self.noise_db)
        elif self._speech_frames >= self.recheck_frames and self._speech_frames % 33 == 0:
            # Nobody talks for 15 s without a pause: a steady sound (music, a fan) has been taken for
            # speech. Speech has quiet moments between words, so its 10th percentile is the background;
            # a steady sound has none, so this moves the background up to it.
            self.noise_db = max(self.noise_db, float(np.percentile(self._recent, 10)))
