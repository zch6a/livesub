"""Voice activity detection: is someone talking right now?

An energy detector with an adaptive noise floor. Each 30 ms frame's level is compared with a running
estimate of the background: the estimate falls quickly (so it finds the quiet between words) and rises
slowly, and only while the level is near it (so speech itself does not drag it up). A frame is "voice"
when it is clearly above the background and above an absolute minimum. Hysteresis turns frames into
speech: a few voice frames in a row to start (a click is not speech), half a second of non-voice to
stop (the pause between words is not the end of speech).

It is deliberately simple — no model, no download — and it only decides *when* to run the recognizer.
Music and steady noise above the floor count as voice; Whisper then hears no words, and nothing is
shown.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audio import RATE


@dataclass(frozen=True)
class VadConfig:
    frame_ms: int = 30
    margin_db: float = 9.0  # how far above the background a frame must be to count as voice
    min_db: float = -50.0  # and never quieter than this (dBFS)
    start_ms: int = 90  # voice this long in a row starts speech
    hangover_ms: int = 500  # non-voice this long in a row ends it


class EnergyVAD:
    def __init__(self, rate: int = RATE, config: VadConfig | None = None):
        self.config = c = config or VadConfig()
        self.frame = int(rate * c.frame_ms / 1000)
        self.start_frames = max(1, round(c.start_ms / c.frame_ms))
        self.hang_frames = max(1, round(c.hangover_ms / c.frame_ms))
        self.noise_db: float | None = None
        self.speaking = False
        self._run = 0
        self._rest = np.zeros(0, np.float32)
        self.level_db = -120.0

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
        if self.noise_db is None:
            self.noise_db = level
        elif level < self.noise_db:
            self.noise_db += 0.3 * (level - self.noise_db)
        elif level < self.noise_db + c.margin_db:
            self.noise_db += 0.05 * (level - self.noise_db)
        voice = level > max(self.noise_db + c.margin_db, c.min_db)
        if voice != self.speaking:
            self._run += 1
            if self._run >= (self.start_frames if voice else self.hang_frames):
                self.speaking = voice
                self._run = 0
        else:
            self._run = 0
        return self.speaking
