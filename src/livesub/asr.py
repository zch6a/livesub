"""Speech recognizers. Anything with a `transcribe` method of this shape will do."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .streaming import Word


class Transcriber(Protocol):
    def transcribe(self, audio: np.ndarray, *, offset: float = 0.0, prompt: str = "", language: str | None = None) -> list[Word]:
        """Words in 16 kHz mono `audio`, with times on the stream's clock (`offset` + time in the audio)."""
        ...


class FasterWhisper:
    """faster-whisper (CTranslate2): runs Whisper several times faster than the reference code, on
    CPU or a CUDA GPU. The model is downloaded once from Hugging Face the first time it is used, and
    everything after that is offline."""

    def __init__(self, model: str = "small", device: str = "auto", compute_type: str = "default", beam_size: int = 5):
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:  # pragma: no cover - depends on the environment
            raise RuntimeError("faster-whisper is not installed: pip install \"livesub[whisper]\"") from e
        self.model = WhisperModel(model, device=device, compute_type=compute_type)
        self.beam_size = beam_size
        self.language: str | None = None

    def transcribe(self, audio: np.ndarray, *, offset: float = 0.0, prompt: str = "", language: str | None = None) -> list[Word]:
        segments, info = self.model.transcribe(
            audio, language=language, initial_prompt=prompt or None, beam_size=self.beam_size,
            word_timestamps=True, vad_filter=False, condition_on_previous_text=False,
        )
        self.language = info.language
        words: list[Word] = []
        for s in segments:
            # Whisper's best-known failure: on silence or music it invents text ("Thanks for watching!").
            # A segment it thinks is probably not speech, and is unsure about, is dropped.
            if s.no_speech_prob > 0.6 and s.avg_logprob < -1.0:
                continue
            words.extend(Word(offset + w.start, offset + w.end, w.word) for w in (s.words or []))
        return words
