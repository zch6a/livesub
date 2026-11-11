"""The live loop: audio in, captions out.

    audio blocks ─▶ VAD ─▶ speech buffer ─(every `step` s)─▶ recognizer ─▶ local agreement
                                                                              │ committed words
                                                         partial text ◀──────┤
                                                                              ▼
                                                                    cue builder ─▶ cues ─▶ translator

While nobody speaks, audio only fills a short pre-roll (so the first syllable is not lost when the VAD
notices speech a few frames late). During speech the buffer grows and the recognizer re-reads it every
`step` seconds. When speech ends, the last guess is committed. A buffer that grows past `max_buffer`
(someone talking without pause) is cut after the last committed word: Whisper slows down and wanders on
long windows, and committed words need no second reading. The same loop serves files and live audio.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from dataclasses import replace

import numpy as np

from .asr import Transcriber
from .audio import RATE
from .captions import Cue, CueBuilder
from .streaming import Agreement, Word, join
from .vad import EnergyVAD


class LiveCaptioner:
    def __init__(
        self,
        transcriber: Transcriber,
        *,
        step: float = 1.0,
        max_buffer: float = 12.0,
        preroll: float = 0.3,
        language: str | None = None,
        vad: EnergyVAD | None = None,
        builder: CueBuilder | None = None,
        on_cue: Callable[[Cue], None] | None = None,
        on_partial: Callable[[str, str], None] | None = None,
    ):
        if not 0.2 <= step <= max_buffer / 2:
            raise ValueError("step must be at least 0.2 s and at most half of max_buffer")
        self.transcriber = transcriber
        self.step, self.max_buffer, self.preroll = step, max_buffer, preroll
        self.language = language
        self.vad = vad or EnergyVAD()
        self.builder = builder or CueBuilder()
        self.on_cue, self.on_partial = on_cue, on_partial
        self.agreement = Agreement()
        self.cues: list[Cue] = []
        self.time = 0.0  # seconds of audio fed so far
        self.runs = 0  # recognizer calls, for tests and statistics
        self.longest_window = 0.0
        self._pre: list[np.ndarray] = []
        self._buf: list[np.ndarray] = []
        self._buf_start = 0.0
        self._active = False
        self._last_run = 0.0

    # -- feeding ------------------------------------------------------------------------------------

    def feed(self, block: np.ndarray) -> None:
        """Feed 16 kHz mono float32 audio, in blocks of any size (about 0.1 s is typical)."""
        block = np.asarray(block, dtype=np.float32)
        self.vad.process(block)
        self.time += len(block) / RATE
        if not self._active:
            self._pre.append(block)
            while len(self._pre) > 1 and sum(map(len, self._pre[1:])) >= self.preroll * RATE:
                self._pre.pop(0)
            if self.vad.speaking:
                self._buf, self._pre = self._pre, []
                self._buf_start = self.time - sum(map(len, self._buf)) / RATE
                self._active = True
                self._last_run = self.time
            return
        self._buf.append(block)
        if not self.vad.speaking:
            self._end_of_speech()
            return
        if self.time - self._last_run >= self.step:
            self._last_run = self.time
            self._recognize()
            if self.time - self._buf_start > self.max_buffer:
                self._trim()

    def close(self) -> list[Cue]:
        """End of the stream: finish what is being said. Returns every cue."""
        if self._active:
            self._end_of_speech()
        self._emit(self.builder.flush())
        return self.cues

    # -- internals ----------------------------------------------------------------------------------

    def _recognize(self) -> None:
        audio = np.concatenate(self._buf) if self._buf else np.zeros(0, np.float32)
        self.runs += 1
        self.longest_window = max(self.longest_window, len(audio) / RATE)
        # Only text whose audio has been cut from the window. With committed words still in the window
        # in its prompt, Whisper tends to skip them and smear the times of what follows.
        prompt = join([w for w in self.agreement.committed if w.end <= self._buf_start + 0.05])[-200:]
        words = self.transcriber.transcribe(audio, offset=self._buf_start, prompt=prompt, language=self.language)
        self.agreement.insert(words)
        self._emit(self.builder.add(self.agreement.flush()))
        self._partial()

    def _end_of_speech(self) -> None:
        self._recognize()
        self._emit(self.builder.add(self.agreement.finish()))
        self._emit(self.builder.flush())
        self._active = False
        self._buf = []
        self._pre = []
        self._partial()

    def _trim(self) -> None:
        cut = self.agreement.cut_point()
        if cut <= self._buf_start + 1.0:
            # Nothing settled in the whole window (or barely): settle the current guess rather than
            # let the window grow without bound.
            self._emit(self.builder.add(self.agreement.finish()))
            cut = self.agreement.cut_point()
        if cut <= self._buf_start:
            cut = self.time - self.step  # no words at all: keep only the last step of audio
        audio = np.concatenate(self._buf)
        drop = min(len(audio), max(0, int((cut - self._buf_start) * RATE)))
        self._buf = [audio[drop:]]
        self._buf_start += drop / RATE

    def _emit(self, cues: list[Cue]) -> None:
        for c in cues:
            self.cues.append(c)
            if self.on_cue:
                self.on_cue(c)

    def _partial(self) -> None:
        if self.on_partial:
            # Committed words of the cue being built, and the words still being recognized.
            self.on_partial(join(self.builder.pending()), join(self.agreement.tentative()))


def run_file(captioner: LiveCaptioner, audio: np.ndarray, block: float = 0.1) -> list[Cue]:
    """Feed a whole recording through the live loop, as if it were playing now."""
    n = int(block * RATE)
    for i in range(0, len(audio), n):
        captioner.feed(audio[i:i + n])
    return captioner.close()


class TranslationWorker:
    """Translates cues on a background thread, so a slow translator never holds up the captions.
    `on_done` receives each cue again with its translation (or unchanged, if translation failed)."""

    def __init__(self, translator, target: str, on_done: Callable[[Cue], None], context: int = 2,
                 on_error: Callable[[Exception], None] | None = None):
        self.translator, self.target, self.on_done, self.on_error = translator, target, on_done, on_error
        self.context: list[str] = []
        self.keep = context
        self.q: queue.Queue[Cue | None] = queue.Queue()
        self.thread = threading.Thread(target=self._loop, daemon=True, name="livesub-translate")
        self.thread.start()

    def submit(self, cue: Cue) -> None:
        self.q.put(cue)

    def close(self, timeout: float | None = None) -> None:
        self.q.put(None)
        self.thread.join(timeout)

    def _loop(self) -> None:
        while (cue := self.q.get()) is not None:
            try:
                text = self.translator.translate(cue.text, self.target, self.context[-self.keep:])
                done = replace(cue, translation=text)
            except Exception as e:  # noqa: BLE001 - a failed line must not stop the captions
                done = cue
                if self.on_error:
                    self.on_error(e)
            self.context.append(cue.text)
            del self.context[:-self.keep]
            self.on_done(done)


__all__ = ["LiveCaptioner", "TranslationWorker", "run_file", "Word"]
