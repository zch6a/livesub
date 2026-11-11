"""`livesub live`: where the captions are shown, what is kept, and the two threads that run it all."""

from __future__ import annotations

import contextlib
import queue
import sys
import threading
from dataclasses import dataclass, field

from .captions import Cue
from .pipeline import LiveCaptioner, TranslationWorker


class ConsoleView:
    """Captions in the terminal: the line being recognized is rewritten in place, finished cues scroll."""

    def show(self, committed: str, tentative: str) -> None:
        line = (committed + " " + tentative).strip()
        print("\r\033[K" + line[-110:], end="", flush=True)

    def show_cue(self, cue: Cue) -> None:
        print("\r\033[K" + cue.text, flush=True)

    def show_translation(self, cue: Cue) -> None:
        print("\r\033[K   → " + (cue.translation or ""), flush=True)

    def run(self, stop: threading.Event) -> None:
        while not stop.wait(0.2):
            pass

    def close(self) -> None:
        pass


class OverlayView:
    """Captions in the floating overlay. Between sentences it keeps showing the last finished cue."""

    def __init__(self, overlay):
        self.overlay = overlay
        self.last = ""

    def show(self, committed: str, tentative: str) -> None:
        if committed or tentative:
            self.overlay.show_text(committed, tentative)
        else:
            self.overlay.show_text(self.last)

    def show_cue(self, cue: Cue) -> None:
        self.last = cue.text

    def show_translation(self, cue: Cue) -> None:
        if cue.translation:
            self.overlay.show_translation(cue.translation)

    def run(self, stop: threading.Event) -> None:
        self.overlay.run()  # returns when the window is closed

    def close(self) -> None:
        self.overlay.close()


@dataclass
class Transcript:
    """Every finished cue, in order, with translations filled in as they arrive. Thread-safe."""

    cues: list[Cue] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, cue: Cue) -> None:
        with self.lock:
            self.cues.append(cue)

    def translated(self, cue: Cue) -> None:
        with self.lock:
            for i in range(len(self.cues) - 1, -1, -1):
                if self.cues[i].start == cue.start and self.cues[i].text == cue.text:
                    self.cues[i] = cue
                    return

    def snapshot(self) -> list[Cue]:
        with self.lock:
            return list(self.cues)


class LiveSession:
    """Two threads: one only reads the sound device (its buffer is small, and audio arriving while
    nobody reads it is lost), the other recognizes. While readings run slow, blocks queue up; the
    recognizer then takes them all in and reads the window once, instead of once per block."""

    def __init__(self, blocks, transcriber, view, *, step: float, language: str | None,
                 translator=None, target: str | None = None):
        self.blocks, self.view = blocks, view
        self.transcript = Transcript()
        self.stop = threading.Event()
        self.failure: str | None = None
        self.audio: queue.Queue = queue.Queue()
        self.worker = TranslationWorker(translator, target, self._translated,
                                        on_error=lambda e: print(f"\ntranslation failed: {e}", file=sys.stderr)) if translator else None
        self.captioner = LiveCaptioner(transcriber, step=step, language=language, on_cue=self._cue, on_partial=view.show)

    def run(self) -> None:
        threads = [threading.Thread(target=self._record, daemon=True, name="livesub-capture"),
                   threading.Thread(target=self._recognize, daemon=True, name="livesub-recognize")]
        for t in threads:
            t.start()
        with contextlib.suppress(KeyboardInterrupt):  # Ctrl+C is how the console view stops
            self.view.run(self.stop)
        self.stop.set()
        self.audio.put(None)
        for t in threads:
            t.join(60)  # the last reading may take a few seconds on a CPU
        if self.worker:
            self.worker.close()  # every cue is queued by now: wait for their translations

    def _cue(self, cue: Cue) -> None:
        self.transcript.add(cue)
        self.view.show_cue(cue)
        if self.worker:
            self.worker.submit(cue)

    def _translated(self, cue: Cue) -> None:
        self.transcript.translated(cue)
        self.view.show_translation(cue)

    def _fail(self, what: str, e: BaseException) -> None:
        self.failure = self.failure or f"{what}: {e}"
        self.stop.set()
        self.view.close()

    def _record(self) -> None:
        try:
            for block in self.blocks:
                self.audio.put(block)
                if self.stop.is_set():
                    break
        except Exception as e:  # noqa: BLE001 - report and stop rather than die silently in a thread
            self._fail("audio capture stopped", e)
        finally:
            self.audio.put(None)

    def _recognize(self) -> None:
        try:
            while (block := self.audio.get()) is not None and not self.stop.is_set():
                self.captioner.feed(block, recognize=self.audio.empty())
            self.captioner.close()  # finish what was being said
        except Exception as e:  # noqa: BLE001
            self._fail("recognition failed", e)
        finally:
            self.stop.set()
            self.view.close()
