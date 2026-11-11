"""Turning a recognizer that sees a growing window into captions that do not flicker.

Live, the recognizer runs again every second on everything said since the last cut. Each run is a
fresh guess, and the end of it is the least reliable part: the last word is often cut off mid-sound and
guessed wrong. Showing every guess makes the captions jump. *Local agreement* (Macháček et al., 2023,
"Turning Whisper into Real-Time Transcription System") commits a word only when two runs in a row agree
on it, from the start: the longest common prefix of the previous guess and the new one. Committed words
never change; the rest is shown as tentative.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Word:
    start: float  # seconds, on the stream's clock
    end: float
    text: str  # as the recognizer gives it: with a leading space between Latin words, none for CJK


def norm(text: str) -> str:
    """Compare words without case or punctuation: "Hello," and "hello" agree."""
    return re.sub(r"[^\w]", "", text.lower())


class Agreement:
    def __init__(self, ngram: int = 5):
        self.committed: list[Word] = []
        self.buffer: list[Word] = []  # the previous guess, after the committed words
        self.new: list[Word] = []
        self.last_start = -1.0  # start of the last committed word
        self.last_end = 0.0
        self.ngram = ngram

    def insert(self, words: list[Word], window_start: float | None = None) -> None:
        """A new guess for the window that starts at `window_start`. Words the committed text already
        covers are dropped."""
        # Where does the new guess continue the committed text? Found by the words, not by their times:
        # Whisper's timestamps shift between readings, and in a real run it once returned only the
        # continuation ("harder than they look") with "harder" placed 0.7 s too early — before the last
        # committed word — so filtering by time silently lost words. Times only break ties.
        new = list(words)
        texts = [norm(w.text) for w in new]
        # Only committed words whose audio is in this window can be repeated by it. After a pause the
        # window starts afresh, and matching the old sentence's last word dropped a new sentence's first
        # ("Thank you." then "[You] know, ...").
        heard = self.committed if window_start is None else [w for w in self.committed if w.end > window_start - 0.3]
        tail = [norm(w.text) for w in heard[-self.ngram:]]
        start = None
        # 1. After the longest committed tail the guess repeats ("... the cat sat" then "the cat sat on").
        for k in range(min(len(tail), len(new)), 0, -1):
            hits = [i for i in range(len(new) - k + 1) if texts[i:i + k] == tail[-k:]]
            hits = [i for i in hits if abs(new[i + k - 1].end - self.last_end) < 2.0]
            if k == 1:
                # One word is weak evidence: "the" or "的" further on would match the tail and drop the
                # words before it. Accept it only at the head of the guess, as whisper_streaming does.
                hits = [i for i in hits if i <= 1]
            if hits:
                start = min(hits, key=lambda i: abs(new[i + k - 1].end - self.last_end)) + k
                break
        # 2. Or where the previous, uncommitted guess began.
        if start is None and self.buffer:
            first = norm(self.buffer[0].text)
            start = next((i for i, w in enumerate(new) if texts[i] == first and abs(w.start - self.buffer[0].start) < 1.5), None)
        # 3. Or, with nothing to align on, by time: words that end after the last committed one began.
        self.new = new[start:] if start is not None else [w for w in new if w.end > self.last_start]

    def flush(self) -> list[Word]:
        """Commit the words on which the last two guesses agree; returns them."""
        out: list[Word] = []
        while self.new and self.buffer and norm(self.new[0].text) == norm(self.buffer[0].text):
            out.append(self.new.pop(0))
            self.buffer.pop(0)
        self.buffer, self.new = self.new, []
        self._commit(out)
        return out

    def sentence_cut(self, after: float) -> float | None:
        """The end of the last committed sentence that ends after `after`, if any: the place to cut the
        audio window so that the next reading starts at the beginning of a sentence (as in the paper's
        "sentence trimming"), with the context Whisper reads best."""
        from .captions import SENTENCE_END

        for i in range(len(self.committed) - 1, -1, -1):
            w = self.committed[i]
            if w.end <= after:
                return None
            if w.text.rstrip().endswith(SENTENCE_END):
                nxt = self.committed[i + 1].start if i + 1 < len(self.committed) else (self.buffer[0].start if self.buffer else w.end)
                return min(w.end, nxt)
        return None

    def cut_point(self) -> float:
        """Where the audio before can be dropped: after the last committed word, but not past the first
        word still being recognized (the committed word's end time may be stretched over it)."""
        cut = self.last_end
        if self.buffer:
            cut = min(cut, self.buffer[0].start)
        return max(cut, self.last_start)

    def tentative(self) -> list[Word]:
        return list(self.buffer)

    def finish(self) -> list[Word]:
        """Speech ended: there will be no second opinion, so commit the last guess as it is."""
        out, self.buffer = self.buffer, []
        self._commit(out)
        return out

    def _commit(self, words: list[Word]) -> None:
        if words:
            self.committed.extend(words)
            self.last_start = words[-1].start
            self.last_end = words[-1].end
            del self.committed[:-50]  # only the tail is ever compared or used as a prompt


def join(words: list[Word]) -> str:
    return re.sub(r"\s+", " ", "".join(w.text for w in words)).strip()
