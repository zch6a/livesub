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
        self.last_end = 0.0
        self.ngram = ngram

    def insert(self, words: list[Word]) -> None:
        """A new guess for the current window. Words the committed text already covers are dropped."""
        new = [w for w in words if w.start > self.last_end - 0.1]
        # The window usually starts before the last committed word ends, so a guess often repeats the
        # committed tail ("... the cat sat" then "sat on the mat"): drop the longest such overlap.
        if new and self.committed and abs(new[0].start - self.last_end) < 1.0:
            for n in range(min(self.ngram, len(self.committed), len(new)), 0, -1):
                if [norm(w.text) for w in self.committed[-n:]] == [norm(w.text) for w in new[:n]]:
                    new = new[n:]
                    break
        self.new = new

    def flush(self) -> list[Word]:
        """Commit the words on which the last two guesses agree; returns them."""
        out: list[Word] = []
        while self.new and self.buffer and norm(self.new[0].text) == norm(self.buffer[0].text):
            out.append(self.new.pop(0))
            self.buffer.pop(0)
        self.buffer, self.new = self.new, []
        self._commit(out)
        return out

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
            self.last_end = words[-1].end
            del self.committed[:-50]  # only the tail is ever compared or used as a prompt


def join(words: list[Word]) -> str:
    return re.sub(r"\s+", " ", "".join(w.text for w in words)).strip()
