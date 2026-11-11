"""From committed words to subtitle cues: line breaking, cue boundaries, SRT and WebVTT.

Subtitle conventions (as in broadcasters' and Netflix's style guides): at most two lines of about 42
characters (16 full-width characters for Chinese and Japanese, which count double here), a cue on
screen for at least a second and at most about seven, and a new cue at the end of a sentence or after
a pause.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, replace

from .streaming import Word, join

SENTENCE_END = tuple(".!?。！？…")
# Kinsoku: punctuation that must not begin a line; it stays with the character before it.
NO_LINE_START = set("，。、！？；：）」』】》〉,.!?;:)]}…%")


def is_wide(ch: str) -> bool:
    return unicodedata.east_asian_width(ch) in "WF"


def width(text: str) -> int:
    """Display width in columns: CJK characters take two, combining marks none."""
    return sum(2 if is_wide(c) else 0 if unicodedata.combining(c) else 1 for c in text)


def _tokens(text: str) -> list[tuple[str, bool]]:
    """Split into breakable units: each wide character alone, runs of other non-space characters
    together. The flag says whether a space came before the unit."""
    out: list[tuple[str, bool]] = []
    run, run_space, space = "", False, False
    for ch in text:
        if ch.isspace():
            if run:
                out.append((run, run_space))
                run = ""
            space = True
        elif is_wide(ch):
            if run:
                out.append((run, run_space))
                run = ""
            out.append((ch, space))
            space = False
        else:
            if not run:
                run_space, space = space, False
            run += ch
    if run:
        out.append((run, run_space))
    return out


def wrap(text: str, max_width: int = 42) -> list[str]:
    """Greedy line breaking that never splits a Latin word (unless it alone is too long) and never
    starts a line with closing punctuation."""
    lines: list[str] = []
    line = ""
    for tok, space in _tokens(text.strip()):
        piece = (" " if space and line else "") + tok
        if line and width(line + piece) > max_width and tok[0] not in NO_LINE_START:
            lines.append(line)
            line, piece = "", tok
        while width(piece) > max_width:  # one unit longer than a line: hard split
            cut = 1
            while cut < len(piece) and width(piece[: cut + 1]) <= max_width:
                cut += 1
            lines.append(piece[:cut])
            piece = piece[cut:]
        line += piece
    if line:
        lines.append(line)
    return lines


@dataclass(frozen=True)
class Cue:
    start: float
    end: float
    text: str
    translation: str | None = None


class CueBuilder:
    """Collects committed words and closes a cue at a sentence end, a pause, or when it is full."""

    def __init__(self, max_width: int = 42, max_lines: int = 2, max_gap: float = 1.2, max_duration: float = 7.0):
        self.max_width, self.max_lines = max_width, max_lines
        self.max_gap, self.max_duration = max_gap, max_duration
        self.words: list[Word] = []

    def add(self, words: list[Word]) -> list[Cue]:
        out: list[Cue] = []
        for w in words:
            if self.words:
                gap = w.start - self.words[-1].end > self.max_gap
                full = len(wrap(join(self.words + [w]), self.max_width)) > self.max_lines
                long = w.end - self.words[0].start > self.max_duration
                if gap or full or long:
                    out.append(self._close())
            self.words.append(w)
            if w.text.rstrip().endswith(SENTENCE_END):
                out.append(self._close())
        return out

    def pending(self) -> list[Word]:
        return list(self.words)

    def flush(self) -> list[Cue]:
        return [self._close()] if self.words else []

    def _close(self) -> Cue:
        cue = Cue(self.words[0].start, self.words[-1].end, join(self.words))
        self.words = []
        return cue


def display_times(cues: list[Cue], min_duration: float = 1.0, lead_out: float = 0.3) -> list[Cue]:
    """Times for showing: each cue stays a little after its last word and at least `min_duration`,
    but never overlaps the next one."""
    out = []
    for i, c in enumerate(cues):
        end = max(c.end + lead_out, c.start + min_duration)
        if i + 1 < len(cues):
            end = min(end, cues[i + 1].start)
        out.append(replace(c, end=max(end, c.start + 0.001)))
    return out


def _timestamp(t: float, sep: str) -> str:
    ms = max(0, round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02}{sep}{ms:03}"


def _body(c: Cue, mode: str, max_width: int) -> str:
    original = "\n".join(wrap(c.text, max_width))
    translated = "\n".join(wrap(c.translation, max_width)) if c.translation else ""
    if mode == "translation":
        return translated or original
    if mode == "both" and translated:
        return f"{original}\n{translated}"
    return original


def to_srt(cues: list[Cue], mode: str = "original", max_width: int = 42) -> str:
    """`mode`: "original", "translation" (falls back to the original where there is none) or "both"."""
    blocks = [f"{i}\n{_timestamp(c.start, ',')} --> {_timestamp(c.end, ',')}\n{_body(c, mode, max_width)}\n"
              for i, c in enumerate(display_times(cues), start=1)]
    return "\n".join(blocks)


def to_vtt(cues: list[Cue], mode: str = "original", max_width: int = 42) -> str:
    blocks = [f"{_timestamp(c.start, '.')} --> {_timestamp(c.end, '.')}\n{_body(c, mode, max_width)}\n" for c in display_times(cues)]
    return "WEBVTT\n\n" + "\n".join(blocks)
