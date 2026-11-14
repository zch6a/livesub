"""From committed words to subtitle cues: line breaking, cue boundaries, SRT and WebVTT.

Following Netflix's timed-text style guides: at most two lines of 42 characters, or 16 for Chinese and
Japanese (which count double here, so 32 columns); a two-line cue broken where it reads best rather than
filled greedily; on screen for at least 5/6 s, at most 7 s, and long enough to read (20 characters per
second, 9 for Chinese); two frames between cues, and no gap shorter than half a second. A new cue starts
at the end of a sentence or after a pause.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, replace

from .streaming import Word, join, norm

SENTENCE_END = tuple(".!?。！？…")
CLAUSE_END = tuple(",;:，、；：")
# Kinsoku: punctuation that must not begin a line (it stays with the character before it), and opening
# brackets and quotes that must not end one (they go with the character after them).
NO_LINE_START = set("，。、！？；：）」』】》〉”’,.!?;:)]}…%")
NO_LINE_END = set("（「『【《〈“‘([{")
LATIN_WIDTH, CJK_WIDTH = 42, 32  # columns: 42 Latin characters, 16 Chinese or Japanese ones


def is_wide(ch: str) -> bool:
    return unicodedata.east_asian_width(ch) in "WF"


def width(text: str) -> int:
    """Display width in columns: CJK characters take two, combining marks none."""
    return sum(2 if is_wide(c) else 0 if unicodedata.combining(c) else 1 for c in text)


def is_cjk_text(text: str) -> bool:
    chars = [c for c in text if not c.isspace()]
    return bool(chars) and sum(is_wide(c) for c in chars) * 3 >= len(chars)


def line_width(text: str) -> int:
    """How many columns a line of this text may take: 16 characters of Chinese, 42 of English."""
    return CJK_WIDTH if is_cjk_text(text) else LATIN_WIDTH


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


def wrap(text: str, max_width: int | None = None) -> list[str]:
    """Break text into lines of at most `max_width` columns (default: by script, see `line_width`).

    Never splits a Latin word (unless it alone is too long), never starts a line with closing
    punctuation or ends one with an opening bracket, and when two lines are needed, breaks where the
    two read best: after punctuation if possible, the lines about equal, the top one rather shorter."""
    max_width = max_width or line_width(text)
    lines = _greedy(text, max_width)
    if len(lines) == 2:
        lines = _balanced(text, max_width) or lines
    return lines


def _balanced(text: str, max_width: int) -> list[str] | None:
    toks = _tokens(text.strip())
    best: tuple[float, list[str]] | None = None
    for k in range(1, len(toks)):
        if toks[k][0][0] in NO_LINE_START or toks[k - 1][0][-1] in NO_LINE_END:
            continue
        top = _joined(toks[:k])
        bottom = _joined([(toks[k][0], False)] + toks[k + 1:])
        w1, w2 = width(top), width(bottom)
        if w1 > max_width or w2 > max_width:
            continue
        score = abs(w1 - w2) + (4 if w1 > w2 else 0) - (12 if top.endswith(SENTENCE_END + CLAUSE_END) else 0)
        if best is None or score < best[0]:
            best = (score, [top, bottom])
    return best[1] if best else None


def _joined(toks: list[tuple[str, bool]]) -> str:
    return "".join((" " if space and i else "") + tok for i, (tok, space) in enumerate(toks))


def _greedy(text: str, max_width: int) -> list[str]:
    lines: list[str] = []
    line = ""
    for tok, space in _tokens(text.strip()):
        piece = (" " if space and line else "") + tok
        if line and width(line + piece) > max_width and tok[0] not in NO_LINE_START:
            carry = ""
            if line[-1] in NO_LINE_END and len(line) > 1:  # an opening bracket goes with what it opens
                line, carry = line[:-1], line[-1]
            lines.append(line)
            line, piece = carry, tok
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

    def __init__(self, max_width: int | None = None, max_lines: int = 2, max_gap: float = 1.2, max_duration: float = 7.0):
        self.max_width, self.max_lines = max_width, max_lines
        self.max_gap, self.max_duration = max_gap, max_duration
        self.words: list[Word] = []

    def add(self, words: list[Word]) -> list[Cue]:
        out: list[Cue] = []
        for w in words:
            if not self.words and not norm(w.text):
                continue  # a lone "." or "," split off by the recognizer would be a cue of its own
            if self.words and w.start - self.words[-1].end > self.max_gap:
                out.append(self._close())
            while self.words and (len(wrap(join(self.words + [w]), self.max_width)) > self.max_lines
                                  or w.end - self.words[0].start > self.max_duration):
                # Full: end the cue at its last clause boundary if there is one in its second part, and
                # carry the rest over, rather than leave "they look." alone on the next cue.
                out.append(self._close(self._clause_end()))
            self.words.append(w)
            if w.text.rstrip().endswith(SENTENCE_END):
                out.append(self._close())
        return out

    def pending(self) -> list[Word]:
        return list(self.words)

    def flush(self) -> list[Cue]:
        return [self._close()] if self.words else []

    def _clause_end(self) -> int | None:
        for j in range(len(self.words) - 2, len(self.words) // 3 - 1, -1):
            if self.words[j].text.rstrip().endswith(CLAUSE_END):
                return j + 1
        return None

    def _close(self, n: int | None = None) -> Cue:
        done, self.words = self.words[:n], self.words[len(self.words) if n is None else n:]
        return Cue(done[0].start, done[-1].end, join(done))


FRAME_GAP = 2 / 24  # two frames between cues, so the eye notices the change


def display_times(cues: list[Cue], min_duration: float = 5 / 6, max_duration: float = 7.0, lead_out: float = 0.3) -> list[Cue]:
    """Times for showing a cue in a subtitle file: from its first word until a little after its last,
    and long enough to read (20 characters a second, 9 for Chinese and Japanese), within 5/6 s to 7 s.
    Never closer than two frames to the next cue; a gap shorter than half a second is closed to two
    frames, since a blink between two cues is only distracting."""
    out = []
    for i, c in enumerate(cues):
        chars = len([ch for ch in c.text if not ch.isspace()])
        reading = chars / (9 if is_cjk_text(c.text) else 20)
        end = max(c.end + lead_out, c.start + min_duration, c.start + reading)
        end = min(end, max(c.end, c.start + max_duration))
        if i + 1 < len(cues):
            limit = cues[i + 1].start - FRAME_GAP
            if limit - c.start < 0.25:
                pass  # the next cue starts almost with this one (timing jitter): overlap, not a 1 ms flash
            elif end > limit:
                end = limit
            elif limit - end < 0.5 and limit - c.start <= max(max_duration, c.end - c.start):
                end = limit  # close a short gap, but not past the longest a cue is shown
        out.append(replace(c, end=max(end, c.start + 0.001)))
    return out


def _timestamp(t: float, sep: str) -> str:
    ms = max(0, round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02}{sep}{ms:03}"


def _body(c: Cue, mode: str, max_width: int | None) -> str:
    original = "\n".join(wrap(c.text, max_width))
    translated = "\n".join(wrap(c.translation, max_width)) if c.translation else ""
    if mode == "translation":
        return translated or original
    if mode == "both" and translated:
        return f"{original}\n{translated}"
    return original


def to_srt(cues: list[Cue], mode: str = "original", max_width: int | None = None) -> str:
    """`mode`: "original", "translation" (falls back to the original where there is none) or "both"
    (each language on up to two lines of its own, so up to four in all)."""
    blocks = [f"{i}\n{_timestamp(c.start, ',')} --> {_timestamp(c.end, ',')}\n{_body(c, mode, max_width)}\n"
              for i, c in enumerate(display_times(cues), start=1)]
    return "\n".join(blocks)


def _vtt_text(text: str) -> str:
    """WebVTT cue text is markup: "&" and "<" must be escaped, and "-->" may not appear at all."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def to_vtt(cues: list[Cue], mode: str = "original", max_width: int | None = None) -> str:
    blocks = [f"{_timestamp(c.start, '.')} --> {_timestamp(c.end, '.')}\n{_vtt_text(_body(c, mode, max_width))}\n" for c in display_times(cues)]
    return "WEBVTT\n\n" + "\n".join(blocks)
