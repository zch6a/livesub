"""Test the tests: plant one plausible bug at a time in a copy of the code and check a test fails.

    python scripts/mutate.py

The unmutated suite must pass first; otherwise every mutant would look "caught". The repository is
never modified: the code is copied to a temporary folder.

Left out as equivalent: starting a speech region where the VAD noticed speech rather than a few frames
earlier (the 1 s padding covers the difference); in display_times, the branch that cuts a cue at the next one's
start (the branch after it does the same whenever the first would: a cue that runs past the next one's start is
either under the longest time a cue is shown, or itself longer).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MUTANTS = [
    # audio
    ("src/livesub/audio.py", "self.h = _lowpass(0.5 * dst / src * 0.9, taps) if dst < src else None", "self.h = None"),
    ("src/livesub/audio.py", "self.pos = nxt - drop", "self.pos = 0.0"),
    ("src/livesub/audio.py", "v = np.where(v >= 1 << 23, v - (1 << 24), v)", "v = v"),
    # vad
    ("src/livesub/vad.py", "self.noise_db += 0.3 * (level - self.noise_db)", "self.noise_db += 0.0 * (level - self.noise_db)"),
    ("src/livesub/vad.py", "elif level < self.noise_db + c.margin_db and not self.speaking:", "elif level < self.noise_db + c.margin_db:"),
    ("src/livesub/vad.py", "voice = level > max(self.noise_db + margin, c.min_db)", "voice = level > self.noise_db + margin"),
    ("src/livesub/vad.py", "margin = c.stay_db if self.speaking else c.margin_db", "margin = c.margin_db"),
    ("src/livesub/vad.py", "if level < c.silence_db:", "if False:"),
    ("src/livesub/vad.py", "self.noise_db = min(level, c.initial_db)", "self.noise_db = level"),
    ("src/livesub/vad.py", "self.noise_db = max(self.noise_db, float(np.percentile(self._recent, 10)))", "self.noise_db = self.noise_db"),
    ("src/livesub/vad.py", "if self._run >= (self.start_frames if voice else self.hang_frames):", "if self._run >= 1:"),
    # local agreement
    ("src/livesub/streaming.py", "start = min(hits, key=lambda i: abs(new[i + k - 1].end - self.last_end)) + k", "start = min(hits, key=lambda i: abs(new[i + k - 1].end - self.last_end))"),
    ("src/livesub/streaming.py", "start = next((i for i, w in enumerate(new) if texts[i] == first and abs(w.start - self.buffer[0].start) < 1.5), None)", "start = None"),
    ("src/livesub/streaming.py", "else [w for w in new if w.end > self.last_start]", "else [w for w in new if w.start > self.last_end - 0.1]"),
    ("src/livesub/asr.py", "if not _load_nvidia_wheel(name, load):", "if True:"),
    ("src/livesub/asr.py", "    if device == \"cuda\":\n        raise RuntimeError", "    if False:\n        raise RuntimeError"),
    ("src/livesub/streaming.py", "while self.new and self.buffer and norm(self.new[0].text) == norm(self.buffer[0].text):", "while self.new and self.buffer:"),
    ("src/livesub/streaming.py", 'return re.sub(r"[^\\w]", "", text.lower())', "return text"),
    ("src/livesub/streaming.py", "out, self.buffer = self.buffer, []", "out, self.buffer = [], []"),
    ("src/livesub/streaming.py", "heard = [w for w in self.committed if w.start >= window_start - 0.1]", "heard = [w for w in self.committed if w.end > window_start - 0.3]"),
    ("src/livesub/streaming.py", "if self._fresh:", "if False:"),
    # captions
    ("src/livesub/captions.py", "return sum(2 if is_wide(c) else 0 if unicodedata.combining(c) else 1 for c in text)", "return len(text)"),
    ("src/livesub/captions.py", "if line and width(line + piece) > max_width and tok[0] not in NO_LINE_START:", "if line and width(line + piece) > max_width:"),
    ("src/livesub/captions.py", "if self.words and w.start - self.words[-1].end > self.max_gap:", "if False:"),
    ("src/livesub/captions.py", "while self.words and (len(wrap(join(self.words + [w]), self.max_width)) > self.max_lines", "while self.words and (False"),
    ("src/livesub/captions.py", "or w.end - self.words[0].start > self.max_duration):", "or False):"),
    ("src/livesub/captions.py", "out.append(self._close(self._clause_end()))", "out.append(self._close())"),
    ("src/livesub/captions.py", "end = min(end, max(c.end, c.start + max_duration))", "end = end"),
    ("src/livesub/captions.py", "end = max(c.end + lead_out, c.start + min_duration, c.start + reading)", "end = c.end + lead_out"),
    # pipeline
    ("src/livesub/pipeline.py", "if self.time - self._buf_start > self.max_buffer:", "if False:"),
    ("src/livesub/pipeline.py", "self._emit(self.builder.add(self.agreement.finish()))\n        self._emit(self.builder.flush())", "self._emit(self.builder.flush())"),
    ("src/livesub/pipeline.py", "self._buf, self._pre = self._pre, []", "self._buf, self._pre = [], []"),
    ("src/livesub/pipeline.py", "prompt = join([w for w in self.agreement.committed if w.end <= self._buf_start + 0.05])[-200:]", 'prompt = ""'),
    ("src/livesub/pipeline.py", "prompt = join([w for w in self.agreement.committed if w.end <= self._buf_start + 0.05])[-200:]", "prompt = join(self.agreement.committed)[-200:]"),
    ("src/livesub/pipeline.py", "self.context.append(cue.text)", "pass"),
    ("src/livesub/pipeline.py", "if recognize and self.time - self._last_run >= self.step:", "if self.time - self._last_run >= self.step:"),
    ("src/livesub/pipeline.py", "while b - a > max_len:", "while False:"),
    # recognizer
    ("src/livesub/asr.py", "if s.compression_ratio > 2.4:", "if False:"),
    ("src/livesub/asr.py", "return drop_loops(words)", "return words"),
    ("src/livesub/asr.py", "prompt = SIMPLIFIED_PROMPT + prompt", "prompt = prompt"),
    # live command
    ("src/livesub/cli.py", "    check_output(args.save, \"--save\", args.overwrite)", "    pass"),
    ("src/livesub/live.py", "self.captioner.feed(block, recognize=self.audio.empty())", "self.captioner.feed(block)"),
    ("src/livesub/live.py", "self.transcript.translated(cue)", "pass"),
    ("src/livesub/live.py", "            self.worker.close(timeout=0.1 if hurry else 15)", "            pass"),
    ("src/livesub/cli.py", "args.llm_key or os.environ.get(\"LIVESUB_LLM_API_KEY\")", "args.llm_key"),
    # round 3: subtitle rules, recognizer settings, trimming
    ("src/livesub/captions.py", "return CJK_WIDTH if is_cjk_text(text) else LATIN_WIDTH", "return LATIN_WIDTH"),
    ("src/livesub/captions.py", "        lines = _balanced(text, max_width) or lines", "        pass"),
    ("src/livesub/captions.py", "if line[-1] in NO_LINE_END and len(line) > 1:", "if False:"),
    ("src/livesub/captions.py", "reading = chars / (9 if is_cjk_text(c.text) else 20)", "reading = 0.0"),
    ("src/livesub/captions.py", "if limit - c.start < 0.25:", "if False:"),
    ("src/livesub/captions.py", 'return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")', "return text"),
    ("src/livesub/vad.py", "if np.isfinite(rms) else -200.0", "if True else -200.0"),
    ("src/livesub/pipeline.py", "if self.q.qsize() - self._closing >= self.max_backlog:", "if self.q.qsize() >= self.max_backlog:"),
    ("src/livesub/cli.py", "if path.is_dir():", "if False:"),
    ("src/livesub/audio.py", "raw = raw[: len(raw) - len(raw) % (width * channels)]", "pass"),
    ("src/livesub/streaming.py", "hits = [i for i in hits if i == 0]", "hits = [i for i in hits if i <= 1]"),
    ("src/livesub/pipeline.py", "cut = self.agreement.sentence_cut(after=self._buf_start + 1.0) or self.agreement.cut_point()", "cut = self.agreement.cut_point()"),
    ("src/livesub/pipeline.py", "cut = _quietest(levels, frame, a + 0.8 * max_len, a + max_len)", "cut = a + max_len"),
    ("src/livesub/cli.py", "temperature = 0.0 if args.command == \"live\" else (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)", "temperature = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)"),
    ("src/livesub/asr.py", "if s.no_speech_prob > 0.9:", "if False:"),
    ("src/livesub/asr.py", "need = (6 if len(unit[0]) == 1 and is_wide(unit[0]) else limit) if n == 1 else 3", "need = limit if n == 1 else 99"),
    # round 5: limits of the real library, shutdown, output checks
    ("src/livesub/asr.py", "return max(16, min(int(8 * min(seconds, 30.0)) + 16, 448 - 5 - prompt_tokens))", "return max(16, min(int(8 * seconds) + 16, 440))"),
    ("src/livesub/asr.py", "prompt = SIMPLIFIED_PROMPT + prompt[-150:]", "prompt = prompt if any(is_wide(c) for c in prompt) else SIMPLIFIED_PROMPT + prompt"),
    ("src/livesub/asr.py", "if len(s.text) < 60 and CREDITS.search(s.text):", "if False:"),
    ("src/livesub/asr.py", "        except FileNotFoundError:", "        except Exception:"),
    ("src/livesub/captions.py", "            if not self.words and not norm(w.text):\n                continue", "            pass"),
    ("src/livesub/cli.py", "    if path.exists() and not overwrite:", "    if False:"),
    ("src/livesub/cli.py", "        t.api_key = args.llm_key or t.api_key", "        pass"),
    ("src/livesub/cli.py", "    t.check()", "    pass"),
    ("src/livesub/live.py", "            hurry = True  # a second Ctrl+C: skip the last reading, but still save what there is", "            raise"),
    ("src/livesub/live.py", "            sys.stdout.reconfigure(errors=\"replace\")", "            pass"),
    ("src/livesub/pipeline.py", "            if self.q.qsize() - self._closing >= self.max_backlog:", "            if False:"),
    # translation
    ("src/livesub/translate.py", "return urllib.request.build_opener(urllib.request.ProxyHandler({}))", "return urllib.request.build_opener()"),
    ("src/livesub/translate.py", 'text = re.sub(r"<think>.*?</think>", "", reply, flags=re.S | re.I)', "text = reply"),
    ("src/livesub/translate.py", "if key in self.cache:", "if False:"),
    ("src/livesub/translate.py", 'user = "".join(f"Earlier: {c}\\n" for c in context) + f"Line: {text}"', 'user = f"Line: {text}"'),
]


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "repo"
        shutil.copytree(ROOT, copy, ignore=shutil.ignore_patterns(".venv", ".git", "__pycache__", ".pytest_cache"))
        env = {**os.environ, "PYTHONPATH": str(copy / "src")}

        def run_suite() -> subprocess.CompletedProcess:
            return subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "tests"],
                                  cwd=copy, env=env, capture_output=True, timeout=300)

        baseline = run_suite()
        if baseline.returncode != 0:
            print("The test suite fails without any mutant; fix that first:\n" + baseline.stdout.decode(errors="replace")[-2000:])
            return 2
        survived = []
        for i, (file, original, mutant) in enumerate(MUTANTS, start=1):
            path = copy / file
            source = path.read_text(encoding="utf-8")
            if original not in source:
                print(f"{i:>2}. NOT FOUND in {file}: {original[:60]} (the code changed; update this list)")
                survived.append(file)
                continue
            path.write_text(source.replace(original, mutant, 1), encoding="utf-8")
            try:
                caught = run_suite().returncode != 0
            except subprocess.TimeoutExpired:
                caught = True  # a hang is a failure too
            path.write_text(source, encoding="utf-8")
            print(f"{i:>2}. {'caught  ' if caught else 'SURVIVED'}  {file}: {original[:70]}", flush=True)
            if not caught:
                survived.append(file)
        print(f"\n{len(MUTANTS) - len(survived)} of {len(MUTANTS)} mutants caught")
        return 1 if survived else 0


if __name__ == "__main__":
    sys.exit(main())
