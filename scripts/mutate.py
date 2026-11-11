"""Test the tests: plant one plausible bug at a time in a copy of the code and check a test fails.

    python scripts/mutate.py

The unmutated suite must pass first; otherwise every mutant would look "caught". The repository is
never modified: the code is copied to a temporary folder.
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
    ("src/livesub/vad.py", "elif level < self.noise_db + c.margin_db:", "else:"),
    ("src/livesub/vad.py", "voice = level > max(self.noise_db + c.margin_db, c.min_db)", "voice = level > self.noise_db + c.margin_db"),
    ("src/livesub/vad.py", "if self._run >= (self.start_frames if voice else self.hang_frames):", "if self._run >= 1:"),
    # local agreement
    ("src/livesub/streaming.py", "new = [w for w in words if w.start > self.last_end - 0.1]", "new = list(words)"),
    ("src/livesub/streaming.py", "new = new[n:]", "new = new"),
    ("src/livesub/streaming.py", "while self.new and self.buffer and norm(self.new[0].text) == norm(self.buffer[0].text):", "while self.new and self.buffer:"),
    ("src/livesub/streaming.py", 'return re.sub(r"[^\\w]", "", text.lower())', "return text"),
    ("src/livesub/streaming.py", "out, self.buffer = self.buffer, []", "out, self.buffer = [], []"),
    # captions
    ("src/livesub/captions.py", "return sum(2 if is_wide(c) else 0 if unicodedata.combining(c) else 1 for c in text)", "return len(text)"),
    ("src/livesub/captions.py", "if line and width(line + piece) > max_width and tok[0] not in NO_LINE_START:", "if line and width(line + piece) > max_width:"),
    ("src/livesub/captions.py", "gap = w.start - self.words[-1].end > self.max_gap", "gap = False"),
    ("src/livesub/captions.py", "full = len(wrap(join(self.words + [w]), self.max_width)) > self.max_lines", "full = False"),
    ("src/livesub/captions.py", "long = w.end - self.words[0].start > self.max_duration", "long = False"),
    ("src/livesub/captions.py", "end = min(end, cues[i + 1].start)", "end = end"),
    ("src/livesub/captions.py", "end = max(c.end + lead_out, c.start + min_duration)", "end = c.end + lead_out"),
    # pipeline
    ("src/livesub/pipeline.py", "if self.time - self._buf_start > self.max_buffer:", "if False:"),
    ("src/livesub/pipeline.py", "self._emit(self.builder.add(self.agreement.finish()))\n        self._emit(self.builder.flush())", "self._emit(self.builder.flush())"),
    ("src/livesub/pipeline.py", "self._buf, self._pre = self._pre, []", "self._buf, self._pre = [], []"),
    ("src/livesub/pipeline.py", "prompt = join(self.agreement.committed)[-200:]", 'prompt = ""'),
    ("src/livesub/pipeline.py", "self.context.append(cue.text)", "pass"),
    # translation
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
