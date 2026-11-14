"""`livesub live` end to end, with a fake capture device, a scripted recognizer and a fake overlay."""

import os
import threading
import time

import pytest

from livesub import cli
from livesub.audio import RATE
from synth import ScriptedTranscriber, render, script

WORDS = script("Hello there.", 0.5) + script("This is live.", 2.5)
AUDIO = render(WORDS, 5.0)


class FakeCapture:
    def __init__(self, audio=AUDIO, realtime=False, fail_at=None):
        self.audio, self.realtime, self.fail_at = audio, realtime, fail_at
        self.opened = []
        self.captured = 0.0  # seconds of audio handed out so far

    def open_source(self, source, device):
        self.opened.append((source, device))
        return "mic"

    def blocks(self, source, device, mic=None):
        n = RATE // 10
        for i in range(0, len(self.audio), n):
            if self.fail_at is not None and i >= self.fail_at * RATE:
                raise OSError("device unplugged")
            if self.realtime:
                time.sleep(0.1)
            self.captured = (i + n) / RATE
            yield self.audio[i:i + n]
        # The stream then goes quiet until livesub is stopped.
        while True:
            time.sleep(0.05)
            yield AUDIO[:n] * 0


class FakeOverlay:
    instances = []

    def __init__(self, font_size):
        self.texts, self.translations = [], []
        self.closed = threading.Event()
        FakeOverlay.instances.append(self)

    def show_text(self, committed, tentative=""):
        self.texts.append((committed, tentative))

    def show_translation(self, text):
        self.translations.append(text)

    def run(self):
        # Like a user who double-clicks the captions once both sentences are up.
        deadline = time.time() + 20
        while not self.closed.is_set() and time.time() < deadline:
            if any("This is live." in c for c, _ in self.texts):
                break
            time.sleep(0.05)

    def close(self):
        self.closed.set()


def run_live(argv, monkeypatch, transcriber=None, capture=None, overlay=FakeOverlay, timeout=30):
    made = []
    monkeypatch.setattr(cli, "make_transcriber", lambda args: made.append(1) or (transcriber or ScriptedTranscriber(WORDS)))
    args = cli.build_parser().parse_args(["live", *argv])
    result = {}

    def target():
        result["code"] = cli.cmd_live(args, capture=capture or FakeCapture(), overlay_class=overlay)

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)
    assert not t.is_alive(), "livesub live did not stop"
    return result.get("code"), made


def test_overlay_mode_captions_and_saves(tmp_path, monkeypatch):
    out = tmp_path / "live.srt"
    code, _ = run_live(["--save", str(out)], monkeypatch)
    assert code == 0
    srt = out.read_text(encoding="utf-8")
    assert "Hello there." in srt and "This is live." in srt
    ov = FakeOverlay.instances[-1]
    assert any(t for _, t in ov.texts), "tentative words were shown"


def test_a_recognizer_failure_stops_with_its_own_message(monkeypatch, capsys):
    class Broken:
        def transcribe(self, *a, **k):
            raise RuntimeError("Library cublas64_12.dll is not found")

    code, _ = run_live([], monkeypatch, transcriber=Broken())
    assert code == 1
    err = capsys.readouterr().err
    assert "recognition failed: Library cublas64_12.dll" in err and "audio capture" not in err


def test_a_capture_failure_is_reported(monkeypatch, capsys):
    code, _ = run_live([], monkeypatch, capture=FakeCapture(fail_at=1.0))
    assert code == 1
    assert "audio capture stopped: device unplugged" in capsys.readouterr().err


def test_an_impossible_save_path_fails_before_the_model_loads(tmp_path, monkeypatch):
    loaded = []
    monkeypatch.setattr(cli, "make_transcriber", lambda args: loaded.append(1))
    for path, message in [(tmp_path / "missing" / "x.srt", "does not exist"), (tmp_path / "x.txt", ".srt or .vtt")]:
        args = cli.build_parser().parse_args(["live", "--save", str(path)])
        with pytest.raises(ValueError, match=message):
            cli.cmd_live(args, capture=FakeCapture(), overlay_class=FakeOverlay)
    assert loaded == []


def test_a_slow_recognizer_reads_the_whole_backlog_at_once_and_loses_nothing(tmp_path, monkeypatch):
    capture = FakeCapture(realtime=True)
    staleness = []

    class Slow(ScriptedTranscriber):
        def transcribe(self, audio, **k):
            # During a sentence, how much captured audio this reading leaves out: it should take in the
            # whole backlog. (A reading at the end of speech is not stale: what came after is not speech.)
            end = k["offset"] + len(audio) / RATE
            if any(a < end < b for a, b in ((0.5, 1.7), (2.5, 3.8))):
                staleness.append(capture.captured - end)
            time.sleep(1.3)  # slower than the 1 s step: audio queues up behind every reading
            return super().transcribe(audio, **k)

    out = tmp_path / "slow.srt"
    code, _ = run_live(["--save", str(out)], monkeypatch, transcriber=Slow(WORDS), capture=capture, timeout=60)
    assert code == 0
    srt = out.read_text(encoding="utf-8")
    assert "Hello there." in srt and "This is live." in srt
    assert staleness and max(staleness) < 0.35, staleness


@pytest.mark.skipif(not (os.environ.get("CI") or os.environ.get("LIVESUB_GUI_TESTS")),
                    reason="opens a real window on top of the screen: runs in CI, or locally with LIVESUB_GUI_TESTS=1")
def test_the_overlay_closes_safely_from_another_thread(monkeypatch):
    pytest.importorskip("tkinter")
    from livesub.overlay import Overlay

    try:
        o = Overlay()
    except RuntimeError:
        pytest.skip("no display")
    errors = []
    monkeypatch.setattr(threading, "excepthook", lambda a: errors.append(a.exc_value))

    def later():
        time.sleep(0.3)
        o.close()
        o.close()  # twice, as the shutdown paths may

    threading.Thread(target=later).start()
    o.run()
    o.close()  # and once more from the window's own thread after it is gone
    assert not errors


def test_translations_reach_the_overlay_and_the_saved_file(tmp_path, monkeypatch):
    from test_translate import FakeLLM

    llm = FakeLLM(reply="译文", delay=0.6)  # slow, like a real model: the last one arrives after stopping
    out = tmp_path / "both.srt"
    try:
        code, _ = run_live(["--save", str(out), "--translate-to", "zh", "--llm-url", llm.url, "--llm-model", "m"], monkeypatch)
    finally:
        llm.close()
    assert code == 0
    srt = out.read_text(encoding="utf-8")
    assert srt.count("译文") == 2, srt  # both cues, translated, as the session ended
    assert "译文" in FakeOverlay.instances[-1].translations


def test_a_second_ctrl_c_skips_the_last_reading_but_keeps_the_transcript(tmp_path, monkeypatch):
    import livesub.live as live_module

    real_join = threading.Thread.join
    calls = []

    def impatient_join(self, timeout=None):
        if self.name == "livesub-capture" and not calls:
            calls.append(1)
            raise KeyboardInterrupt  # pressed again while livesub waits for the last reading
        return real_join(self, timeout)

    monkeypatch.setattr(live_module.threading.Thread, "join", impatient_join)
    out = tmp_path / "kept.srt"
    code, _ = run_live(["--save", str(out)], monkeypatch)
    assert code == 0 and calls
    assert "Hello there." in out.read_text(encoding="utf-8")


def test_the_console_survives_a_terminal_that_cannot_show_the_text(monkeypatch):
    import io

    from livesub.captions import Cue
    from livesub.live import ConsoleView

    raw = io.BytesIO()
    monkeypatch.setattr("sys.stdout", io.TextIOWrapper(raw, encoding="cp1252"))
    view = ConsoleView()
    view.show("大家好", "今天")
    view.show_cue(Cue(0, 1, "大家好。"))
    import sys

    sys.stdout.flush()
    assert b"?" in raw.getvalue()


def test_a_translator_that_cannot_keep_up_falls_behind_by_a_few_lines_only():
    from livesub.captions import Cue
    from livesub.pipeline import TranslationWorker

    class Slow:
        def translate(self, text, target, context):
            time.sleep(0.2)
            return text.upper()

    done = []
    w = TranslationWorker(Slow(), "zh", done.append, max_backlog=2)
    for i in range(12):
        w.submit(Cue(i, i + 1, f"line {i}."))
    t = time.time()
    w.close(timeout=10)
    assert len(done) == 12 and w.skipped > 0
    assert time.time() - t < 2.0, "lines that queued up were passed on, not all translated in turn"
    assert done[-1].translation == "LINE 11.", "the newest line is still translated"
