"""Defects found by an independent review in the second polishing cycle, each pinned by a test."""

import os
import time
import wave

import numpy as np
import pytest

from livesub.audio import RATE, read_wav, resample
from livesub.captions import Cue, display_times, to_vtt
from livesub.cli import check_output
from livesub.pipeline import LiveCaptioner, TranslationWorker, run_file
from livesub.vad import EnergyVAD
from synth import ScriptedTranscriber, render, script


@pytest.mark.parametrize("first, second", [("Thank you.", "You know, this is a test."), ("I said no.", "Well no, I did not.")])
@pytest.mark.parametrize("gap", [0.6, 0.8, 1.0])
def test_live_keeps_the_first_words_after_a_short_pause(first, second, gap):
    # Whisper stretches the last word of a reading to the window's end, so the old sentence's last word
    # looked "heard" in the next window and matched the new sentence's first word, which was dropped.
    words = script(first, 1.0)
    words += script(second, words[-1].end + gap)
    cues = run_file(LiveCaptioner(ScriptedTranscriber(words)), render(words, words[-1].end + 3))
    assert [c.text for c in cues] == [first, second]


def test_one_nan_does_not_deafen_the_voice_detector():
    words = script("Hello there. This is a test.", 0.5)
    x = render(words, words[-1].end + 1)
    x[0] = np.nan  # before: the background level became NaN and no frame ever counted as speech
    from livesub.audio import to_mono

    vad = EnergyVAD()
    assert any(vad.process(to_mono(x)))
    vad = EnergyVAD()
    assert any(vad.process(x))  # even if NaN reaches the detector itself


def test_resampling_in_pieces_is_the_same_as_in_one_go():
    from livesub.audio import Resampler

    x = np.random.default_rng(0).standard_normal(48_000 * 25).astype(np.float32)
    assert np.array_equal(resample(x, 48_000, RATE), Resampler(48_000, RATE)(x))
    assert resample(np.zeros(0, np.float32), 48_000, RATE).shape == (0,)


def test_webvtt_text_is_escaped():
    vtt = to_vtt([Cue(0, 1, "R&D <3 costs --> profit")])
    assert "R&amp;D &lt;3 costs --&gt; profit" in vtt
    assert vtt.count("-->") == 1  # only the timing line's


def test_display_times_stay_within_seven_seconds_and_never_flash():
    a, b = display_times([Cue(0, 6.9, "A long sentence that was spoken slowly."), Cue(7.3, 8, "Next.")])
    assert a.end <= 7.0  # closing the short gap had pushed it to 7.217 s
    a, b = display_times([Cue(1.0, 1.2, "x."), Cue(1.05, 2, "y.")])
    assert a.end - a.start >= 0.5  # it was shown for 1 ms


def test_stopping_does_not_count_the_stop_marker_as_a_waiting_line():
    class Slow:
        def translate(self, text, target, context):
            time.sleep(0.2)
            return text.upper()

    done = []
    worker = TranslationWorker(Slow(), "xx", done.append, max_backlog=3)
    for i in range(3):
        worker.submit(Cue(i, i + 1, f"line {i}."))
    worker.close(15)  # three lines and the stop marker queued: the marker counted as a third waiting line
    assert worker.skipped == 0, [c.translation for c in done]  # the oldest line was skipped
    assert [c.translation for c in done] == ["LINE 0.", "LINE 1.", "LINE 2."]


def test_output_path_problems_are_found_before_transcribing(tmp_path):
    folder = tmp_path / "subs.srt"
    folder.mkdir()
    with pytest.raises(ValueError, match="folder"):
        check_output(folder, "-o", True)  # failed with "Permission denied" only after transcribing


def wav_bytes(path, rate, data=b"", width=2, channels=1, truncate=0):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(max(rate, 1))
        w.writeframes(data)
    if rate <= 0 or truncate:
        raw = bytearray(path.read_bytes())
        if rate <= 0:
            raw[24:28] = (0).to_bytes(4, "little")
        if truncate:
            raw = raw[:-truncate]
        path.write_bytes(bytes(raw))


def test_bad_wav_files_are_reported_with_their_name(tmp_path):
    zero = tmp_path / "zero.wav"
    wav_bytes(zero, 0, b"\x00\x00" * 100)
    with pytest.raises(ValueError, match="zero.wav"):
        read_wav(zero)
    cut = tmp_path / "cut.wav"
    wav_bytes(cut, 16_000, b"\x01\x00\x02\x00" * 50, width=3, channels=1, truncate=1)  # 24-bit, one byte short
    x, rate = read_wav(cut)  # was: "cannot reshape array of size ... into shape (3)"
    assert rate == 16_000 and len(x) > 0


def test_the_window_test_is_opt_in_outside_ci():
    import test_live

    marks = [m for m in getattr(test_live.test_the_overlay_closes_safely_from_another_thread, "pytestmark", []) if m.name == "skipif"]
    assert marks and marks[0].args[0] == (not (os.environ.get("CI") or os.environ.get("LIVESUB_GUI_TESTS")))


# ---- tests for mutants the round's first mutation run left alive ----

from livesub.streaming import Agreement, Word  # noqa: E402


def _ws(*items):
    return [Word(s, e, t) for t, s, e in items]


def test_a_new_utterance_keeps_a_first_word_that_ends_the_last_one():
    # "... thank you." then, after speech ended, "you know what": the new "you" is not the old one repeated
    a = Agreement()
    a.insert(_ws((" thank", 0.0, 0.4), (" you.", 0.4, 0.8)))
    a.finish()
    a.insert(_ws((" you", 1.2, 1.4), (" know", 1.4, 1.7), (" what", 1.7, 2.0)))  # no window start: all of it is heard
    assert [w.text for w in a.new] == [" you", " know", " what"]


def test_one_repeated_word_counts_only_at_the_head_of_the_guess():
    # "I said no." committed, then "Well no, ..." : "no" at the second word must not throw away "Well"
    a = Agreement()
    for _ in range(2):  # two readings agree: committed
        a.insert(_ws((" I", 0.0, 0.2), (" said", 0.2, 0.5), (" no.", 0.5, 0.8)))
        a.flush()
    assert [w.text for w in a.committed][-1] == " no."
    a.insert(_ws((" well", 0.9, 1.1), (" no,", 1.1, 1.3), (" then", 1.3, 1.6)), window_start=0.0)
    assert [w.text for w in a.new][0] == " well"


def test_a_wav_cut_in_the_middle_of_a_sample_is_read(tmp_path):
    p = tmp_path / "cut.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes((np.arange(100, dtype="<i2") * 100).tobytes())
    data = p.read_bytes()
    p.write_bytes(data[:-1])  # the last sample has one byte of its two
    x, rate = read_wav(p)
    assert rate == RATE and len(x) == 99
