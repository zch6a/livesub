import threading

import numpy as np

from livesub.audio import RATE
from livesub.captions import Cue
from livesub.pipeline import LiveCaptioner, TranslationWorker, run_file
from livesub.streaming import join

from synth import ScriptedTranscriber, cjk_script, render, script

TALK = (
    script("Good morning everyone. Today we are going to look at live captions.", 1.0)
    + script("They have to keep up with speech, and they must not flicker.", 8.0)
    + script("That is the whole problem.", 14.5)
)


def caption(timeline, duration, **kw):
    asr = ScriptedTranscriber(timeline)
    partials = []
    cap = LiveCaptioner(asr, on_partial=lambda done, maybe: partials.append((done, maybe)), **kw)
    cues = run_file(cap, render(timeline, duration))
    return cap, asr, cues, partials


def test_every_word_appears_once_in_order_and_no_wrong_guess_survives():
    cap, asr, cues, _ = caption(TALK, 18.0)
    assert " ".join(c.text for c in cues) == join(TALK)
    assert not any("-" in c.text for c in cues)


def test_cues_follow_the_speech_in_time():
    _, _, cues, _ = caption(TALK, 18.0)
    for c in cues:
        assert c.start < c.end
        spoken = [w for w in TALK if c.start - 1e-9 <= w.start and w.end <= c.end + 1e-9]
        assert join(spoken) == c.text
    assert all(a.end <= b.start for a, b in zip(cues, cues[1:]))


def test_silence_costs_nothing_and_speech_about_one_run_per_step():
    cap, asr, _, _ = caption(TALK, 18.0)
    speech = sum(w.end - w.start + 0.08 for w in TALK)
    assert cap.runs <= speech / cap.step + 3 * 2  # plus one final read per utterance
    # The recognizer never sees the long silences before, between and after the utterances.
    assert all(end - start < 6.5 for start, end, _ in asr.calls)


def test_partial_text_shows_tentative_words_before_they_are_committed():
    _, _, cues, partials = caption(TALK, 18.0)
    assert any(maybe.endswith("-") for _, maybe in partials), "the unstable guess is shown as tentative"
    assert not any(done.endswith("-") for done, _ in partials), "and never as committed"
    assert any(done and maybe for done, maybe in partials)
    assert partials[-1] == ("", "")  # cleared when speech ends


def test_long_unbroken_speech_keeps_the_window_bounded():
    text = " ".join(f"w{i}" for i in range(120))  # 48 s without a pause
    words = script(text, 0.5)
    cap, asr, cues, _ = caption(words, 50.0, max_buffer=8.0)
    assert " ".join(c.text for c in cues) == text
    assert cap.longest_window <= 8.0 + cap.step + 0.2


def test_the_prompt_carries_the_committed_text():
    _, asr, _, _ = caption(TALK, 18.0)
    later = [p for _, _, p in asr.calls if p]
    assert later and "must not flicker." in later[-1]


def test_chinese():
    words = cjk_script("大家好。今天我们来看实时字幕。", 1.0) + cjk_script("它不能闪烁。", 6.0)
    _, _, cues, _ = caption(words, 9.0)
    assert [c.text for c in cues] == ["大家好。", "今天我们来看实时字幕。", "它不能闪烁。"]


def test_nothing_to_say():
    cap = LiveCaptioner(ScriptedTranscriber([]))
    assert run_file(cap, render([], 5.0)) == []
    assert cap.runs == 0


class Upper:
    def __init__(self, fail_on=None):
        self.fail_on = fail_on
        self.contexts = []

    def translate(self, text, target, context):
        self.contexts.append(list(context))
        if text == self.fail_on:
            raise RuntimeError("down")
        return f"[{target}] {text.upper()}"


def test_translation_worker_translates_in_order_with_context_and_survives_failures():
    done, errors = [], []
    finished = threading.Event()
    t = Upper(fail_on="b.")
    w = TranslationWorker(t, "zh", lambda c: (done.append(c), len(done) == 3 and finished.set()), on_error=errors.append)
    for i, text in enumerate(["a.", "b.", "c."]):
        w.submit(Cue(i, i + 1, text))
    assert finished.wait(5)
    w.close(5)
    assert [c.translation for c in done] == ["[zh] A.", None, "[zh] C."]
    assert t.contexts == [[], ["a."], ["a.", "b."]]
    assert len(errors) == 1


def test_bad_settings_are_refused():
    import pytest

    with pytest.raises(ValueError):
        LiveCaptioner(ScriptedTranscriber([]), step=0.1)
    with pytest.raises(ValueError):
        LiveCaptioner(ScriptedTranscriber([]), step=5, max_buffer=8)
