import threading

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
        spoken = [w for w in TALK if c.start - 0.15 <= w.start and w.end <= c.end + 0.15]  # timestamps jitter
        assert join(spoken) == c.text
    assert all(a.end <= b.start for a, b in zip(cues, cues[1:], strict=False))


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


def test_the_prompt_never_repeats_words_still_in_the_window():
    # Given words it can still hear in its prompt, Whisper tends to skip them (seen in a real run).
    _, asr, _, _ = caption(TALK, 18.0)
    for offset, end, prompt in asr.calls:
        inside = [w for w in TALK if w.start >= offset and w.end <= end]
        if len(inside) >= 3:
            assert join(inside[:3]) not in prompt, (offset, end, prompt)


def test_sentences_that_share_a_word_across_a_pause():
    words = script("Thank you.", 1.0) + script("You know, this is a test.", 3.0)
    _, _, cues, _ = caption(words, 6.0)
    assert [c.text for c in cues] == ["Thank you.", "You know, this is a test."]


def test_speech_regions_cover_every_word_and_skip_the_silence():
    from livesub.pipeline import speech_regions

    regions = speech_regions(render(TALK, 18.0))
    for w in TALK:
        assert any(a <= w.start and w.end <= b for a, b in regions), w
    assert sum(b - a for a, b in regions) < 0.8 * 18.0
    assert all(b - a <= 30.0 for a, b in regions)


def test_long_speech_is_split_into_regions_whisper_can_read():
    from livesub.pipeline import speech_regions

    words = script(" ".join(f"w{i}" for i in range(150)), 0.5)  # 60 s without a pause
    regions = speech_regions(render(words, 62.0))
    assert len(regions) >= 2 and all(b - a <= 30.0 for a, b in regions)


def test_a_file_is_read_once_per_region_and_gives_the_same_captions():
    from livesub.pipeline import transcribe_file

    asr = ScriptedTranscriber(TALK, unstable_tail=False)
    cues = transcribe_file(asr, render(TALK, 18.0))
    assert " ".join(c.text for c in cues) == join(TALK)
    assert len(asr.calls) == 3  # one per utterance, against dozens in the live loop
    assert "flicker." in asr.calls[-1][2]  # earlier text is the prompt
