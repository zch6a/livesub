from livesub.captions import Cue, CueBuilder, display_times, to_srt, to_vtt, width, wrap
from livesub.streaming import Word

from synth import cjk_script, script


def test_width_counts_cjk_double():
    assert width("abc") == 3
    assert width("字幕") == 4
    assert width("é") == 1  # e + combining acute


def test_wrap_breaks_latin_between_words():
    lines = wrap("The quick brown fox jumps over the lazy dog and keeps on running far away", 42)
    assert all(width(l) <= 42 for l in lines)
    assert " ".join(lines) == "The quick brown fox jumps over the lazy dog and keeps on running far away"


def test_wrap_breaks_cjk_anywhere_but_never_before_closing_punctuation():
    text = "今天我们来聊一聊离线实时字幕是怎么做出来的，以及它为什么不会闪烁。"
    lines = wrap(text, 16)
    assert "".join(lines) == text
    assert all(width(l) <= 18 for l in lines)  # a kept punctuation mark may overhang by one character
    assert not any(l[0] in "，。" for l in lines)


def test_wrap_mixed_scripts_and_overlong_words():
    assert wrap("用 Python 写一个字幕工具", 12) == ["用 Python 写", "一个字幕工具"]
    assert wrap("a" * 50, 20) == ["a" * 20, "a" * 20, "a" * 10]


def test_cues_close_at_sentence_ends_pauses_and_when_full():
    b = CueBuilder()
    cues = b.add(script("Hello there. How are", 0.0))
    assert [c.text for c in cues] == ["Hello there."]
    cues = b.add(script("you", 1.6) + script("later", 5.0))  # 3 s pause before "later"
    assert [c.text for c in cues] == ["How are you"]
    long = script(" ".join(["word"] * 30), 6.0)
    cues = b.add(long) + b.flush()
    assert len(cues) >= 2
    assert all(len(wrap(c.text)) <= 2 for c in cues)
    assert all(c.end - c.start <= 7.0 for c in cues)
    assert " ".join(c.text for c in cues) == "later " + " ".join(["word"] * 30)
    assert b.flush() == []
    # Short words, so the two-line limit is reached before the time limit.
    b2 = CueBuilder(max_width=20)
    full = b2.add(script(" ".join(["ab"] * 30), 0.0, word=0.1, gap=0.02))
    assert full and all(len(wrap(c.text, 20)) <= 2 for c in full)
    assert len(wrap(full[0].text + " ab", 20)) > 2  # each cue is as full as it can be


def test_cjk_cues():
    b = CueBuilder(max_width=16)
    cues = b.add(cjk_script("你好。今天天气怎么样？", 0.0))
    assert [c.text for c in cues] == ["你好。", "今天天气怎么样？"]


def test_display_times_hold_short_cues_without_overlapping_the_next():
    cues = [Cue(0.0, 0.2, "Hi."), Cue(0.5, 1.0, "Yes."), Cue(5.0, 5.5, "Bye.")]
    shown = display_times(cues)
    assert shown[0].end == 0.5  # wanted 1.0, but the next cue starts at 0.5
    assert shown[1].end == 1.5  # min 1 s from its start
    assert shown[2].end == 6.0


def test_srt_and_vtt_format():
    cues = [Cue(1.0, 2.5, "Hello there.", "你好。"), Cue(3661.2, 3663.0, "Bye.")]
    assert to_srt(cues) == "1\n00:00:01,000 --> 00:00:02,800\nHello there.\n\n2\n01:01:01,200 --> 01:01:03,300\nBye.\n"
    assert to_srt(cues, mode="both").split("\n")[2:4] == ["Hello there.", "你好。"]
    assert "\nBye.\n" in to_srt(cues, mode="translation")  # no translation: falls back to the original
    vtt = to_vtt(cues)
    assert vtt.startswith("WEBVTT\n\n00:00:01.000 --> 00:00:02.800\nHello there.\n")


def test_words_keep_their_own_spacing():
    b = CueBuilder()
    b.add([Word(0, 0.2, " It"), Word(0.2, 0.4, "'s"), Word(0.4, 0.6, " fine.")])
    assert b.words == []
