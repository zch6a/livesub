import numpy as np
import pytest

from livesub.audio import RATE
from livesub.vad import EnergyVAD

from synth import render, script


def timeline(x, vad=None, block=0.1):
    vad = vad or EnergyVAD()
    states = []
    n = int(block * RATE)
    for i in range(0, len(x), n):
        states += vad.process(x[i:i + n])
    return np.array(states), vad.frame / RATE


def edges(states, frame):
    d = np.diff(states.astype(int), prepend=0)
    return [round(i * frame, 2) for i in np.flatnonzero(d == 1)], [round(i * frame, 2) for i in np.flatnonzero(d == -1)]


@pytest.mark.parametrize("noise_db", [-80, -65, -50])
def test_finds_each_utterance_promptly_and_holds_through_the_pauses_between_words(noise_db):
    words = script("one two three four", 1.0) + script("five six seven", 4.0)
    x = render(words, 7.0, noise_db=noise_db, speech_db=noise_db + 30 if noise_db > -60 else -20)
    states, frame = timeline(x)
    on, off = edges(states, frame)
    assert len(on) == 2 and len(off) == 2, (on, off)
    assert 1.0 < on[0] < 1.2 and 4.0 < on[1] < 4.2
    for start, last_word_end in zip(off, [words[3].end, words[-1].end]):
        assert last_word_end < start < last_word_end + 0.7


def test_quiet_background_alone_never_counts_as_speech():
    states, _ = timeline(render([], 5.0, noise_db=-45))
    assert not states.any()


def test_a_click_is_not_speech():
    x = render([], 3.0)
    x[RATE: RATE + 480] += 0.5  # 30 ms
    states, _ = timeline(x)
    assert not states.any()


def test_digital_silence_then_speech():
    words = script("hello there", 0.5)
    x = render(words, 2.5)
    x[: int(0.5 * RATE)] = 0
    states, frame = timeline(x)
    on, _ = edges(states, frame)
    assert len(on) == 1 and 0.5 < on[0] < 0.7


def test_frames_carry_over_between_blocks_of_any_size():
    x = render(script("a b c", 0.4), 2.0)
    one, _ = timeline(x, block=10)  # all at once
    many, _ = timeline(x, block=0.013)
    assert one.tolist() == many.tolist()


def test_starting_in_the_middle_of_something_loud():
    # livesub started while a video was already playing loudly: the first frames set the floor high, and
    # it must come down within the quiet that follows, or later speech goes unnoticed.
    words = script("are you still there", 3.0)
    x = render(words, 5.0, speech_db=-30)
    x[: RATE] += np.random.default_rng(1).standard_normal(RATE).astype(np.float32) * 10 ** (-22 / 20)
    states, frame = timeline(x)
    on, _ = edges(states, frame)
    assert on and 3.0 < on[-1] < 3.2


def test_a_long_monologue_does_not_raise_the_floor_until_speech_is_lost():
    words = script(" ".join(["talking"] * 40), 0.5)  # 16 s without a pause longer than 80 ms
    states, frame = timeline(render(words, 17.5))
    on, off = edges(states, frame)
    assert len(on) == 1 and len(off) == 1 and off[0] > words[-1].end


def test_a_faint_hiss_after_digital_silence_is_not_speech():
    # Muted output, then a very quiet hiss: far above the floor of exact zeros, but below any speech.
    x = render([], 3.0, noise_db=-72)
    x[:RATE] = 0
    states, _ = timeline(x)
    assert not states.any()


def test_sound_that_never_dips_to_the_background_keeps_counting_as_speech():
    # Speech over a music bed never falls back to the noise floor between words. The floor must not
    # climb towards it during speech, or captions would stop after a second.
    from livesub.streaming import Word

    states, frame = timeline(render([Word(0.5, 6.5, " sustained")], 7.5))
    on, off = edges(states, frame)
    assert len(on) == 1 and len(off) == 1 and off[0] > 6.0
