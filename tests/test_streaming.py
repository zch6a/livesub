from livesub.streaming import Agreement, Word, join, norm


def words(*pairs):
    return [Word(s, s + 0.3, t) for s, t in pairs]


def test_commits_only_what_two_guesses_in_a_row_agree_on():
    a = Agreement()
    a.insert(words((0.0, " the"), (0.4, " ca-")))
    assert a.flush() == []  # nothing to compare with yet
    a.insert(words((0.0, " the"), (0.4, " cat"), (0.8, " s-")))
    assert join(a.flush()) == "the"  # "ca-" and "cat" disagree
    a.insert(words((0.4, " cat"), (0.8, " sat"), (1.2, " on")))
    assert join(a.flush()) == "cat"
    assert join(a.tentative()) == "sat on"
    assert join(a.finish()) == "sat on"
    assert join(a.committed) == "the cat sat on"


def test_case_and_punctuation_do_not_block_agreement():
    a = Agreement()
    a.insert(words((0.0, " Hello"), (0.4, " world")))
    a.flush()
    a.insert(words((0.0, " hello,"), (0.4, " world.")))
    assert join(a.flush()) == "hello, world."  # the newer wording wins
    assert norm(" Hello,") == norm("hello")


def test_words_already_committed_are_not_committed_again():
    a = Agreement()
    for _ in range(2):
        a.insert(words((0.0, " one"), (0.4, " two")))
        a.flush()
    assert join(a.committed) == "one two"
    # A later window that starts earlier and repeats the committed tail, with drifted timestamps.
    a.insert(words((0.65, " two"), (0.95, " three")))
    a.flush()
    a.insert(words((0.65, " two"), (0.95, " three"), (1.35, " four")))
    a.flush()
    assert join(a.committed) == "one two three"


def test_a_contradicting_guess_resets_the_comparison():
    a = Agreement()
    a.insert(words((0.0, " I"), (0.3, " scream")))
    a.flush()
    a.insert(words((0.0, " ice"), (0.3, " cream")))
    assert a.flush() == []
    a.insert(words((0.0, " ice"), (0.3, " cream")))
    assert join(a.flush()) == "ice cream"


def test_replay_of_a_real_whisper_run_loses_no_words():
    # faster-whisper "small" on Windows text-to-speech, readings 1 s apart (times as it gave them).
    # On the last reading it returned only the continuation, with "harder" put 0.7 s too early; an
    # earlier version filtered words by time and lost "harder than".
    def W(s):  # noqa: N802 - reads like the data it parses
        return [Word(float(a), float(b), " " + t) for t, a, b in (x.split("|") for x in s.split())]
    readings = [
        "Today|3.50|3.84 we|3.84|4.08 are|4.08|4.24 going|4.24|4.46 to|4.46|4.72 talk...|4.72|4.88",
        "Today|3.50|3.82 we|3.82|4.08 are|4.08|4.24 going|4.24|4.46 to|4.46|4.70 talk|4.70|4.94 about|4.94|5.18 live|5.18|5.46 captions.|5.46|5.78",
        "today|3.50|3.82 we|3.82|4.08 are|4.08|4.24 going|4.24|4.46 to|4.46|4.70 talk|4.70|4.96 about|4.96|5.18 live|5.18|5.44 captions|5.44|5.88 and|5.88|6.60 why|6.60|6.82 they|6.82|7.00 are|7.00|7.08",
        "Today|3.50|3.82 we|3.82|4.08 are|4.08|4.24 going|4.24|4.46 to|4.46|4.70 talk|4.70|4.94 about|4.94|5.18 live|5.18|5.46 captions,|5.46|5.88 and|6.46|6.62 why|6.62|6.82 they|6.82|7.00 are|7.00|7.16 harder|7.16|7.42 than|7.42|7.66 they|7.66|7.80 look.|7.80|8.00",
        "harder|6.44|7.04 than|7.04|7.64 they|7.64|7.80 look.|7.80|8.04",
    ]
    a = Agreement()
    for r in readings:
        a.insert(W(r))
        a.flush()
    a.finish()
    assert join(a.committed) == "Today we are going to talk about live captions and why they are harder than they look."


def test_a_reworded_last_word_is_not_lost_either():
    a = Agreement()
    for _ in range(2):
        a.insert(words((0.0, " we"), (0.4, " were")))
        a.flush()
    # The next reading rewords the committed tail and continues: no textual anchor, so times decide.
    a.insert(words((0.45, " are"), (0.8, " here")))
    a.flush()
    a.insert(words((0.45, " are"), (0.8, " here")))
    a.flush()
    assert join(a.committed) == "we were are here"  # a possible duplicate, rather than a lost word


def test_a_continuation_smeared_far_back_is_anchored_on_the_previous_guess():
    # As in the real run, but with "harder" smeared to end before the last committed word even began:
    # no time rule can tell it from an old word; the previous guess, which began with "harder", can.
    a = Agreement()
    a.insert([Word(6.6, 6.8, " why"), Word(6.82, 7.0, " they"), Word(7.0, 7.16, " are"), Word(7.16, 7.42, " har-")])
    a.flush()
    a.insert([Word(6.6, 6.8, " why"), Word(6.82, 7.0, " they"), Word(7.0, 7.16, " are"), Word(7.16, 7.42, " harder"), Word(7.42, 7.66, " than")])
    assert join(a.flush()) == "why they are"
    a.insert([Word(6.3, 6.9, " harder"), Word(6.9, 7.6, " than"), Word(7.64, 7.8, " they"), Word(7.8, 8.04, " look.")])
    a.flush()
    a.finish()
    assert join(a.committed) == "why they are harder than they look."


def test_a_new_sentence_may_begin_with_the_last_word_of_the_previous_one():
    # Found with real Whisper: "Thank you." then, after a pause, "You know, this is a test."
    a = Agreement()
    for _ in range(2):
        a.insert(words((0.5, " Thank"), (0.8, " you.")), window_start=0.3)
        a.flush()
    a.finish()
    nxt = words((2.6, " You"), (2.9, " know,"), (3.2, " this"))
    for _ in range(2):
        a.insert(nxt, window_start=2.4)  # a new window, after the pause
        a.flush()
    a.finish()
    assert join(a.committed) == "Thank you. You know, this"
