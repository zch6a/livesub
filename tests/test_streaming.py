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
