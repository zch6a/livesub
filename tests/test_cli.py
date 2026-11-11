import re

import pytest

from livesub import cli
from livesub.audio import write_wav
from synth import ScriptedTranscriber, render, script
from test_translate import FakeLLM

WORDS = script("Hello and welcome.", 0.5) + script("This file has two sentences.", 3.0)


@pytest.fixture
def recording(tmp_path, monkeypatch):
    path = tmp_path / "talk.wav"
    write_wav(path, render(WORDS, 6.0))
    monkeypatch.setattr(cli, "make_transcriber", lambda args: ScriptedTranscriber(WORDS))
    return path


def test_file_writes_srt_next_to_the_input(recording):
    assert cli.main(["file", str(recording)]) == 0
    srt = recording.with_suffix(".srt").read_text(encoding="utf-8")
    assert re.match(r"1\n00:00:00,[3-6]\d\d --> ", srt)  # 0.5 s, give or take the recognizer's jitter
    assert "Hello and welcome.\n\n2\n" in srt and "This file has two sentences." in srt


def test_file_with_translation_writes_both_lines(recording, tmp_path):
    llm = FakeLLM(reply="<think>ok</think>译文")
    try:
        out = tmp_path / "out.vtt"
        assert cli.main(["file", str(recording), "-o", str(out), "--translate-to", "zh", "--llm-url", llm.url, "--llm-model", "m"]) == 0
    finally:
        llm.close()
    vtt = out.read_text(encoding="utf-8")
    assert vtt.startswith("WEBVTT") and "Hello and welcome.\n译文\n" in vtt
    assert len(llm.requests) == 2


def test_translation_without_a_model_explains_how_to_get_one(recording, monkeypatch):
    monkeypatch.delenv("LIVESUB_LLM_BASE_URL", raising=False)
    with pytest.raises(SystemExit, match="ollama"):
        cli.main(["file", str(recording), "--translate-to", "zh"])


def test_bad_input_is_a_clear_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "make_transcriber", lambda args: ScriptedTranscriber([]))
    (tmp_path / "x.wav").write_bytes(b"nope")
    assert cli.main(["file", str(tmp_path / "x.wav")]) == 1
    err = capsys.readouterr().err
    try:
        import av  # noqa: F401 - with a decoder installed, the decoder's own message
        assert "cannot read audio" in err
    except ImportError:
        assert "ffmpeg" in err


def test_help_and_version(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--version"])
    assert "livesub 0.1.0" in capsys.readouterr().out


def test_the_live_loop_is_still_available_for_files(recording):
    assert cli.main(["file", str(recording), "--live-loop"]) == 0
    assert "This file has two sentences." in recording.with_suffix(".srt").read_text(encoding="utf-8")


def test_llm_url_still_takes_the_key_and_model_from_the_environment(monkeypatch):
    from livesub.translate import ChatTranslator

    monkeypatch.setattr(ChatTranslator, "check", lambda self: None)  # the hosted endpoint is made up
    monkeypatch.setenv("LIVESUB_LLM_API_KEY", "secret")
    monkeypatch.setenv("LIVESUB_LLM_MODEL", "deepseek-chat")
    args = cli.build_parser().parse_args(["file", "x.wav", "--translate-to", "zh", "--llm-url", "https://api.example.com/v1"])
    t = cli.make_translator(args)
    assert t.api_key == "secret" and t.model == "deepseek-chat"


def test_beam_is_checked_when_the_arguments_are_read(capsys):
    with pytest.raises(SystemExit):
        cli.main(["file", "x.wav", "--beam", "0"])
    assert "must be 1 to 10" in capsys.readouterr().err


def test_output_problems_are_found_before_the_model_loads(tmp_path, monkeypatch, capsys):
    loaded = []
    monkeypatch.setattr(cli, "make_transcriber", lambda args: loaded.append(1))
    talk = tmp_path / "talk.wav"
    write_wav(talk, render(WORDS, 6.0))
    (tmp_path / "talk.srt").write_text("subtitles someone made by hand", encoding="utf-8")
    for argv, message in [
        (["file", str(talk)], "pass --overwrite"),  # would replace the existing talk.srt
        (["file", str(talk), "-o", str(tmp_path / "missing" / "x.srt")], "does not exist"),
        (["file", str(talk), "-o", str(tmp_path / "x.txt")], ".srt or .vtt"),
    ]:
        assert cli.main(argv) == 1
        assert message in capsys.readouterr().err, argv
    assert loaded == []
    assert (tmp_path / "talk.srt").read_text(encoding="utf-8") == "subtitles someone made by hand"


def test_overwrite_replaces_the_file(recording):
    recording.with_suffix(".srt").write_text("old", encoding="utf-8")
    assert cli.main(["file", str(recording), "--overwrite"]) == 0
    assert "Hello and welcome." in recording.with_suffix(".srt").read_text(encoding="utf-8")


def test_settings_are_checked_when_the_arguments_are_read(capsys):
    for argv, message in [(["live", "--step", "0.1"], "must be 0.2 to 6"), (["file", "x.wav", "--step", "abc"], "not a number")]:
        with pytest.raises(SystemExit):
            cli.main(argv)
        assert message in capsys.readouterr().err


def test_translation_options_without_a_target_are_an_error(recording, capsys):
    assert cli.main(["file", str(recording), "--llm-url", "http://localhost:11434/v1"]) == 1
    assert "only matter with --translate-to" in capsys.readouterr().err
    assert cli.main(["file", str(recording), "--subs", "translation"]) == 1
    assert "needs --translate-to" in capsys.readouterr().err


def test_an_unreachable_translator_fails_before_the_model_loads(recording, monkeypatch, capsys):
    loaded = []
    monkeypatch.setattr(cli, "make_transcriber", lambda args: loaded.append(1))
    assert cli.main(["file", str(recording), "--overwrite", "--translate-to", "zh", "--llm-url", "http://127.0.0.1:9/v1"]) == 1
    assert "cannot reach the translation endpoint" in capsys.readouterr().err and loaded == []


def test_llm_key_also_applies_to_an_endpoint_from_the_environment(monkeypatch):
    from livesub.translate import ChatTranslator

    monkeypatch.setattr(ChatTranslator, "check", lambda self: None)
    monkeypatch.setenv("LIVESUB_LLM_BASE_URL", "https://api.example.com/v1")
    monkeypatch.delenv("LIVESUB_LLM_API_KEY", raising=False)
    args = cli.build_parser().parse_args(["live", "--translate-to", "zh", "--llm-key", "sk-from-flag"])
    t = cli.make_translator(args)
    assert t.api_key == "sk-from-flag" and t.timeout == 10.0  # live: a short timeout
