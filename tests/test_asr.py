import numpy as np
import pytest

from livesub import asr


def test_a_gpu_without_its_libraries_is_not_usable():
    def missing(name):
        raise OSError(name)

    ok, why = asr.cuda_usable(gpus=1, load=missing)
    assert not ok and "pip install nvidia-cublas-cu12" in why


def test_a_gpu_with_its_libraries_is():
    assert asr.cuda_usable(gpus=1, load=lambda name: None) == (True, "CUDA")
    assert asr.cuda_usable(gpus=0, load=lambda name: None)[0] is False


def test_auto_falls_back_to_the_cpu_and_cuda_explains(monkeypatch, capsys):
    monkeypatch.setattr(asr, "cuda_usable", lambda: (False, "cublas64_12.dll not found; " + asr.GPU_HINT))
    assert asr.resolve_device("auto") == "cpu"
    assert "running on the CPU" in capsys.readouterr().err
    with pytest.raises(RuntimeError, match="nvidia-cublas"):
        asr.resolve_device("cuda")
    with pytest.raises(ValueError):
        asr.resolve_device("gpu")


def test_no_gpu_at_all_is_quiet(monkeypatch, capsys):
    monkeypatch.setattr(asr, "cuda_usable", lambda: (False, "no CUDA GPU found"))
    assert asr.resolve_device("auto") == "cpu"
    assert capsys.readouterr().err == ""


def test_libraries_installed_as_nvidia_wheels_are_found(tmp_path, monkeypatch):
    import os
    import site
    import sys

    sub = "bin" if os.name == "nt" else "lib"
    names = ["cublas64_12.dll", "cudnn_ops64_9.dll"] if sys.platform == "win32" else ["libcublas.so.12", "libcudnn_ops.so.9"]
    for pkg, name in zip(["cublas", "cudnn"], names, strict=False):
        d = tmp_path / "nvidia" / pkg / sub
        d.mkdir(parents=True)
        (d / name).write_bytes(b"")
    monkeypatch.setattr(site, "getsitepackages", lambda: [str(tmp_path)])
    monkeypatch.setattr(site, "getusersitepackages", lambda: str(tmp_path / "user"))
    if os.name == "nt":
        monkeypatch.setattr(os, "add_dll_directory", lambda p: None, raising=False)
    monkeypatch.setenv("PATH", os.environ.get("PATH", ""))
    loaded = []

    def load(name):
        if os.sep not in name and "/" not in name:
            raise OSError(name)  # not on the search path
        loaded.append(name)

    assert asr.cuda_usable(gpus=1, load=load) == (True, "CUDA")
    assert len(loaded) == 2 and all(str(tmp_path) in p for p in loaded)


def test_a_recognizer_loop_is_cut_but_real_repetition_is_kept():
    from livesub.streaming import Word, join

    looped = [Word(i * 0.2, i * 0.2 + 0.2, t) for i, t in enumerate(["规则", "看,", "看,", "看,", "看,", "看,", "看", "谢谢"])]
    assert join(asr.drop_loops(looped)) == "规则看,谢谢"
    said = [Word(i * 0.3, i * 0.3 + 0.3, t) for i, t in enumerate([" no,", " no,", " no!", " stop"])]
    assert asr.drop_loops(said) == said  # three in a row is still speech
    assert asr.drop_loops([]) == []


class FakeSegment:
    def __init__(self, words, no_speech=0.0, logprob=-0.2, compression=1.2):
        from types import SimpleNamespace

        self.words = [SimpleNamespace(start=a, end=b, word=t) for a, b, t in words]
        self.text = "".join(t for _, _, t in words)
        self.no_speech_prob, self.avg_logprob, self.compression_ratio = no_speech, logprob, compression


class FakeTokenizer:
    """Worst case for the token count: one token per character."""

    def encode(self, text):
        from types import SimpleNamespace

        return SimpleNamespace(ids=list(range(len(text))))


class FakeModel:
    def __init__(self, segments, language="en"):
        self.segments, self.language, self.calls = segments, language, []
        self.hf_tokenizer = FakeTokenizer()

    def transcribe(self, audio, **kw):
        from types import SimpleNamespace

        self.calls.append(kw)
        # faster-whisper's own rule (transcribe.py): the prompt, cut to its last 223 tokens, plus the
        # task tokens, plus max_new_tokens must not exceed 448. An earlier fake without it let a cap
        # through that crashed every long recording.
        prompt = kw.get("initial_prompt")
        prompt_tokens = (min(223, len(self.hf_tokenizer.encode(" " + prompt.strip()).ids)) if prompt else 0) + 4
        if kw.get("max_new_tokens") is not None and prompt_tokens + kw["max_new_tokens"] > 448:
            raise ValueError(f"The length of the prompt is {prompt_tokens}, and the max_new_tokens {kw['max_new_tokens']} exceeds 448")
        return iter(self.segments), SimpleNamespace(language=self.language)


def whisper(segments, language="en"):
    w = asr.FasterWhisper.__new__(asr.FasterWhisper)  # skip loading a real model
    w.model, w.beam_size, w.language, w.device, w.temperature = FakeModel(segments, language), 5, None, "cpu", 0.0
    return w


def test_transcribe_puts_words_on_the_stream_clock_and_drops_hallucinations():
    from livesub.streaming import join

    w = whisper([
        FakeSegment([(0.0, 0.3, " Hello"), (0.3, 0.6, " there.")]),
        FakeSegment([(1.0, 1.5, " Thanks for watching!")], no_speech=0.95, logprob=-0.3),  # invented in silence, confidently
        FakeSegment([(2.0, 2.1, " the")] * 3, compression=3.1),  # looping
    ])
    words = w.transcribe(np.zeros(16000, np.float32), offset=10.0)
    assert join(words) == "Hello there." and words[0].start == 10.0
    kw = w.model.calls[0]
    assert kw["word_timestamps"] and not kw["vad_filter"] and not kw["condition_on_previous_text"]
    assert kw["max_new_tokens"] == 8 + 16  # one second of audio
    w.transcribe(np.zeros(16000 * 30, np.float32))
    assert w.model.calls[-1]["max_new_tokens"] == 256
    # With a long prompt the cap shrinks so that prompt + cap fit Whisper's 448 tokens (the fake model
    # raises otherwise), and a long recording is capped per 30 s chunk.
    w.transcribe(np.zeros(16000 * 90, np.float32), prompt="word " * 200)
    assert w.model.calls[-1]["max_new_tokens"] == 448 - 5 - 223
    assert w.language == "en"


def test_chinese_always_gets_the_simplified_prompt():
    w = whisper([], language="zh")
    w.transcribe(np.zeros(1600, np.float32), language="zh")
    assert w.model.calls[-1]["initial_prompt"].startswith(asr.SIMPLIFIED_PROMPT)
    # Even after Chinese text, which may itself be Traditional and would carry the drift on.
    w.transcribe(np.zeros(1600, np.float32), prompt="大家好，這是繁體", language="zh")
    assert w.model.calls[-1]["initial_prompt"] == asr.SIMPLIFIED_PROMPT + "大家好，這是繁體"
    w.transcribe(np.zeros(1600, np.float32), language=None)  # detected as Chinese last time
    assert w.model.calls[-1]["initial_prompt"].startswith(asr.SIMPLIFIED_PROMPT)
    e = whisper([], language="en")
    e.transcribe(np.zeros(1600, np.float32), language="en")
    assert e.model.calls[-1]["initial_prompt"] is None


def test_a_loop_that_compresses_normally_is_still_cut():
    from livesub.streaming import join

    loop = [(2.0 + i * 0.2, 2.2 + i * 0.2, "看,") for i in range(6)]
    w = whisper([FakeSegment([(0.0, 0.5, "规则")] + loop + [(3.5, 3.9, "谢谢")], compression=1.6)], language="zh")
    assert join(w.transcribe(np.zeros(16000, np.float32), language="zh")) == "规则看,谢谢"


def test_phrase_loops_are_cut_and_real_chinese_repetition_is_kept():
    from livesub.streaming import Word, join

    def ws(texts):
        return [Word(i * 0.2, i * 0.2 + 0.2, t) for i, t in enumerate(texts)]

    looped = ws([" Thank", " you.", " Thank", " you.", " Thank", " you.", " Bye."])
    assert join(asr.drop_loops(looped)) == "Thank you. Bye."
    twice = ws([" Thank", " you.", " Thank", " you."])
    assert asr.drop_loops(twice) == twice  # twice is still speech
    for said in ("对对对对", "哈哈哈哈哈"):
        words = ws(list(said))
        assert asr.drop_loops(words) == words, said
    assert join(asr.drop_loops(ws(list("好好好好好好好")))) == "好"


def test_live_decodes_once_and_files_may_retry():
    from livesub import cli

    seen = {}

    class Recorder:
        def __init__(self, *a, temperature=None, **k):
            seen["temperature"] = temperature

    import livesub.asr as mod

    original = mod.FasterWhisper
    mod.FasterWhisper = Recorder
    try:
        for command, expected in (("live", 0.0), ("file", (0.0, 0.2, 0.4, 0.6, 0.8, 1.0))):
            args = cli.build_parser().parse_args([command] + (["x.wav"] if command == "file" else []) + ["--device", "cpu"])
            cli.make_transcriber(args)
            assert seen["temperature"] == expected, command
    finally:
        mod.FasterWhisper = original


def test_a_long_recording_with_real_faster_whisper():
    """The real library, where it is installed with the tiny model cached (skipped elsewhere, such as
    in CI): long speech regions, each with the earlier text as prompt, in English and Chinese. This is
    what crashed with "prompt + max_new_tokens exceeds 448" before."""
    pytest.importorskip("faster_whisper")
    from livesub.pipeline import transcribe_file

    try:
        w = asr.FasterWhisper("tiny", device="cpu")
    except Exception as e:  # noqa: BLE001 - not cached, no network
        pytest.skip(f"tiny model not available: {e}")
    rng = np.random.default_rng(0)
    t = np.arange(16000 * 75) / 16000
    # A long stretch of speech-like sound with no pause: one region of the full length before splitting.
    audio = (0.1 * rng.standard_normal(len(t)) * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t))).astype(np.float32)
    audio[: 16000 * 4] *= 0.01
    audio[16000 * 4: 16000 * 6] = 0
    for language in ("en", "zh"):
        transcribe_file(w, audio, language=language)  # must not raise
    long_prompt = "这是一段很长的提示文字。" * 40
    w.transcribe(audio[: 16000 * 30], prompt=long_prompt, language="zh")


def test_subtitle_credits_whisper_invents_are_dropped():
    from livesub.streaming import join

    w = whisper([
        FakeSegment([(0.0, 0.5, "速度要快。")]),
        FakeSegment([(5.0, 5.6, "字幕製作曲")]),
        FakeSegment([(6.0, 6.8, " Subtitles by the Amara.org community")]),
    ], language="zh")
    assert join(w.transcribe(np.zeros(16000 * 7, np.float32), language="zh")) == "速度要快。"


def test_only_a_missing_model_leads_to_a_download(monkeypatch, capsys):
    import sys
    import types

    calls = []

    class Model:
        def __init__(self, name, device, compute_type, local_files_only=False):
            calls.append(local_files_only)
            if compute_type == "float16":
                raise ValueError("Requested float16 compute type, but the target device does not support it")
            if local_files_only:
                raise FileNotFoundError("not in the cache")

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=Model))
    with pytest.raises(ValueError, match="float16"):
        asr.FasterWhisper("tiny", device="cpu", compute_type="float16")
    assert calls == [True], "a bad compute type must not turn into a download"
    asr.FasterWhisper("tiny", device="cpu")
    assert calls[1:] == [True, False] and "downloading" in capsys.readouterr().err
