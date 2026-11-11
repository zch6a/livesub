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
        self.no_speech_prob, self.avg_logprob, self.compression_ratio = no_speech, logprob, compression


class FakeModel:
    def __init__(self, segments, language="en"):
        self.segments, self.language, self.calls = segments, language, []

    def transcribe(self, audio, **kw):
        from types import SimpleNamespace

        self.calls.append(kw)
        return iter(self.segments), SimpleNamespace(language=self.language)


def whisper(segments, language="en"):
    w = asr.FasterWhisper.__new__(asr.FasterWhisper)  # skip loading a real model
    w.model, w.beam_size, w.language, w.device = FakeModel(segments, language), 5, None, "cpu"
    return w


def test_transcribe_puts_words_on_the_stream_clock_and_drops_hallucinations():
    from livesub.streaming import join

    w = whisper([
        FakeSegment([(0.0, 0.3, " Hello"), (0.3, 0.6, " there.")]),
        FakeSegment([(1.0, 1.5, " Thanks for watching!")], no_speech=0.9, logprob=-1.4),  # invented in silence
        FakeSegment([(2.0, 2.1, " the")] * 3, compression=3.1),  # looping
    ])
    words = w.transcribe(np.zeros(16000, np.float32), offset=10.0)
    assert join(words) == "Hello there." and words[0].start == 10.0
    kw = w.model.calls[0]
    assert kw["word_timestamps"] and not kw["vad_filter"] and not kw["condition_on_previous_text"]
    assert w.language == "en"


def test_chinese_gets_a_simplified_prompt_unless_the_prompt_is_already_chinese():
    w = whisper([], language="zh")
    w.transcribe(np.zeros(1600, np.float32), language="zh")
    assert w.model.calls[-1]["initial_prompt"].startswith(asr.SIMPLIFIED_PROMPT)
    w.transcribe(np.zeros(1600, np.float32), prompt="大家好", language="zh")
    assert w.model.calls[-1]["initial_prompt"] == "大家好"
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
