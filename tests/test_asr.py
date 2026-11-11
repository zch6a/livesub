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
    for pkg, name in zip(["cublas", "cudnn"], names):
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
