"""Speech recognizers. Anything with a `transcribe` method of this shape will do."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .captions import is_wide
from .streaming import Word

SIMPLIFIED_PROMPT = "以下是普通话的句子，使用简体中文。"


class Transcriber(Protocol):
    def transcribe(self, audio: np.ndarray, *, offset: float = 0.0, prompt: str = "", language: str | None = None) -> list[Word]:
        """Words in 16 kHz mono `audio`, with times on the stream's clock (`offset` + time in the audio)."""
        ...


class FasterWhisper:
    """faster-whisper (CTranslate2): runs Whisper several times faster than the reference code, on
    CPU or a CUDA GPU. The model is downloaded once from Hugging Face the first time it is used, and
    everything after that is offline."""

    def __init__(self, model: str = "small", device: str = "auto", compute_type: str = "default", beam_size: int = 5):
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:  # pragma: no cover - depends on the environment
            raise RuntimeError("faster-whisper is not installed: pip install \"livesub[whisper]\"") from e
        self.device = resolve_device(device)
        if compute_type == "default":
            compute_type = "float16" if self.device == "cuda" else "int8"
        self.model = WhisperModel(model, device=self.device, compute_type=compute_type)
        self.beam_size = beam_size
        self.language: str | None = None

    def transcribe(self, audio: np.ndarray, *, offset: float = 0.0, prompt: str = "", language: str | None = None) -> list[Word]:
        if (language or self.language) == "zh" and not any(is_wide(c) for c in prompt):
            # Whisper often writes Mandarin in Traditional characters; a Simplified prompt steers it.
            prompt = SIMPLIFIED_PROMPT + prompt
        segments, info = self.model.transcribe(
            audio, language=language, initial_prompt=prompt or None, beam_size=self.beam_size,
            word_timestamps=True, vad_filter=False, condition_on_previous_text=False,
        )
        self.language = info.language
        words: list[Word] = []
        for s in segments:
            # Whisper's best-known failure: on silence or music it invents text ("Thanks for watching!").
            # A segment it thinks is probably not speech, and is unsure about, is dropped.
            if s.no_speech_prob > 0.6 and s.avg_logprob < -1.0:
                continue
            words.extend(Word(offset + w.start, offset + w.end, w.word) for w in (s.words or []))
        return words


GPU_HINT = ("the GPU needs CUDA 12 cuBLAS and cuDNN 9: pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 "
            "(or install the CUDA toolkit)")


def cuda_usable(gpus=None, load=None) -> tuple[bool, str]:
    """Can CTranslate2 actually run on a GPU here? Having one is not enough: it also needs the CUDA
    libraries, and without them it loads the model fine and fails on the first sentence."""
    import ctypes
    import sys

    if gpus is None:
        try:
            import ctranslate2

            gpus = ctranslate2.get_cuda_device_count()
        except Exception:  # noqa: BLE001 - no CTranslate2, no GPU
            gpus = 0
    if not gpus:
        return False, "no CUDA GPU found"
    names = ["cublas64_12.dll", "cudnn_ops64_9.dll"] if sys.platform == "win32" else ["libcublas.so.12", "libcudnn_ops.so.9"]
    load = load or ctypes.CDLL
    for name in names:
        try:
            load(name)
        except OSError:
            # pip-installed NVIDIA wheels live in site-packages/nvidia/*/bin (Windows) or lib (Linux).
            if not _load_nvidia_wheel(name, load):
                return False, f"{name} not found; {GPU_HINT}"
    return True, "CUDA"


def _load_nvidia_wheel(name: str, load) -> bool:
    import os
    import site
    from pathlib import Path

    for root in site.getsitepackages() + [site.getusersitepackages()]:
        for lib in Path(root, "nvidia").glob("*/bin" if os.name == "nt" else "*/lib"):
            if (lib / name).exists():
                if os.name == "nt":
                    os.add_dll_directory(str(lib))
                    os.environ["PATH"] = str(lib) + os.pathsep + os.environ.get("PATH", "")
                try:
                    load(str(lib / name))
                    return True
                except OSError:
                    return False
    return False


def resolve_device(device: str) -> str:
    """"auto" becomes "cuda" only if the GPU can really be used; asking for "cuda" when it cannot is
    an error that says what to install."""
    if device not in ("auto", "cuda", "cpu"):
        raise ValueError('device must be auto, cuda or cpu')
    if device == "cpu":
        return "cpu"
    ok, why = cuda_usable()
    if ok:
        return "cuda"
    if device == "cuda":
        raise RuntimeError(f"cannot use the GPU: {why}")
    if why != "no CUDA GPU found":
        import sys

        print(f"note: running on the CPU — {why}", file=sys.stderr)
    return "cpu"
