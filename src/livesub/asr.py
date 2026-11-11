"""Speech recognizers. Anything with a `transcribe` method of this shape will do."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .captions import is_wide
from .streaming import Word, norm

SIMPLIFIED_PROMPT = "以下是普通话的句子，使用简体中文。"


class Transcriber(Protocol):
    def warm_up(self, language: str | None = None) -> None:
        """One throwaway reading, so that the first real one is not slowed by setting up."""
        self.model.transcribe(np.zeros(16000, np.float32), language=language or "en", beam_size=1, temperature=0.0)

    def transcribe(self, audio: np.ndarray, *, offset: float = 0.0, prompt: str = "", language: str | None = None) -> list[Word]:
        """Words in 16 kHz mono `audio`, with times on the stream's clock (`offset` + time in the audio)."""
        ...


class FasterWhisper:
    """faster-whisper (CTranslate2): runs Whisper several times faster than the reference code, on
    CPU or a CUDA GPU. The model is downloaded once from Hugging Face the first time it is used, and
    everything after that is offline."""

    def __init__(self, model: str = "small", device: str = "auto", compute_type: str = "default", beam_size: int = 5,
                 temperature: float | tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)):
        """`temperature`: a tuple lets Whisper decode again, more randomly, when a result looks wrong
        (repetitive or improbable): up to six decodes for one window. Good for files; live, pass 0.0,
        so that a bad window costs one decode and the captions do not stall."""
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:  # pragma: no cover - depends on the environment
            raise RuntimeError("faster-whisper is not installed: pip install \"livesub[whisper]\"") from e
        self.device = resolve_device(device)
        if compute_type == "default":
            compute_type = "float16" if self.device == "cuda" else "int8"
        try:
            # From the local cache, without asking Hugging Face: once a model is downloaded, livesub
            # starts with no network at all. (Otherwise faster-whisper checks for updates every time.)
            self.model = WhisperModel(model, device=self.device, compute_type=compute_type, local_files_only=True)
        except Exception:  # noqa: BLE001 - not cached yet (faster-whisper raises various errors here)
            import sys

            print(f"downloading Whisper '{model}' from Hugging Face (once; after this livesub runs offline)...", file=sys.stderr)
            self.model = WhisperModel(model, device=self.device, compute_type=compute_type)
        self.beam_size = beam_size
        self.temperature = temperature
        self.language: str | None = None

    def warm_up(self, language: str | None = None) -> None:
        """One throwaway reading, so that the first real one is not slowed by setting up."""
        self.model.transcribe(np.zeros(16000, np.float32), language=language or "en", beam_size=1, temperature=0.0)

    def transcribe(self, audio: np.ndarray, *, offset: float = 0.0, prompt: str = "", language: str | None = None) -> list[Word]:
        if (language or self.language) == "zh" and not any(is_wide(c) for c in prompt):
            # Whisper often writes Mandarin in Traditional characters; a Simplified prompt steers it.
            prompt = SIMPLIFIED_PROMPT + prompt
        segments, info = self.model.transcribe(
            audio, language=language, initial_prompt=prompt or None, beam_size=self.beam_size,
            temperature=self.temperature, word_timestamps=True, vad_filter=False, condition_on_previous_text=False,
            # Speech rarely needs more than 8 tokens a second. Without a cap, a window ending on a word
            # cut in half sometimes made the decoder run on for 2.5 s whatever the window's length,
            # which set the worst-case lag of live captions (3.7 s readings, down to 1.4 s with it).
            max_new_tokens=min(440, int(8 * len(audio) / 16000) + 16),
        )
        self.language = info.language
        words: list[Word] = []
        for s in segments:
            # Whisper's best-known failure: on silence or music it invents text ("Thanks for watching!").
            # faster-whisper already drops a window it thinks is probably not speech *and* is unsure
            # about (no_speech_prob > 0.6 with avg_logprob < -1); this also drops a segment it is
            # nearly sure is not speech, however confident the text (as whisper_streaming does).
            if s.no_speech_prob > 0.9:
                continue
            # Its other one: looping ("看,看,看,看,看,看,看" in a pause, seen in a real run).
            if s.compression_ratio > 2.4:
                continue
            words.extend(Word(offset + w.start, offset + w.end, w.word) for w in (s.words or []))
        return drop_loops(words)


def drop_loops(words: list[Word], limit: int = 4) -> list[Word]:
    """Remove a recognizer loop, keeping the first occurrence: a word repeated `limit` or more times in
    a row, or a phrase of 2 to 4 words repeated 3 or more times ("Thank you. Thank you. Thank you.").
    A single Chinese or Japanese character may repeat up to five times, because people do say
    对对对对 and 哈哈哈哈哈; Whisper's loops of one are longer."""
    keys = [norm(w.text) for w in words]
    out: list[Word] = []
    i = 0
    while i < len(words):
        best = None  # (phrase length, words covered)
        for n in range(1, 5):
            unit = keys[i:i + n]
            if len(unit) < n or not all(unit):
                continue
            r = 1
            while keys[i + r * n:i + (r + 1) * n] == unit:
                r += 1
            need = (6 if len(unit[0]) == 1 and is_wide(unit[0]) else limit) if n == 1 else 3
            if r >= need and (best is None or r * n > best[1]):
                best = (n, r * n)
        if best:
            out.extend(words[i:i + best[0]])
            i += best[1]
        else:
            out.append(words[i])
            i += 1
    return out


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
