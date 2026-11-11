"""Audio plumbing: WAV in and out, mono mixdown, and a streaming resampler to the 16 kHz Whisper expects."""

from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np

RATE = 16_000


def to_mono(x: np.ndarray) -> np.ndarray:
    """Average the channels of a (frames, channels) array; 1-D input is returned as float32."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return np.ascontiguousarray(x, dtype=np.float32)


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a PCM WAV file (8, 16, 24 or 32-bit integer) as mono float32 in [-1, 1]."""
    try:
        with wave.open(str(path), "rb") as w:
            channels, width, rate, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
            raw = w.readframes(n)
    except (wave.Error, EOFError) as e:  # a file shorter than a WAV header raises EOFError
        raise ValueError(f"{path}: not a PCM WAV file ({e}); install faster-whisper (pip install 'livesub[whisper]') to read "
                         "other formats, or convert it: ffmpeg -i in.mp4 -ar 16000 -ac 1 out.wav") from e
    if width == 1:
        x = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128) / 128
    elif width == 2:
        x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768
    elif width == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        v = np.where(v >= 1 << 23, v - (1 << 24), v)
        x = v.astype(np.float32) / (1 << 23)
    elif width == 4:
        x = (np.frombuffer(raw, "<i4").astype(np.float64) / 2**31).astype(np.float32)
    else:
        raise ValueError(f"{path}: unsupported sample width {width * 8} bits")
    return to_mono(x.reshape(-1, channels)), rate


def write_wav(path: str | Path, x: np.ndarray, rate: int = RATE) -> None:
    """Write mono float audio as 16-bit PCM."""
    pcm = (np.clip(np.asarray(x, dtype=np.float32), -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def _lowpass(cutoff: float, taps: int) -> np.ndarray:
    """Windowed-sinc FIR low-pass; `cutoff` in cycles per sample (0 to 0.5)."""
    k = np.arange(taps) - (taps - 1) / 2
    h = 2 * cutoff * np.sinc(2 * cutoff * k) * np.blackman(taps)
    return (h / h.sum()).astype(np.float32)


class Resampler:
    """Resample a stream of blocks from `src` Hz to `dst` Hz.

    Feeding a signal in blocks gives the same output as feeding it at once: the low-pass filter and
    the interpolation position carry over between calls. Downsampling filters first, so tones above
    the new Nyquist frequency are removed rather than folded back as aliases.
    """

    def __init__(self, src: int, dst: int = RATE, taps: int = 127):
        if src <= 0 or dst <= 0:
            raise ValueError("sample rates must be positive")
        self.src, self.dst = src, dst
        self.step = src / dst
        self.h = _lowpass(0.5 * dst / src * 0.9, taps) if dst < src else None
        self.hist = np.zeros(taps - 1 if self.h is not None else 0, np.float32)
        self.data = np.zeros(0, np.float32)  # filtered samples not yet passed by the read position
        self.pos = 0.0  # read position within self.data

    def __call__(self, block: np.ndarray) -> np.ndarray:
        block = np.asarray(block, dtype=np.float32)
        if self.src == self.dst:
            return block
        if self.h is not None:
            buf = np.concatenate([self.hist, block])
            filtered = np.convolve(buf, self.h, mode="valid").astype(np.float32)
            self.hist = buf[len(buf) - len(self.hist):]
        else:
            filtered = block
        data = np.concatenate([self.data, filtered])
        last = len(data) - 1
        n = int(math.floor((last - self.pos) / self.step)) + 1 if last >= self.pos else 0
        t = self.pos + np.arange(n) * self.step
        out = np.interp(t, np.arange(len(data)), data).astype(np.float32) if n else np.zeros(0, np.float32)
        nxt = self.pos + n * self.step
        drop = min(int(math.floor(nxt)), len(data))
        self.data = data[drop:]
        self.pos = nxt - drop
        return out


def resample(x: np.ndarray, src: int, dst: int = RATE) -> np.ndarray:
    return Resampler(src, dst)(x)


def read_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Any recording as mono float32: WAV directly, and anything else (MP3, M4A, MP4, MKV, ...) through
    PyAV, which faster-whisper installs."""
    try:
        return read_wav(path)
    except ValueError as wav_error:
        try:
            import av
        except ImportError:
            raise wav_error from None
        if not Path(path).is_file():
            raise FileNotFoundError(path) from None
        try:
            with av.open(str(path)) as container:
                stream = next((s for s in container.streams if s.type == "audio"), None)
                if stream is None:
                    raise ValueError(f"{path}: has no audio track")
                resampler = av.AudioResampler(format="flt", layout="mono", rate=RATE)
                chunks = [f.to_ndarray().reshape(-1) for frame in container.decode(stream) for f in resampler.resample(frame)]
                chunks += [f.to_ndarray().reshape(-1) for f in resampler.resample(None)]
        except av.error.FFmpegError as e:
            raise ValueError(f"{path}: cannot read audio from it; is it an audio or video file? ({e})") from e
        return (np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, np.float32)), RATE
