import wave

import numpy as np
import pytest

from livesub.audio import RATE, Resampler, read_wav, resample, to_mono, write_wav


def tone(freq, rate, seconds=1.0, amp=0.5):
    t = np.arange(int(rate * seconds)) / rate
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def peak_hz(x, rate):
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return np.argmax(spectrum) * rate / len(x)


def test_wav_round_trip(tmp_path):
    x = tone(440, RATE, 0.5)
    write_wav(tmp_path / "a.wav", x)
    y, rate = read_wav(tmp_path / "a.wav")
    assert rate == RATE and len(y) == len(x)
    assert np.max(np.abs(y - x)) < 1e-4


@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_reads_every_integer_width_and_mixes_stereo(tmp_path, width):
    left, right = tone(300, 8000, 0.2, 0.4), tone(300, 8000, 0.2, 0.2)
    scale = 2 ** (8 * width - 1) - 1
    frames = np.stack([left, right], axis=1).reshape(-1)
    ints = np.round(frames * scale).astype(np.int64)
    if width == 1:
        raw = (ints + 128).astype(np.uint8).tobytes()
    else:
        raw = b"".join(int(v).to_bytes(width, "little", signed=True) for v in ints)
    with wave.open(str(tmp_path / "s.wav"), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(width)
        w.setframerate(8000)
        w.writeframes(raw)
    y, rate = read_wav(tmp_path / "s.wav")
    assert rate == 8000
    assert np.max(np.abs(y - (left + right) / 2)) < (0.02 if width == 1 else 1e-3)


@pytest.mark.parametrize("junk", [b"not audio", b"nope", b""])
def test_not_a_wav_gives_a_useful_error(tmp_path, junk):
    (tmp_path / "x.wav").write_bytes(junk)
    with pytest.raises(ValueError, match="ffmpeg"):
        read_wav(tmp_path / "x.wav")


@pytest.mark.parametrize("src", [44100, 48000, 22050, 8000])
def test_resampling_keeps_pitch_and_length(src):
    y = resample(tone(1000, src), src)
    assert abs(len(y) - RATE) <= 2
    assert abs(peak_hz(y, RATE) - 1000) < 3


def test_streaming_in_blocks_equals_one_shot():
    x = tone(700, 48000, 0.7) + tone(3100, 48000, 0.7, 0.2)
    whole = resample(x, 48000)
    r = Resampler(48000)
    sizes = [480, 1, 4800, 333, 1024]
    parts, i, k = [], 0, 0
    while i < len(x):
        parts.append(r(x[i:i + sizes[k % len(sizes)]]))
        i += sizes[k % len(sizes)]
        k += 1
    streamed = np.concatenate(parts)
    assert len(streamed) == len(whole)
    assert np.allclose(streamed, whole, atol=1e-5)


def test_downsampling_removes_tones_above_the_new_nyquist_instead_of_aliasing_them():
    # 11 kHz at 48 kHz would fold to 5 kHz at 16 kHz without the low-pass filter.
    y = resample(tone(11000, 48000), 48000)[2000:-2000]
    passband = resample(tone(5000, 48000), 48000)[2000:-2000]
    attenuation_db = 20 * np.log10(np.sqrt(np.mean(passband**2)) / (np.sqrt(np.mean(y**2)) + 1e-12))
    assert attenuation_db > 40, attenuation_db


def test_to_mono():
    assert to_mono(np.array([[1.0, 0.0], [0.5, 0.5]])).tolist() == [0.5, 0.5]
    assert to_mono([0.1, 0.2]).dtype == np.float32
