<div align="center">

# livesub

**Offline live captions for anything your computer plays or hears, with optional translation.**

<a href="https://github.com/zch6a/livesub/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/zch6a/livesub/actions/workflows/ci.yml/badge.svg"></a>
<img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-3776ab?logo=python&logoColor=white">
<img alt="Runs offline" src="https://img.shields.io/badge/runs-offline-2ea44f">
<a href="LICENSE"><img alt="MIT license" src="https://img.shields.io/badge/license-MIT-blue"></a>

**English** · [简体中文](README.zh-CN.md)

</div>

---

A video call, a lecture stream, a game, a podcast: livesub recognizes the speech on your own machine
with Whisper and shows it in a caption bar that floats over every window, optionally with a
translation underneath.

- **Offline.** Speech recognition runs locally with [faster-whisper](https://github.com/SYSTRAN/faster-whisper),
  on the CPU or a CUDA GPU. Translation goes to an OpenAI-compatible endpoint *you* choose. Point it at
  a local model (Ollama, LM Studio, llama.cpp) and nothing leaves your computer.
- **Captions that don't flicker.** A word is shown dimmed while it is being recognized. It turns white
  once two readings in a row agree on it, and from then on it never changes. (At the end of speech
  the last reading is taken as it is, since no second one will come.)
- **System-wide.** It captures what your speakers play (WASAPI loopback on Windows, monitor sources on
  Linux) or a microphone.
- **Subtitle files too.** `livesub file` writes SRT or WebVTT for a recording (WAV, MP3, M4A, MP4, MKV, …):
  original, translated, or both.

> **Status: 0.1, in development.** Recognition has been checked end to end with real Whisper on English
> and Chinese speech from text-to-speech, including fast, noisy, music-backed and mixed-language
> speech. The rest of the pipeline, live mode included, is covered by 111 tests on synthetic audio. Live
> capture from a real sound card has not been tested by the author yet, and translation only against a
> stand-in endpoint, not a real model. See [POLISH_LOG.md](POLISH_LOG.md).

## Install

```bash
pip install "livesub[all] @ git+https://github.com/zch6a/livesub"
```

`[all]` adds faster-whisper and soundcard. The Whisper model is downloaded once, the first time you use it.

**GPU (optional).** CTranslate2 needs CUDA 12 cuBLAS and cuDNN 9. If they are missing, livesub says so
and runs on the CPU:

```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

## Use

```bash
livesub live                                  # caption what the computer plays, in an overlay
livesub live --source mic --console           # a microphone, in the terminal
livesub live --language en --translate-to zh --llm-url http://localhost:11434/v1 --llm-model qwen2.5:7b
livesub live --save meeting.srt               # keep the captions when you stop
livesub file lecture.mp4                      # subtitle a recording: writes lecture.srt
livesub devices
```

Drag the caption bar anywhere. Double-click it or press Esc to stop.

### Which model

Live, the recognizer re-reads the current sentence as it grows, so the time one reading takes is how
far the captions run behind. Here is the time to read one 8-second window on a laptop CPU (int8, an
otherwise idle machine). A busy machine can take twice as long.

| model | size | 8 s window on a CPU | live on a CPU |
|---|---|---|---|
| `tiny` | 75 MB | 0.33 s | quick, least accurate |
| `base` | 145 MB | 0.60 s | about 1 s behind (**default on a CPU**) |
| `small` | 480 MB | 1.7 s | about 2 s behind (**default on a GPU and for files**) |
| `medium`, `large-v3`, `turbo` | 1.5–3 GB | — | GPU only |

How accurate they are on test speech made with text-to-speech (WER, word errors; CER, character errors
for Chinese). That is easier than real speech, so treat these numbers as a floor:

| | clean | fast | 60 s talk | noise at 10 dB SNR | music bed | Chinese | Chinese + English |
|---|---|---|---|---|---|---|---|
| `base`, live | 2.1% | 0% | 1.9% | 5.3% | 4.3% | 10.3% | 13.3% |
| `base`, live, `--beam 1` | 3.2% | 0% | 3.8% | 5.3% | 3.2% | 13.8% | 30% |
| `small`, `livesub file` | 1.1% | 0% | 1.3% | 2.1% | 1.1% | 1.1% | 10% |

- **Files.** `livesub file` reads each stretch of speech once. With `small` on a CPU, a recording takes
  about a fifth of its length. `--live-loop` runs it through the live loop instead, as `livesub live`
  would hear it, which is slower.
- **Chinese.** `base` is weak on Chinese. Use `--model small` if a GPU is available, or accept the lag.
  livesub steers Whisper towards Simplified characters, which it otherwise often mixes with
  Traditional ones.
- **On a slow CPU** `--beam 1` roughly halves the work, at some cost in accuracy, much more in Chinese.
- Pass `--language` when you know the language: it skips detection and avoids mistakes on short
  phrases.

**Offline.** Once a model has been downloaded, livesub starts without contacting anyone. **macOS** has no
system loopback: send the sound to a virtual device such as [BlackHole](https://github.com/ExistentialAudio/BlackHole)
and use `--source mic --input-device BlackHole`.

## How it works

```
audio ─▶ resample to 16 kHz ─▶ voice activity ─▶ speech buffer ─(every second)─▶ Whisper
                                                                                    │
     overlay ◀── tentative words ◀── local agreement ◀───────────── words + times ◀─┘
        ▲                                 │ committed words
        └── translation ◀── cues ◀── line breaking (2 lines, CJK-aware)
```

- **Voice activity.** An energy detector with an adaptive estimate of the background decides *when* to
  run Whisper, so silence costs nothing. It is simple and needs no model, and it can be fooled: music
  or steady noise counts as voice until, after 15 s without a pause, the background is re-estimated.
  Whisper then hears no words, and its invented text in such stretches ("Thanks for watching!",
  repeated phrases) is filtered out. Live readings decode once (no temperature fallback), so a bad
  window costs one decode and cannot stall the captions.
- **Local agreement** ([Macháček et al., 2023](https://arxiv.org/abs/2307.14743)). The recognizer re-reads
  the growing buffer, and the words that two consecutive readings agree on are committed. New words
  are lined up with the committed text by their *words*, not their timestamps: Whisper's times shift
  between readings, and in a real run, aligning by time silently dropped words. A buffer longer than
  12 s is cut at the end of the last committed sentence (or, without one, after the last committed
  word), so recognition time stays bounded however long someone talks.
- **Cues**, following [Netflix's timed-text style guides](https://partnerhelp.netflixstudios.com/hc/en-us/articles/215758617-Timed-Text-Style-Guide-General-Requirements).
  - A cue closes at the end of a sentence, at a pause, or when its two lines are full. In that last
    case it ends at a comma when it can, so a short tail isn't left on its own.
  - Lines hold 42 characters, or 16 in Chinese and Japanese. Two lines are broken where they read
    best: after punctuation, and about equally long, rather than by filling the first line.
  - Latin words are never split. Closing punctuation never starts a line, and an opening bracket
    never ends one.
  - In subtitle files, a cue stays up long enough to read (20 characters a second, 9 in Chinese),
    between 5/6 s and 7 s, with two frames between cues.
  - Bilingual files give each language up to two lines of its own, so a cue can have four.

## Develop

```bash
uv venv && uv pip install numpy pytest
uv run pytest
python scripts/mutate.py      # test the tests: 57 planted bugs, each must make a test fail
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for how the tests work and how to check against real Whisper
without recording anyone.

## License

[MIT](LICENSE)
