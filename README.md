# livesub

Live captions for anything your computer plays or hears — a video call, a lecture stream, a game, a
podcast — recognized on your own machine with Whisper, optionally translated, in a caption bar that
floats over every window.

- **Offline.** Speech recognition runs locally ([faster-whisper](https://github.com/SYSTRAN/faster-whisper),
  CPU or CUDA). Translation goes to an OpenAI-compatible endpoint *you* choose; point it at a local
  model (Ollama, LM Studio, llama.cpp) and nothing leaves your computer.
- **Captions that do not flicker.** Each word is shown dimmed while it is being recognized and turns
  white only when two readings in a row agree on it (*local agreement*); committed words never change.
- **System-wide.** Captures what the speakers play (WASAPI loopback on Windows, monitor sources on
  Linux) or a microphone.
- **Subtitles too.** `livesub file` runs a recording through the same live loop and writes SRT or WebVTT,
  original, translated, or both.

> Status: 0.1, in development. The pipeline is tested end to end with synthetic audio and a scripted
> recognizer; see [POLISH_LOG.md](POLISH_LOG.md).

## Install

```bash
pip install "livesub[all] @ git+https://github.com/zch6a/livesub"
```

`[all]` adds faster-whisper and soundcard. The Whisper model (`small` by default, about 0.5 GB) is
downloaded once, on first use.

## Use

```bash
livesub live                                  # caption what the computer plays, in an overlay
livesub live --source mic --console           # a microphone, in the terminal
livesub live --language en --translate-to zh --llm-url http://localhost:11434/v1 --llm-model qwen2.5:7b
livesub live --save meeting.srt               # keep the captions when you stop
livesub file talk.wav -o talk.srt             # subtitle a recording
livesub devices
```

Drag the caption bar anywhere; double-click it or press Esc to stop. `--model` picks the Whisper size
(`tiny` … `large-v3`, `turbo`): larger is more accurate and slower. `--step` sets how often the
recognizer re-reads what is being said (default every second).

## How it works

```
audio ─▶ resample to 16 kHz ─▶ voice activity ─▶ speech buffer ─(every second)─▶ Whisper
                                                                                    │
     overlay ◀── tentative words ◀── local agreement ◀───────────── words + times ◀─┘
        ▲                                 │ committed words
        └── translation ◀── cues ◀── line breaking (2 lines, CJK-aware)
```

- **Voice activity**: an energy detector with an adaptive noise floor decides *when* to run Whisper,
  so silence costs nothing and Whisper is not asked to transcribe it (which is when it invents text).
- **Local agreement** (Macháček et al., 2023): the recognizer re-reads the growing buffer; the words
  two readings agree on, from the start, are committed. A buffer that grows past 12 s is cut after the
  last committed word, so recognition time stays bounded however long someone talks.
- **Cues** close at a sentence end, a pause, or when two lines are full; lines break between Latin
  words and anywhere in Chinese or Japanese, but never before closing punctuation.

## Develop

```bash
uv venv && uv pip install numpy pytest
uv run pytest
```

## License

[MIT](LICENSE)
