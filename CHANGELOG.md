# Changelog

## 0.1.0 (unreleased)

First version.

- `livesub live`: captions for what the computer plays (WASAPI loopback on Windows, monitor sources on
  Linux) or a microphone, in a floating overlay or the terminal. Optional translation, and `--save`
  to SRT or WebVTT.
- `livesub file`: subtitles for a recording, in one pass over its speech, or through the live loop
  with `--live-loop`.
- Local agreement: a word is committed only when two readings agree on it. New words are aligned to the
  committed text by their words, not their timestamps.
- A voice activity detector with two thresholds and an adaptive background estimate that ignores
  digital silence and re-learns steady sounds.
- Whisper's two well-known failures are filtered out: invented text in silence and looping on a word.
  Chinese is steered towards Simplified characters.
- Startup is offline once a model is cached. `--device auto` uses the GPU only when CUDA's libraries are
  really there, and otherwise says what to install.
- Translation through any OpenAI-compatible endpoint. A model on this machine is reached directly, even
  when a proxy is configured.
