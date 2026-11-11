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
- Whisper's best-known inventions are filtered out: text it is nearly sure is not speech, loops of a
  word or phrase, and subtitle credits from its training data. Others can get through. Chinese is
  steered towards Simplified characters in every 30 s window.
- Startup is offline once a model is cached. `--device auto` uses the GPU only when CUDA's libraries are
  really there, and otherwise says what to install.
- Translation through any OpenAI-compatible endpoint. A model on this machine is reached directly, even
  when a proxy is configured. The endpoint is checked before the model loads, and a slow translator
  falls behind by a few lines at most.
- Lag measured end to end: with `base` on a laptop CPU a word appears about 1.2 s after it is said and
  is final at 1.8 s. A cap on new tokens per reading stops the decoder running on after a word cut in
  half.
- Subtitle files follow Netflix's timing and line rules (16 characters a line in Chinese, reading
  speed, two frames between cues). `livesub file` reads any audio or video format, in regions of at
  most 30 s, and will not replace an existing file without `--overwrite`.
- Every setting that can be wrong (output path, device, display, translator, `--step`) is checked
  before the model loads. Ctrl+C twice still saves the transcript.
