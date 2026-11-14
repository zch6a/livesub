# Changelog

## 0.2.0 — 2026-11-14

Fixes from a review of the whole program, each with a test that fails without it:

- Live mode kept losing the first word of a sentence after a short pause: the new sentence's first word was matched
  against the old sentence's last ("Thank you." then "you know..."). Words heard before the window count by where
  they start, and after speech ends the next reading starts a new utterance.
- One matching word is accepted only at the very start of a reading ("I said no." then "Well no, ..." lost "Well").
- One NaN or infinite sample (a broken float WAV, a driver glitch) made the voice detector deaf for the whole run.
- A WAV file cut off in the middle of a sample no longer fails; files with a zero sample rate or no channels are
  reported by name. A two-hour file is resampled ten seconds at a time (one call took about 14 GB).
- Cue display times: a cue never flashes for a millisecond when the next starts at almost the same time, and a short
  gap is closed only up to the longest time a cue is shown. WebVTT text is escaped (`&`, `<`, `-->`).
- Stopping no longer counts its own stop marker as a line waiting to be translated.
- `--save` problems (a folder, a folder that is not writable) and the translator's options are checked before a
  long recording is decoded.
- The window test runs only in CI or with `LIVESUB_GUI_TESTS=1` (it opens a window). CI installs from the lock file
  (`uv sync --locked`), with ruff pinned.

143 tests; `scripts/mutate.py` plants 76 bugs, all caught.

## 0.1.0

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
