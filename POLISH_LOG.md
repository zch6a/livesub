# Polish log

Each round goes through the whole project: code, tests, docs, and actually running it.

## Development

- **Scope**: live captions for system audio or a microphone, offline recognition with faster-whisper,
  optional translation through any OpenAI-compatible endpoint (a local model keeps it offline), a
  floating caption overlay, and SRT/WebVTT export of recordings through the same live loop.
- **Pipeline**:
  - a streaming resampler (windowed-sinc low-pass, then interpolation; block-by-block output equals one-shot output);
  - an energy VAD with an adaptive noise floor and hysteresis;
  - a speech buffer re-read every `step` seconds;
  - local agreement, which commits only the words two readings agree on and drops the committed tail when a new window repeats it;
  - a cue builder (sentence end, pause, two full lines, 7 s);
  - CJK-aware line breaking with kinsoku;
  - display timing (at least 1 s on screen, no overlaps);
  - a translation worker thread with context lines, a cache, and cleanup of `<think>` blocks and quotes.
- **Tests (59)** use synthetic audio: noise bursts shaped like syllables over background noise, and a scripted recognizer that returns a wrong, truncated guess for any word cut off at the window's end, as Whisper does.
  - The pipeline test checks that every word appears exactly once and in order, that no wrong guess survives, and that cues line up with the speech.
  - Other pipeline checks: silence never reaches the recognizer, the window stays bounded through 48 s of unbroken speech, tentative words are shown but never as committed, the prompt carries the committed text, and Chinese captions work.
  - The translator is tested against a local fake endpoint, including failure, empty-reply and cache cases.
  - The CLI is tested end to end on a WAV file, with and without translation.
- **Found while testing**: a WAV file shorter than its header made `wave` raise `EOFError` rather than its own error, so the friendly "convert it with ffmpeg" message was skipped. Both are caught now.
- **Not yet verified**: real Whisper (the model is not downloaded yet), and live capture with `soundcard` (not installed). The overlay was smoke-tested: it shows committed, tentative and translated text and closes cleanly.
