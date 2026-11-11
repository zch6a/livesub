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
- **At that point not yet verified**: real Whisper and live capture. The overlay was smoke-tested: it shows committed, tentative and translated text and closes cleanly.

### Development, part 2: the first real runs

- **Mutation testing** (`scripts/mutate.py`) refuses to start unless the unmutated suite passes. The first run caught 23 of 27 mutants. The 4 survivors were real gaps, now covered by tests:
  - the noise floor never falling (livesub started during something loud);
  - the floor creeping up during speech (speech over a music bed never dips to the background);
  - the absolute level minimum (a faint hiss after muted output);
  - the 7-second cue limit (slow speech).
- **Real Whisper** (`small`, CPU int8) on English speech generated with Windows text-to-speech, so no one's voice was recorded. The first run **lost words mid-sentence**: "harder than", "keep", "the end of", "screen is", "as". Logging every reading found two causes.
  1. Committed words still in the window were also in the prompt. Whisper then skipped them and returned only the continuation, with its first word smeared 0.7 s early, before the last committed word. As in whisper_streaming, the prompt now carries only text whose audio has already been cut from the window.
  2. Agreement decided which words were new by their timestamps, which shift between readings. New words are now lined up with the committed text by their words: after the longest committed tail they repeat, or else where the previous guess began, with time only as a last resort.
  - The failing sequence of readings is a regression test: the old code loses "harder than" on it, the new code keeps every word. The scripted recognizer now also jitters times and stretches the last word, as Whisper does. Two guesses at the cause that were wrong (an inflated last end time; mild jitter) are recorded here and in the tests' comments, not in the code.
  - Rerun on the whole recording: every word of the script is right.
- **Cue breaks**: a full cue now ends at its last comma in its second part rather than orphaning the tail. From the real run: "…harder than" followed by a cue of just "they look." became "…live captions," followed by "and why they are harder than they look.".
- **Speed**, measured on an 8 s window on a laptop CPU, int8: tiny 0.33 s, base 0.60 s, small 1.72 s. Small is too slow to keep up live with a 1 s step: captions would lag about 2 s. The default model is now small on a usable GPU and for files, and base for live captions on a CPU.
- **GPU**: with the default `--device auto`, CTranslate2 picked the GPU on this machine and then crashed on the first sentence, because cuBLAS was missing. A probe now checks cuBLAS and cuDNN, including copies installed as NVIDIA pip wheels. Auto falls back to the CPU with a note on what to install; `--device cuda` stops with that same note.
- **Chinese** (Microsoft Huihui text-to-speech): correct apart from one misheard word ("实时" heard as "时时"). Whisper first wrote Traditional characters; a Simplified Chinese prompt, used when the language is Chinese, fixed that. The first Chinese file was pure silence: a PowerShell variable named `$s` overwrote `$S`. livesub correctly produced no captions for it.
- 74 tests, 32 of 32 mutants caught. CI workflow (Ubuntu, Windows and macOS × Python 3.10 and 3.13), README in English and Chinese.
- **Still not verified**: live capture with `soundcard`, since that would record whatever this machine is playing; and the GPU path, since the CUDA libraries are not installed here.
