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

## Round 1

- **Measured on harder speech** (all generated with Windows text-to-speech; scored by word or character error rate against the script):
  - fast speech, a 60-second talk;
  - white noise 10 dB below the speech;
  - a chord "music bed" under the speech and the pauses;
  - Chinese, and Chinese mixed with English.

  The first run found three problems:
  - **Noise.** With noise at 10 dB SNR, quiet syllables fell below the single 9 dB threshold, speech kept stopping mid-sentence, and Whisper got 3-second scraps: 20% WER. The VAD now uses two thresholds: 9 dB to start, 4.5 dB to go on. While speech lasts, the background estimate does not rise. Noise is now 5.3% WER, music 4.3% (was 9.6%), Chinese 10.3% CER (was 14.9%).
  - **Loops.** In a pause, Whisper wrote "看,看,看,看,看,看,看". Segments it rates as highly repetitive (compression ratio above 2.4) are dropped, and so is any run of four or more identical words. Three in a row ("no, no, no") is kept.
  - **Speed.** `base` ran at 2.3× real time on Chinese in the live loop (see the live-mode fix below).
- **A blind review** (a fresh agent with no history) reported 17 findings. All are fixed, each with a test that fails without the fix:
  1. *Live capture lost audio while Whisper ran.* Reading the device and recognizing shared one thread, with a 0.1 s device buffer. Capture now has its own thread, a queue, and a 1 s buffer. When readings fall behind, the recognizer takes in the whole backlog and reads once. The test uses a recognizer slower than the step and checks that no mid-sentence reading leaves out more than one block.
  2. *The README's speeds.* The reviewer measured about twice my numbers, but its measurements ran while my scoring run was also using the CPU. The README now says the figures are from an idle machine and that a busy one can take twice as long. It also explains that a slow reading adds delay rather than piling up.
  3. *Stuck "speaking" after digital silence.* Exact zeros taught the background estimate −200 dB, and after that any hum counted as speech. Zeros are now ignored, and the first background estimate is capped at −45 dB. After 15 s of "speech" without a pause, the background is re-estimated from the 10th percentile of the last 5 s, so a steady sound stops counting.
  4. *A new sentence lost its first word when it repeated the previous sentence's last* ("Thank you." / "[You] know"). The text alignment now matches only committed words whose audio is still in the window.
  5. *A recognition failure hung live mode.* `close()` ran Whisper again and failed again. It now stops with "recognition failed: …".
  6. *A traceback on every overlay close.* Tk was touched from the audio thread. `Overlay.close()` is now safe from any thread and safe to call more than once.
  7. *A bad `--save` path lost the whole session at the end.* It is checked before the model loads, and so are the capture device and the display.
  8. *A recording starting mid-word lost its first word.* Fixed by the capped first background estimate (item 3).
  9. *`livesub file` was about 9× slower than needed.* It re-read every window each second. It now finds the speech regions (joined across short gaps, at most 30 s each) and reads each once: 0.13–0.21× real time with `small` on a CPU, against about 2.5× before. It is also the most accurate configuration: Chinese 1.1% CER, noise 2.1% WER. `--live-loop` keeps the old behaviour.
  10. *"Offline" was not true at startup.* faster-whisper asked Hugging Face on every start. The model now loads from the cache first, and the network is used only to download it the first time.
  11. *`--llm-url` ignored `LIVESUB_LLM_API_KEY` and `LIVESUB_LLM_MODEL`.* It now uses them. The help text recommends the environment variable over `--llm-key`, which ends up in shell history.
  12. *macOS loopback failed cryptically after the model loaded.* It now fails at once and points to BlackHole; the README says so too.
  13. *No tkinter or no display gave a traceback.* It is now a clear message suggesting `--console`; `main` also catches `OSError`.
  14. *Shutdown could drop the last caption's translation.* Now the threads finish first, then the translation worker drains completely, then the file is saved.
  15. *An `http_proxy` setting sent captions for a local model through the proxy.* Loopback hosts are now reached directly. The test points every proxy variable at a dead port: the old code failed there, and the new code succeeds.
  16. *Untested paths.* There are now live-mode tests end to end, with a fake capture device and overlay. They cover recognition and capture failures, a bad save path and a slow recognizer. `FasterWhisper.transcribe` is tested against a fake model: the offset, both hallucination filters, and the Simplified-Chinese prompt. The VAD has new tests for every case above.
  17. *CI never installed the package.* CI now installs it, runs `livesub --version` and `livesub file --help` and ruff, and builds the wheel and sdist. Locally, the wheel contains exactly the package and its license.
- **Beam size**, measured live with `base`: greedy decoding (`--beam 1`, new option) is about 1.5–2× faster. It is a little less accurate in English and much less in Chinese (mixed language 30% CER vs 13%). The default stays 5.
- **Mutation testing**: 46 mutants. The first run of the new set left 7 alive, each a real test gap, all now covered. Example: no test had a stretch of soft speech long enough for a rising background estimate to matter. One mutant is equivalent (the speech-region start, which the 0.3 s padding covers) and was removed, with a note in the runner.
- **Engineering**: ruff (E, F, W, B, UP, SIM, I) is clean. A `.gitattributes` keeps LF line endings: a `git stash pop` had rewritten files with CRLF, which broke multi-line edits. CONTRIBUTING (how the tests work; checking against real Whisper without recording anyone) and CHANGELOG were added.
- 99 tests, 46 of 46 mutants caught.
- **Still not verified**: capture from a real sound card (it would record whatever this machine plays), and the GPU path (CUDA libraries not installed here).

## Round 2

A new user's first run: a fresh clone into an empty folder, a new virtual environment, `pip install -e ".[all]"` (23 s), and then every command in the README.

- **Recordings other than WAV were refused.** That is the first thing most people try (`livesub file lecture.mp4`), and it failed with "not a PCM WAV file". faster-whisper already installs a decoder (PyAV), so `livesub file` now reads MP3, M4A, MP4, MKV and the rest directly, resampled to 16 kHz mono. Checked with an AAC file made from the test speech, which transcribed correctly. Without PyAV the old message remains, now suggesting `livesub[whisper]` before ffmpeg. A test builds an M4A file and checks its pitch and length, an unreadable file and a missing one.
- **`livesub --help` showed no examples.** It now lists four, and the README's file example uses `lecture.mp4`.
- **`--beam 0` was checked only after the input file had been read.** argparse now rejects it at once ("must be 1 to 10").
- **Checked and fine:**
  - the early errors: wrong device, bad `--save` name, `--translate-to` without a model. Each arrives in about a second, before the model loads, and says what to do;
  - `livesub devices`;
  - a missing input file;
  - the offline claim: with every proxy variable pointing at a dead port, `livesub file` started from the cached model in 7.7 s and never tried the network.
- **A false alarm, recorded so it is not chased again**: a wrong `--input-device` seemed to exit silently. The message was there; my own output filter had dropped it.
- **Known limitation**: on fast speech, Whisper `small` sometimes writes no punctuation at all. Every word is right, but cues then break only when they are full, sometimes mid-phrase. This comes from the model, not from livesub; it is not worked around.
- **Not testable here**: translation with a real model (no Ollama, LM Studio or llama.cpp server on this machine; only the fake endpoint in the tests), live capture, and the GPU. The README now says all three.
- 101 tests, ruff clean.
