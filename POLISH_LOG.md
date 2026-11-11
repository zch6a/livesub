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

## Round 3

Checked against the literature and comparable projects: an agent read the Macháček et al. paper and the whisper_streaming code, WhisperLive and whisper.cpp's stream example, faster-whisper's defaults, and Netflix's timed-text guides (English and Simplified Chinese). It found 12 issues, and all were acted on.

**Wrong or misleading**
- **Chinese lines held 21 characters**, while the module's own docstring said 16. The width was 42 columns for every script. Lines are now 16 characters in Chinese and Japanese (32 columns) and 42 in other scripts, chosen by the text itself.
- **Temperature fallback could stall live captions.** faster-whisper re-decodes a window up to 6 times when the result looks bad, and the README said a reading's time is the lag. Live readings now decode once (`temperature=0.0`); files keep the fallback. Measured on the live loop with `base`: Chinese went from 2.25× to 1.49× real time, and its error rate fell from 10.3% to 6.9% CER.
- **"Whisper is never asked to transcribe silence" was overstated.** Music and steady noise count as voice until the 15 s re-estimate. Both READMEs now say so. (Using faster-whisper's bundled Silero VAD as the gate is left as a possible improvement.)
- **My `no_speech_prob > 0.6 and avg_logprob < -1` filter did nothing**: faster-whisper already applies exactly that rule. It is replaced by one that adds something: drop a segment Whisper is over 90% sure is not speech (as whisper_streaming does).
- **Long file regions were cut every 30 s**, which can split a word. faster-whisper moves its own 30 s window along by timestamps, so regions now go to it whole. Only a region longer than 2 minutes is split, at its quietest frame; a test checks no word is cut.
- **"Turns white only once two readings agree" has exceptions**: end of speech, and a forced trim. The README now says so.

**Could be better, now done**
- **A single-word match away from the head of a guess dropped words.** Committed "… sat on the", then a guess holding only the continuation ("mat and the dog"): the later "the" matched the tail, and "mat and" was lost. A one-word match now counts only at the head, as in whisper_streaming; longer matches may be anywhere near. The old code fails the new test.
- **Sentence trimming.** A full buffer is now cut at the end of the last committed sentence, so the next window starts with a sentence, falling back to the last committed word. A test runs a 40 s talk with an 8 s window and checks that every trimmed window starts at a sentence.
- **Loops of phrases**: "Thank you. Thank you. Thank you." is now caught, at three or more repeats of a 2–4 word phrase. A single Chinese character may repeat up to five times (对对对对, 哈哈哈哈哈), while Whisper's own loops are longer.
- **Line breaking**: two lines are now broken where they read best (after punctuation; about equal, the top a little shorter), not filled greedily. Closing quotes ” ’ joined the "never starts a line" set, and opening brackets and quotes never end a line.
- **Display timing in files**:
  - long enough to read: 20 characters a second, 9 in Chinese;
  - between 5/6 s and 7 s on screen;
  - two frames before the next cue, with gaps under half a second closed.
- **Bilingual files** can have four lines (two per language); the docstring and the README now say so.

**Also this round**
- **Missing punctuation on fast speech** (round 2's known limitation) came from livesub, not the model. With 0.3 s of lead-in before a region, Whisper wrote the fast clip with no punctuation at all; with 1 s it punctuated normally. Regions now get 1 s before and 0.5 s after. They are joined before padding, so the padding never takes in a neighbouring region's audio.
- **Re-scored with real Whisper.** Live with `base`: music 3.2%, Chinese 6.9%, the rest unchanged. Files with `small`: clean English and music now 0%, Chinese 1.1%. The README tables are updated.
- Every change has a test. Several fail on the old code: the single-word match, the opening bracket on a three-line cue, and sentence trimming (checked with mutants).
- 111 tests, 57 of 57 mutants caught, ruff clean.

## Round 4

Performance and a full read of the code.

- **An hour of speech** (synthetic, with the scripted recognizer) through the live loop:
  - every word came out once and in order;
  - the loop itself costs about 3 ms per second of audio;
  - memory grows only by the list of finished cues (0.2 MB per 10 minutes, kept for `--save`); the
    committed words kept for alignment stay at 50.
- **Real lag, measured.** A minute of English and the Chinese recording were played in real time through the same capture thread, queue and deferral as `livesub live`, with real Whisper. For each word I measured how long after it is said it first appears and when its cue is final. The README's "about 1 s behind" for `base` was wrong: words appeared 2.5 s after being said (median) and cues were final after 3.8 s, with single readings of up to 4.3 s (6.0 s in Chinese).
  - **Fixed: the slow readings.** Logging them showed about 2.5 s whatever the window's length, even on a 1.5 s window, and nearly all ended on a word cut in half. The decoder ran on without finding the end of the text. Capping new tokens at 8 a second of audio (plus 16) removed the spikes: the slowest reading dropped from 3.7 s to 1.4 s. Accuracy on every test recording is unchanged, and live mode now runs faster than real time on all of them (Chinese 1.49× → 0.88×).
  - **A warm-up reading at start**, so the first real reading does not pay for setting up.
  - Tried and **not adopted**: a 0.5 s step (words appear a little sooner, but are final later) and an 8 s buffer (worse on both counts, since shorter context means more re-reading).
  - Now, with `base` on a laptop CPU (median): a word appears 1.2 s after it is said and is final at 1.8 s; in Chinese, 1.6 s and 2.6 s. With `tiny`: 0.7 s and 1.1 s. With `small`: 2.6 s and 3.7 s. The README tables show these measured figures instead of guesses.
- **Code read**:
  - `cmd_live` had grown to 103 statements, with complexity 37: display, bookkeeping and threading all in one closure-heavy function. It is now `live.py`: a `ConsoleView` and an `OverlayView` with the same four methods, a thread-safe `Transcript`, and a `LiveSession` that owns the two threads. `cmd_live` only checks the setup and starts the session.
  - Ruff's complexity rules (C901, PLR0912, PLR0915) pass on all of `src`.
- **Tests**: live translation end to end. A slow fake translator (0.6 s a reply) checks that the overlay shows translations, and that the saved file has every cue translated even when the last translation arrives after stopping. Without that slowness, the mutant that skips waiting for translations had survived. The token cap is checked too.
- 112 tests, 60 of 60 mutants caught, ruff clean.

## Round 5

A final blind review: a fresh agent with none of the earlier rounds' conclusions read everything, ran the tests and the CLI, and made its own recordings with text-to-speech. It found 14 problems, the worst of them critical. All are fixed, each with a test; 14 of the 15 new tests fail on the code as it was before this round.

- **Critical: `livesub file` crashed on any real lecture and wrote nothing.** Round 4's token cap allowed up to 440 new tokens, and every region after the first passes the earlier text as its prompt. faster-whisper refuses a prompt plus cap over 448 tokens. My tests used a fake model with no such limit, so they could not see it. Now:
  - the cap is computed per 30 s chunk, minus the prompt's actual token count (as faster-whisper cuts it, to 223 tokens, plus 4 task tokens);
  - the fake model enforces the 448 rule;
  - a test runs the real faster-whisper (`tiny`, where it is installed) on 75 s of continuous speech in English and Chinese, with a long Chinese prompt;
  - the reviewer's 68 s English lecture now gives 18 captions.
- **Chinese turned Traditional after the first 30 s.** faster-whisper applies a prompt only to the first window of a call, and the Simplified steer was skipped whenever the prompt already held Chinese, which could itself be Traditional. Files are read in regions of at most 30 s again, split at the quietest moment in their last fifth so no word is cut, and the steer is always given for Chinese. The reviewer's 78 s Chinese lecture now has no Traditional characters, and its invented ending "字幕製作曲" is gone.
- **Ctrl+C twice lost the whole `--save` transcript.** Only the first press was handled. The first now prints what is happening, a second skips the last reading but still saves, and `main` turns a stray interrupt into exit code 130 instead of a traceback.
- **A slow or unreachable translator stalled shutdown**: 30 s per queued line, and the queue had no limit. Now:
  - a line that waits behind three newer ones is passed on untranslated;
  - live requests time out after 10 s;
  - the shutdown wait is capped at 15 s;
  - errors are reported once, with a count of untranslated lines at the end;
  - the endpoint is checked with a bare TCP connection (nothing sent) before the model loads.
- **Any failure loading a cached model started a download**, with a wrong message: a bad `--compute-type` reached the network. Only `FileNotFoundError`, which is what "not in the cache" raises, leads to a download now.
- **`--llm-key` was ignored** with `LIVESUB_LLM_BASE_URL`. Fixed. `--llm-*` options without `--translate-to`, and `--subs translation` without a translator, are now errors instead of being silently ignored.
- **A cue holding only "."** appeared in a real run. A punctuation-only word can no longer start a cue.
- **`file -o` was unchecked, and the default `movie.srt` silently replaced subtitles already next to `movie.mkv`.** Output paths for `file` and `live --save` are checked before any work: suffix, folder, and an existing file, which needs `--overwrite`.
- **Console mode died on a terminal that cannot show Chinese** (cp1252 redirect). Unprintable characters now become "?".
- **`--step 0.1` loaded the model before failing.** `--step` is now range-checked by argparse, and the display is checked before the model loads. The overlay window opens only after the model is ready, so it is not frozen during the load.
- **Subtitle credits Whisper learned from its training data** ("Subtitles by the Amara.org community", "字幕由…提供") are filtered out. The README and CHANGELOG no longer claim more than this.
- **Docs**:
  - the translation environment variables and `--overwrite` are now in the README;
  - the test count is corrected;
  - `uv run python scripts/mutate.py` replaces `python scripts/mutate.py`;
  - the runner's docstring gives the current padding;
  - the CHANGELOG is brought up to date.

**Publishing.** The other five projects went to GitHub this round, which brought its own findings:
- GitHub blocked llm-redact's push over a Stripe-shaped demo key, so the demo now uses an obviously made-up JWT, rewritten through history.
- agent-blackbox's CI found a real quadratic regex on its first run, fixed in 1.0.1.

livesub goes up the same way: noreply commit email, `main`, public.

- 125 tests, ruff clean. The real-library test runs where faster-whisper and the `tiny` model are present; CI, without them, skips it.
