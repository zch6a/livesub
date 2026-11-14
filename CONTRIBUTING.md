# Contributing

Thanks for helping. livesub is small on purpose: plain Python and numpy, with faster-whisper and
soundcard as optional extras. Bug reports with a short recording (or a way to make one) are the most
useful thing you can send.

## Set up

```bash
git clone https://github.com/zch6a/livesub && cd livesub
uv sync --all-extras     # the versions in uv.lock, as CI uses them
uv run pytest            # about 145 tests, no model and no audio device needed
uv run ruff check src tests scripts
uv run python scripts/mutate.py  # plants each bug in its list in turn; every one must make a test fail
```

## How the tests work

- **No real voices.** Tests render synthetic speech-like audio (`tests/synth.py`): noise bursts shaped
  like syllables, over background noise.
- **A scripted recognizer instead of Whisper.** `ScriptedTranscriber` "hears" the scripted words that fall
  inside the window. Like Whisper, it guesses a word cut off at the window's end wrongly, shifts
  timestamps between readings, and stretches the last word.
- **Real recognizer behaviour, as data.** When real Whisper does something the fake doesn't, record the
  readings (times and words) and replay them as a test. `test_replay_of_a_real_whisper_run_loses_no_words`
  is an example.
- **Live mode with fakes.** `livesub live` is tested end to end with a fake capture device and a fake
  overlay (`tests/test_live.py`). One test opens the real overlay window for a moment; it runs in CI and
  is skipped locally unless `LIVESUB_GUI_TESTS=1`, so a test run does not put a window over your screen.

## Checking against real Whisper

Never commit recordings of real people. To try the recognizer, generate speech with your operating
system's text-to-speech (Windows: `System.Speech`; macOS: `say -o out.aiff`; Linux: `espeak-ng -w out.wav`).
Convert it to 16 kHz mono WAV, and run `livesub file out.wav`.

## Pull requests

- Add a test that fails without your change.
- If you change a rule the mutation runner covers, update `scripts/mutate.py`. It refuses to start
  until the unchanged suite passes.
- Keep the README's numbers honest: say what you measured and on what.
