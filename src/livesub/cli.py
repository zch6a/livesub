"""livesub command line.

    livesub live                       captions for whatever the computer is playing, in an overlay
    livesub live --source mic --console
    livesub file talk.wav -o talk.srt  subtitle a recording
    livesub devices
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import __version__
from .audio import RATE, read_audio, resample
from .captions import Cue, to_srt, to_vtt
from .pipeline import LiveCaptioner, run_file, transcribe_file
from .translate import ChatTranslator, TranslationError


def make_transcriber(args):
    from .asr import FasterWhisper, resolve_device

    device = resolve_device(args.device)
    model = args.model
    if model is None:
        # Live, the model re-reads the sentence as it grows: on a laptop CPU "small" takes about 1.7 s per
        # 8 s window, "base" 0.6 s. A file is read once per stretch of speech, so "small" is fast enough.
        model = "small" if device == "cuda" or args.command == "file" else "base"
    print(f"loading Whisper '{model}' on the {device.upper()} (downloaded once, on first use)...", file=sys.stderr)
    # Live, no temperature fallback: one decode per reading, so a bad window cannot stall the captions.
    temperature = 0.0 if args.command == "live" else (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    return FasterWhisper(model, device=device, compute_type=args.compute_type, beam_size=args.beam, temperature=temperature)


def make_translator(args):
    if not args.translate_to:
        return None
    if args.llm_url:
        model = args.llm_model or os.environ.get("LIVESUB_LLM_MODEL", "qwen2.5:7b")
        return ChatTranslator(args.llm_url, model, args.llm_key or os.environ.get("LIVESUB_LLM_API_KEY"))
    t = ChatTranslator.from_env()
    if t is None:
        raise SystemExit("--translate-to needs a translation model: start a local one (e.g. `ollama serve`) and pass "
                         "--llm-url http://localhost:11434/v1 --llm-model qwen2.5:7b, or set LIVESUB_LLM_BASE_URL")
    if args.llm_model:
        t.model = args.llm_model
    return t


def _int_range(lo: int, hi: int):
    def parse(text: str) -> int:
        try:
            v = int(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{text!r} is not a whole number") from None
        if not lo <= v <= hi:
            raise argparse.ArgumentTypeError(f"must be {lo} to {hi}")
        return v

    return parse


def _common(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("recognition")
    g.add_argument("--model", default=None, help="Whisper model: tiny, base, small, medium, large-v3, turbo, distil-large-v3 "
                   "(default: small, or base for live captions on a CPU)")
    g.add_argument("--device", default="auto", help="auto, cpu or cuda")
    g.add_argument("--compute-type", default="default", help="e.g. int8 on CPU, float16 on a GPU")
    g.add_argument("--beam", type=_int_range(1, 10), default=5, help="beam size: 1 is about twice as fast on a CPU, and less accurate (default: 5)")
    g.add_argument("--language", help="spoken language code (en, zh, ja, ...); default: detect")
    g.add_argument("--step", type=float, default=1.0, help="seconds between re-readings (lower = quicker captions, more work)")
    t = p.add_argument_group("translation (optional)")
    t.add_argument("--translate-to", metavar="LANG", help="translate captions into this language, e.g. zh, en, ja")
    t.add_argument("--llm-url", help="OpenAI-compatible endpoint, e.g. http://localhost:11434/v1 for Ollama")
    t.add_argument("--llm-model", help="model name at that endpoint")
    t.add_argument("--llm-key", help="API key, if the endpoint needs one; better set LIVESUB_LLM_API_KEY, which stays out of shell history")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="livesub", description="Offline live captions, with optional translation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="examples:\n"
               "  livesub live                          captions for what this computer plays\n"
               "  livesub live --source mic --console   a microphone, in the terminal\n"
               "  livesub file lecture.mp4              write lecture.srt\n"
               "  livesub live --translate-to zh --llm-url http://localhost:11434/v1 --llm-model qwen2.5:7b\n"
               "\nrun `livesub live --help` or `livesub file --help` for all options")
    p.add_argument("--version", action="version", version=f"livesub {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    live = sub.add_parser("live", help="caption live audio")
    live.add_argument("--source", choices=["loopback", "mic"], default="loopback", help="what the computer plays (default) or a microphone")
    live.add_argument("--input-device", help="part of a device name (see `livesub devices`)")
    live.add_argument("--console", action="store_true", help="print captions in the terminal instead of the overlay")
    live.add_argument("--save", type=Path, help="also write the captions to this .srt or .vtt file when you stop")
    live.add_argument("--font-size", type=int, default=22)
    _common(live)

    f = sub.add_parser("file", help="subtitle a recording (WAV, or any audio/video format once faster-whisper is installed)")
    f.add_argument("input", type=Path)
    f.add_argument("-o", "--output", type=Path, help="output .srt or .vtt (default: next to the input, .srt)")
    f.add_argument("--subs", choices=["original", "translation", "both"], default=None,
                   help="what the subtitle file shows (default: both when translating)")
    f.add_argument("--live-loop", action="store_true", help="run the recording through the live loop, as `livesub live` would hear it (slower)")
    _common(f)

    sub.add_parser("devices", help="list audio devices")
    return p


def write_subtitles(path: Path, cues: list[Cue], mode: str) -> None:
    text = to_vtt(cues, mode) if path.suffix.lower() == ".vtt" else to_srt(cues, mode)
    path.write_text(text, encoding="utf-8")


def translate_all(translator, cues: list[Cue], target: str) -> list[Cue]:
    from dataclasses import replace

    out, context, failures = [], [], 0
    for c in cues:
        try:
            out.append(replace(c, translation=translator.translate(c.text, target, context[-2:])))
        except TranslationError as e:
            failures += 1
            if failures == 1:
                print(f"warning: {e}", file=sys.stderr)
            out.append(c)
        context.append(c.text)
    return out


def cmd_file(args) -> int:
    audio, rate = read_audio(args.input)
    audio = resample(audio, rate, RATE)
    translator = make_translator(args)
    transcriber = make_transcriber(args)
    print(f"{args.input}: {len(audio) / RATE:.1f} s of audio", file=sys.stderr)
    show = lambda c: print(f"[{c.start:7.2f}] {c.text}", file=sys.stderr)  # noqa: E731
    if args.live_loop:
        cues = run_file(LiveCaptioner(transcriber, step=args.step, language=args.language, on_cue=show), audio)
    else:
        cues = transcribe_file(transcriber, audio, language=args.language, on_cue=show)
    if translator:
        cues = translate_all(translator, cues, args.translate_to)
    out = args.output or args.input.with_suffix(".srt")
    mode = args.subs or ("both" if translator else "original")
    write_subtitles(out, cues, mode)
    print(f"{len(cues)} captions -> {out}", file=sys.stderr)
    return 0


def check_save_path(path: Path | None) -> None:
    """Fail now, not after an hour of captions, if they could not be saved."""
    if path is None:
        return
    if path.suffix.lower() not in (".srt", ".vtt"):
        raise ValueError(f"--save {path}: the file must end in .srt or .vtt")
    if not path.resolve().parent.is_dir():
        raise ValueError(f"--save {path}: the folder {path.resolve().parent} does not exist")


def cmd_live(args, capture=None, overlay_class=None) -> int:
    from .live import ConsoleView, LiveSession, OverlayView

    if capture is None:
        from . import capture
    # Everything that can be wrong is found before the model loads: the save path, the translator,
    # the sound device, the display.
    check_save_path(args.save)
    translator = make_translator(args)
    mic = capture.open_source(args.source, args.input_device)
    if args.console:
        view = ConsoleView()
    else:
        if overlay_class is None:
            from .overlay import Overlay as overlay_class
        view = OverlayView(overlay_class(font_size=args.font_size))
    transcriber = make_transcriber(args)
    if hasattr(transcriber, "warm_up"):
        transcriber.warm_up(args.language)
    session = LiveSession(capture.blocks(args.source, args.input_device, mic=mic), transcriber, view,
                          step=args.step, language=args.language, translator=translator, target=args.translate_to)
    src = "what this computer plays" if args.source == "loopback" else "the microphone"
    how = "Ctrl+C" if args.console else "double-click the captions or press Esc"
    print(f"listening to {src}; {how} to stop", file=sys.stderr)
    session.run()
    cues = session.transcript.snapshot()
    if args.save:
        write_subtitles(args.save, cues, "both" if translator else "original")
        print(f"\n{len(cues)} captions -> {args.save}", file=sys.stderr)
    if session.failure:
        print(f"\nlivesub: {session.failure}", file=sys.stderr)
        return 1
    return 0


def cmd_devices(_args) -> int:
    from .capture import list_devices

    for kind, names in list_devices().items():
        print(f"{kind}:")
        for n in names:
            print(f"  {n}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return {"live": cmd_live, "file": cmd_file, "devices": cmd_devices}[args.command](args)
    except (RuntimeError, ValueError, OSError) as e:
        print(f"livesub: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
