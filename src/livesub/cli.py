"""livesub command line.

    livesub live                       captions for whatever the computer is playing, in an overlay
    livesub live --source mic --console
    livesub file talk.wav -o talk.srt  subtitle a recording through the same live loop
    livesub devices
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

from . import __version__
from .audio import RATE, read_wav, resample
from .captions import Cue, to_srt, to_vtt
from .pipeline import LiveCaptioner, TranslationWorker, run_file
from .translate import ChatTranslator, TranslationError


def make_transcriber(args):
    from .asr import FasterWhisper, resolve_device

    device = resolve_device(args.device)
    model = args.model
    if model is None:
        # Live, the model must read a window faster than it grows: on a laptop CPU "small" takes about
        # 1.7 s per 8 s window (captions fall 2 s behind), "base" 0.6 s. A file has no such deadline.
        model = "small" if device == "cuda" or args.command == "file" else "base"
    print(f"loading Whisper '{model}' on the {device.upper()} (downloaded once, on first use)...", file=sys.stderr)
    return FasterWhisper(model, device=device, compute_type=args.compute_type)


def make_translator(args):
    if not args.translate_to:
        return None
    if args.llm_url:
        return ChatTranslator(args.llm_url, args.llm_model or "qwen2.5:7b", args.llm_key)
    t = ChatTranslator.from_env()
    if t is None:
        raise SystemExit("--translate-to needs a translation model: start a local one (e.g. `ollama serve`) and pass "
                         "--llm-url http://localhost:11434/v1 --llm-model qwen2.5:7b, or set LIVESUB_LLM_BASE_URL")
    if args.llm_model:
        t.model = args.llm_model
    return t


def _common(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("recognition")
    g.add_argument("--model", default=None, help="Whisper model: tiny, base, small, medium, large-v3, turbo, distil-large-v3 (default: small, or base for live captions on a CPU)")
    g.add_argument("--device", default="auto", help="auto, cpu or cuda")
    g.add_argument("--compute-type", default="default", help="e.g. int8 on CPU, float16 on a GPU")
    g.add_argument("--language", help="spoken language code (en, zh, ja, ...); default: detect")
    g.add_argument("--step", type=float, default=1.0, help="seconds between re-readings (lower = quicker captions, more work)")
    t = p.add_argument_group("translation (optional)")
    t.add_argument("--translate-to", metavar="LANG", help="translate captions into this language, e.g. zh, en, ja")
    t.add_argument("--llm-url", help="OpenAI-compatible endpoint, e.g. http://localhost:11434/v1 for Ollama")
    t.add_argument("--llm-model", help="model name at that endpoint")
    t.add_argument("--llm-key", help="API key, if the endpoint needs one (or set LIVESUB_LLM_API_KEY)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="livesub", description="Offline live captions, with optional translation.")
    p.add_argument("--version", action="version", version=f"livesub {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    live = sub.add_parser("live", help="caption live audio")
    live.add_argument("--source", choices=["loopback", "mic"], default="loopback", help="what the computer plays (default) or a microphone")
    live.add_argument("--input-device", help="part of a device name (see `livesub devices`)")
    live.add_argument("--console", action="store_true", help="print captions in the terminal instead of the overlay")
    live.add_argument("--save", type=Path, help="also write the captions to this .srt or .vtt file when you stop")
    live.add_argument("--font-size", type=int, default=22)
    _common(live)

    f = sub.add_parser("file", help="subtitle a WAV recording")
    f.add_argument("input", type=Path)
    f.add_argument("-o", "--output", type=Path, help="output .srt or .vtt (default: next to the input, .srt)")
    f.add_argument("--subs", choices=["original", "translation", "both"], default=None,
                   help="what the subtitle file shows (default: both when translating)")
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
    audio, rate = read_wav(args.input)
    audio = resample(audio, rate, RATE)
    translator = make_translator(args)
    transcriber = make_transcriber(args)
    print(f"{args.input}: {len(audio) / RATE:.1f} s of audio", file=sys.stderr)
    cap = LiveCaptioner(transcriber, step=args.step, language=args.language,
                        on_cue=lambda c: print(f"[{c.start:7.2f}] {c.text}", file=sys.stderr))
    cues = run_file(cap, audio)
    if translator:
        cues = translate_all(translator, cues, args.translate_to)
    out = args.output or args.input.with_suffix(".srt")
    mode = args.subs or ("both" if translator else "original")
    write_subtitles(out, cues, mode)
    print(f"{len(cues)} captions -> {out}", file=sys.stderr)
    return 0


def cmd_live(args) -> int:
    from . import capture

    translator = make_translator(args)
    transcriber = make_transcriber(args)
    stop = threading.Event()
    done: list[Cue] = []
    lock = threading.Lock()
    last = {"cue": ""}

    if args.console:
        def show(committed: str, tentative: str) -> None:
            line = (committed + " " + tentative).strip()
            print("\r\033[K" + line[-110:], end="", flush=True)

        def show_cue(c: Cue) -> None:
            print("\r\033[K" + c.text, flush=True)

        def show_translation(c: Cue) -> None:
            print("\r\033[K   → " + (c.translation or ""), flush=True)

        overlay = None
    else:
        from .overlay import Overlay

        overlay = Overlay(font_size=args.font_size)

        def show(committed: str, tentative: str) -> None:
            if committed or tentative:
                overlay.show_text(committed, tentative)
            else:
                overlay.show_text(last["cue"])

        def show_cue(c: Cue) -> None:
            pass

        def show_translation(c: Cue) -> None:
            if c.translation:
                overlay.show_translation(c.translation)

    def translated(c: Cue) -> None:
        with lock:
            for i in range(len(done) - 1, -1, -1):
                if done[i].start == c.start and done[i].text == c.text:
                    done[i] = c
                    break
        show_translation(c)

    worker = TranslationWorker(translator, args.translate_to, translated,
                               on_error=lambda e: print(f"\ntranslation failed: {e}", file=sys.stderr)) if translator else None

    def on_cue(c: Cue) -> None:
        last["cue"] = c.text
        with lock:
            done.append(c)
        show_cue(c)
        if worker:
            worker.submit(c)

    cap = LiveCaptioner(transcriber, step=args.step, language=args.language, on_cue=on_cue, on_partial=show)

    def loop() -> None:
        try:
            for block in capture.blocks(args.source, args.input_device):
                if stop.is_set():
                    break
                cap.feed(block)
        except Exception as e:  # noqa: BLE001 - report and stop rather than die silently in a thread
            print(f"\naudio capture stopped: {e}", file=sys.stderr)
        finally:
            cap.close()
            stop.set()
            if overlay:
                overlay.close()

    thread = threading.Thread(target=loop, daemon=True, name="livesub-audio")
    thread.start()
    src = "what this computer plays" if args.source == "loopback" else "the microphone"
    print(f"listening to {src}; {'Ctrl+C' if args.console else 'double-click the captions or press Esc'} to stop", file=sys.stderr)
    try:
        if overlay:
            overlay.run()
        else:
            while not stop.wait(0.2):
                pass
    except KeyboardInterrupt:
        pass
    stop.set()
    thread.join(5)
    if worker:
        worker.close(10)
    if args.save:
        with lock:
            write_subtitles(args.save, list(done), "both" if translator else "original")
        print(f"\n{len(done)} captions -> {args.save}", file=sys.stderr)
    return 0


def cmd_devices(args) -> int:
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
    except (RuntimeError, ValueError, FileNotFoundError) as e:
        print(f"livesub: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
