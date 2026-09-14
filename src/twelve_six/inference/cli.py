from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .contracts import GenerationConfig
from .generation import generate
from .loader import load_backend


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="twelve-six-generate",
        description="Generate text from a local 12-6 AI checkpoint.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--backend-loader",
        required=True,
        metavar="MODULE:CALLABLE",
        help="factory that accepts checkpoint Path and returns the D07 inference backend",
    )
    parser.add_argument("--prompt", help="prompt text; when omitted, read all prompt text from stdin")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--greedy", dest="sample", action="store_false", help="greedy decoding")
    mode.add_argument("--sample", dest="sample", action="store_true", help="sample from logits")
    parser.set_defaults(sample=False)

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--stop", action="append", default=[], help="repeatable text stop sequence")
    parser.add_argument(
        "--stop-token-id",
        action="append",
        type=int,
        default=[],
        help="repeatable token-id stop condition",
    )
    parser.add_argument(
        "--keep-stop-string",
        action="store_true",
        help="keep matched text stop sequence in emitted text",
    )
    parser.add_argument("--json", action="store_true", help="emit one JSON object to stdout")
    return parser


def _read_prompt(parser: argparse.ArgumentParser, prompt: str | None) -> str:
    if prompt is not None:
        return prompt
    if sys.stdin.isatty():
        parser.error("provide --prompt or pipe prompt text on stdin")
    return sys.stdin.read()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    prompt = _read_prompt(parser, args.prompt)

    try:
        config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            sample=args.sample,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            seed=args.seed,
            stop_token_ids=tuple(args.stop_token_id),
            stop_strings=tuple(args.stop),
            strip_stop_strings=not args.keep_stop_string,
        )
        backend = load_backend(args.backend_loader, args.checkpoint)
        result = generate(backend, prompt, config)
    except (ImportError, AttributeError, FileNotFoundError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    mode = "sample" if config.sample else "greedy"
    print(
        "generation: "
        f"mode={mode} seed={config.seed} "
        f"prompt_tokens={len(result.prompt_token_ids)} "
        f"new_tokens={len(result.generated_token_ids)} "
        f"stop={result.stop_reason}",
        file=sys.stderr,
    )

    if args.json:
        payload = {
            "text": result.text,
            "prompt_token_ids": result.prompt_token_ids,
            "generated_token_ids": result.generated_token_ids,
            "stop_reason": result.stop_reason,
            "seed": config.seed,
            "mode": mode,
        }
        print(json.dumps(payload, ensure_ascii=False))
    else:
        print(result.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
