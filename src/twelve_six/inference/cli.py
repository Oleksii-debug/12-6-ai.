from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .contracts import GenerationConfig
from .generation import generate
from .loader import load_backend

DEFAULT_BACKEND_LOADER = "twelve_six.inference.first_party:load_first_party_backend"
DEFAULT_MAX_PROMPT_CHARS = 1_048_576
PROMPT_READ_CHUNK_CHARS = 65_536


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="twelve-six-generate",
        description="Generate a raw Base completion from a local 12-6 AI checkpoint.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--backend-loader",
        default=DEFAULT_BACKEND_LOADER,
        metavar="MODULE:CALLABLE",
        help=(
            "checkpoint backend factory; defaults to the verified first-party "
            "D01+D04+D05 adapter"
        ),
    )
    parser.add_argument("--prompt", help="prompt text; when omitted, read all prompt text from stdin")
    parser.add_argument(
        "--max-prompt-chars",
        type=_positive_int,
        default=DEFAULT_MAX_PROMPT_CHARS,
        metavar="N",
        help=(
            "maximum prompt characters accepted from --prompt or stdin; "
            f"default: {DEFAULT_MAX_PROMPT_CHARS}"
        ),
    )
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


def _read_bounded_stdin(max_prompt_chars: int) -> str:
    """Read stdin through a strict character ceiling without silent truncation."""

    remaining = max_prompt_chars + 1
    chunks: list[str] = []
    while remaining > 0:
        chunk = sys.stdin.read(min(PROMPT_READ_CHUNK_CHARS, remaining))
        if chunk == "":
            break
        if not isinstance(chunk, str):
            raise TypeError("stdin text stream must return strings")
        chunks.append(chunk)
        remaining -= len(chunk)
    return "".join(chunks)


def _read_prompt(
    parser: argparse.ArgumentParser,
    prompt: str | None,
    *,
    max_prompt_chars: int,
) -> str:
    if not isinstance(max_prompt_chars, int) or isinstance(max_prompt_chars, bool):
        raise TypeError("max_prompt_chars must be a positive integer")
    if max_prompt_chars <= 0:
        raise ValueError("max_prompt_chars must be a positive integer")

    if prompt is not None:
        if len(prompt) > max_prompt_chars:
            parser.error(
                "--prompt exceeds --max-prompt-chars "
                f"({len(prompt)} > {max_prompt_chars})"
            )
        return prompt
    if sys.stdin.isatty():
        parser.error("provide --prompt or pipe prompt text on stdin")

    stdin_prompt = _read_bounded_stdin(max_prompt_chars)
    if len(stdin_prompt) > max_prompt_chars:
        parser.error(
            "stdin prompt exceeds --max-prompt-chars "
            f"(limit={max_prompt_chars})"
        )
    return stdin_prompt


def _backend_diagnostics(backend: object) -> dict[str, object] | None:
    diagnostics = getattr(backend, "diagnostics", None)
    if diagnostics is None:
        return None
    if not callable(diagnostics):
        raise TypeError("backend diagnostics attribute must be callable")
    payload = diagnostics()
    if not isinstance(payload, dict):
        raise TypeError("backend diagnostics must return a dictionary")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    prompt = _read_prompt(
        parser,
        args.prompt,
        max_prompt_chars=args.max_prompt_chars,
    )

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
        backend_diagnostics = _backend_diagnostics(backend)
        result = generate(backend, prompt, config)
    except (
        ImportError,
        AttributeError,
        FileNotFoundError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    mode = "sample" if config.sample else "greedy"
    if backend_diagnostics is not None:
        print(
            "backend: "
            f"kind={backend_diagnostics.get('backend')} "
            f"checkpoint_id={backend_diagnostics.get('checkpoint_id')} "
            f"model_spec={backend_diagnostics.get('model_spec_sha256')} "
            f"tokenizer={backend_diagnostics.get('tokenizer_config_sha256')}",
            file=sys.stderr,
        )
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
            "backend": backend_diagnostics,
        }
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        print(result.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
