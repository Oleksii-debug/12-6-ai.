from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .api import FirstPartyInference
from .contracts import GenerationConfig
from .generation import generate, generate_token_ids
from .loader import load_backend

DEFAULT_BACKEND_LOADER = "twelve_six.inference.first_party:load_first_party_backend"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="twelve-six-generate",
        description="Generate one raw Base completion with the local first-party decoder.",
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--checkpoint",
        type=Path,
        help="D05 checkpoint directory; works for compatible learned ModelSpec-v1 checkpoints",
    )
    source.add_argument(
        "--random-init-stage",
        type=Path,
        help="mechanics-only StageConfig, e.g. configs/candidates/model341_20m_candidate_a.json",
    )
    parser.add_argument(
        "--init-seed",
        type=int,
        default=0,
        help="model initialization seed used only with --random-init-stage",
    )
    parser.add_argument(
        "--backend-loader",
        default=DEFAULT_BACKEND_LOADER,
        metavar="MODULE:CALLABLE",
        help=(
            "checkpoint backend factory retained for compatibility; the default is the "
            "verified first-party D01+D04+D05 adapter"
        ),
    )

    prompt = parser.add_mutually_exclusive_group()
    prompt.add_argument(
        "--prompt",
        help="raw prompt text; when no input option is supplied, read all text from stdin",
    )
    prompt.add_argument(
        "--token-ids",
        metavar="IDS",
        help="comma- or whitespace-separated raw prompt token IDs, e.g. '72,101,108,108,111'",
    )

    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0, help="generation sampling seed")
    parser.add_argument(
        "--cache-mode",
        choices=("static", "stateless"),
        default="static",
        help="explicit first-party decode path; default: static",
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--greedy", dest="sample", action="store_false", help="greedy decoding")
    mode.add_argument("--sample", dest="sample", action="store_true", help="sample from logits")
    parser.set_defaults(sample=False)

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--stop", action="append", default=[], help="repeatable raw text stop sequence")
    parser.add_argument(
        "--stop-token-id",
        action="append",
        type=int,
        default=[],
        help="repeatable token-ID stop condition",
    )
    parser.add_argument(
        "--keep-stop-string",
        action="store_true",
        help="keep a matched raw text stop sequence in emitted text",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="compatibility/debug output; default stdout is generated plain text only",
    )
    return parser


def _read_text_prompt(parser: argparse.ArgumentParser, prompt: str | None) -> str:
    if prompt is not None:
        return prompt
    if sys.stdin.isatty():
        parser.error("provide --prompt, --token-ids, or pipe raw prompt text on stdin")
    return sys.stdin.read()


def _parse_token_ids(parser: argparse.ArgumentParser, raw: str) -> tuple[int, ...]:
    parts = raw.replace(",", " ").split()
    if not parts:
        parser.error("--token-ids must contain at least one integer token ID")
    values: list[int] = []
    for part in parts:
        try:
            value = int(part, 10)
        except ValueError:
            parser.error(f"invalid token ID {part!r}; use decimal integers separated by commas/spaces")
        values.append(value)
    return tuple(values)


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

    if args.random_init_stage is not None and args.backend_loader != DEFAULT_BACKEND_LOADER:
        parser.error("--backend-loader is only valid with --checkpoint")

    input_kind = "token_ids" if args.token_ids is not None else "text"
    prompt_token_ids = (
        _parse_token_ids(parser, args.token_ids) if args.token_ids is not None else None
    )
    prompt_text = None if prompt_token_ids is not None else _read_text_prompt(parser, args.prompt)

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

        first_party: FirstPartyInference | None = None
        if args.random_init_stage is not None:
            first_party = FirstPartyInference.from_random_init_stage(
                args.random_init_stage,
                seed=args.init_seed,
            )
            backend = first_party.backend
            backend_diagnostics = first_party.diagnostics()
        elif args.backend_loader == DEFAULT_BACKEND_LOADER:
            if args.checkpoint is None:
                raise ValueError("checkpoint path is missing")
            first_party = FirstPartyInference.from_checkpoint(args.checkpoint)
            backend = first_party.backend
            backend_diagnostics = first_party.diagnostics()
        else:
            if args.checkpoint is None:
                raise ValueError("checkpoint path is missing")
            backend = load_backend(args.backend_loader, args.checkpoint)
            backend_diagnostics = _backend_diagnostics(backend)

        if prompt_token_ids is not None:
            if first_party is not None:
                result = first_party.generate_token_ids(
                    prompt_token_ids,
                    config,
                    cache_mode=args.cache_mode,
                )
            else:
                result = generate_token_ids(
                    backend,
                    prompt_token_ids,
                    config,
                    cache_mode=args.cache_mode,
                )
        else:
            if prompt_text is None:
                raise ValueError("text prompt is missing")
            if first_party is not None:
                result = first_party.generate_text(
                    prompt_text,
                    config,
                    cache_mode=args.cache_mode,
                )
            else:
                result = generate(
                    backend,
                    prompt_text,
                    config,
                    cache_mode=args.cache_mode,
                )
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

    mode_name = "sample" if config.sample else "greedy"
    if backend_diagnostics is not None:
        print(
            "backend: "
            f"kind={backend_diagnostics.get('backend')} "
            f"source={backend_diagnostics.get('source_kind', 'checkpoint')} "
            f"checkpoint_id={backend_diagnostics.get('checkpoint_id')} "
            f"model_spec={backend_diagnostics.get('model_spec_sha256')} "
            f"tokenizer={backend_diagnostics.get('tokenizer_config_sha256')}",
            file=sys.stderr,
        )
    print(
        "generation: "
        f"input={input_kind} cache={args.cache_mode} mode={mode_name} seed={config.seed} "
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
            "mode": mode_name,
            "cache_mode": args.cache_mode,
            "input_kind": input_kind,
            "backend": backend_diagnostics,
        }
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        print(result.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
