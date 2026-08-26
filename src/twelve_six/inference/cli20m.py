from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .contracts import GenerationConfig
from .twenty_m import TwentyMInference, load_20m_model_spec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="twelve-six-20m-generate",
        description=(
            "Generate a raw completion with the maintained first-party ~20M "
            "runtime. No chat template or role semantics are applied."
        ),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="verified first-party checkpoint directory")
    source.add_argument(
        "--random-init-spec",
        type=Path,
        help="20M ModelSpec JSON; random-init mechanics only",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--init-seed", type=int, default=0)
    parser.add_argument("--prompt", help="raw prompt text; otherwise read stdin")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--greedy", dest="sample", action="store_false")
    mode.add_argument("--sample", dest="sample", action="store_true")
    parser.set_defaults(sample=False)

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--stop", action="append", default=[])
    parser.add_argument("--stop-token-id", action="append", type=int, default=[])
    parser.add_argument("--keep-stop-string", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def _read_prompt(parser: argparse.ArgumentParser, prompt: str | None) -> str:
    if prompt is not None:
        return prompt
    if sys.stdin.isatty():
        parser.error("provide --prompt or pipe raw prompt text on stdin")
    return sys.stdin.read()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    prompt = _read_prompt(parser, args.prompt)

    try:
        if args.checkpoint is not None:
            session = TwentyMInference.from_checkpoint(args.checkpoint, device=args.device)
        else:
            spec = load_20m_model_spec(args.random_init_spec)
            session = TwentyMInference.from_random_init(
                spec,
                seed=args.init_seed,
                device=args.device,
            )

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
        result = session.generate(prompt, config)
        diagnostics = session.diagnostics()
    except (
        FileNotFoundError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    mode = "sample" if config.sample else "greedy"
    print(
        "backend: "
        f"source={diagnostics.get('source')} "
        f"parameters={diagnostics.get('parameter_count')} "
        f"model_spec={diagnostics.get('model_spec_sha256')} "
        f"device={diagnostics.get('device')}",
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
        print(
            json.dumps(
                {
                    "text": result.text,
                    "prompt_token_ids": result.prompt_token_ids,
                    "generated_token_ids": result.generated_token_ids,
                    "stop_reason": result.stop_reason,
                    "seed": config.seed,
                    "mode": mode,
                    "backend": diagnostics,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    else:
        print(result.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
