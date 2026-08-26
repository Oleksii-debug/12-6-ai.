#!/usr/bin/env python3
"""Train byte-level BPE candidates and compare tokenizer scale economics.

This tool is intentionally outside the core runtime dependency set. Use the locked tokenizer
experiment environment documented by the project when running BPE training or loading tokenizer.json
artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from twelve_six.tokenization.byte import ByteTokenizer
from twelve_six.tokenization.scale_metrics import measure_tokenizer, vocabulary_parameter_cost

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _iter_jsonl_text(path: Path, *, text_field: str) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict) or not isinstance(row.get(text_field), str):
                raise ValueError(
                    f"{path}:{line_number} must contain string field {text_field!r}"
                )
            yield row[text_field]


def _iter_many(paths: Iterable[Path], *, text_field: str) -> Iterator[str]:
    for path in paths:
        yield from _iter_jsonl_text(path, text_field=text_field)


class HFTokenizerAdapter:
    """Minimal adapter around Hugging Face tokenizers.Tokenizer."""

    pad_id = None
    bos_id = None
    eos_id = None
    version = "experiment-hf-tokenizer-json"

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self.vocab_size = int(tokenizer.get_vocab_size(with_added_tokens=True))

    @property
    def identity(self) -> Any:
        raise NotImplementedError("experiment adapter has no checkpoint identity")

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        if add_bos or add_eos:
            raise ValueError("experiment adapter does not inject BOS/EOS")
        return list(self._tokenizer.encode(text, add_special_tokens=False).ids)

    def decode(
        self,
        token_ids: Iterable[int],
        *,
        skip_special_tokens: bool = True,
        errors: str = "strict",
    ) -> str:
        del errors
        return str(
            self._tokenizer.decode(
                list(token_ids), skip_special_tokens=skip_special_tokens
            )
        )


def _load_hf_tokenizer(path: Path) -> HFTokenizerAdapter:
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise RuntimeError(
            "tokenizers is required; use linux-x86_64-tokenizer-experiment"
        ) from exc
    return HFTokenizerAdapter(Tokenizer.from_file(str(path)))


def _train_bpe(args: argparse.Namespace) -> int:
    try:
        import tokenizers as tokenizers_package
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    except ImportError as exc:
        raise RuntimeError(
            "tokenizers is required; use linux-x86_64-tokenizer-experiment"
        ) from exc

    train_paths = [Path(value).resolve() for value in args.train_jsonl]
    for path in train_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=[],
        show_progress=False,
    )
    tokenizer.train_from_iterator(
        _iter_many(train_paths, text_field=args.text_field), trainer=trainer
    )
    tokenizer.save(str(output), pretty=False)

    manifest = {
        "schema": "12-6.tokenizer-experiment-candidate.v1",
        "kind": "byte-level-bpe",
        "tokenizers_version": tokenizers_package.__version__,
        "tokenizer_json": str(output),
        "tokenizer_json_sha256": _hash_file(output),
        "vocab_size_requested": args.vocab_size,
        "vocab_size_actual": tokenizer.get_vocab_size(with_added_tokens=True),
        "min_frequency": args.min_frequency,
        "text_field": args.text_field,
        "training_inputs": [
            {"path": str(path), "sha256": _hash_file(path)} for path in train_paths
        ],
        "normalization": "none",
        "pre_tokenizer": "ByteLevel(add_prefix_space=false,use_regex=true)",
        "decoder": "ByteLevel",
        "promotion_authority": False,
    }
    manifest_path = Path(args.manifest or f"{output}.manifest.json").resolve()
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


def _candidate_spec(raw: str) -> tuple[str, Path]:
    name, separator, path = raw.partition("=")
    if not separator or not name or not path:
        raise ValueError("candidate must use NAME=TOKENIZER_JSON")
    return name, Path(path).resolve()


def _measure_one(
    *,
    name: str,
    tokenizer: Any,
    input_path: Path,
    text_field: str,
    context_length: int,
    d_model: int,
    target_parameters: int | None,
    tied_embeddings: bool,
    artifact_path: Path | None,
) -> dict[str, Any]:
    measurement = measure_tokenizer(
        tokenizer,
        _iter_jsonl_text(input_path, text_field=text_field),
        context_length=context_length,
    )
    result: dict[str, Any] = {
        "name": name,
        "input": str(input_path),
        "input_sha256": _hash_file(input_path),
        "measurement": measurement.to_dict(),
        "vocabulary_cost": vocabulary_parameter_cost(
            vocab_size=int(tokenizer.vocab_size),
            d_model=d_model,
            tied_embeddings=tied_embeddings,
            target_parameters=target_parameters,
        ),
    }
    if artifact_path is not None:
        result["artifact"] = {
            "path": str(artifact_path),
            "sha256": _hash_file(artifact_path),
        }
    return result


def _measure(args: argparse.Namespace) -> int:
    input_paths = [Path(value).resolve() for value in args.input]
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    candidates: list[tuple[str, Any, Path | None]] = [("byte", ByteTokenizer(), None)]
    for raw in args.candidate:
        name, path = _candidate_spec(raw)
        if not path.is_file():
            raise FileNotFoundError(path)
        candidates.append((name, _load_hf_tokenizer(path), path))

    results: list[dict[str, Any]] = []
    for input_path in input_paths:
        for name, tokenizer, artifact_path in candidates:
            results.append(
                _measure_one(
                    name=name,
                    tokenizer=tokenizer,
                    input_path=input_path,
                    text_field=args.text_field,
                    context_length=args.context_length,
                    d_model=args.d_model,
                    target_parameters=args.target_parameters,
                    tied_embeddings=args.tied_embeddings,
                    artifact_path=artifact_path,
                )
            )

    report = {
        "schema": "12-6.tokenizer-scale-experiment-report.v1",
        "context_length": args.context_length,
        "d_model": args.d_model,
        "target_parameters": args.target_parameters,
        "tied_embeddings": args.tied_embeddings,
        "text_field": args.text_field,
        "results": results,
        "interpretation_boundary": {
            "sequence_metrics_are_diagnostic": True,
            "dense_attention_pair_ratio_is_an_architecture_agnostic_proxy": True,
            "downstream_language_model_quality_measured": False,
            "tokenizer_promotion_authority": False,
        },
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"report": str(output), "results": len(results)}, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train-bpe")
    train.add_argument("--train-jsonl", action="append", required=True)
    train.add_argument("--text-field", default="text")
    train.add_argument("--vocab-size", type=int, required=True)
    train.add_argument("--min-frequency", type=int, default=2)
    train.add_argument("--output", required=True)
    train.add_argument("--manifest")
    train.set_defaults(handler=_train_bpe)

    measure = sub.add_parser("measure")
    measure.add_argument("--input", action="append", required=True)
    measure.add_argument("--candidate", action="append", default=[])
    measure.add_argument("--text-field", default="text")
    measure.add_argument("--context-length", type=int, required=True)
    measure.add_argument("--d-model", type=int, required=True)
    measure.add_argument("--target-parameters", type=int)
    embeddings = measure.add_mutually_exclusive_group()
    embeddings.add_argument("--tied-embeddings", action="store_true", dest="tied_embeddings")
    embeddings.add_argument("--untied-embeddings", action="store_false", dest="tied_embeddings")
    measure.set_defaults(tied_embeddings=True)
    measure.add_argument("--output", required=True)
    measure.set_defaults(handler=_measure)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if getattr(args, "vocab_size", 1) <= 0:
        raise ValueError("vocab-size must be positive")
    if getattr(args, "min_frequency", 1) <= 0:
        raise ValueError("min-frequency must be positive")
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
