#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any

EXPECTED_VERSION = "0.23.1"
EXPECTED_UPSTREAM_COMMIT = "7f1623b90b5adfb9bc327d4c3468d2f70bbce262"
EXPECTED_WHEEL_SHA256 = "5075b405006415ea148a992d093699c66eb01952bf59f4d5727089a98bda45a4"
SCHEMA = "12-6.hf-tokenizers-runtime-qualification.v1"

CORPUS = [
    "Українська мова потребує точного відтворення байтів і меж документів.",
    "Deterministic tokenization must preserve exact text and stable identities.",
    "def add(a, b):\n    return a + b\n",
    "Модель навчається лише на дозволеному корпусі без витоку тестових даних.",
    "Packing, masking, and resume state must be deterministic across restarts.",
    "for item in records:\n    assert item is not None\n",
]
PROBES = [CORPUS[0], CORPUS[1], CORPUS[2], "Україна + English + code_123"]


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def corpus_sha256() -> str:
    return canonical_sha256(CORPUS)


def _build_family(family: str):
    import tokenizers
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    if tokenizers.__version__ != EXPECTED_VERSION:
        raise RuntimeError(
            f"tokenizers version mismatch: {tokenizers.__version__} != {EXPECTED_VERSION}"
        )

    if family == "BPE":
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        trainer = trainers.BpeTrainer(
            vocab_size=384,
            min_frequency=1,
            special_tokens=["[UNK]"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )
    elif family == "UNIGRAM":
        tokenizer = Tokenizer(models.Unigram())
        trainer = trainers.UnigramTrainer(
            vocab_size=384,
            special_tokens=["[UNK]"],
            unk_token="[UNK]",
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )
    else:
        raise ValueError(f"unsupported family: {family}")

    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(CORPUS, trainer=trainer)
    return tokenizer


def _family_evidence(family: str) -> dict[str, Any]:
    first = _build_family(family)
    second = _build_family(family)
    first_json = first.to_str()
    second_json = second.to_str()
    if first_json != second_json:
        raise RuntimeError(f"{family} double-build tokenizer JSON differs")

    unk_id = first.token_to_id("[UNK]")
    if unk_id is None:
        raise RuntimeError(f"{family} missing required [UNK] id")

    probe_rows: list[dict[str, Any]] = []
    total_bytes = 0
    total_tokens = 0
    total_unk = 0
    for text in PROBES:
        encoded = first.encode(text)
        decoded = first.decode(encoded.ids, skip_special_tokens=False)
        if decoded != text:
            raise RuntimeError(
                f"{family} encode/decode invariant failed: {decoded!r} != {text!r}"
            )
        unk_count = sum(token_id == unk_id for token_id in encoded.ids)
        total_unk += unk_count
        total_bytes += len(text.encode("utf-8"))
        total_tokens += len(encoded.ids)
        probe_rows.append(
            {
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "ids_sha256": canonical_sha256(encoded.ids),
                "token_count": len(encoded.ids),
                "unk_count": unk_count,
            }
        )

    if total_tokens <= 0:
        raise RuntimeError(f"{family} produced zero tokens")

    start = time.perf_counter()
    iterations = 1000
    encoded_count = 0
    for _ in range(iterations):
        for text in PROBES:
            encoded_count += len(first.encode(text).ids)
    elapsed = time.perf_counter() - start

    return {
        "family": family,
        "tokenizer_json_sha256": hashlib.sha256(first_json.encode("utf-8")).hexdigest(),
        "vocab_size": first.get_vocab_size(),
        "unk_token_id": unk_id,
        "double_build_byte_identical": True,
        "encode_decode_roundtrip": True,
        "probe_identity_sha256": canonical_sha256(probe_rows),
        "probe_rows": probe_rows,
        "total_probe_bytes": total_bytes,
        "total_probe_tokens": total_tokens,
        "bytes_per_token": total_bytes / total_tokens,
        "oov_unk_count": total_unk,
        "benchmark": {
            "iterations": iterations,
            "encoded_token_count": encoded_count,
            "elapsed_seconds": elapsed,
            "tokens_per_second": encoded_count / elapsed if elapsed > 0 else None,
        },
    }


def build_report(expected_head_sha: str) -> dict[str, Any]:
    import tokenizers

    if tokenizers.__version__ != EXPECTED_VERSION:
        raise RuntimeError("exact tokenizers runtime is not installed")

    families = [_family_evidence("BPE"), _family_evidence("UNIGRAM")]
    deterministic_families = []
    for row in families:
        deterministic_families.append(
            {
                key: value
                for key, value in row.items()
                if key not in {"benchmark", "bytes_per_token"}
            }
        )

    semantic_core = {
        "schema_version": SCHEMA,
        "repository": "Oleksii-debug/12-6-ai.",
        "exact_head_sha": expected_head_sha,
        "upstream_version": EXPECTED_VERSION,
        "upstream_commit": EXPECTED_UPSTREAM_COMMIT,
        "wheel_sha256": EXPECTED_WHEEL_SHA256,
        "corpus_sha256": corpus_sha256(),
        "families": deterministic_families,
        "canonical_base_changed": False,
        "token_ids_changed": False,
        "training_executed": False,
        "final_test_accessed": False,
        "paid_compute_used": False,
    }
    report = dict(semantic_core)
    report.update(
        {
            "runtime": {
                "tokenizers_version": tokenizers.__version__,
                "python": sys.version,
                "platform": platform.platform(),
                "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            },
            "family_metrics": families,
            "semantic_identity_sha256": canonical_sha256(semantic_core),
            "status": "RUNTIME_QUALIFIED_OPTIONAL_COMPONENT",
            "promotion_boundary": (
                "HF Tokenizers is runtime-qualified for optional BPE/Unigram experiments only; "
                "canonical s0-byte-v1 remains unchanged until a corpus-bound evidence decision."
            ),
        }
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-head-sha", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if len(args.expected_head_sha) != 40:
        raise SystemExit("expected head SHA must be exact 40-hex git SHA")
    int(args.expected_head_sha, 16)
    report = build_report(args.expected_head_sha.lower())
    Path(args.output).write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "semantic_identity_sha256": report["semantic_identity_sha256"],
                "families": [row["family"] for row in report["family_metrics"]],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
