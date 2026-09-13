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
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
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


def _probe_rows(tokenizer, *, unk_id: int, family: str) -> tuple[list[dict[str, Any]], int, int, int]:
    rows: list[dict[str, Any]] = []
    total_bytes = 0
    total_tokens = 0
    total_unk = 0
    for text in PROBES:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded.ids, skip_special_tokens=False)
        if decoded != text:
            raise RuntimeError(
                f"{family} encode/decode invariant failed: {decoded!r} != {text!r}"
            )
        unk_count = sum(token_id == unk_id for token_id in encoded.ids)
        total_unk += unk_count
        total_bytes += len(text.encode("utf-8"))
        total_tokens += len(encoded.ids)
        rows.append(
            {
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "ids": encoded.ids,
                "ids_sha256": canonical_sha256(encoded.ids),
                "token_count": len(encoded.ids),
                "unk_count": unk_count,
            }
        )
    if total_tokens <= 0:
        raise RuntimeError(f"{family} produced zero tokens")
    return rows, total_bytes, total_tokens, total_unk


def _family_evidence(family: str) -> dict[str, Any]:
    first = _build_family(family)
    second = _build_family(family)
    first_json = first.to_str()
    second_json = second.to_str()
    first_model = json.loads(first_json)
    second_model = json.loads(second_json)

    first_unk_id = first.token_to_id("[UNK]")
    second_unk_id = second.token_to_id("[UNK]")
    if first_unk_id is None or second_unk_id is None:
        raise RuntimeError(f"{family} missing required [UNK] id")

    first_rows, total_bytes, total_tokens, total_unk = _probe_rows(
        first,
        unk_id=first_unk_id,
        family=family,
    )
    second_rows, second_bytes, second_tokens, second_unk = _probe_rows(
        second,
        unk_id=second_unk_id,
        family=family,
    )

    first_vocab = sorted(first.get_vocab().items())
    second_vocab = sorted(second.get_vocab().items())
    canonical_model_a = canonical_sha256(first_model)
    canonical_model_b = canonical_sha256(second_model)
    vocab_a = canonical_sha256(first_vocab)
    vocab_b = canonical_sha256(second_vocab)
    probes_a = canonical_sha256(first_rows)
    probes_b = canonical_sha256(second_rows)

    semantic_equal = (
        canonical_model_a == canonical_model_b
        and vocab_a == vocab_b
        and probes_a == probes_b
        and first_unk_id == second_unk_id
        and first.get_vocab_size() == second.get_vocab_size()
    )
    roundtrip_ok = True
    oov_zero = total_unk == 0 and second_unk == 0
    qualified = semantic_equal and roundtrip_ok and oov_zero

    rejection_reasons: list[str] = []
    if canonical_model_a != canonical_model_b:
        rejection_reasons.append("canonical_model_identity_differs")
    if vocab_a != vocab_b:
        rejection_reasons.append("vocab_id_mapping_differs")
    if probes_a != probes_b:
        rejection_reasons.append("probe_encodings_differ")
    if first_unk_id != second_unk_id:
        rejection_reasons.append("unk_id_differs")
    if not oov_zero:
        rejection_reasons.append("probe_oov_nonzero")

    start = time.perf_counter()
    iterations = 1000
    encoded_count = 0
    for _ in range(iterations):
        for text in PROBES:
            encoded_count += len(first.encode(text).ids)
    elapsed = time.perf_counter() - start

    return {
        "family": family,
        "qualification_status": (
            "QUALIFIED_DETERMINISTIC_SEMANTICS"
            if qualified
            else "UNQUALIFIED_NONDETERMINISTIC"
        ),
        "qualified_deterministic": qualified,
        "raw_serialization_byte_identical": first_json == second_json,
        "canonical_model_identity_equal": canonical_model_a == canonical_model_b,
        "vocab_id_mapping_equal": vocab_a == vocab_b,
        "probe_encodings_equal": probes_a == probes_b,
        "encode_decode_roundtrip": roundtrip_ok,
        "oov_zero": oov_zero,
        "rejection_reasons": rejection_reasons,
        "canonical_model_sha256_a": canonical_model_a,
        "canonical_model_sha256_b": canonical_model_b,
        "raw_json_sha256_a": hashlib.sha256(first_json.encode("utf-8")).hexdigest(),
        "raw_json_sha256_b": hashlib.sha256(second_json.encode("utf-8")).hexdigest(),
        "vocab_mapping_sha256_a": vocab_a,
        "vocab_mapping_sha256_b": vocab_b,
        "probe_identity_sha256_a": probes_a,
        "probe_identity_sha256_b": probes_b,
        "vocab_size": first.get_vocab_size(),
        "unk_token_id": first_unk_id,
        "total_probe_bytes": total_bytes,
        "total_probe_tokens": total_tokens,
        "bytes_per_token": total_bytes / total_tokens,
        "oov_unk_count": total_unk,
        "second_build_total_probe_bytes": second_bytes,
        "second_build_total_probe_tokens": second_tokens,
        "second_build_oov_unk_count": second_unk,
        "benchmark": {
            "iterations": iterations,
            "encoded_token_count": encoded_count,
            "elapsed_seconds": elapsed,
            "tokens_per_second": encoded_count / elapsed if elapsed > 0 else None,
        },
    }


def _stable_family_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "family": row["family"],
        "qualification_status": row["qualification_status"],
        "qualified_deterministic": row["qualified_deterministic"],
        "canonical_model_identity_equal": row["canonical_model_identity_equal"],
        "vocab_id_mapping_equal": row["vocab_id_mapping_equal"],
        "probe_encodings_equal": row["probe_encodings_equal"],
        "encode_decode_roundtrip": row["encode_decode_roundtrip"],
        "oov_zero": row["oov_zero"],
        "rejection_reasons": row["rejection_reasons"],
        "vocab_size": row["vocab_size"],
        "unk_token_id": row["unk_token_id"],
    }


def build_report(expected_head_sha: str) -> dict[str, Any]:
    import tokenizers

    if tokenizers.__version__ != EXPECTED_VERSION:
        raise RuntimeError("exact tokenizers runtime is not installed")

    families = [_family_evidence("BPE"), _family_evidence("UNIGRAM")]
    by_family = {row["family"]: row for row in families}
    if not by_family["BPE"]["qualified_deterministic"]:
        raise RuntimeError("BPE failed deterministic runtime qualification")

    unigram_qualified = by_family["UNIGRAM"]["qualified_deterministic"]
    status = (
        "RUNTIME_QUALIFIED_BPE_AND_UNIGRAM"
        if unigram_qualified
        else "RUNTIME_QUALIFIED_BPE_ONLY_UNIGRAM_REJECTED"
    )
    stable_families = [_stable_family_summary(row) for row in families]

    qualification_core = {
        "schema_version": SCHEMA,
        "repository": "Oleksii-debug/12-6-ai.",
        "exact_head_sha": expected_head_sha,
        "upstream_version": EXPECTED_VERSION,
        "upstream_commit": EXPECTED_UPSTREAM_COMMIT,
        "wheel_sha256": EXPECTED_WHEEL_SHA256,
        "corpus_sha256": corpus_sha256(),
        "status": status,
        "families": stable_families,
        "canonical_base_changed": False,
        "token_ids_changed": False,
        "training_executed": False,
        "final_test_accessed": False,
        "paid_compute_used": False,
    }
    report = dict(qualification_core)
    report.update(
        {
            "runtime": {
                "tokenizers_version": tokenizers.__version__,
                "python": sys.version,
                "platform": platform.platform(),
                "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            },
            "family_metrics": families,
            "qualification_identity_sha256": canonical_sha256(qualification_core),
            "promotion_boundary": (
                "HF Tokenizers 0.23.1 is runtime-qualified only for family rows marked "
                "QUALIFIED_DETERMINISTIC_SEMANTICS. Any rejected family remains unavailable "
                "for canonical training. Canonical s0-byte-v1 and token IDs remain unchanged "
                "until a terminal corpus-bound tokenizer decision explicitly replaces them."
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
                "qualification_identity_sha256": report["qualification_identity_sha256"],
                "families": {
                    row["family"]: row["qualification_status"]
                    for row in report["family_metrics"]
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
