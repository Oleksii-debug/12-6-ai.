"""Diagnose exact Unigram reproducibility under the locked tokenizer experiment runtime."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import random
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from twelve_six.tokenization import real_experiments as real
from twelve_six.tokenization.experiments import measure_probe, train_hf_tokenizer

SCHEMA = "12-6.tokenizer-unigram-repro.v1"
DECISION_REJECT = "REJECT_UNIGRAM_CANONICAL_TOKENIZERS_0_23_1"
UPSTREAM_TRAINER_BLOB_SHA = "ff5ca9428ab7c7ca9b96065046f32b42246dc234"
UPSTREAM_PYTHON_BINDING_BLOB_SHA = "df0b11ec57ef129515008f44dfae7c539f45ff46"
AHASH_VERSION = "0.8.11"
_EXACT_SEED_RE = re.compile(r"(?<![0-9A-Za-z_])seed(?![0-9A-Za-z_])")


class UnigramReproError(ValueError):
    """Fail-closed Unigram reproducibility evidence error."""


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _artifact_to_dict(adapter: Any) -> dict[str, object]:
    artifact = adapter.artifact_identity
    return {
        "algorithm": artifact.algorithm,
        "tokenizers_version": artifact.tokenizers_version,
        "training_manifest_sha256": artifact.training_manifest_sha256,
        "tokenizer_json_sha256": artifact.tokenizer_json_sha256,
        "vocab_sha256": artifact.vocab_sha256,
        "vocab_size": artifact.vocab_size,
        "config_sha256": artifact.config_sha256,
        "special_tokens": dict(artifact.special_tokens),
    }


def _ordered_training_texts(dataset: dict[str, Any], order: str) -> tuple[str, ...]:
    records = dataset["train_records"]
    texts: list[str] = []
    for record in records:
        text = record.get("text")
        if not isinstance(text, str):
            raise UnigramReproError("training record text must be str")
        texts.append(text)
    if order == "reversed":
        texts.reverse()
    elif order != "manifest":
        raise UnigramReproError("order must be manifest or reversed")
    return tuple(texts)


def _trainer_seed_probe(tokenizers: Any) -> dict[str, object]:
    """Inspect the advertised API without passing an ignored **kwargs seed option."""
    trainer_type = tokenizers.trainers.UnigramTrainer
    text_signature = getattr(trainer_type, "__text_signature__", None)
    doc = getattr(trainer_type, "__doc__", "") or ""
    advertised = "\n".join(part for part in (text_signature, doc) if isinstance(part, str))
    supported = _EXACT_SEED_RE.search(advertised) is not None
    return {
        "public_seed_argument_supported": supported,
        "probe": "advertised UnigramTrainer public signature/documentation",
        "text_signature": text_signature,
        "exact_seed_argument_advertised": supported,
        "unknown_kwargs_policy": "unknown kwargs are printed-and-ignored by pinned Python binding",
    }


def _single(order: str) -> dict[str, object]:
    random.seed(0)
    dataset = real._dataset_contract()
    manifest = real._training_manifest("unigram", dataset, vocab_size=512)
    texts = _ordered_training_texts(dataset, order)
    adapter = train_hf_tokenizer(manifest, texts)

    runtime = adapter._tokenizer
    tokenizer_json = runtime.to_str()
    tokenizer_json_repeat = runtime.to_str()

    tokenizers = importlib.import_module("tokenizers")
    reloaded_json = tokenizers.Tokenizer.from_str(tokenizer_json).to_str()
    payload = json.loads(tokenizer_json)
    if not isinstance(payload, dict):
        raise TypeError("serialized tokenizer must be a JSON object")
    model = payload.get("model")
    if not isinstance(model, dict) or model.get("type") != "Unigram":
        raise UnigramReproError("serialized tokenizer must contain a Unigram model")
    vocab = model.get("vocab")
    if not isinstance(vocab, list):
        raise UnigramReproError("serialized Unigram model must expose ordered vocab")
    tokens: list[str] = []
    scores: list[float] = []
    for entry in vocab:
        if not isinstance(entry, list) or len(entry) != 2:
            raise UnigramReproError("Unigram vocab entry must be [token, score]")
        token, score = entry
        if not isinstance(token, str) or not isinstance(score, int | float):
            raise UnigramReproError("Unigram vocab token/score types are invalid")
        tokens.append(token)
        scores.append(float(score))

    probes = real._held_out_probes(dataset)
    probe_results = [
        measure_probe(adapter, probe, unknown_token_id=adapter.unk_id) for probe in probes
    ]
    probe_ids = [adapter.encode(probe.text) for probe in probes]
    if not all(result.round_trip_exact for result in probe_results):
        raise UnigramReproError("Unigram failed strict held-out round trip")
    if any(result.unknown_tokens for result in probe_results):
        raise UnigramReproError("Unigram emitted held-out unknown tokens")

    text_order_identity = _sha256_text(_canonical_json(texts))
    text_multiset_identity = _sha256_text(_canonical_json(sorted(texts)))
    return {
        "artifact": _artifact_to_dict(adapter),
        "internals": {
            "model_type": "Unigram",
            "ordered_model_vocab_sha256": _sha256_text(_canonical_json(vocab)),
            "ordered_model_tokens_sha256": _sha256_text(_canonical_json(tokens)),
            "ordered_model_scores_sha256": _sha256_text(_canonical_json(scores)),
            "added_tokens_sha256": _sha256_text(_canonical_json(payload.get("added_tokens", []))),
            "pre_tokenizer_sha256": _sha256_text(_canonical_json(payload.get("pre_tokenizer"))),
            "decoder_sha256": _sha256_text(_canonical_json(payload.get("decoder"))),
            "serialization_repeat_exact": tokenizer_json == tokenizer_json_repeat,
            "serialize_reload_serialize_exact": tokenizer_json == reloaded_json,
            "serialized_bytes": len(tokenizer_json.encode("utf-8")),
        },
        "training": {
            "order": order,
            "ordered_texts_sha256": text_order_identity,
            "text_multiset_sha256": text_multiset_identity,
            "manifest_sha256": manifest.sha256,
            "record_ids": list(dataset["train_record_ids"]),
            "python_random_seed": 0,
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "tokenizers_parallelism": os.environ.get("TOKENIZERS_PARALLELISM"),
            "rayon_num_threads": os.environ.get("RAYON_NUM_THREADS"),
        },
        "held_out": {
            "probe_ids": probe_ids,
            "language_summary": real.summarize_by_language(probe_results),
            "strict_round_trip_all": True,
            "unknown_tokens": 0,
        },
        "seed_probe": _trainer_seed_probe(tokenizers),
        "runtime": {
            "python": platform.python_version(),
            "tokenizers": tokenizers.__version__,
            "platform": platform.platform(),
        },
    }


def _child_environment(*, serial: bool) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    if serial:
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["RAYON_NUM_THREADS"] = "1"
    else:
        env["TOKENIZERS_PARALLELISM"] = "true"
        env["RAYON_NUM_THREADS"] = "4"
    return env


def _run_child(*, order: str, serial: bool) -> dict[str, object]:
    process = subprocess.run(
        [sys.executable, "-m", "twelve_six.tokenization.unigram_repro", "single", "--order", order],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=_child_environment(serial=serial),
    )
    value = json.loads(process.stdout)
    if not isinstance(value, dict):
        raise TypeError("child result must be a JSON object")
    return value


def _pairwise_equal(values: Sequence[object]) -> bool:
    return all(value == values[0] for value in values[1:]) if values else True


def _regime_summary(runs: Sequence[dict[str, object]]) -> dict[str, object]:
    if len(runs) < 2:
        raise UnigramReproError("a reproducibility regime needs at least two runs")
    artifacts = [run["artifact"] for run in runs]
    internals = [run["internals"] for run in runs]
    held_out = [run["held_out"] for run in runs]
    tokenizer_json_hashes = [artifact["tokenizer_json_sha256"] for artifact in artifacts]
    vocab_hashes = [artifact["vocab_sha256"] for artifact in artifacts]
    model_vocab_hashes = [item["ordered_model_vocab_sha256"] for item in internals]
    probe_ids = [item["probe_ids"] for item in held_out]
    return {
        "runs": list(runs),
        "exact_artifact_identity_equal": _pairwise_equal(artifacts),
        "tokenizer_json_identity_equal": _pairwise_equal(tokenizer_json_hashes),
        "ordered_vocab_identity_equal": _pairwise_equal(vocab_hashes),
        "ordered_model_vocab_identity_equal": _pairwise_equal(model_vocab_hashes),
        "held_out_encoding_equal": _pairwise_equal(probe_ids),
        "serialization_repeat_exact_all": all(
            bool(item["serialization_repeat_exact"]) for item in internals
        ),
        "serialize_reload_serialize_exact_all": all(
            bool(item["serialize_reload_serialize_exact"]) for item in internals
        ),
        "strict_round_trip_all": all(bool(item["strict_round_trip_all"]) for item in held_out),
        "zero_unknown_tokens_all": all(item["unknown_tokens"] == 0 for item in held_out),
        "unique_tokenizer_json_artifacts": len(set(tokenizer_json_hashes)),
        "unique_ordered_vocabularies": len(set(vocab_hashes)),
    }


def build_report(source_sha: str) -> dict[str, object]:
    real._require_git_sha(source_sha, "source_sha")
    serial_manifest = [_run_child(order="manifest", serial=True) for _ in range(3)]
    serial_reversed = [_run_child(order="reversed", serial=True) for _ in range(2)]
    parallel_manifest = [_run_child(order="manifest", serial=False) for _ in range(2)]

    serial = _regime_summary(serial_manifest)
    reversed_serial = _regime_summary(serial_reversed)
    parallel = _regime_summary(parallel_manifest)
    first = serial_manifest[0]
    seed_supported = bool(first["seed_probe"]["public_seed_argument_supported"])

    serial_drift = not bool(serial["exact_artifact_identity_equal"])
    vocab_drift = not bool(serial["ordered_vocab_identity_equal"])
    encoding_drift = not bool(serial["held_out_encoding_equal"])
    serialization_stable = bool(serial["serialization_repeat_exact_all"]) and bool(
        serial["serialize_reload_serialize_exact_all"]
    )

    if not serial_drift:
        raise UnigramReproError(
            "serial locked probe did not reproduce the known Unigram drift; do not change decision"
        )
    if not serialization_stable:
        raise UnigramReproError(
            "serialization itself drifted; root-cause classification is ambiguous"
        )
    if seed_supported:
        raise UnigramReproError("runtime unexpectedly advertises a public UnigramTrainer seed")

    report: dict[str, object] = {
        "schema": SCHEMA,
        "source": {
            "repository": "Oleksii-debug/12-6-ai.",
            "source_sha": source_sha,
            "observed_head_sha": real._git_head(),
        },
        "runtime": {
            "python": first["runtime"]["python"],
            "tokenizers": first["runtime"]["tokenizers"],
            "experiment_lock": real.verify_experiment_lock(),
        },
        "training_identity": {
            "manifest_sha256": first["training"]["manifest_sha256"],
            "record_ids": first["training"]["record_ids"],
            "manifest_ordered_texts_sha256": first["training"]["ordered_texts_sha256"],
            "text_multiset_sha256": first["training"]["text_multiset_sha256"],
            "requested_vocab_size": 512,
            "actual_vocab_size": first["artifact"]["vocab_size"],
        },
        "minimal_reproduction": {
            "serial_manifest_order": serial,
            "serial_reversed_input_order": reversed_serial,
            "parallel_manifest_order": parallel,
            "manifest_vs_reversed_first_artifact_equal": (
                serial_manifest[0]["artifact"] == serial_reversed[0]["artifact"]
            ),
        },
        "root_cause": {
            "status": "CONFIRMED_BACKEND_HASH_ORDER_NONDETERMINISM",
            "supported_seed_control": False,
            "seed_probe": first["seed_probe"],
            "fixed_manifest_input_order_still_drifts": serial_drift,
            "threading_eliminated_as_sufficient_fix": serial_drift,
            "serialization_metadata_eliminated_as_primary_cause": serialization_stable,
            "input_order_is_not_a_sufficient_control": serial_drift,
            "floating_reduction_may_be_secondary": True,
            "upstream_source": {
                "package": "tokenizers",
                "version": real.TOKENIZERS_VERSION,
                "trainer_path": "tokenizers/src/models/unigram/trainer.rs",
                "trainer_blob_sha": UPSTREAM_TRAINER_BLOB_SHA,
                "python_binding_path": "bindings/python/src/trainers.rs",
                "python_binding_blob_sha": UPSTREAM_PYTHON_BINDING_BLOB_SHA,
                "ahash_version": AHASH_VERSION,
                "mechanism": (
                    "UnigramTrainer feed/train uses randomized AHashMap/AHashSet iteration before "
                    "EM training. The Python binding exposes no seed control and ignores unknown "
                    "kwargs such as seed. Parallel floating reductions are an additional possible "
                    "source, but exact drift persists with tokenizer parallelism disabled and one "
                    "Rayon thread."
                ),
            },
        },
        "semantic_equivalence": {
            "safe_for_checkpoint_identity": False,
            "ordered_vocab_drift": vocab_drift,
            "held_out_encoding_drift": encoding_drift,
            "round_trip_preserved": bool(serial["strict_round_trip_all"]),
            "zero_unknown_tokens": bool(serial["zero_unknown_tokens_all"]),
            "reason": (
                "different token-ID vocabulary and/or held-out token sequences change "
                "embedding/LM-head semantics even when decoded text is identical; finite probe "
                "agreement would not prove global tokenizer equivalence"
            ),
        },
        "decision": {
            "status": DECISION_REJECT,
            "canonical_use": "FAIL",
            "semantic_equivalence_identity_allowed": False,
            "canonical_s0_unchanged": True,
            "reason": (
                "pinned tokenizers 0.23.1 exposes no supported Unigram seed and repeated "
                "serial locked training still changes exact vocabulary/artifact identity; "
                "semantic equivalence is unsafe"
            ),
        },
    }
    if report["source"]["observed_head_sha"] != source_sha:
        raise UnigramReproError("source head changed during Unigram reproducibility run")
    report["evidence_sha256"] = _sha256_text(_canonical_json(report))
    return report


def validate_report(report: dict[str, object], *, expected_source_sha: str | None = None) -> None:
    if report.get("schema") != SCHEMA:
        raise UnigramReproError("unexpected Unigram repro schema")
    evidence_sha = report.get("evidence_sha256")
    if not isinstance(evidence_sha, str):
        raise UnigramReproError("missing evidence_sha256")
    body = dict(report)
    body.pop("evidence_sha256")
    if evidence_sha != _sha256_text(_canonical_json(body)):
        raise UnigramReproError("Unigram repro self-hash mismatch")
    source = report.get("source")
    if not isinstance(source, dict):
        raise TypeError("source must be an object")
    if expected_source_sha is not None and source.get("source_sha") != expected_source_sha:
        raise UnigramReproError("Unigram repro source SHA mismatch")
    decision = report.get("decision")
    semantic = report.get("semantic_equivalence")
    root_cause = report.get("root_cause")
    if (
        not isinstance(decision, dict)
        or not isinstance(semantic, dict)
        or not isinstance(root_cause, dict)
    ):
        raise TypeError("decision, semantic_equivalence and root_cause must be objects")
    if decision.get("status") != DECISION_REJECT or decision.get("canonical_use") != "FAIL":
        raise UnigramReproError("observed Unigram drift cannot be promoted to canonical PASS")
    if decision.get("semantic_equivalence_identity_allowed") is not False:
        raise UnigramReproError("semantic-equivalence identity must remain disabled")
    if semantic.get("safe_for_checkpoint_identity") is not False:
        raise UnigramReproError("semantic equivalence is unsafe when token IDs/encodings drift")
    if root_cause.get("supported_seed_control") is not False:
        raise UnigramReproError("pinned public UnigramTrainer seed control is not supported")


def _write_report(path: Path, report: dict[str, object]) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    single = subparsers.add_parser("single")
    single.add_argument("--order", choices=("manifest", "reversed"), required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--source-sha", required=True)
    run.add_argument("--output", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--input", type=Path, required=True)
    validate.add_argument("--source-sha", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "single":
        print(_canonical_json(_single(args.order)))
        return 0
    if args.command == "run":
        report = build_report(args.source_sha)
        validate_report(report, expected_source_sha=args.source_sha)
        _write_report(args.output, report)
        return 0
    report = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise TypeError("Unigram repro report must be a JSON object")
    validate_report(report, expected_source_sha=args.source_sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
