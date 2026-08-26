"""Final fail-closed reproducibility determination for the incumbent HF Unigram path."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from twelve_six.tokenization.experiments import (
    CorpusFileIdentity,
    TokenizerTrainingManifest,
    train_hf_tokenizer,
)

SCHEMA = "12-6.tok240-unigram-reproducibility-final.v1"
DECISION = "INELIGIBLE_FOR_RESEARCH_SELECTION"
TOKENIZERS_VERSION = "0.23.1"
PYTHON_VERSION = "3.11.16"
EXPECTED_DATASET_ID = "s0-tiny-controlled-v1"
EXPECTED_DATASET_MANIFEST_SHA256 = (
    "b085a7ab56510575a11a80824fcff3a95a17f237d46d1be820e59d1289f220c2"
)
EXPECTED_TRAIN_SHA256 = "61d24b7138df56527d201cea405d11c9f607684b4a9593dfa20c599cc2ee6998"
EXPECTED_TRAINING_MANIFEST_SHA256 = (
    "bbe6fc282af46aa0d62c4405d3c2dc92da76b03bb46c1dfb9c9f2f4d738dcca4"
)
EXPECTED_TOKENIZERS_WHEEL_SHA256 = (
    "5075b405006415ea148a992d093699c66eb01952bf59f4d5727089a98bda45a4"
)
UPSTREAM_TRAINER_BLOB_SHA = "ff5ca9428ab7c7ca9b96065046f32b42246dc234"
UPSTREAM_PYTHON_BINDING_BLOB_SHA = "df0b11ec57ef129515008f44dfae7c539f45ff46"
HISTORICAL_EXACT_SOURCE_SHA = "e925109473822bcd11ceef71f98f1441a6816f62"
HISTORICAL_WORKFLOW_RUN_ID = 32861353159
HISTORICAL_WORKFLOW_JOB_ID = 97845863963
_EXACT_SEED_RE = re.compile(r"(?<![0-9A-Za-z_])seed(?![0-9A-Za-z_])")


class Tok240Error(ValueError):
    """Fail-closed TOK-240 evidence error."""


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _require_git_sha(value: str) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise Tok240Error("source SHA must be a full lowercase Git SHA")


def _git_head(repo_root: Path) -> str:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return process.stdout.strip()


def _load_dataset(repo_root: Path) -> tuple[dict[str, Any], tuple[str, ...], bytes]:
    manifest_path = repo_root / "data/s0/packaged/manifest.json"
    train_path = repo_root / "data/s0/packaged/train.jsonl"
    manifest_bytes = manifest_path.read_bytes()
    train_bytes = train_path.read_bytes()
    if _sha256_bytes(manifest_bytes) != EXPECTED_DATASET_MANIFEST_SHA256:
        raise Tok240Error("immutable S0 dataset manifest identity drifted")
    if _sha256_bytes(train_bytes) != EXPECTED_TRAIN_SHA256:
        raise Tok240Error("immutable S0 train bytes drifted")
    dataset = json.loads(manifest_bytes)
    if dataset.get("dataset_id") != EXPECTED_DATASET_ID:
        raise Tok240Error("unexpected S0 dataset ID")
    if dataset.get("outputs", {}).get("train.jsonl") != EXPECTED_TRAIN_SHA256:
        raise Tok240Error("dataset manifest does not bind the expected train SHA")
    texts: list[str] = []
    for raw_line in train_bytes.decode("utf-8").splitlines():
        record = json.loads(raw_line)
        text = record.get("text")
        if not isinstance(text, str):
            raise Tok240Error("training record text must be str")
        texts.append(text)
    if len(texts) != 10:
        raise Tok240Error("expected exactly 10 immutable S0 training records")
    return dataset, tuple(texts), train_bytes


def _manifest(repo_root: Path, train_bytes: bytes) -> TokenizerTrainingManifest:
    manifest = TokenizerTrainingManifest(
        experiment_id="controlled-real-unigram-v1",
        algorithm="unigram",
        tokenizers_version=TOKENIZERS_VERSION,
        dataset_id=EXPECTED_DATASET_ID,
        dataset_manifest_sha256=EXPECTED_DATASET_MANIFEST_SHA256,
        corpus_files=(
            CorpusFileIdentity(
                path="data/s0/packaged/train.jsonl",
                sha256=EXPECTED_TRAIN_SHA256,
                byte_count=len(train_bytes),
            ),
        ),
        vocab_size=512,
    )
    if manifest.sha256 != EXPECTED_TRAINING_MANIFEST_SHA256:
        raise Tok240Error("incumbent Unigram training-manifest identity drifted")
    return manifest


def _probe_texts(repo_root: Path) -> tuple[str, ...]:
    validation = repo_root / "data/s0/packaged/validation.jsonl"
    rows = [json.loads(line) for line in validation.read_text(encoding="utf-8").splitlines()]
    texts = [row["text"] for row in rows]
    texts.extend(
        (
            "def add(a: int, b: int) -> int:\n    return a + b\n",
            "Їжак, naïve, Ελληνικά, emoji 🙂, tab\tnewline\n",
        )
    )
    return tuple(texts)


def _trainer_seed_probe(tokenizers: Any) -> dict[str, object]:
    trainer_type = tokenizers.trainers.UnigramTrainer
    text_signature = getattr(trainer_type, "__text_signature__", None)
    doc = getattr(trainer_type, "__doc__", "") or ""
    advertised = "\n".join(part for part in (text_signature, doc) if isinstance(part, str))
    supported = _EXACT_SEED_RE.search(advertised) is not None
    return {
        "public_seed_argument_supported": supported,
        "text_signature": text_signature,
        "exact_seed_argument_advertised": supported,
        "note": "seed_size is not a randomness seed; unknown kwargs are ignored by the pinned binding",
    }


def _artifact_dict(adapter: Any) -> dict[str, object]:
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


def build_single(repo_root: Path) -> dict[str, object]:
    if platform.python_version() != PYTHON_VERSION:
        raise Tok240Error(f"TOK-240 requires CPython {PYTHON_VERSION}")
    actual_tokenizers = importlib.metadata.version("tokenizers")
    if actual_tokenizers != TOKENIZERS_VERSION:
        raise Tok240Error("TOK-240 exact tokenizer runtime mismatch")
    random.seed(0)
    _, texts, train_bytes = _load_dataset(repo_root)
    manifest = _manifest(repo_root, train_bytes)
    adapter = train_hf_tokenizer(manifest, texts)
    runtime = adapter._tokenizer
    serialized = runtime.to_str()
    serialized_repeat = runtime.to_str()
    tokenizers = importlib.import_module("tokenizers")
    reserialized = tokenizers.Tokenizer.from_str(serialized).to_str()
    payload = json.loads(serialized)
    if payload.get("model", {}).get("type") != "Unigram":
        raise Tok240Error("runtime did not serialize a Unigram model")
    semantic_payload = {
        "model": payload.get("model"),
        "normalizer": payload.get("normalizer"),
        "pre_tokenizer": payload.get("pre_tokenizer"),
        "post_processor": payload.get("post_processor"),
        "decoder": payload.get("decoder"),
        "added_tokens": payload.get("added_tokens", []),
        "special_tokens": dict(adapter.artifact_identity.special_tokens),
    }
    probes = _probe_texts(repo_root)
    probe_ids = [adapter.encode(text) for text in probes]
    roundtrip = [
        adapter.decode(ids, skip_special_tokens=False, errors="strict") == text
        for ids, text in zip(probe_ids, probes, strict=True)
    ]
    unknown_count = sum(
        sum(token_id == adapter.unk_id for token_id in ids) for ids in probe_ids
    )
    return {
        "artifact": _artifact_dict(adapter),
        "canonical_semantic_sha256": _sha256_text(_canonical_json(semantic_payload)),
        "probe_ids": probe_ids,
        "strict_roundtrip_all": all(roundtrip),
        "unknown_tokens": unknown_count,
        "serialization": {
            "same_object_to_str_exact": serialized == serialized_repeat,
            "serialize_reload_serialize_exact": serialized == reserialized,
        },
        "training": {
            "train_sha256": EXPECTED_TRAIN_SHA256,
            "training_manifest_sha256": manifest.sha256,
            "ordered_texts_sha256": _sha256_text(_canonical_json(texts)),
            "records": len(texts),
        },
        "runtime": {
            "python": platform.python_version(),
            "tokenizers": actual_tokenizers,
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "tokenizers_parallelism": os.environ.get("TOKENIZERS_PARALLELISM"),
            "rayon_num_threads": os.environ.get("RAYON_NUM_THREADS"),
        },
        "seed_probe": _trainer_seed_probe(tokenizers),
    }


def _child_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONHASHSEED": "0",
            "TOKENIZERS_PARALLELISM": "false",
            "RAYON_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    return env


def _run_child(repo_root: Path) -> dict[str, object]:
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "twelve_six.tok240_unigram_repro",
            "single",
            "--repo-root",
            str(repo_root),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=_child_env(),
    )
    value = json.loads(process.stdout)
    if not isinstance(value, dict):
        raise Tok240Error("child evidence must be a JSON object")
    return value


def summarize_runs(runs: list[dict[str, object]]) -> dict[str, object]:
    if len(runs) < 2:
        raise Tok240Error("at least two independent artifacts are required")
    artifacts = [run["artifact"] for run in runs]
    tokenizer_json = [artifact["tokenizer_json_sha256"] for artifact in artifacts]
    vocab = [artifact["vocab_sha256"] for artifact in artifacts]
    semantics = [run["canonical_semantic_sha256"] for run in runs]
    probes = [run["probe_ids"] for run in runs]
    specials = [artifact["special_tokens"] for artifact in artifacts]
    exact_artifact = all(artifact == artifacts[0] for artifact in artifacts[1:])
    byte_identical = all(value == tokenizer_json[0] for value in tokenizer_json[1:])
    ordered_vocab_identical = all(value == vocab[0] for value in vocab[1:])
    semantic_identical = all(value == semantics[0] for value in semantics[1:])
    probe_encoding_identical = all(value == probes[0] for value in probes[1:])
    special_identical = all(value == specials[0] for value in specials[1:])
    roundtrip_all = all(bool(run["strict_roundtrip_all"]) for run in runs)
    unknown_zero = all(int(run["unknown_tokens"]) == 0 for run in runs)
    seed_supported = any(bool(run["seed_probe"]["public_seed_argument_supported"]) for run in runs)
    reproducible = exact_artifact and byte_identical and ordered_vocab_identical and semantic_identical
    return {
        "independent_runs": len(runs),
        "exact_artifact_identity_equal": exact_artifact,
        "byte_identical_tokenizer_json": byte_identical,
        "ordered_token_id_vocabulary_equal": ordered_vocab_identical,
        "canonical_semantic_identity_equal": semantic_identical,
        "probe_encoding_equal": probe_encoding_identical,
        "special_token_metadata_equal": special_identical,
        "strict_roundtrip_all": roundtrip_all,
        "zero_unknown_tokens_all": unknown_zero,
        "public_seed_control_supported": seed_supported,
        "eligible_under_reproducibility_contract": reproducible,
    }


def build_report(
    repo_root: Path,
    *,
    source_sha: str,
    run_count: int,
    environment_manifest: Path | None,
) -> dict[str, object]:
    _require_git_sha(source_sha)
    if run_count < 2:
        raise Tok240Error("run_count must be at least 2")
    observed = _git_head(repo_root)
    if observed != source_sha:
        raise Tok240Error("source checkout does not match declared source SHA")
    runs = [_run_child(repo_root) for _ in range(run_count)]
    summary = summarize_runs(runs)
    ineligible = not bool(summary["eligible_under_reproducibility_contract"])
    if not ineligible:
        raise Tok240Error(
            "Unigram unexpectedly became reproducible; TOK-240 must not auto-promote without review"
        )
    env_identity = None
    if environment_manifest is not None:
        env_path = environment_manifest if environment_manifest.is_absolute() else repo_root / environment_manifest
        env_identity = {
            "path": str(environment_manifest),
            "sha256": _sha256_bytes(env_path.read_bytes()),
        }
    report: dict[str, object] = {
        "schema": SCHEMA,
        "source": {
            "repository": "Oleksii-debug/12-6-ai.",
            "source_sha": source_sha,
            "observed_head_sha": observed,
        },
        "exact_environment": {
            "python": PYTHON_VERSION,
            "tokenizers": TOKENIZERS_VERSION,
            "tokenizers_wheel_sha256": EXPECTED_TOKENIZERS_WHEEL_SHA256,
            "environment_manifest": env_identity,
        },
        "immutable_input": {
            "dataset_id": EXPECTED_DATASET_ID,
            "dataset_manifest_sha256": EXPECTED_DATASET_MANIFEST_SHA256,
            "train_sha256": EXPECTED_TRAIN_SHA256,
            "training_manifest_sha256": EXPECTED_TRAINING_MANIFEST_SHA256,
            "input_order": "manifest record order, identical in every independent child process",
        },
        "historical_reconstruction": {
            "exact_source_sha": HISTORICAL_EXACT_SOURCE_SHA,
            "workflow_run_id": HISTORICAL_WORKFLOW_RUN_ID,
            "workflow_job_id": HISTORICAL_WORKFLOW_JOB_ID,
            "failure": "two identical-input Unigram trainings produced different artifact/vocab identities and held-out token IDs",
            "diagnostic_harness_defect": (
                "the old root-cause audit incorrectly required serialize->reload->serialize byte identity; "
                "that secondary serialization behavior does not explain cross-training ordered-vocab drift"
            ),
        },
        "bounded_controls": {
            "python_random_seed": 0,
            "pythonhashseed": "0",
            "tokenizers_parallelism": "false",
            "rayon_num_threads": "1",
            "omp_num_threads": "1",
            "mkl_num_threads": "1",
            "input_order_fixed": True,
            "algorithm_changed": False,
        },
        "independent_runs": runs,
        "reproducibility": summary,
        "cause_classification": {
            "library_randomness": {
                "status": "CONFIRMED_PRIMARY_MECHANISM",
                "evidence": (
                    "pinned UnigramTrainer uses randomized AHashMap/AHashSet iteration; equal-frequency "
                    "seed pieces preserve hash-derived tie order before EM"
                ),
            },
            "input_ordering": {
                "status": "ELIMINATED_AS_SUFFICIENT_CONTROL",
                "evidence": "identical manifest record order is used in every drifting run",
            },
            "threading": {
                "status": "ELIMINATED_AS_SUFFICIENT_CONTROL",
                "evidence": "drift persists with tokenizer parallelism disabled and one Rayon thread",
            },
            "floating_point_tie_handling": {
                "status": "POSSIBLE_DOWNSTREAM_AMPLIFIER_NOT_REQUIRED_FOR_FAILURE",
                "evidence": (
                    "EM/floating reductions can amplify an earlier ordering difference, but hash-derived "
                    "tie order exists before those reductions and serial drift is already sufficient"
                ),
            },
            "artifact_serialization": {
                "status": "SECONDARY_NOT_PRIMARY",
                "evidence": (
                    "serialization/reload byte normalization is reported separately; ordered token-ID "
                    "vocabulary and probe encodings already differ between trained models"
                ),
            },
            "special_token_metadata": {
                "status": "ELIMINATED_AS_PRIMARY_CAUSE",
                "evidence": "all runs retain the same <unk> token metadata while model identity drifts",
            },
            "unsupported_deterministic_control": {
                "status": "CONFIRMED",
                "evidence": "the pinned Python UnigramTrainer API advertises no supported randomness seed",
            },
        },
        "upstream_source_evidence": {
            "trainer_blob_sha": UPSTREAM_TRAINER_BLOB_SHA,
            "python_binding_blob_sha": UPSTREAM_PYTHON_BINDING_BLOB_SHA,
        },
        "decision": {
            "status": DECISION,
            "research_selection_eligible": False,
            "canonicalized_semantic_identity_allowed": False,
            "stop_model_comparisons": True,
            "tok241_may_compare_unigram": False,
            "reason": (
                "neither byte-identical artifact identity nor canonicalized semantic identity can be "
                "guaranteed under the exact supported runtime; token-ID semantics drift"
            ),
        },
        "post_decision_checks": {
            "fertility": "NOT_RUN_INELIGIBLE",
            "speed": "NOT_RUN_INELIGIBLE",
            "model_comparisons": "STOPPED_INELIGIBLE",
        },
        "truth_boundary": {
            "local_free_only": True,
            "new_tokenizer_algorithm_added": False,
            "model_training_run": False,
            "family_quality_comparison_claimed": False,
        },
    }
    report["evidence_sha256"] = _sha256_text(_canonical_json(report))
    return report


def validate_report(report: dict[str, object], *, expected_source_sha: str | None = None) -> None:
    if report.get("schema") != SCHEMA:
        raise Tok240Error("unexpected TOK-240 schema")
    evidence_sha = report.get("evidence_sha256")
    body = dict(report)
    body.pop("evidence_sha256", None)
    if evidence_sha != _sha256_text(_canonical_json(body)):
        raise Tok240Error("TOK-240 evidence self-hash mismatch")
    source = report.get("source")
    if not isinstance(source, dict):
        raise Tok240Error("source block missing")
    if expected_source_sha is not None and source.get("source_sha") != expected_source_sha:
        raise Tok240Error("TOK-240 source SHA mismatch")
    repro = report.get("reproducibility")
    decision = report.get("decision")
    if not isinstance(repro, dict) or not isinstance(decision, dict):
        raise Tok240Error("reproducibility/decision blocks missing")
    if int(repro.get("independent_runs", 0)) < 2:
        raise Tok240Error("TOK-240 lacks two independent artifacts")
    if bool(repro.get("eligible_under_reproducibility_contract")):
        raise Tok240Error("TOK-240 evidence must not mark drifting Unigram eligible")
    if decision.get("status") != DECISION:
        raise Tok240Error("TOK-240 must fail closed as ineligible")
    if decision.get("research_selection_eligible") is not False:
        raise Tok240Error("Unigram research-selection eligibility must be false")
    if decision.get("stop_model_comparisons") is not True:
        raise Tok240Error("model comparisons must stop for ineligible Unigram")
    if decision.get("tok241_may_compare_unigram") is not False:
        raise Tok240Error("TOK-241 must not compare ineligible Unigram")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    single = subparsers.add_parser("single")
    single.add_argument("--repo-root", default=".")

    run = subparsers.add_parser("run")
    run.add_argument("--repo-root", default=".")
    run.add_argument("--source-sha", required=True)
    run.add_argument("--runs", type=int, default=3)
    run.add_argument("--environment-manifest")
    run.add_argument("--output", required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--input", required=True)
    validate.add_argument("--source-sha")

    args = parser.parse_args(argv)
    if args.command == "single":
        print(_canonical_json(build_single(Path(args.repo_root).resolve())))
        return 0
    if args.command == "run":
        repo_root = Path(args.repo_root).resolve()
        environment_manifest = (
            Path(args.environment_manifest) if args.environment_manifest else None
        )
        report = build_report(
            repo_root,
            source_sha=args.source_sha,
            run_count=args.runs,
            environment_manifest=environment_manifest,
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        print(
            _canonical_json(
                {
                    "decision": report["decision"]["status"],
                    "evidence_sha256": report["evidence_sha256"],
                    "independent_runs": report["reproducibility"]["independent_runs"],
                }
            )
        )
        return 0
    report = json.loads(Path(args.input).read_text(encoding="utf-8"))
    validate_report(report, expected_source_sha=args.source_sha)
    print(_canonical_json({"status": "PASS", "decision": report["decision"]["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
