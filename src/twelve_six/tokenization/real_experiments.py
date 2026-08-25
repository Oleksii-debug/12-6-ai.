"""Run and validate real maintained-library tokenizer experiments on the controlled train split."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from twelve_six.tokenization.experiments import (
    CorpusFileIdentity,
    TokenizerProbe,
    TokenizerTrainingManifest,
    measure_probe,
    summarize_by_language,
    train_hf_tokenizer,
    vocabulary_parameter_cost,
)

SCHEMA = "12-6.tokenizer-real-comparison.v2"
AUTHORITY = "CONTROLLED_S0_TRAIN_SPLIT_MECHANICS_ONLY_NOT_S1_CORPUS_OR_FREEZE"
TOKENIZERS_VERSION = "0.23.1"
TOKENIZERS_WHEEL_SHA256 = (
    "5075b405006415ea148a992d093699c66eb01952bf59f4d5727089a98bda45a4"
)
LOCK_PATH = Path("requirements/experiments/tokenizers-linux-x86_64.lock.txt")
DATASET_MANIFEST_PATH = Path("data/s0/packaged/manifest.json")
TRAIN_PATH = Path("data/s0/packaged/train.jsonl")
VALIDATION_PATH = Path("data/s0/packaged/validation.jsonl")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class TokenizerEvidenceError(ValueError):
    """Fail-closed tokenizer experiment evidence error."""


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be str")
    if _SHA256_RE.fullmatch(value) is None:
        raise TokenizerEvidenceError(f"{field} must be lowercase SHA-256")
    return value


def _require_git_sha(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be str")
    if _GIT_SHA_RE.fullmatch(value) is None:
        raise TokenizerEvidenceError(f"{field} must be full lowercase Git SHA")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"{path}:{line_number} must contain a JSON object")
        records.append(value)
    if not records:
        raise TokenizerEvidenceError(f"{path} must not be empty")
    return records


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, encoding="utf-8"
    ).strip()


def verify_experiment_lock(path: Path = LOCK_PATH) -> dict[str, object]:
    """Require one exact package/version/hash and no floating experiment dependency."""
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    expected = (
        f"tokenizers=={TOKENIZERS_VERSION} "
        f"--hash=sha256:{TOKENIZERS_WHEEL_SHA256}"
    )
    if lines != [expected]:
        raise TokenizerEvidenceError("experiment lock must contain exactly the admitted wheel hash")
    return {
        "path": path.as_posix(),
        "sha256": _sha256_file(path),
        "package": "tokenizers",
        "version": TOKENIZERS_VERSION,
        "wheel_sha256": TOKENIZERS_WHEEL_SHA256,
        "install_policy": "pip --require-hashes --only-binary=:all: --no-deps",
    }


def _record_ids(records: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for record in records:
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id:
            raise TokenizerEvidenceError("every packaged record must have a non-empty string id")
        ids.append(record_id)
    if len(ids) != len(set(ids)):
        raise TokenizerEvidenceError("record ids must be unique within a split")
    return ids


def _dataset_contract() -> dict[str, Any]:
    manifest = _load_json(DATASET_MANIFEST_PATH)
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise TokenizerEvidenceError("dataset manifest outputs must be an object")
    train_sha = _require_sha256(outputs.get(TRAIN_PATH.name), "train output sha256")
    validation_sha = _require_sha256(
        outputs.get(VALIDATION_PATH.name), "validation output sha256"
    )
    if _sha256_file(TRAIN_PATH) != train_sha:
        raise TokenizerEvidenceError("train split does not match dataset manifest")
    if _sha256_file(VALIDATION_PATH) != validation_sha:
        raise TokenizerEvidenceError("validation split does not match dataset manifest")

    dataset_id = manifest.get("dataset_id")
    dataset_identity = _require_sha256(
        manifest.get("dataset_identity_sha256"), "dataset_identity_sha256"
    )
    if not isinstance(dataset_id, str) or not dataset_id:
        raise TokenizerEvidenceError("dataset_id must be a non-empty string")

    train_records = _load_jsonl(TRAIN_PATH)
    validation_records = _load_jsonl(VALIDATION_PATH)
    train_ids = _record_ids(train_records)
    validation_ids = _record_ids(validation_records)
    if set(train_ids) & set(validation_ids):
        raise TokenizerEvidenceError("train and validation record ids must be disjoint")

    assignments = manifest.get("document_assignments")
    if not isinstance(assignments, list):
        raise TokenizerEvidenceError("document_assignments must be a list")
    expected_by_split: dict[str, list[str]] = {"train": [], "validation": []}
    for assignment in assignments:
        if not isinstance(assignment, dict):
            raise TypeError("document assignment must be an object")
        split = assignment.get("split")
        record_id = assignment.get("id")
        if split not in expected_by_split or not isinstance(record_id, str):
            raise TokenizerEvidenceError("document assignment has unsupported split or id")
        expected_by_split[split].append(record_id)
    if train_ids != expected_by_split["train"]:
        raise TokenizerEvidenceError("train JSONL order/ids do not match manifest assignments")
    if validation_ids != expected_by_split["validation"]:
        raise TokenizerEvidenceError("validation JSONL order/ids do not match manifest assignments")

    return {
        "dataset_id": dataset_id,
        "dataset_identity_sha256": dataset_identity,
        "dataset_manifest_sha256": _sha256_file(DATASET_MANIFEST_PATH),
        "train_path": TRAIN_PATH.as_posix(),
        "train_sha256": train_sha,
        "validation_path": VALIDATION_PATH.as_posix(),
        "validation_sha256": validation_sha,
        "train_records": train_records,
        "validation_records": validation_records,
        "train_record_ids": train_ids,
        "validation_record_ids": validation_ids,
        "train_validation_overlap": [],
    }


def _training_manifest(
    algorithm: str,
    dataset: dict[str, Any],
    *,
    vocab_size: int,
) -> TokenizerTrainingManifest:
    train_bytes = TRAIN_PATH.read_bytes()
    return TokenizerTrainingManifest(
        experiment_id=f"controlled-real-{algorithm}-v1",
        algorithm=algorithm,
        tokenizers_version=TOKENIZERS_VERSION,
        dataset_id=dataset["dataset_id"],
        dataset_manifest_sha256=dataset["dataset_manifest_sha256"],
        corpus_files=(
            CorpusFileIdentity(
                TRAIN_PATH.as_posix(),
                dataset["train_sha256"],
                len(train_bytes),
            ),
        ),
        vocab_size=vocab_size,
        min_frequency=2 if algorithm == "bpe" else None,
    )


def _held_out_probes(dataset: dict[str, Any]) -> list[TokenizerProbe]:
    probes: list[TokenizerProbe] = []
    for record in dataset["validation_records"]:
        text = record.get("text")
        record_id = record.get("id")
        language = record.get("language")
        if not all(isinstance(value, str) for value in (text, record_id, language)):
            raise TokenizerEvidenceError("validation record probe fields must be strings")
        probes.append(
            TokenizerProbe(
                name=record_id,
                language=language,
                category="held-out-d03-validation",
                text=text,
            )
        )
    probes.extend(
        [
            TokenizerProbe(
                name="project-authored-code-probe",
                language="code",
                category="project-authored-held-out-probe",
                text="def add(a: int, b: int) -> int:\n    return a + b\n",
            ),
            TokenizerProbe(
                name="project-authored-unicode-probe",
                language="multi",
                category="project-authored-held-out-probe",
                text="Україна 🇺🇦 — naïve café; 数学; مرحبا; é",
            ),
        ]
    )
    return probes


def _probe_to_dict(result: object) -> dict[str, object]:
    fields = (
        "name",
        "language",
        "category",
        "codepoints",
        "utf8_bytes",
        "tokens",
        "fertility_tokens_per_codepoint",
        "tokens_per_utf8_byte",
        "round_trip_exact",
        "unknown_tokens",
    )
    return {field: getattr(result, field) for field in fields}


def _artifact_to_dict(adapter: object) -> dict[str, object]:
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


def _run_algorithm(
    algorithm: str,
    dataset: dict[str, Any],
    *,
    vocab_size: int,
) -> dict[str, Any]:
    manifest = _training_manifest(algorithm, dataset, vocab_size=vocab_size)
    train_texts = [str(record["text"]) for record in dataset["train_records"]]
    first = train_hf_tokenizer(manifest, train_texts)
    second = train_hf_tokenizer(manifest, train_texts)
    first_artifact = _artifact_to_dict(first)
    second_artifact = _artifact_to_dict(second)
    identity_equal = first_artifact == second_artifact
    drift_fields = sorted(
        field
        for field in first_artifact
        if first_artifact[field] != second_artifact[field]
    )

    probes = _held_out_probes(dataset)
    first_results = [
        measure_probe(first, probe, unknown_token_id=first.unk_id) for probe in probes
    ]
    second_results = [
        measure_probe(second, probe, unknown_token_id=second.unk_id) for probe in probes
    ]
    all_results = first_results + second_results
    if not all(result.round_trip_exact for result in all_results):
        raise TokenizerEvidenceError(f"{algorithm} failed held-out strict round trip")
    if any(result.unknown_tokens for result in all_results):
        raise TokenizerEvidenceError(f"{algorithm} emitted unknown tokens on held-out probes")

    first_probe_ids = [first.encode(probe.text) for probe in probes]
    second_probe_ids = [second.encode(probe.text) for probe in probes]
    probe_encoding_equal = first_probe_ids == second_probe_ids

    byte_baseline = sum(len(probe.text.encode("utf-8")) for probe in probes)
    first_tokens = sum(result.tokens for result in first_results)
    second_tokens = sum(result.tokens for result in second_results)
    cost = vocabulary_parameter_cost(
        vocab_size=first.vocab_size,
        d_model=48,
        tied_lm_head=True,
    )
    return {
        "status": "PASS",
        "algorithm": algorithm,
        "training_manifest": manifest.to_dict(),
        "training_manifest_sha256": manifest.sha256,
        "training_input": {
            "split": "train",
            "records": len(dataset["train_records"]),
            "record_ids": dataset["train_record_ids"],
            "validation_used_for_training": False,
        },
        "artifact": first_artifact,
        "repeat_build_artifact": second_artifact,
        "repeated_build_identity_equal": identity_equal,
        "repeatability_status": "PASS" if identity_equal else "FAIL",
        "artifact_drift_fields": drift_fields,
        "held_out_probe_encoding_equal": probe_encoding_equal,
        "held_out": {
            "probes": [_probe_to_dict(result) for result in first_results],
            "repeat_build_probes": [_probe_to_dict(result) for result in second_results],
            "strict_round_trip_all": True,
            "unknown_tokens": 0,
            "tokens": first_tokens,
            "repeat_build_tokens": second_tokens,
            "byte_baseline_tokens": byte_baseline,
            "token_reduction_vs_bytes": (
                (byte_baseline - first_tokens) / byte_baseline if byte_baseline else 0.0
            ),
            "repeat_build_token_reduction_vs_bytes": (
                (byte_baseline - second_tokens) / byte_baseline if byte_baseline else 0.0
            ),
            "language_summary": summarize_by_language(first_results),
            "repeat_build_language_summary": summarize_by_language(second_results),
        },
        "non_frozen_s1_parameter_cost": {
            "d_model": 48,
            "tied_lm_head": True,
            "vocab_size": first.vocab_size,
            "total_vocabulary_parameters": cost.total_vocabulary_parameters,
        },
        "locked_for_s1": False,
    }


def build_report(*, source_sha: str, vocab_size: int = 512) -> dict[str, Any]:
    source_sha = _require_git_sha(source_sha, "source_sha")
    if vocab_size < 257:
        raise TokenizerEvidenceError("vocab_size must cover byte alphabet plus <unk>")
    observed_head = _git_head()
    if observed_head != source_sha:
        raise TokenizerEvidenceError("checkout HEAD does not match requested source_sha")

    lock = verify_experiment_lock()
    actual_runtime_version = importlib.metadata.version("tokenizers")
    if actual_runtime_version != TOKENIZERS_VERSION:
        raise TokenizerEvidenceError("installed tokenizers version does not match experiment lock")

    dataset = _dataset_contract()
    bpe = _run_algorithm("bpe", dataset, vocab_size=vocab_size)
    unigram = _run_algorithm("unigram", dataset, vocab_size=vocab_size)
    if bpe["training_input"]["record_ids"] != unigram["training_input"]["record_ids"]:
        raise TokenizerEvidenceError("BPE and Unigram did not use the same train records")

    repeatable = (
        bpe["repeated_build_identity_equal"] is True
        and unigram["repeated_build_identity_equal"] is True
    )
    repeatability_gate = "PASS" if repeatable else "FAIL"
    decision_status = (
        "NO_FREEZE_CONTROLLED_MECHANICS_ONLY"
        if repeatable
        else "NO_FREEZE_REPEATABILITY_BLOCKED"
    )
    if repeatable:
        decision_reason = (
            "real maintained-library mechanics are exercised only on the controlled S0 train "
            "fixture; representative S1 corpus and rights/decontamination evidence are absent"
        )
    else:
        decision_reason = (
            "at least one maintained-library algorithm produced different exact artifact "
            "identity across identical repeated builds; tokenizer freeze is blocked even before "
            "representative S1 corpus and model-quality comparisons"
        )

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "source": {
            "repository": "Oleksii-debug/12-6-ai.",
            "source_sha": source_sha,
            "observed_head_sha": observed_head,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "package": "tokenizers",
            "version": actual_runtime_version,
            "experiment_lock": lock,
        },
        "dataset": {
            "dataset_id": dataset["dataset_id"],
            "dataset_identity_sha256": dataset["dataset_identity_sha256"],
            "dataset_manifest_sha256": dataset["dataset_manifest_sha256"],
            "train_path": dataset["train_path"],
            "train_sha256": dataset["train_sha256"],
            "validation_path": dataset["validation_path"],
            "validation_sha256": dataset["validation_sha256"],
            "train_records": len(dataset["train_records"]),
            "validation_records": len(dataset["validation_records"]),
            "train_validation_record_overlap": dataset["train_validation_overlap"],
            "representative_s1_corpus": False,
            "external_sources_training_approved": False,
        },
        "requested_vocab_size": vocab_size,
        "algorithms": {"bpe": bpe, "unigram": unigram},
        "gates": {
            "exact_source_binding": "PASS",
            "hash_locked_experiment_runtime": "PASS",
            "same_train_corpus_for_algorithms": "PASS",
            "train_validation_separation": "PASS",
            "real_bpe_execution": "PASS",
            "real_unigram_execution": "PASS",
            "bpe_repeatable_artifact_identity": bpe["repeatability_status"],
            "unigram_repeatable_artifact_identity": unigram["repeatability_status"],
            "repeatable_artifact_identity": repeatability_gate,
            "held_out_strict_round_trip": "PASS",
            "held_out_zero_unknown_tokens": "PASS",
            "representative_s1_corpus": "NOT_TESTED",
            "external_source_rights_approval": "NOT_TESTED",
            "s1_tokenizer_freeze": "NOT_TESTED",
            "model_quality": "NOT_TESTED",
        },
        "decision": {
            "status": decision_status,
            "winner": None,
            "reason": decision_reason,
        },
        "truth_boundary": {
            "canonical_s0_tokenizer_unchanged": True,
            "s1_tokenizer_frozen": False,
            "external_sources_training_approved": False,
            "representative_corpus_claimed": False,
            "model_quality_claimed": False,
            "paid_compute_used": False,
            "foreign_pretrained_weights_used": False,
            "audit_pass_claimed": False,
            "candidate_or_stable_claimed": False,
        },
    }
    report["evidence_sha256"] = _sha256_bytes(_canonical_json(report).encode())
    validate_report(report, expected_source_sha=source_sha)
    return report


def _require_gate(report: dict[str, Any], name: str, expected: str) -> None:
    gates = report.get("gates")
    if not isinstance(gates, dict) or gates.get(name) != expected:
        raise TokenizerEvidenceError(f"gate {name} must be {expected}")


def validate_report(
    report: dict[str, Any],
    *,
    expected_source_sha: str | None = None,
) -> dict[str, Any]:
    if report.get("schema") != SCHEMA or report.get("authority") != AUTHORITY:
        raise TokenizerEvidenceError("unexpected tokenizer evidence schema or authority")
    source = report.get("source")
    if not isinstance(source, dict):
        raise TokenizerEvidenceError("source must be an object")
    source_sha = _require_git_sha(source.get("source_sha"), "source.source_sha")
    if source.get("observed_head_sha") != source_sha:
        raise TokenizerEvidenceError("observed HEAD/source binding failed")
    if expected_source_sha is not None and source_sha != _require_git_sha(
        expected_source_sha, "expected_source_sha"
    ):
        raise TokenizerEvidenceError("evidence source SHA is stale")

    claimed_hash = _require_sha256(report.get("evidence_sha256"), "evidence_sha256")
    payload = dict(report)
    del payload["evidence_sha256"]
    actual_hash = _sha256_bytes(_canonical_json(payload).encode())
    if actual_hash != claimed_hash:
        raise TokenizerEvidenceError("evidence self-hash mismatch")

    dataset = report.get("dataset")
    if not isinstance(dataset, dict):
        raise TokenizerEvidenceError("dataset must be an object")
    if dataset.get("train_validation_record_overlap") != []:
        raise TokenizerEvidenceError("train/validation record overlap must be empty")
    if dataset.get("representative_s1_corpus") is not False:
        raise TokenizerEvidenceError(
            "controlled fixture cannot be labeled representative S1 corpus"
        )
    if dataset.get("external_sources_training_approved") is not False:
        raise TokenizerEvidenceError("this evidence cannot approve external training sources")

    algorithms = report.get("algorithms")
    if not isinstance(algorithms, dict) or set(algorithms) != {"bpe", "unigram"}:
        raise TokenizerEvidenceError("evidence must contain exactly BPE and Unigram results")
    training_ids: list[list[str]] = []
    repeatability: dict[str, bool] = {}
    for name in ("bpe", "unigram"):
        algorithm = algorithms[name]
        if not isinstance(algorithm, dict) or algorithm.get("status") != "PASS":
            raise TokenizerEvidenceError(f"{name} execution must PASS")
        training_input = algorithm.get("training_input")
        if not isinstance(training_input, dict):
            raise TokenizerEvidenceError(f"{name} training input must be an object")
        if training_input.get("split") != "train":
            raise TokenizerEvidenceError(f"{name} must train only on train split")
        if training_input.get("validation_used_for_training") is not False:
            raise TokenizerEvidenceError(f"{name} must not train on validation")
        ids = training_input.get("record_ids")
        if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
            raise TokenizerEvidenceError(f"{name} training record ids must be strings")
        training_ids.append(ids)

        identity_equal = algorithm.get("repeated_build_identity_equal")
        if not isinstance(identity_equal, bool):
            raise TokenizerEvidenceError(f"{name} repeatability must be boolean")
        expected_repeatability = "PASS" if identity_equal else "FAIL"
        if algorithm.get("repeatability_status") != expected_repeatability:
            raise TokenizerEvidenceError(f"{name} repeatability status is inconsistent")
        drift_fields = algorithm.get("artifact_drift_fields")
        if not isinstance(drift_fields, list) or not all(
            isinstance(field, str) for field in drift_fields
        ):
            raise TokenizerEvidenceError(f"{name} artifact drift fields must be strings")
        if identity_equal and drift_fields:
            raise TokenizerEvidenceError(f"{name} repeatable artifact cannot report drift fields")
        if not identity_equal and not drift_fields:
            raise TokenizerEvidenceError(f"{name} drift must identify changed artifact fields")
        repeatability[name] = identity_equal

        held_out = algorithm.get("held_out")
        if not isinstance(held_out, dict):
            raise TokenizerEvidenceError(f"{name} held-out evidence must be an object")
        if held_out.get("strict_round_trip_all") is not True:
            raise TokenizerEvidenceError(f"{name} held-out strict round trip must PASS")
        if held_out.get("unknown_tokens") != 0:
            raise TokenizerEvidenceError(f"{name} held-out unknown tokens must be zero")
        if algorithm.get("locked_for_s1") is not False:
            raise TokenizerEvidenceError(f"{name} cannot be locked for S1 by this evidence")
    if training_ids[0] != training_ids[1]:
        raise TokenizerEvidenceError("BPE and Unigram training record identities differ")

    for gate in (
        "exact_source_binding",
        "hash_locked_experiment_runtime",
        "same_train_corpus_for_algorithms",
        "train_validation_separation",
        "real_bpe_execution",
        "real_unigram_execution",
        "held_out_strict_round_trip",
        "held_out_zero_unknown_tokens",
    ):
        _require_gate(report, gate, "PASS")

    for name in ("bpe", "unigram"):
        expected = "PASS" if repeatability[name] else "FAIL"
        _require_gate(report, f"{name}_repeatable_artifact_identity", expected)
    expected_global_repeat = "PASS" if all(repeatability.values()) else "FAIL"
    _require_gate(report, "repeatable_artifact_identity", expected_global_repeat)

    for gate in (
        "representative_s1_corpus",
        "external_source_rights_approval",
        "s1_tokenizer_freeze",
        "model_quality",
    ):
        _require_gate(report, gate, "NOT_TESTED")

    decision = report.get("decision")
    if not isinstance(decision, dict):
        raise TokenizerEvidenceError("decision must be an object")
    expected_decision = (
        "NO_FREEZE_CONTROLLED_MECHANICS_ONLY"
        if all(repeatability.values())
        else "NO_FREEZE_REPEATABILITY_BLOCKED"
    )
    if decision.get("status") != expected_decision:
        raise TokenizerEvidenceError("decision status does not match observed repeatability")
    if decision.get("winner") is not None:
        raise TokenizerEvidenceError(
            "controlled mechanics evidence cannot choose a tokenizer winner"
        )

    boundary = report.get("truth_boundary")
    if not isinstance(boundary, dict):
        raise TokenizerEvidenceError("truth_boundary must be an object")
    required_true = {"canonical_s0_tokenizer_unchanged"}
    required_false = {
        "s1_tokenizer_frozen",
        "external_sources_training_approved",
        "representative_corpus_claimed",
        "model_quality_claimed",
        "paid_compute_used",
        "foreign_pretrained_weights_used",
        "audit_pass_claimed",
        "candidate_or_stable_claimed",
    }
    if any(boundary.get(field) is not True for field in required_true):
        raise TokenizerEvidenceError("canonical S0 tokenizer must remain unchanged")
    if any(boundary.get(field) is not False for field in required_false):
        raise TokenizerEvidenceError("truth boundary contains an unauthorized positive claim")
    return report


def _write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _command_run(args: argparse.Namespace) -> int:
    report = build_report(source_sha=args.source_sha, vocab_size=args.vocab_size)
    _write_report(report, args.output)
    print(json.dumps(report, sort_keys=True))
    return 0


def _command_validate(args: argparse.Namespace) -> int:
    report = _load_json(args.input)
    validate_report(report, expected_source_sha=args.source_sha)
    print(
        "tokenizer_evidence=pass "
        f"decision={report['decision']['status']} sha256={report['evidence_sha256']}"
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run real controlled tokenizer comparison")
    run.add_argument("--source-sha", required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--vocab-size", type=int, default=512)
    run.set_defaults(func=_command_run)
    validate = subparsers.add_parser("validate", help="validate existing tokenizer comparison")
    validate.add_argument("--input", type=Path, required=True)
    validate.add_argument("--source-sha")
    validate.set_defaults(func=_command_validate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (TokenizerEvidenceError, TypeError, OSError, json.JSONDecodeError) as exc:
        print(f"tokenizer_evidence=fail error={exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
