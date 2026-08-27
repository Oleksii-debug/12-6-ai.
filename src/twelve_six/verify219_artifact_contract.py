"""Fail-closed structural preflight for the immutable LEARN-191 artifact.

This module deliberately performs no model training or checkpoint writes.  It
binds producer metadata that the scientific verifier must not silently inherit:
InitSpec, tokenizer, selection subset identity, budget/exposure and final
fresh-load counters.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from twelve_six import milestone100_first_learned as m100
from twelve_six.checkpoint import hash_json
from twelve_six.model import InitSpec, ModelSpec
from twelve_six.tokenization import ByteTokenizer

PRODUCER_SHA = "a75920cef8bde37a8c590e34095be83c97b75f1d"
EXPECTED_MODEL_SPEC_SHA256 = (
    "462c85da80a3c0d7d6a4f1a570b87d208b1847d8a57b12a4d9be7e36846b65dc"
)
EXPECTED_INIT_SPEC_SHA256 = (
    "86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5"
)
EXPECTED_PARAMETER_COUNT = 3_213_120
EXPECTED_CORPUS_ID = "422f545da93526ceba2c0ff738e0b9deac65f35bfb31d87d2aab068cda091fc8"
EXPECTED_M150_EVAL_ID = (
    "7189e6df053574beb686727c94e684cdbaf08a34ef33aa953eff7cdae0320113"
)
EXPECTED_TARGETS = [16_632, 65_772, 131_292]
EXPECTED_MIDPOINT = 65_772
SELECTION_LIMITS = {"uk": 256, "en": 192, "code": 128}


class Verify219ArtifactContractError(RuntimeError):
    """Raised when immutable producer metadata violates the verifier contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Verify219ArtifactContractError(message)


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Verify219ArtifactContractError(f"cannot read {path}: {exc}") from exc
    _require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def _verify_self_hash(value: Mapping[str, Any], *, label: str) -> str:
    supplied = value.get("identity_sha256")
    _require(
        isinstance(supplied, str) and len(supplied) == 64,
        f"{label} identity_sha256 missing",
    )
    unsigned = dict(value)
    unsigned.pop("identity_sha256", None)
    _require(hash_json(unsigned) == supplied, f"{label} self-hash mismatch")
    return supplied


def selection_identity(tokenizer: ByteTokenizer) -> dict[str, Any]:
    """Reconstruct LEARN-191's preregistered selection-validation identity."""

    value: dict[str, Any] = {
        "schema": "12-6.learn191-selection-validation.v1",
        "corpus_identity_sha256": EXPECTED_CORPUS_ID,
        "split": "validation",
        "packing_version": m100.PACKING_VERSION,
        "sequence_length": 128,
        "cross_document": False,
        "ordered_strata": ["uk", "en", "code"],
        "packed_example_limits": dict(SELECTION_LIMITS),
        "selection_rule": (
            "first-N deterministic document-isolated packed examples per stratum"
        ),
        "tokenizer_config_sha256": tokenizer.identity.config_sha256,
        "tokenizer_vocab_sha256": tokenizer.identity.vocab_sha256,
        "purpose": "checkpoint selection only; not final test",
    }
    value["identity_sha256"] = hash_json(value)
    return value


def validate_artifact_contract(artifact_root: Path) -> dict[str, Any]:
    root = artifact_root.resolve() / "learn191-evidence"
    report = _read(root / "learn191-real-3m-report.json")
    run = _read(root / "run-manifest.json")
    truth = _read(root / "truth.json")
    phase1 = _read(root / "3m" / "phase1.json")
    resume = _read(root / "3m" / "resume.json")
    fresh = _read(root / "3m" / "final-fresh-load-proof.json")
    best = _read(root / "3m" / "best-checkpoint.json")
    final = _read(root / "3m" / "final-checkpoint.json")

    hashes = {
        "report": _verify_self_hash(report, label="report"),
        "run_manifest": _verify_self_hash(run, label="run manifest"),
        "truth": _verify_self_hash(truth, label="truth"),
        "phase1": _verify_self_hash(phase1, label="phase1"),
        "resume": _verify_self_hash(resume, label="resume"),
        "fresh": _verify_self_hash(fresh, label="final fresh load"),
    }

    for label, value in (
        ("report", report.get("source_sha")),
        ("run", run.get("source_sha")),
        ("truth", truth.get("source_sha")),
        ("phase1", phase1.get("source_sha")),
        ("resume", resume.get("source_sha")),
        ("fresh", fresh.get("source_sha")),
    ):
        _require(value == PRODUCER_SHA, f"{label} source SHA mismatch")

    _require(report.get("worker_id") == "LEARN-191-REAL-3M", "producer worker mismatch")
    _require(run.get("worker_id") == "LEARN-191-REAL-3M", "run worker mismatch")
    _require(truth.get("worker_id") == "LEARN-191-REAL-3M", "truth worker mismatch")

    model = report.get("model")
    _require(isinstance(model, Mapping), "report model missing")
    _require(
        model.get("spec_sha256") == EXPECTED_MODEL_SPEC_SHA256,
        "report ModelSpec hash mismatch",
    )
    _require(
        int(model.get("parameter_count", -1)) == EXPECTED_PARAMETER_COUNT,
        "report parameter count mismatch",
    )
    _require(
        model.get("init_spec_sha256") == EXPECTED_INIT_SPEC_SHA256,
        "report InitSpec hash mismatch",
    )
    _require(
        ModelSpec.from_dict(dict(model["spec"])).identity_sha256()
        == EXPECTED_MODEL_SPEC_SHA256,
        "report ModelSpec semantic identity mismatch",
    )
    _require(
        InitSpec.from_dict(dict(model["init_spec"])).identity_sha256()
        == EXPECTED_INIT_SPEC_SHA256,
        "report InitSpec semantic identity mismatch",
    )
    _require(
        run.get("model_spec_sha256") == EXPECTED_MODEL_SPEC_SHA256,
        "run ModelSpec hash mismatch",
    )
    _require(
        run.get("init_spec_sha256") == EXPECTED_INIT_SPEC_SHA256,
        "run InitSpec hash mismatch",
    )
    _require(
        int(run.get("parameter_count", -1)) == EXPECTED_PARAMETER_COUNT,
        "run parameter count mismatch",
    )

    tokenizer = ByteTokenizer()
    tokenizer_record = run.get("tokenizer")
    _require(isinstance(tokenizer_record, Mapping), "run tokenizer record missing")
    _require(tokenizer_record.get("version") == tokenizer.identity.version, "tokenizer version drift")
    _require(
        tokenizer_record.get("config_sha256") == tokenizer.identity.config_sha256,
        "tokenizer config drift",
    )
    _require(
        tokenizer_record.get("vocab_sha256") == tokenizer.identity.vocab_sha256,
        "tokenizer vocabulary drift",
    )
    _require(
        report.get("tokenizer") == tokenizer_record,
        "report/run tokenizer records disagree",
    )

    _require(run.get("corpus_identity_sha256") == EXPECTED_CORPUS_ID, "run corpus drift")
    _require(
        report.get("corpus_identity_sha256") == EXPECTED_CORPUS_ID,
        "report corpus drift",
    )
    _require(
        truth.get("corpus_identity_sha256") == EXPECTED_CORPUS_ID,
        "truth corpus drift",
    )
    _require(
        run.get("retained_m150_evaluation_identity_sha256") == EXPECTED_M150_EVAL_ID,
        "run M150 evaluation identity drift",
    )
    _require(
        report.get("retained_m150_evaluation_identity_sha256") == EXPECTED_M150_EVAL_ID,
        "report M150 evaluation identity drift",
    )

    expected_selection = selection_identity(tokenizer)
    run_selection = run.get("selection_validation")
    report_selection = report.get("selection_validation")
    _require(isinstance(run_selection, Mapping), "run selection identity missing")
    _require(isinstance(report_selection, Mapping), "report selection identity missing")
    _require(
        dict(run_selection) == expected_selection,
        "run selection-validation identity is not independently reproducible",
    )
    _require(
        dict(report_selection) == expected_selection,
        "report selection-validation identity is not independently reproducible",
    )
    _require(
        truth.get("selection_validation_identity_sha256")
        == expected_selection["identity_sha256"],
        "truth selection-validation identity mismatch",
    )

    _require(run.get("optimized_token_targets") == EXPECTED_TARGETS, "run target budget drift")
    _require(truth.get("optimized_token_targets") == EXPECTED_TARGETS, "truth target budget drift")
    _require(
        int(run.get("fresh_process_resume_target", -1)) == EXPECTED_MIDPOINT,
        "run midpoint target drift",
    )
    _require(
        int(truth.get("midpoint_resume_target", -1)) == EXPECTED_MIDPOINT,
        "truth midpoint target drift",
    )
    for label, exposure in (
        ("report", report.get("source_exposure_fraction")),
        ("resume", resume.get("source_exposure_fraction")),
        ("truth", truth.get("source_exposure_fraction_at_final_target")),
    ):
        _require(
            isinstance(exposure, (int, float)) and not isinstance(exposure, bool),
            f"{label} source exposure missing",
        )
        _require(0.0 < float(exposure) < 0.01, f"{label} source exposure boundary violated")

    checkpoints_raw = report.get("checkpoints")
    _require(isinstance(checkpoints_raw, list), "report checkpoint list missing")
    checkpoints = {
        int(item["target_optimized_tokens"]): item
        for item in checkpoints_raw
        if isinstance(item, Mapping)
    }
    _require(sorted(checkpoints) == EXPECTED_TARGETS, "report checkpoint target set drift")
    final_target = EXPECTED_TARGETS[-1]
    final_checkpoint = checkpoints[final_target]
    _require(
        int(final.get("target_optimized_tokens", -1)) == final_target,
        "final checkpoint target drift",
    )
    _require(
        final.get("checkpoint_path") == f"checkpoint-t{final_target:06d}",
        "final checkpoint path drift",
    )
    _require(
        int(fresh.get("optimizer_step", -1))
        == int(final_checkpoint.get("optimizer_step", -2)),
        "fresh-load optimizer step disagrees with final checkpoint",
    )
    _require(
        int(fresh.get("optimized_tokens", -1))
        == int(final_checkpoint.get("actual_optimized_tokens", -2)),
        "fresh-load optimized tokens disagree with final checkpoint",
    )
    _require(
        fresh.get("checkpoint_id") == final_checkpoint.get("checkpoint_id"),
        "fresh-load checkpoint ID disagrees with final checkpoint",
    )
    _require(
        isinstance(fresh.get("model_state_sha256"), str)
        and len(fresh["model_state_sha256"]) == 64,
        "fresh-load model-state digest missing",
    )
    _require(
        resume.get("fresh_process_resume_passed") is True,
        "producer fresh-process resume flag is not PASS",
    )
    _require(
        int(resume.get("process_pid", -1)) != int(phase1.get("process_pid", -1)),
        "phase1 and resume process IDs are not distinct",
    )
    _require(
        int(best.get("target_optimized_tokens", -1)) in EXPECTED_TARGETS,
        "best checkpoint target is outside preregistered candidates",
    )

    result = {
        "schema": "12-6.verify219-artifact-contract-preflight.v1",
        "producer_sha": PRODUCER_SHA,
        "evidence_hashes": hashes,
        "model_spec_sha256": EXPECTED_MODEL_SPEC_SHA256,
        "init_spec_sha256": EXPECTED_INIT_SPEC_SHA256,
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "tokenizer_config_sha256": tokenizer.identity.config_sha256,
        "tokenizer_vocab_sha256": tokenizer.identity.vocab_sha256,
        "corpus_identity_sha256": EXPECTED_CORPUS_ID,
        "m150_evaluation_identity_sha256": EXPECTED_M150_EVAL_ID,
        "selection_validation_identity_sha256": expected_selection["identity_sha256"],
        "optimized_token_targets": list(EXPECTED_TARGETS),
        "midpoint_target": EXPECTED_MIDPOINT,
        "final_checkpoint_id": final_checkpoint["checkpoint_id"],
        "final_optimizer_step": int(final_checkpoint["optimizer_step"]),
        "final_optimized_tokens": int(final_checkpoint["actual_optimized_tokens"]),
        "training_executed": False,
        "optimizer_updates": 0,
    }
    result["identity_sha256"] = hash_json(result)
    return result
