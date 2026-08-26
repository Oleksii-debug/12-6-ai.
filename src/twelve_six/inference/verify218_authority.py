"""Fail-closed VERIFY-218 authority gate for the learned ~10M Transformers consumer."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

VERIFY218_WORKER_ID = "VERIFY-218-LEARNED-10M-INDEPENDENT"
VERIFY218_STATUS = "VERIFIED_LEARNED_10M"
GATE_SCHEMA = "12-6.runtime225-verify218-consumer-gate.v1"
EXPECTED_PARAMETER_COUNT = 10_000_640
EXPECTED_MODEL_SPEC_SHA256 = "61caa5469123e23b9b72fc2024140bfca84c4c480dcb0a7e712ba800a4f22998"
EXPECTED_TOKENIZER_VERSION = "s0-byte-v1"
EXPECTED_TOKENIZER_CONFIG_SHA256 = "b04055c1061dd641dcab7cb9d62a931f09b8d1a070140a926ceb4e91d73ca8e1"
EXPECTED_TOKENIZER_VOCAB_SHA256 = "905ed40bb42cc4d550e228ff5f24158d504b38e8ed5974dfa3077bd5867ad571"
EXPECTED_CORPUS_SHA256 = "422f545da93526ceba2c0ff738e0b9deac65f35bfb31d87d2aab068cda091fc8"
EXPECTED_SOURCE_ARTIFACT_NAME = "scale141-10m-learned-fallback"
_REQUIRED_GATES = (
    "checkpoint_integrity",
    "fresh_process_resume",
    "finite_first_party_logits",
    "heldout_bpb",
    "evaluation_non_mutation",
    "greedy_generation",
    "best_final_role_resolution",
)
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


class Verify218AuthorityError(RuntimeError):
    pass


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json_object(path: str | Path, *, label: str) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Verify218AuthorityError(f"{label} is not readable UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise Verify218AuthorityError(f"{label} must be a JSON object")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Verify218AuthorityError(f"{label} must be an object")
    return value


def _positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise Verify218AuthorityError(f"{label} must be a positive integer")
    return value


def _sha40(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HEX40.fullmatch(value) is None:
        raise Verify218AuthorityError(f"{label} must be a lowercase 40-hex SHA")
    return value


def _sha64(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise Verify218AuthorityError(f"{label} must be a lowercase 64-hex SHA-256")
    return value


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise Verify218AuthorityError(f"{label} must be sha256:<64 lowercase hex>")
    return value


def _validate_verifier_transport(
    artifact: Mapping[str, Any],
    run: Mapping[str, Any],
    *,
    artifact_id: int,
    artifact_digest: str,
    run_id: int,
    source_sha: str,
) -> None:
    if artifact.get("id") != artifact_id:
        raise Verify218AuthorityError("VERIFY-218 artifact id mismatch")
    if artifact.get("digest") != artifact_digest:
        raise Verify218AuthorityError("VERIFY-218 artifact digest mismatch")
    if artifact.get("expired") is not False:
        raise Verify218AuthorityError("VERIFY-218 artifact is expired")
    workflow_run = _mapping(artifact.get("workflow_run"), "VERIFY-218 artifact workflow_run")
    if workflow_run.get("id") != run_id or workflow_run.get("head_sha") != source_sha:
        raise Verify218AuthorityError("VERIFY-218 artifact workflow provenance mismatch")
    if run.get("id") != run_id or run.get("head_sha") != source_sha:
        raise Verify218AuthorityError("VERIFY-218 run identity mismatch")
    if run.get("status") != "completed" or run.get("conclusion") != "success":
        raise Verify218AuthorityError("VERIFY-218 run is not terminal SUCCESS")


def validate_verify218_authority(
    manifest: Mapping[str, Any],
    verifier_artifact: Mapping[str, Any],
    verifier_run: Mapping[str, Any],
    *,
    verifier_artifact_id: int,
    verifier_artifact_digest: str,
    verifier_run_id: int,
    verifier_source_sha: str,
) -> dict[str, Any]:
    """Return exact learned-source coordinates only for a valid VERIFY-218 authority."""

    _positive_int(verifier_artifact_id, "VERIFY-218 artifact id")
    _digest(verifier_artifact_digest, "VERIFY-218 artifact digest")
    _positive_int(verifier_run_id, "VERIFY-218 run id")
    _sha40(verifier_source_sha, "VERIFY-218 source SHA")
    _validate_verifier_transport(
        verifier_artifact,
        verifier_run,
        artifact_id=verifier_artifact_id,
        artifact_digest=verifier_artifact_digest,
        run_id=verifier_run_id,
        source_sha=verifier_source_sha,
    )

    schema = manifest.get("schema")
    if not isinstance(schema, str) or not schema.startswith("12-6.verify218-"):
        raise Verify218AuthorityError("VERIFY-218 manifest schema is not a VERIFY-218 authority schema")
    if manifest.get("worker_id") != VERIFY218_WORKER_ID:
        raise Verify218AuthorityError("VERIFY-218 worker identity mismatch")
    if manifest.get("status") != VERIFY218_STATUS:
        raise Verify218AuthorityError("VERIFY-218 did not emit VERIFIED_LEARNED_10M")
    if manifest.get("verified_learned_10m") is not True:
        raise Verify218AuthorityError("VERIFY-218 learned-10M verification flag is not true")
    if manifest.get("foreign_pretrained_weights") is not False:
        raise Verify218AuthorityError("VERIFY-218 authority admits foreign/pretrained weights")
    if manifest.get("mechanics_only_checkpoint") is not False:
        raise Verify218AuthorityError("VERIFY-218 authority admits a mechanics-only checkpoint")
    if manifest.get("one_step_smoke") is not False:
        raise Verify218AuthorityError("VERIFY-218 authority admits a one-step smoke checkpoint")

    gates = _mapping(manifest.get("gates"), "VERIFY-218 gates")
    missing_gates = [name for name in _REQUIRED_GATES if gates.get(name) is not True]
    if missing_gates:
        raise Verify218AuthorityError(f"VERIFY-218 required gates are incomplete: {missing_gates}")

    model = _mapping(manifest.get("model"), "VERIFY-218 model")
    if model.get("model_spec_sha256") != EXPECTED_MODEL_SPEC_SHA256:
        raise Verify218AuthorityError("VERIFY-218 ModelSpec is not the exact admitted 10M spec")
    if model.get("parameter_count") != EXPECTED_PARAMETER_COUNT:
        raise Verify218AuthorityError("VERIFY-218 parameter count mismatch")

    tokenizer = _mapping(manifest.get("tokenizer"), "VERIFY-218 tokenizer")
    expected_tokenizer = {
        "version": EXPECTED_TOKENIZER_VERSION,
        "config_sha256": EXPECTED_TOKENIZER_CONFIG_SHA256,
        "vocab_sha256": EXPECTED_TOKENIZER_VOCAB_SHA256,
    }
    for field, expected in expected_tokenizer.items():
        if tokenizer.get(field) != expected:
            raise Verify218AuthorityError(f"VERIFY-218 tokenizer mismatch: {field}")
    if manifest.get("corpus_identity_sha256") != EXPECTED_CORPUS_SHA256:
        raise Verify218AuthorityError("VERIFY-218 corpus identity mismatch")

    source = _mapping(manifest.get("source"), "VERIFY-218 learned source")
    source_artifact_id = _positive_int(source.get("artifact_id"), "learned source artifact id")
    source_artifact_digest = _digest(source.get("artifact_digest"), "learned source artifact digest")
    source_run_id = _positive_int(source.get("workflow_run_id"), "learned source workflow run id")
    source_sha = _sha40(source.get("source_sha"), "learned source SHA")
    source_artifact_name = source.get("artifact_name")
    if source_artifact_name != EXPECTED_SOURCE_ARTIFACT_NAME:
        raise Verify218AuthorityError("VERIFY-218 source artifact is not the maintained SCALE-141 learned artifact")

    checkpoint = _mapping(manifest.get("checkpoint"), "VERIFY-218 checkpoint")
    if checkpoint.get("role") != "best":
        raise Verify218AuthorityError("VERIFY-218 consumer checkpoint role must be best")
    checkpoint_id = _sha64(checkpoint.get("checkpoint_id"), "VERIFY-218 checkpoint id")
    checkpoint_step = _positive_int(checkpoint.get("step"), "VERIFY-218 checkpoint step")
    checkpoint_tokens_seen = _positive_int(
        checkpoint.get("tokens_seen"), "VERIFY-218 checkpoint tokens_seen"
    )

    output: dict[str, Any] = {
        "schema": GATE_SCHEMA,
        "status": "PASS",
        "authority": {
            "worker_id": VERIFY218_WORKER_ID,
            "manifest_schema": schema,
            "artifact_id": verifier_artifact_id,
            "artifact_digest": verifier_artifact_digest,
            "workflow_run_id": verifier_run_id,
            "source_sha": verifier_source_sha,
        },
        "learned_source": {
            "artifact_id": source_artifact_id,
            "artifact_name": source_artifact_name,
            "artifact_digest": source_artifact_digest,
            "workflow_run_id": source_run_id,
            "source_sha": source_sha,
            "checkpoint_role": "best",
            "checkpoint_id": checkpoint_id,
            "checkpoint_step": checkpoint_step,
            "checkpoint_tokens_seen": checkpoint_tokens_seen,
            "model_spec_sha256": EXPECTED_MODEL_SPEC_SHA256,
            "parameter_count": EXPECTED_PARAMETER_COUNT,
            "tokenizer_version": EXPECTED_TOKENIZER_VERSION,
            "tokenizer_config_sha256": EXPECTED_TOKENIZER_CONFIG_SHA256,
            "tokenizer_vocab_sha256": EXPECTED_TOKENIZER_VOCAB_SHA256,
            "corpus_identity_sha256": EXPECTED_CORPUS_SHA256,
        },
        "truth_boundary": {
            "source_selected_by_verify218": True,
            "source_substitution_allowed": False,
            "foreign_pretrained_weights": False,
            "paid_compute": False,
        },
    }
    output["identity_sha256"] = _canonical_sha256(output)
    return output
