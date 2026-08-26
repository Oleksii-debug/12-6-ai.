"""Fail-closed EVAL-289 code evaluation rights and reservation verifier."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

WORKER_ID = "EVAL-289-CODE-EVALUATION-RIGHTS-RESERVATION"
SCHEMA = "12-6.eval289-code-evaluation-rights-reservation.v1"
AUTHORITY_PATH = Path("evidence/eval289/code-evaluation-rights-reservation.json")
BLOCKER = "BLOCKED_NO_PRISTINE_CODE_OBJECTS_WITH_EXPLICIT_EVALUATION_AUTHORITY"


class Eval289Error(RuntimeError):
    pass


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def hash_json(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def assess_candidate(candidate: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if candidate.get("evaluation_use_explicitly_authorized") is not True:
        reasons.append("NO_EXPLICIT_EVALUATION_USE_AUTHORITY")
    exposure = candidate.get("training_exposure")
    if isinstance(exposure, dict) and exposure.get("exposed") is True:
        reasons.append("ALREADY_EXPOSED_TO_MODEL_TRAINING")
    return reasons


def validate_authority(value: dict[str, Any]) -> None:
    if value.get("schema_version") != SCHEMA or value.get("worker_id") != WORKER_ID:
        raise Eval289Error("unsupported EVAL-289 authority identity")

    unsigned = dict(value)
    identity = unsigned.pop("authority_identity_sha256", None)
    if identity != hash_json(unsigned):
        raise Eval289Error("authority self-hash mismatch")

    source = value.get("wave1_source_authority", {})
    if source.get("data227_head_sha") != "8ebdb2e132ed7bae5245e9d4c140752640ab9885":
        raise Eval289Error("DATA-227 head identity changed")
    if source.get("dedicated_workflow_run") != 32956209865:
        raise Eval289Error("DATA-227 terminal workflow identity changed")
    if source.get("dedicated_workflow_conclusion") != "success":
        raise Eval289Error("DATA-227 dedicated workflow is not terminal-success evidence")
    trainer = source.get("trainer_proof", {})
    if trainer.get("passed") is not True or trainer.get("optimizer_steps") != 4:
        raise Eval289Error("DATA-227 optimizer exposure proof changed")
    if trainer.get("tokens_seen") != 252:
        raise Eval289Error("DATA-227 optimized-token exposure changed")

    candidates = value.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 2:
        raise Eval289Error("expected exactly two terminal Wave-1 code candidates")
    observed_families = {str(item.get("source_family")) for item in candidates}
    if len(observed_families) != value.get("observed_source_family_count"):
        raise Eval289Error("observed source-family count mismatch")
    if value.get("observed_source_family_count") < 2:
        raise Eval289Error("Wave-1 source-family diversity unexpectedly disappeared")

    for candidate in candidates:
        reasons = assess_candidate(candidate)
        if candidate.get("evaluation_eligible") is not False:
            raise Eval289Error("contaminated Wave-1 code candidate marked evaluation-eligible")
        if candidate.get("ineligibility_reasons") != reasons:
            raise Eval289Error("candidate ineligibility reasons do not match fail-closed gates")
        if reasons != [
            "NO_EXPLICIT_EVALUATION_USE_AUTHORITY",
            "ALREADY_EXPOSED_TO_MODEL_TRAINING",
        ]:
            raise Eval289Error("Wave-1 code candidate blocker weakened")

    if value.get("status") != BLOCKER:
        raise Eval289Error("Wave-1 exact code objects must remain blocked")
    if value.get("eligible_object_count") != 0:
        raise Eval289Error("ineligible code object counted as eligible")
    if value.get("eligible_source_family_count") != 0:
        raise Eval289Error("ineligible source family counted as evaluation-eligible")
    reservation = value.get("reservation", {})
    if reservation.get("active") is not False or reservation.get("objects") != []:
        raise Eval289Error("contaminated code object was reserved")
    if value.get("training_corpus_mutation_authorized") is not False:
        raise Eval289Error("EVAL-289 must not authorize training-corpus mutation")
    if value.get("model_training_executed_by_eval289") is not False:
        raise Eval289Error("EVAL-289 must not execute model training")


def verify(repo_root: Path) -> dict[str, Any]:
    path = repo_root / AUTHORITY_PATH
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Eval289Error(f"unable to read authority: {path}") from exc
    if not isinstance(value, dict):
        raise Eval289Error("authority must be a JSON object")
    validate_authority(value)
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("verify",))
    args = parser.parse_args(argv)
    if args.command == "verify":
        verify(Path("."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
