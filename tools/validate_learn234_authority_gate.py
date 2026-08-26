from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _canonical_identity(payload: dict[str, object]) -> str:
    body = dict(payload)
    claimed = body.pop("gate_identity_sha256", None)
    if not isinstance(claimed, str) or len(claimed) != 64:
        raise ValueError("missing gate_identity_sha256")
    raw = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def validate(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "12-6.learn234-authority-gate.v1":
        raise ValueError("wrong schema")
    if payload.get("worker_id") != "LEARN-234-EXTERNAL-REAL-500K":
        raise ValueError("wrong worker id")
    claimed = payload.get("gate_identity_sha256")
    if _canonical_identity(payload) != claimed:
        raise ValueError("gate identity mismatch")

    data230 = payload["required_authorities"]["data230"]
    execution = payload["execution"]
    if not data230["terminal_deterministic"]:
        if payload["status"] != "BLOCKED_NO_TERMINAL_DATA230":
            raise ValueError("missing fail-closed DATA-230 status")
        if execution["training_started"] or execution["optimizer_updates"] != 0:
            raise ValueError("training occurred without terminal DATA-230")

    incumbent = payload["incumbent_500k"]
    if incumbent["parameters"] != 467808:
        raise ValueError("500K parameter incumbent drift")
    if incumbent["model_spec_sha256"] != (
        "208ac8ca113388e76f280d0154cae815785bee7705546f4d854d9447b9dd1f4a"
    ):
        raise ValueError("500K ModelSpec incumbent drift")
    if not incumbent["random_initialization_only"]:
        raise ValueError("random initialization requirement lost")

    budget = payload["budget_preregistration"]
    if budget["artificial_corpus_repetition_allowed"]:
        raise ValueError("artificial corpus repetition enabled")
    if budget["external_real_budget_rule"] != (
        "min(948504, DATA230_one_pass_unique_train_optimized_targets)"
    ):
        raise ValueError("budget preregistration drift")

    for forbidden in ("foreign_weights", "sft", "rlhf", "dpo", "paid_compute"):
        if execution[forbidden]:
            raise ValueError(f"forbidden execution mode enabled: {forbidden}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("gate", type=Path)
    args = parser.parse_args()
    validate(args.gate)
    print("LEARN-234 authority gate: PASS (blocked before training as required)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
