#!/usr/bin/env python3
"""Validate R01 exposure semantics and optionally assess one exposure request."""

from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

from twelve_six.training_exposure import assess_training_exposure

DEFAULT_CONTRACT = Path("configs/research/r01_training_exposure_semantics_v1.json")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _contract_sha256(document: dict[str, Any]) -> str:
    payload = dict(document)
    payload.pop("contract_sha256", None)
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _git_blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()  # noqa: S324 - Git object identity


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _experiment(r01: dict[str, Any], experiment_id: str) -> dict[str, Any]:
    for item in r01.get("experiment_matrix", []):
        if item.get("id") == experiment_id:
            return item
    raise ValueError(f"missing R01 experiment {experiment_id}")


def validate_contract(repo_root: Path, contract_path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    contract = json.loads((repo_root / contract_path).read_text(encoding="utf-8"))
    expected_hash = contract.get("contract_sha256")
    _require(expected_hash == _contract_sha256(contract), "contract_sha256 mismatch")
    _require(contract.get("schema_version") == 1, "unexpected schema_version")
    _require(
        contract.get("contract_id") == "R01-TRAINING-EXPOSURE-SEMANTICS-V1",
        "unexpected contract_id",
    )

    base = contract["base_r01"]
    r01_path = repo_root / base["path"]
    r01_bytes = r01_path.read_bytes()
    _require(
        _git_blob_sha1(r01_bytes) == base["git_blob_sha1"],
        "base R01 Git blob identity mismatch",
    )
    r01 = json.loads(r01_bytes.decode("utf-8"))
    parameters = base["model_parameters"]
    planned_tpp = base["planned_tokens_per_parameter"]
    _require(parameters == r01["authority"]["parameter_count"], "parameter count drift")
    _require(planned_tpp == [10, 20, 40], "planned token-per-parameter vector drift")
    _require(
        _experiment(r01, "R01-E20")["planned_tokens_per_parameter"] == planned_tpp,
        "R01-E20 exposure vector drift",
    )
    _require(
        _experiment(r01, "R01-E30")["planned_tokens_per_parameter"] == planned_tpp,
        "R01-E30 exposure vector drift",
    )

    planning = contract["model341_planning"]
    _require(planning["parameter_count"] == parameters, "planning parameter count drift")
    arms = planning["arms"]
    _require([arm["tokens_per_parameter"] for arm in arms] == planned_tpp, "arm order drift")
    for arm in arms:
        expected = parameters * arm["tokens_per_parameter"]
        _require(
            arm["total_training_exposures"] == expected,
            "total training exposure arithmetic mismatch",
        )

    floor = planning["illustrative_unique_floor_positions"]
    _require(floor == 20_000_000, "illustrative floor drift")
    _require(planning["illustrative_floor_is_authority"] is False, "floor became authority")
    epoch_rows = planning["effective_epochs_at_illustrative_floor"]
    _require(len(epoch_rows) == len(arms), "epoch row count mismatch")
    for arm, row in zip(arms, epoch_rows, strict=True):
        expected_epochs = Fraction(arm["total_training_exposures"], floor)
        _require(row["tokens_per_parameter"] == arm["tokens_per_parameter"], "epoch arm drift")
        _require(row["numerator"] == expected_epochs.numerator, "epoch numerator mismatch")
        _require(row["denominator"] == expected_epochs.denominator, "epoch denominator mismatch")
        _require(
            row["decimal"] == f"{float(expected_epochs):.5f}",
            "epoch decimal mismatch",
        )
    reference_epochs = contract["research_reference"]["repeat_epoch_reference"]
    _require(reference_epochs == 4, "repeat research reference drift")
    _require(
        planning["four_epoch_reference_capacity_at_illustrative_floor"]
        == floor * reference_epochs,
        "four-epoch reference capacity mismatch",
    )

    rules = contract["hard_rules"]
    required_true = (
        "source_bytes_are_not_loss_positions",
        "repeat_exposures_are_not_unique",
        "padding_is_not_loss_position",
        "masked_or_ignored_targets_are_not_loss_positions",
        "planned_tokens_per_parameter_is_not_unique_data_claim",
        "repeat_training_requires_explicit_preregistered_cap",
        "repeat_training_is_not_authorized_by_this_contract",
    )
    for key in required_true:
        _require(rules.get(key) is True, f"hard rule {key} must remain true")
    _require(rules.get("long_training_authorized") is False, "long training was authorized")
    _require(rules.get("paid_compute_authorized") is False, "paid compute was authorized")

    truth = contract["truth_boundary"]
    _require(truth.get("optimizer_updates") == 0, "optimizer_updates must remain zero")
    for key, value in truth.items():
        if key == "optimizer_updates":
            continue
        _require(value is False, f"truth boundary {key} must remain false")

    return {
        "status": "PASS",
        "contract_sha256": expected_hash,
        "r01_git_blob_sha1": base["git_blob_sha1"],
        "parameter_count": parameters,
        "planned_tokens_per_parameter": planned_tpp,
        "long_training_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--unique-loss-positions", type=int)
    parser.add_argument("--requested-total-exposures", type=int)
    parser.add_argument("--max-repeat-epochs")
    args = parser.parse_args()

    result: dict[str, Any] = {"contract": validate_contract(args.repo_root, args.contract)}
    supplied = (args.unique_loss_positions, args.requested_total_exposures)
    if any(value is not None for value in supplied):
        if any(value is None for value in supplied):
            parser.error("supply both --unique-loss-positions and --requested-total-exposures")
        cap = None if args.max_repeat_epochs is None else Fraction(args.max_repeat_epochs)
        assessment = assess_training_exposure(
            unique_loss_positions=args.unique_loss_positions,
            requested_total_exposures=args.requested_total_exposures,
            max_repeat_epochs=cap,
        )
        result["assessment"] = assessment.as_dict()

    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
