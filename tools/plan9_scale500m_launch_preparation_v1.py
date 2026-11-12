"""Bind the accepted Plan 2 data to the primary 500M campaign preparation.

The receipt is preparation evidence. It does not launch training or authorize
compute, and it does not substitute a development recipe for an approved one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from twelve_six.model import load_stage_config

SCHEMA = "12-6.plan9-scale500m-launch-preparation.v1"
PROFILE = "SCALE-500M"
PROFILE_FILE = "configs/stages/scale_500m.json"
PROFILE_SHA256 = "1007a6b2519086a3e52fc06eb57a44a38ee38992bacbb922bb72b82b423b0339"
RUNTIME_PATHS = {
    "trainer": "src/twelve_six/training/trainer.py",
    "trainer_config": "src/twelve_six/training/config.py",
    "checkpoint_binding": "src/twelve_six/checkpoint/run_binding.py",
    "checkpoint_resume": "src/twelve_six/checkpoint/d04_resume_binding.py",
    "reference_inference": "tools/inference_runtime.py",
}


class LaunchPreparationDenied(ValueError):
    """The exact campaign preparation inputs did not match."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise LaunchPreparationDenied(reason)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: dict[str, Any]) -> bytes:
    return (json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ) + "\n").encode("utf-8")


def _read(path: Path, limit: int = 2_000_000) -> bytes:
    _require(
        not path.is_symlink() and path.is_file(), f"missing or linked input: {path.name}",
    )
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    _require(len(raw) <= limit, f"oversized input: {path.name}")
    return raw


def _read_pinned_json(path: Path, expected: str) -> dict[str, Any]:
    raw = _read(path)
    _require(_sha(raw) == expected, f"pinned input changed: {path.name}")
    value = json.loads(raw)
    _require(type(value) is dict and _canonical(value) == raw, f"noncanonical input: {path.name}")
    return value


def _open_release(root: Path, release: Path) -> dict[str, Any]:
    try:
        from tools.plan2_s18_physical_release_v1 import open_release
    except ModuleNotFoundError as exc:
        if exc.name != "tools.plan2_s18_physical_release_v1":
            raise
        raise LaunchPreparationDenied("Plan 2 release consumer is unavailable") from exc
    return open_release(root, release)


def _load_profile(root: Path) -> tuple[Any, dict[str, Any]]:
    path = root / PROFILE_FILE
    raw = _read(path).replace(b"\r\n", b"\n")
    _require(_sha(raw) == PROFILE_SHA256, "primary scale profile differs from the accepted source")
    config = load_stage_config(path)
    seed = json.loads(raw)["base_init_manifest"]
    _require(
        config.stage == PROFILE and config.canonical_base == "random_init"
        and seed["weights_origin"] == "random_init" and seed["ancestor_checkpoint_sha256"] is None
        and seed["model_spec_sha256"] == config.model.identity_sha256()
        and seed["init_spec_sha256"] == config.init.identity_sha256()
        and type(seed["seed"]) is int and seed["seed"] >= 0,
        "primary scale profile is not a fresh canonical Base",
    )
    return config, seed


def build_preparation(root: Path, release: Path) -> dict[str, Any]:
    """Open the real release and produce an immutable, non-launching receipt."""
    config, seed = _load_profile(root)
    authority = _open_release(root, release)
    _require(authority["plan9_data_release_verified"] is True and authority["campaign_authorized"] is False,
             "expected a verified Plan 2 data handoff without campaign authority")
    release_manifest = _read_pinned_json(release / "release-manifest.json", authority["release_manifest_sha256"])
    members = release_manifest["members"]
    binding = _read_pinned_json(release / "plan9-binding.json", members["plan9-binding.json"])
    dataset = _read_pinned_json(release / "dataset-manifest.json", members["dataset-manifest.json"])
    tokenizer = _read_pinned_json(release / "tokenizer-manifest.json", members["tokenizer-manifest.json"])
    split = _read_pinned_json(release / "split-manifest.json", members["split-manifest.json"])
    consumer = binding["physical_consumer"]
    compatibility = consumer["profile_compatibility"][PROFILE]
    _require(
        binding["campaign_authorized"] is False
        and consumer["model_training_executed"] is False
        and consumer["plan4_reference_inference_verified"] is True
        and consumer["weights_origin"] == "random_init"
        and compatibility["model_spec_sha256"] == config.model.identity_sha256()
        and compatibility["init_spec_sha256"] == config.init.identity_sha256()
        and compatibility["parameters"] == config.model.parameter_count()
        and compatibility["vocab_size"] == tokenizer["tokenizer_identity"]["vocab_size"] == config.model.vocab_size,
        "released batch compatibility does not bind the primary ModelSpec and InitSpec",
    )
    count = consumer["ordered_target_count"]
    _require(type(count) is int and count > 0 and count == authority["training_targets"]
             and consumer["tokenizer_manifest_sha256"] == tokenizer["source_manifest_sha256"],
             "released tokenizer or ordered target count differs from Plan 2 authority")
    roles = split["roles"]
    _require(
        set(roles) == {"train", "validation", "test"}
        and split["manifest_sha256"] == consumer["split_manifest_sha256"]
        and split["s16_admission_manifest_sha256"] == dataset["manifest_sha256"]
        and roles["train"]["source_membership_sha256"] == dataset["manifest_sha256"]
        and roles["train"]["records"] == dataset["records"]
        and roles["validation"]["manifest_sha256"] != roles["test"]["manifest_sha256"],
        "training and evaluation reservations do not bind the admitted corpus",
    )
    runtime = {
        name: {"path": path, "sha256": _sha(_read(root / path).replace(b"\r\n", b"\n"))}
        for name, path in RUNTIME_PATHS.items()
    }
    core = {
        "schema": SCHEMA,
        "decision": "INPUTS_BOUND_CAMPAIGN_PENDING",
        "profile": {
            "id": PROFILE,
            "source_sha256": PROFILE_SHA256,
            "model_spec_sha256": config.model.identity_sha256(),
            "init_spec_sha256": config.init.identity_sha256(),
            "parameters": config.model.parameter_count(),
            "initialization": seed,
        },
        "data": {
            "release_manifest_sha256": authority["release_manifest_sha256"],
            "release_authority_sha256": authority["authority_sha256"],
            "dataset_manifest_sha256": dataset["manifest_sha256"],
            "tokenizer_identity": tokenizer["tokenizer_identity"],
            "packing_manifest_sha256": consumer["packing_manifest_sha256"],
            "exposure_ledger_sha256": consumer["exposure_manifest_sha256"],
            "ordered_causal_loss_positions": count,
            "physical_consumer_sha256": consumer["manifest_sha256"],
        },
        "evaluation": {
            "split_manifest_sha256": split["manifest_sha256"],
            "validation_manifest_sha256": roles["validation"]["manifest_sha256"],
            "final_test_manifest_sha256": roles["test"]["manifest_sha256"],
            "final_test_payload_opened": False,
        },
        "runtime_source": runtime,
        "recipe_sha256": None,
        "campaign_checkpoint_sha256": None,
        "measured_host_receipt_sha256": None,
        "production_champion_sha256": None,
        "pending": [
            "accepted_500m_campaign_recipe_data_budget_and_stop_criteria",
            "exact_500m_host_resource_and_checkpoint_resume_evidence",
            "explicit_campaign_training_and_compute_authority",
        ],
        "campaign_authorized": False,
        "training_executed": False,
        "paid_compute_used": False,
    }
    return {**core, "preparation_sha256": _sha(_canonical(core))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--release", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = build_preparation(args.root.resolve(strict=True), args.release)
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"decision": "DENIED", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
