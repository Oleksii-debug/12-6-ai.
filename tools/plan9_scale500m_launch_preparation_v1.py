"""Bind the accepted Plan 2 data to the primary 500M campaign preparation.

The receipt is preparation evidence. It does not launch training or authorize
compute, and it does not substitute a development recipe for an approved one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from twelve_six.checkpoint.d04_resume_binding import (
    D04_RESUME_BINDING_SCHEMA,
    D04_RESUME_SHA_FIELDS,
)
from twelve_six.model import StageConfig, load_stage_config
from twelve_six.training.config import TrainingRecipe

if TYPE_CHECKING:
    from torch import Tensor

SCHEMA = "12-6.plan9-scale500m-launch-preparation.v1"
CURSOR_SCHEMA = "12-6.plan9-s18-physical-resume-cursor.v1"
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


def _read_pinned_json(path: Path, expected: str, *, limit: int = 2_000_000) -> dict[str, Any]:
    raw = _read(path, limit)
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


def _load_profile(root: Path) -> tuple[StageConfig, dict[str, Any]]:
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


def build_preparation(
    root: Path, release: Path, *, recipe: TrainingRecipe | None = None,
) -> dict[str, Any]:
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
    packing = _read_pinned_json(
        release / "packing-manifest.json", members["packing-manifest.json"], limit=15_000_000,
    )
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
    block_size, block_count = packing["block_size"], packing["block_count"]
    _require(
        packing["manifest_sha256"] == consumer["packing_manifest_sha256"]
        and packing["split"] == "train"
        and packing["cluster_split_sha256"] == split["manifest_sha256"]
        and packing["tokenizer_manifest_sha256"] == tokenizer["source_manifest_sha256"]
        and packing["target_count"] == count
        and type(block_size) is int and 2 <= block_size <= config.model.max_seq_len
        and type(block_count) is int and block_count > 0 and count <= block_size * block_count,
        "physical packing width, count, or identity differs from the accepted release",
    )
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
    training_fields = None
    if recipe is not None:
        _require(recipe.sequence_length == block_size,
                 f"candidate recipe sequence length must match the released width ({block_size})")
        _require(recipe.target_unique_loss_positions <= count,
                 "candidate recipe unique target exceeds the released ordered loss positions")
        training_fields = recipe.run_training_fields()
        _require(
            recipe.target_unique_loss_positions <= training_fields["global_batch_tokens"] * recipe.trainer.max_steps,
            "candidate recipe unique target exceeds configured batch and step capacity",
        )
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
            "packed_block_size": block_size,
            "packed_block_count": block_count,
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
        "recipe_sha256": None if recipe is None else recipe.identity_sha256(),
        "candidate_training_fields": training_fields,
        "campaign_recipe_accepted": False,
        "checkpoint_resume_contract": {
            "plan3_entrypoint": "twelve_six.checkpoint.bind_checkpoint_identity",
            "d04_schema": D04_RESUME_BINDING_SCHEMA,
            "d04_required_data_fields": list(D04_RESUME_SHA_FIELDS),
            "d04_plan_and_checkpoint_bound": False,
            "s18_physical_cursor_schema": CURSOR_SCHEMA,
        },
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


def _open_batch_stream(
    root: Path, release: Path, recipe: TrainingRecipe, next_batch_index: int | None,
) -> tuple[dict[str, Any], Iterator[dict[str, Tensor]]]:
    from tools.plan2_s18_physical_release_v1 import iter_release_batches

    preparation = build_preparation(root, release, recipe=recipe)
    data = preparation["data"]
    blocks = data["packed_block_count"]
    batch_count = (blocks + recipe.microbatch_size - 1) // recipe.microbatch_size
    if type(next_batch_index) is not int or not 0 <= next_batch_index <= batch_count:
        raise LaunchPreparationDenied("resume position is outside the sealed one-pass batch stream")
    batches = iter_release_batches(root, release, batch_size=recipe.microbatch_size)
    active_targets = 0
    for _ in range(next_batch_index):
        batch = next(batches, None)
        if batch is None:
            raise LaunchPreparationDenied("sealed batch stream ended before the requested cursor")
        active_targets += int(batch["loss_mask"].sum().item())
    _require(active_targets <= recipe.maximum_total_exposures,
             "resume prefix exceeds the candidate recipe exposure maximum")
    consumed_blocks = min(blocks, next_batch_index * recipe.microbatch_size)
    core = {
        "schema": CURSOR_SCHEMA,
        "release_manifest_sha256": data["release_manifest_sha256"],
        "packing_manifest_sha256": data["packing_manifest_sha256"],
        "exposure_ledger_sha256": data["exposure_ledger_sha256"],
        "recipe_sha256": preparation["recipe_sha256"],
        "next_batch_index": next_batch_index,
        "consumed_blocks": consumed_blocks,
        "consumed_loss_positions": active_targets,
        "complete": consumed_blocks == blocks,
    }
    return {**core, "cursor_sha256": _sha(_canonical(core))}, batches


def build_resume_cursor(
    root: Path, release: Path, *, recipe: TrainingRecipe, next_batch_index: int,
) -> dict[str, Any]:
    """Verify the ordered prefix and record its exact position in the sealed stream."""
    cursor, _ = _open_batch_stream(root, release, recipe, next_batch_index)
    return cursor


def iter_resumed_release_batches(
    root: Path, release: Path, *, recipe: TrainingRecipe, cursor: dict[str, Any],
    expected_cursor_sha256: str,
) -> Iterator[dict[str, Tensor]]:
    """Resume the physical stream after revalidating the trusted cursor's exact prefix."""
    _require(type(cursor) is dict and cursor.get("cursor_sha256") == expected_cursor_sha256,
             "resume cursor differs from the trusted identity")
    core = {key: value for key, value in cursor.items() if key != "cursor_sha256"}
    _require(_sha(_canonical(core)) == expected_cursor_sha256, "resume cursor digest mismatch")
    verified, batches = _open_batch_stream(root, release, recipe, cursor.get("next_batch_index"))
    _require(verified == cursor, "resume cursor differs from the verified physical prefix")
    yield from batches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, help="Explicit candidate TrainingRecipe v1 JSON")
    parser.add_argument("--next-batch-index", type=int, help="Inspect the sealed stream cursor")
    args = parser.parse_args()
    try:
        root = args.root.resolve(strict=True)
        recipe = None if args.recipe is None else TrainingRecipe.from_dict(json.loads(_read(args.recipe)))
        if args.next_batch_index is None:
            receipt = build_preparation(root, args.release, recipe=recipe)
        else:
            if recipe is None:
                raise LaunchPreparationDenied("--next-batch-index requires an explicit candidate --recipe")
            receipt = build_resume_cursor(
                root, args.release, recipe=recipe, next_batch_index=args.next_batch_index,
            )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(json.dumps({"decision": "DENIED", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
