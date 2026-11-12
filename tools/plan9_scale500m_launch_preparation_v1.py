"""Bind the accepted Plan 2 data to the primary 500M campaign preparation.

The receipt is preparation evidence. It does not launch training or authorize
compute, and it does not substitute a development recipe for an approved one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Iterator, Mapping
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from twelve_six.checkpoint import (
    CheckpointIdentity,
    bind_checkpoint_identity,
    detect_git_sha,
    hash_json,
)
from twelve_six.checkpoint.d04_resume_binding import (
    D04_RESUME_BINDING_SCHEMA,
    D04_RESUME_SHA_FIELDS,
)
from twelve_six.checkpoint.transaction import TRANSACTION_SCHEMA
from twelve_six.model import StageConfig, load_stage_config
from twelve_six.training import runtime_release
from twelve_six.training.config import TrainingRecipe

if TYPE_CHECKING:
    from torch import Tensor

    from twelve_six.training.run_control import RunReceipt
    from twelve_six.training.trainer import Trainer

SCHEMA = "12-6.plan9-scale500m-launch-preparation.v1"
CURSOR_SCHEMA = "12-6.plan9-s18-physical-resume-cursor.v1"
PACKET_SCHEMA = "12-6.plan9-scale500m-candidate-packet.v1"
DATA_BUDGET_SCHEMA = "12-6.plan9-scale500m-one-pass-data-budget.v1"
PROFILE = "SCALE-500M"
PROFILE_FILE = "configs/stages/scale_500m.json"
PACKET_TOOL_FILE = "tools/plan9_scale500m_launch_preparation_v1.py"
PROFILE_SHA256 = "1007a6b2519086a3e52fc06eb57a44a38ee38992bacbb922bb72b82b423b0339"
RUNTIME_PATHS = {
    "trainer": "src/twelve_six/training/trainer.py",
    "trainer_config": "src/twelve_six/training/config.py",
    "run_control": "src/twelve_six/training/run_control.py",
    "runtime_release_verifier": "src/twelve_six/training/runtime_release.py",
    "runtime_release_descriptor": runtime_release.RELEASE_PATH,
    "checkpoint_binding": "src/twelve_six/checkpoint/run_binding.py",
    "checkpoint_resume": "src/twelve_six/checkpoint/d04_resume_binding.py",
    "checkpoint_transaction": "src/twelve_six/checkpoint/transaction.py",
    "model": "src/twelve_six/model.py",
    "release_reader": "tools/plan2_s18_physical_release_v1.py",
    "release_consumer": "tools/plan2_s18_plan9_consumer_v1.py",
    "release_integrity": "tools/plan2_s18_release_integrity_v1.py",
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


def _candidate_checkout_sha(root: Path) -> str:
    sha = detect_git_sha(root)
    if sha is None:
        raise LaunchPreparationDenied("a Git checkout is required to bind the candidate source")
    _require(Path(__file__).resolve() == (root / PACKET_TOOL_FILE).resolve(),
             "run the packet tool from the candidate checkout")
    for path in (PROFILE_FILE, PACKET_TOOL_FILE, *RUNTIME_PATHS.values()):
        try:
            committed = subprocess.check_output(
                ["git", "show", f"{sha}:{path}"], cwd=root, stderr=subprocess.PIPE, timeout=10,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise LaunchPreparationDenied(f"candidate source not present at Git HEAD: {path}") from exc
        worktree = _read(root / path)
        _require(committed.replace(b"\r\n", b"\n") == worktree.replace(b"\r\n", b"\n"),
                 f"candidate source differs from Git HEAD: {path}")
    return sha


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
    released_runtime = runtime_release.verify_runtime_release(root)
    runtime_handoff = released_runtime["plan2_handoff"]
    _require(
        released_runtime["qualification"] == "LOCAL_COMPONENT"
        and released_runtime["full_profile_measurement_status"] == "OPEN_EXTERNAL"
        and released_runtime["campaign_authorized"] is False
        and runtime_handoff["release_manifest_sha256"] == authority["release_manifest_sha256"]
        and runtime_handoff["release_authority_sha256"] == authority["authority_sha256"]
        and runtime_handoff["plan9_binding_sha256"] == members["plan9-binding.json"]
        and runtime_handoff["training_targets"] == count
        and runtime_handoff["vocab_size"] == config.model.vocab_size,
        "released Plan 3 runtime does not bind the exact Plan 2 handoff",
    )
    block_size, block_count = packing["block_size"], packing["block_count"]
    _require(
        packing["manifest_sha256"] == consumer["packing_manifest_sha256"]
        and packing["split"] == "train"
        and packing["cluster_split_sha256"] == split["manifest_sha256"]
        and packing["tokenizer_manifest_sha256"] == tokenizer["source_manifest_sha256"]
        and packing["target_count"] == count
        and type(block_size) is int and 2 <= block_size <= config.model.max_seq_len
        and runtime_handoff["first_batch_shape"] == [2, block_size]
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
            "packing_version": packing["schema_version"],
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
        "plan3_runtime_release": {
            "descriptor_path": runtime_release.RELEASE_PATH,
            "descriptor_sha256": runtime_release.RELEASE_SHA256,
            "qualification": released_runtime["qualification"],
            "full_profile_measurement_status": released_runtime["full_profile_measurement_status"],
            "plan2_handoff": runtime_handoff,
        },
        "recipe_sha256": None if recipe is None else recipe.identity_sha256(),
        "candidate_training_fields": training_fields,
        "campaign_recipe_accepted": False,
        "checkpoint_resume_contract": {
            "plan3_entrypoint": "twelve_six.checkpoint.bind_checkpoint_identity",
            "trainer_restore_entrypoint": "twelve_six.checkpoint.load_trainer_checkpoint",
            "legacy_cursor_checkpoint_expectation": "expected_training_config_hash",
            "controller_input_entrypoint": "prepare_run_controller_input",
            "controller_transaction_schema": TRANSACTION_SCHEMA,
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
) -> tuple[dict[str, Any], dict[str, Any], Iterator[dict[str, Tensor]]]:
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
    return preparation, {**core, "cursor_sha256": _sha(_canonical(core))}, batches


def build_resume_cursor(
    root: Path, release: Path, *, recipe: TrainingRecipe, next_batch_index: int,
) -> dict[str, Any]:
    """Verify the ordered prefix and record its exact position in the sealed stream."""
    _, cursor, _ = _open_batch_stream(root, release, recipe, next_batch_index)
    return cursor


def _verify_resume_cursor(
    root: Path, release: Path, recipe: TrainingRecipe, cursor: dict[str, Any],
    expected_cursor_sha256: str,
) -> tuple[dict[str, Any], Iterator[dict[str, Tensor]]]:
    _require(type(cursor) is dict and cursor.get("cursor_sha256") == expected_cursor_sha256,
             "resume cursor differs from the trusted identity")
    core = {key: value for key, value in cursor.items() if key != "cursor_sha256"}
    _require(_sha(_canonical(core)) == expected_cursor_sha256, "resume cursor digest mismatch")
    preparation, verified, batches = _open_batch_stream(root, release, recipe, cursor.get("next_batch_index"))
    _require(verified == cursor, "resume cursor differs from the verified physical prefix")
    return preparation, batches


def iter_resumed_release_batches(
    root: Path, release: Path, *, recipe: TrainingRecipe, cursor: dict[str, Any],
    expected_cursor_sha256: str,
) -> Iterator[dict[str, Tensor]]:
    """Resume the physical stream after revalidating the trusted cursor's exact prefix."""
    _, batches = _verify_resume_cursor(root, release, recipe, cursor, expected_cursor_sha256)
    yield from batches


def bind_resume_checkpoint_identity(
    root: Path, release: Path, *, recipe: TrainingRecipe, cursor: dict[str, Any],
    expected_cursor_sha256: str, candidate_git_sha: str, environment_lock_sha256: str,
    run_id: str, optimizer_step: int,
) -> CheckpointIdentity:
    """Prepare Plan 3 checkpoint identity without producing a checkpoint or launch grant."""
    preparation, _ = _verify_resume_cursor(root, release, recipe, cursor, expected_cursor_sha256)
    _, _, _, identity = _bind_checkpoint_preflight(
        root, preparation, recipe, cursor,
        candidate_git_sha=candidate_git_sha, environment_lock_sha256=environment_lock_sha256,
        run_id=run_id, optimizer_step=optimizer_step,
    )
    return _bind_legacy_cursor(identity, cursor)


def _bind_checkpoint_preflight(
    root: Path, preparation: dict[str, Any], recipe: TrainingRecipe, cursor: dict[str, Any],
    *, candidate_git_sha: str, environment_lock_sha256: str, run_id: str, optimizer_step: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], CheckpointIdentity]:
    accumulation = recipe.trainer.gradient_accumulation_steps
    windows, partial = divmod(cursor["next_batch_index"], accumulation)
    _require(partial == 0 and type(optimizer_step) is int and 0 <= optimizer_step <= windows,
             "physical cursor is not at the declared optimizer checkpoint boundary")

    profile, seed = _load_profile(root)
    data = preparation["data"]
    tokenizer = data["tokenizer_identity"]
    packing = {
        "config_sha256": data["packing_manifest_sha256"],
        "version": data["packing_version"],
        "sequence_length": data["packed_block_size"],
    }
    run_manifest = {
        "run_id": run_id,
        "stage": PROFILE,
        "run_kind": "plan9_pretraining_candidate",
        "candidate": {
            "git_sha": candidate_git_sha,
            "modelspec_sha256": profile.model.identity_sha256(),
            "initspec_sha256": profile.init.identity_sha256(),
            "parameter_count": profile.model.parameter_count(),
            "initialization": seed,
        },
        "data": {
            "dataset_manifest_sha256": data["dataset_manifest_sha256"],
            "split_identity": preparation["evaluation"]["split_manifest_sha256"],
            "tokenizer_sha256": tokenizer["config_sha256"],
            "tokenizer_vocab_sha256": tokenizer["vocab_sha256"],
            "tokenizer_version": tokenizer["version"],
            "packing_sha256": packing["config_sha256"],
            "packing_version": packing["version"],
            "release_manifest_sha256": data["release_manifest_sha256"],
            "exposure_ledger_sha256": data["exposure_ledger_sha256"],
            "physical_cursor_schema": CURSOR_SCHEMA,
        },
        "training": preparation["candidate_training_fields"],
        "environment": {"lock_sha256": environment_lock_sha256},
        "evaluation": preparation["evaluation"],
        "runtime_source": preparation["runtime_source"],
        "plan3_runtime_release": preparation["plan3_runtime_release"],
        "campaign_authorized": False,
    }
    model_spec, init_spec = profile.model.to_dict(), profile.init.to_dict()
    identity = bind_checkpoint_identity(
        run_manifest=run_manifest,
        model_spec=model_spec,
        init_spec=init_spec,
        tokenizer_identity=tokenizer,
        packing_identity=packing,
        step=optimizer_step,
        tokens_seen=cursor["consumed_loss_positions"],
        environment_lock_hash=environment_lock_sha256,
    )
    return run_manifest, model_spec, init_spec, identity


def _bind_legacy_cursor(identity: CheckpointIdentity, cursor: dict[str, Any]) -> CheckpointIdentity:
    """Preserve the older progress-hash verifier independently of RunController."""
    bound = dict(identity.training_config)
    bound["data"] = {**bound["data"], "physical_resume_cursor_sha256": cursor["cursor_sha256"]}
    return replace(identity, training_config=bound)


def build_candidate_packet(
    root: Path, release: Path, *, recipe: TrainingRecipe,
    environment_lock_sha256: str, run_id: str, next_batch_index: int = 0,
    optimizer_step: int = 0,
) -> dict[str, Any]:
    """Assemble one portable candidate record from verified inputs and cursor evidence."""
    candidate_git_sha = _candidate_checkout_sha(root)
    preparation, cursor, _ = _open_batch_stream(root, release, recipe, next_batch_index)
    handoff = runtime_release.verify_plan2_handoff(root, release)
    _require(
        handoff["runtime_release_sha256"] == preparation["plan3_runtime_release"]["descriptor_sha256"]
        and handoff["plan2_handoff"] == preparation["plan3_runtime_release"]["plan2_handoff"]
        and type(handoff["optimizer_steps_executed"]) is int
        and handoff["optimizer_steps_executed"] == 0
        and handoff["campaign_authorized"] is False,
        "physical Plan 2 handoff differs from the released Plan 3 runtime",
    )
    run_manifest, model, init, identity = _bind_checkpoint_preflight(
        root, preparation, recipe, cursor,
        candidate_git_sha=candidate_git_sha, environment_lock_sha256=environment_lock_sha256,
        run_id=run_id, optimizer_step=optimizer_step,
    )
    identity.validate()
    core = {
        "schema": PACKET_SCHEMA,
        "decision": "CANDIDATE_PACKET_CAMPAIGN_PENDING",
        "preparation": preparation,
        "model_spec": model,
        "init_spec": init,
        "run_manifest": run_manifest,
        "run_manifest_sha256": identity.run_manifest_hash,
        "physical_cursor": cursor,
        "plan3_physical_handoff": {
            "aligned_loss_verified": True,
            "optimizer_steps_executed": handoff["optimizer_steps_executed"],
        },
        "checkpoint_preflight": {
            "legacy_expected_training_config_hash": hash_json(
                _bind_legacy_cursor(identity, cursor).training_config,
            ),
            "controller_base_training_config_sha256": hash_json(identity.training_config),
            "optimizer_step": identity.step,
            "tokens_seen": identity.tokens_seen,
            "checkpoint_artifact_verified": False,
            "checkpoint_transaction_verified": False,
        },
        "environment_lock_qualification": "PINNED_BYTES_ONLY",
        "campaign_authorized": False,
        "training_executed": False,
        "paid_compute_used": False,
    }
    return {**core, "packet_sha256": _sha(_canonical(core))}


def verify_candidate_packet(
    root: Path, release: Path, *, packet: dict[str, Any], expected_packet_sha256: str,
    environment_lock_sha256: str,
) -> dict[str, Any]:
    """Rebuild the pinned candidate against the exact current inputs before accepting it."""
    _require(type(packet) is dict and packet.get("packet_sha256") == expected_packet_sha256,
             "packet differs from the trusted identity")
    core = {key: value for key, value in packet.items() if key != "packet_sha256"}
    _require(_sha(_canonical(core)) == expected_packet_sha256, "candidate packet digest mismatch")
    run = packet["run_manifest"]
    actual = build_candidate_packet(
        root, release, recipe=TrainingRecipe.from_dict(run["training"]["recipe"]),
        environment_lock_sha256=environment_lock_sha256,
        run_id=run["run_id"], next_batch_index=packet["physical_cursor"]["next_batch_index"],
        optimizer_step=packet["checkpoint_preflight"]["optimizer_step"],
    )
    _require(actual["packet_sha256"] == expected_packet_sha256,
             "candidate packet differs from the verified release, source, lock or physical prefix")
    return actual


class PhysicalRunInput:
    """Connect the sealed S18 stream to a single RunController writer."""

    def __init__(self, root: Path, release: Path, recipe: TrainingRecipe) -> None:
        self._root = root
        self._release = release
        self._recipe = recipe
        self._cursor: dict[str, Any] | None = None
        self._iterator: Iterator[dict[str, Tensor]] | None = None
        self._total_blocks = 0

    def restore_exposure_cursor(self, cursor: Mapping[str, Any]) -> None:
        """Reopen the real release and its exact committed prefix before reading input."""
        self._cursor = None
        self._iterator = None
        supplied = dict(cursor)
        digest = supplied.get("cursor_sha256")
        _require(type(digest) is str, "checkpoint exposure cursor lacks its physical digest")
        preparation, batches = _verify_resume_cursor(
            self._root, self._release, self._recipe, supplied, digest,
        )
        self._total_blocks = preparation["data"]["packed_block_count"]
        self._cursor, self._iterator = supplied, batches

    def batches(self, receipt: RunReceipt) -> Iterator[dict[str, Tensor]]:
        cursor, source = self._cursor, self._iterator
        if cursor is None or source is None:
            raise LaunchPreparationDenied("restore the verified exposure cursor before opening batches")
        accumulation = self._recipe.trainer.gradient_accumulation_steps
        _require(
            cursor["next_batch_index"] == receipt.committed_step * accumulation
            and cursor["consumed_loss_positions"] == receipt.tokens_seen,
            "run receipt progress differs from the verified exposure cursor",
        )
        self._iterator = None
        for batch in source:
            positions = cursor["consumed_loss_positions"] + int(batch["loss_mask"].sum().item())
            _require(positions <= self._recipe.maximum_total_exposures,
                     "next batch would exceed the sealed recipe exposure budget")
            blocks = cursor["consumed_blocks"] + int(batch["input_ids"].shape[0])
            core = {key: value for key, value in cursor.items() if key != "cursor_sha256"}
            core.update(
                next_batch_index=cursor["next_batch_index"] + 1,
                consumed_blocks=blocks,
                consumed_loss_positions=positions,
                complete=blocks == self._total_blocks,
            )
            cursor = {**core, "cursor_sha256": _sha(_canonical(core))}
            self._cursor = cursor
            yield batch

    def read_exposure_cursor(self, trainer: Trainer) -> dict[str, Any]:
        """Return the position to publish with the successful optimizer state."""
        cursor = self._cursor
        if cursor is None:
            raise LaunchPreparationDenied("no verified exposure cursor is available")
        _require(
            cursor["next_batch_index"] == trainer.micro_step
            and cursor["consumed_loss_positions"] == trainer.tokens_seen,
            "trainer progress differs from the physical exposure cursor",
        )
        return dict(cursor)


def prepare_run_controller_input(
    root: Path, release: Path, *, packet: dict[str, Any], expected_packet_sha256: str,
    environment_lock_sha256: str,
) -> tuple[CheckpointIdentity, dict[str, Any], PhysicalRunInput]:
    """Return an initial identity and cursor hooks without running the controller."""
    verified = verify_candidate_packet(
        root, release, packet=packet, expected_packet_sha256=expected_packet_sha256,
        environment_lock_sha256=environment_lock_sha256,
    )
    cursor = verified["physical_cursor"]
    _require(cursor["next_batch_index"] == 0, "controller setup requires the initial candidate packet")
    run = verified["run_manifest"]
    recipe = TrainingRecipe.from_dict(run["training"]["recipe"])
    _, _, _, identity = _bind_checkpoint_preflight(
        root, verified["preparation"], recipe, cursor,
        candidate_git_sha=run["candidate"]["git_sha"], environment_lock_sha256=environment_lock_sha256,
        run_id=run["run_id"], optimizer_step=0,
    )
    return identity, cursor, PhysicalRunInput(root, release, recipe)


def check_one_pass_data_budget(
    root: Path, release: Path, *, recipe: TrainingRecipe,
) -> dict[str, Any]:
    """Check a nominal optimizer horizon against actual unique training positions."""
    trainer = recipe.trainer
    planned_batches = trainer.max_steps * trainer.gradient_accumulation_steps
    preparation, cursor, _ = _open_batch_stream(root, release, recipe, planned_batches)
    unique_positions = cursor["consumed_loss_positions"]
    _require(unique_positions >= recipe.target_unique_loss_positions,
             "sealed masks cannot meet the unique target within the candidate optimizer horizon")

    data = preparation["data"]
    manifest = _read_pinned_json(release / "release-manifest.json", data["release_manifest_sha256"])
    members = manifest["members"]
    dataset = _read_pinned_json(release / "dataset-manifest.json", members["dataset-manifest.json"])
    rights_pin = members["rights-attribution.json"]
    rights = _read_pinned_json(release / "rights-attribution.json", rights_pin)
    notices = rights["notice_sha256"]
    _require(
        rights["scope"] == "LOCAL_PLAN9_TRAINING_DATA_HANDOFF"
        and rights["evaluation_payloads_included"] is False
        and rights["public_redistribution_authorized"] is False
        and rights["dataset_sha256"] == members["dataset-manifest.json"]
        and bool(rights["selected_source_terms"])
        and rights["selected_source_terms"] == dataset["selected_source_terms"]
        and type(notices) is dict
        and bool(notices)
        and all(name.startswith("notices/") and members.get(name) == digest for name, digest in notices.items()),
        "rights and notice scope differs from the accepted local training data handoff",
    )
    for name, digest in notices.items():
        _require(_sha(_read(release / name)) == digest, f"pinned notice changed: {name}")
    core = {
        "schema": DATA_BUDGET_SCHEMA,
        "decision": "ONE_PASS_DATA_BUDGET_VERIFIED_CAMPAIGN_PENDING",
        "release_manifest_sha256": data["release_manifest_sha256"],
        "release_authority_sha256": data["release_authority_sha256"],
        "rights_attribution_sha256": rights_pin,
        "source_family_count": len(rights["selected_source_terms"]),
        "notice_count": len(notices),
        "data_scope": rights["scope"],
        "evaluation_payloads_included": False,
        "public_redistribution_authorized": False,
        "recipe_sha256": preparation["recipe_sha256"],
        "physical_cursor_sha256": cursor["cursor_sha256"],
        "target_unique_loss_positions": recipe.target_unique_loss_positions,
        "maximum_total_exposures": recipe.maximum_total_exposures,
        "nominal_optimizer_steps": trainer.max_steps,
        "planned_microbatches": planned_batches,
        "exact_one_pass_unique_loss_positions": unique_positions,
        "exact_one_pass_total_exposures": unique_positions,
        "requires_no_skipped_updates_to_reach_nominal_optimizer_steps": True,
        "target_environment_qualified": False,
        "campaign_stop_policy_accepted": False,
        "campaign_authorized": False,
        "training_executed": False,
        "paid_compute_used": False,
    }
    return {**core, "data_budget_sha256": _sha(_canonical(core))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, help="Explicit candidate TrainingRecipe v1 JSON")
    parser.add_argument("--next-batch-index", type=int, help="Inspect the sealed stream cursor")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--build-packet", action="store_true", help="Emit a canonical candidate packet")
    modes.add_argument("--verify-packet", type=Path, help="Reopen a pinned candidate packet")
    modes.add_argument("--check-one-pass-budget", action="store_true", help="Check rights and exact data budget")
    parser.add_argument("--packet-sha256", help="Trusted packet digest for verification")
    parser.add_argument("--environment-lock", type=Path, help="Environment lock bytes to pin without host qualification")
    parser.add_argument("--run-id", help="Candidate run identity for packet construction")
    parser.add_argument("--optimizer-step", type=int, default=0, help="Candidate checkpoint position")
    args = parser.parse_args()
    try:
        root = args.root.resolve(strict=True)
        recipe = None if args.recipe is None else TrainingRecipe.from_dict(json.loads(_read(args.recipe)))
        if args.check_one_pass_budget:
            if recipe is None:
                raise LaunchPreparationDenied("--check-one-pass-budget requires --recipe")
            sys.stdout.buffer.write(_canonical(check_one_pass_data_budget(root, args.release, recipe=recipe)))
            return 0
        if args.build_packet or args.verify_packet is not None:
            if args.environment_lock is None:
                raise LaunchPreparationDenied("an explicit --environment-lock is required")
            lock_sha = _sha(_read(args.environment_lock))
            if args.build_packet:
                if recipe is None or args.run_id is None:
                    raise LaunchPreparationDenied("--build-packet requires --recipe and --run-id")
                receipt = build_candidate_packet(
                    root, args.release, recipe=recipe,
                    environment_lock_sha256=lock_sha, run_id=args.run_id,
                    next_batch_index=0 if args.next_batch_index is None else args.next_batch_index,
                    optimizer_step=args.optimizer_step,
                )
            else:
                if args.packet_sha256 is None:
                    raise LaunchPreparationDenied("--verify-packet requires a trusted --packet-sha256")
                packet_bytes = _read(args.verify_packet)
                packet = json.loads(packet_bytes)
                _require(type(packet) is dict and _canonical(packet) == packet_bytes,
                         "candidate packet must use canonical JSON bytes")
                receipt = verify_candidate_packet(
                    root, args.release, packet=packet,
                    expected_packet_sha256=args.packet_sha256,
                    environment_lock_sha256=lock_sha,
                )
            sys.stdout.buffer.write(_canonical(receipt))
            return 0
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
