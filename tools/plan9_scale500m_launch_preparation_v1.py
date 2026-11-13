"""Bind the accepted Plan 2 data to the primary 500M campaign preparation.

The receipt is preparation evidence. It does not launch training or authorize
compute, and it does not substitute a development recipe for an approved one.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import re
import stat
import subprocess
import sys
from collections.abc import Iterator, Mapping
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import FunctionType, ModuleType
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
from twelve_six.integration import dependency_lock
from twelve_six.model import ModelSpec, StageConfig, load_stage_config
from twelve_six.training import runtime_release
from twelve_six.training.config import TrainingRecipe
from twelve_six.training.resource_envelope import (
    HostResources,
    assess_resource_envelope,
    observe_host_resources,
)

if TYPE_CHECKING:
    from torch import Tensor

    from twelve_six.training.run_control import RunReceipt
    from twelve_six.training.trainer import Trainer

SCHEMA = "12-6.plan9-scale500m-launch-preparation.v1"
CURSOR_SCHEMA = "12-6.plan9-s18-physical-resume-cursor.v1"
PACKET_SCHEMA = "12-6.plan9-scale500m-candidate-packet.v1"
DATA_BUDGET_SCHEMA = "12-6.plan9-scale500m-one-pass-data-budget.v1"
NATIVE_PREFLIGHT_SCHEMA = "12-6.plan9-scale500m-native-launch-preflight.v1"
EXTERNAL_GPU_SCHEMA = "12-6.plan9-scale500m-external-gpu-candidate.v1"
GPU_PROFILE_PLAN_SCHEMA = "12-6.plan9-external-gpu-profile-plan.v1"
CYBER_GPU_PREPARATION_SCHEMA = "12-6.plan9-cyber-external-gpu-preparation.v1"
CYBER_EVIDENCE = "evidence/plan9_cyber_candidate"
PLAN3_RUNTIME_RELEASE_VERSION = 2
EXTERNAL_GPU_PROFILES = ("linux-x86_64", "linux-aarch64")
LOCK_INDEX = "requirements/locks/index.json"
LOCK_INDEX_SHA256 = "870587414a9108fc5d924a0caad37cd228515ad32df6639538deba48dc901b5a"
ONE_PASS_RECIPE_FILE = "configs/plan9/scale500m_one_pass_recipe_v1.json"
ONE_PASS_RECIPE_SHA256 = "edc679f9c2fd301fdac52ccfca9e7d6374894f2a735bb1e202eb88e1e3e6f1af"
ONE_PASS_BUDGET_FILE = "evidence/plan9_s1/scale500m_one_pass_data_budget_v1.json"
ONE_PASS_BUDGET_SHA256 = "979cc86b15699773819f8c815d7c536bd446621b486fdbdb2b1a399ce34b4d04"
PROFILE = "SCALE-500M"
PROFILE_FILE = "configs/stages/scale_500m.json"
PACKET_TOOL_FILE = "tools/plan9_scale500m_launch_preparation_v1.py"
PROFILE_SHA256 = "1007a6b2519086a3e52fc06eb57a44a38ee38992bacbb922bb72b82b423b0339"
GPU_PROFILE_SOURCES = {
    PROFILE: (PROFILE_FILE, PROFILE_SHA256),
    "SCALE-1B": (
        "configs/stages/scale_1b.json", "cda103c731761388f76deec4a1a3eb6baec7a1466072bc6366d45bedb4bf3221",
    ),
    "SCALE-2B": (
        "configs/stages/scale_2b.json", "6d860f3a8867212e4f09bf275accbb8be371d8868e08f4ff1b85f98c020daba6",
    ),
}
RUNTIME_PATHS = {
    "trainer": "src/twelve_six/training/trainer.py",
    "trainer_config": "src/twelve_six/training/config.py",
    "run_control": "src/twelve_six/training/run_control.py",
    "runtime_release_verifier": "src/twelve_six/training/runtime_release.py",
    "runtime_release_descriptor": runtime_release.RELEASE_V2_PATH,
    "dependency_lock_verifier": "src/twelve_six/integration/dependency_lock.py",
    "checkpoint_binding": "src/twelve_six/checkpoint/run_binding.py",
    "checkpoint_resume": "src/twelve_six/checkpoint/d04_resume_binding.py",
    "checkpoint_transaction": "src/twelve_six/checkpoint/transaction.py",
    "model": "src/twelve_six/model.py",
    "release_reader": "tools/plan2_s18_physical_release_v1.py",
    "release_consumer": "tools/plan2_s18_plan9_consumer_v1.py",
    "release_integrity": "tools/plan2_s18_release_integrity_v1.py",
    "reference_inference": "tools/inference_runtime.py",
}
_FIRST_PARTY_MODULE = re.compile(r"^(?:tools|twelve_six)(?:\.|$)")


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
    dependencies = set(sys.modules) & {
        "tools", "twelve_six", "twelve_six.checkpoint", "twelve_six.checkpoint.run_binding",
        "twelve_six.integration.dependency_lock", "twelve_six.model",
        "twelve_six.training.config", "twelve_six.training.runtime_release",
        "twelve_six.training.resource_envelope",
    }
    _loaded_checkout_paths(root, dependencies)
    # The release reader imports the physical consumer lazily when iteration starts.
    for module in ("tools.plan2_s18_physical_release_v1", "tools.plan2_s18_plan9_consumer_v1"):
        importlib.import_module(module)
        dependencies.add(module)

    _verify_loaded_checkout_sources(
        root, sha, dependencies,
        required_paths={PROFILE_FILE, PACKET_TOOL_FILE, *RUNTIME_PATHS.values()},
    )
    return sha


def _loaded_checkout_paths(root: Path, dependencies: set[str]) -> set[str]:
    """Require first-party Python imports to resolve within this checkout."""
    root = root.resolve()
    paths = set()
    checked = set()
    pending = sorted(dependencies)
    while pending:
        name = pending.pop()
        if name in checked:
            continue
        checked.add(name)
        module = sys.modules[name]
        package = name.partition(".")[0]
        relative = ("" if package == "tools" else "src/") + name.replace(".", "/")
        parent = name.rpartition(".")[0]
        if parent and parent in sys.modules:
            pending.append(parent)
        origin = getattr(module, "__file__", None)
        locations = getattr(module, "__path__", None)
        if locations is not None:
            _require({Path(path).resolve() for path in locations} == {root / relative},
                     f"loaded package path differs from candidate checkout: {name}")
        if origin is None:
            _require(locations is not None,
                     f"loaded namespace differs from candidate checkout: {name}")
            continue
        expected = relative + ("/__init__.py" if locations is not None else ".py")
        spec = getattr(module, "__spec__", None)
        _require(Path(origin).resolve() == root / expected,
                 f"loaded source differs from candidate checkout: {name}")
        _require(spec is not None and spec.origin is not None and Path(spec.origin).resolve() == root / expected,
                 f"loaded source specification differs from candidate checkout: {name}")
        paths.add(expected)
        for value in tuple(vars(module).values()):
            if isinstance(value, ModuleType):
                dependency = value.__name__
                # Import machinery attaches unrelated child modules to packages.
                if locations is not None and dependency.startswith(name + "."):
                    continue
            elif isinstance(value, (FunctionType, type)):
                dependency = getattr(value, "__module__", "") or ""
            else:
                continue
            if _FIRST_PARTY_MODULE.match(dependency) and dependency in sys.modules:
                pending.append(dependency)
    return paths


def _verify_loaded_checkout_sources(root: Path, sha: str, dependencies: set[str], *, required_paths: set[str]) -> None:
    paths = _loaded_checkout_paths(root, dependencies) | required_paths
    try:
        records = subprocess.check_output(
            ["git", "ls-tree", "-r", "-z", sha, "--", *sorted(paths)],
            cwd=root, stderr=subprocess.PIPE, timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise LaunchPreparationDenied("cannot read candidate source identities from Git HEAD") from exc
    blobs = {}
    for record in records.split(b"\0"):
        if record:
            metadata, path = record.split(b"\t", 1)
            mode, kind, digest = metadata.decode("ascii").split()
            if kind == "blob" and mode in {"100644", "100755"}:
                blobs[path.decode("utf-8")] = digest
    algorithm = "sha1" if len(sha) == 40 else "sha256"
    for path in sorted(paths):
        _require(path in blobs, f"candidate source not present at Git HEAD: {path}")
        raw = _read(root / path)
        # Git identifies a blob by its header and bytes. Accept a CRLF checkout
        # of the same LF blob while retaining the committed byte identity.
        matches = any(
            hashlib.new(algorithm, f"blob {len(data)}\0".encode("ascii") + data).hexdigest() == blobs[path]
            for data in (raw, raw.replace(b"\r\n", b"\n"))
        )
        _require(matches, f"candidate source differs from Git HEAD: {path}")


def _exact_checkout_bytes(root: Path, sha: str, path: str) -> bytes:
    try:
        committed = subprocess.check_output(
            ["git", "show", f"{sha}:{path}"], cwd=root, stderr=subprocess.PIPE, timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise LaunchPreparationDenied(f"candidate source not present at Git HEAD: {path}") from exc
    worktree = _read(root / path)
    _require(committed.replace(b"\r\n", b"\n") == worktree.replace(b"\r\n", b"\n"),
             f"candidate source differs from Git HEAD: {path}")
    return committed


def _open_release(root: Path, release: Path) -> dict[str, Any]:
    try:
        from tools.plan2_s18_physical_release_v1 import open_release
    except ModuleNotFoundError as exc:
        if exc.name != "tools.plan2_s18_physical_release_v1":
            raise
        raise LaunchPreparationDenied("Plan 2 release consumer is unavailable") from exc
    return open_release(root, release)


def _load_profile(root: Path, profile: str = PROFILE) -> tuple[StageConfig, dict[str, Any]]:
    path, source_sha256 = GPU_PROFILE_SOURCES[profile]
    raw = _read(root / path).replace(b"\r\n", b"\n")
    _require(_sha(raw) == source_sha256, f"{profile} profile differs from the accepted source")
    document = json.loads(raw)
    config = load_stage_config(root / path)
    seed = document["base_init_manifest"]
    _require(
        config.stage == profile and document["profile_id"] == profile
        and config.canonical_base == "random_init"
        and seed["weights_origin"] == "random_init" and seed["ancestor_checkpoint_sha256"] is None
        and seed["model_spec_sha256"] == config.model.identity_sha256()
        and seed["init_spec_sha256"] == config.init.identity_sha256()
        and type(seed["seed"]) is int and seed["seed"] >= 0,
        f"{profile} profile is not a fresh canonical Base",
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
    released_runtime = runtime_release.verify_runtime_release(root, release_version=PLAN3_RUNTIME_RELEASE_VERSION)
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
            "descriptor_path": runtime_release.RELEASE_V2_PATH,
            "descriptor_sha256": runtime_release.RELEASE_V2_SHA256,
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
    handoff = runtime_release.verify_plan2_handoff(root, release, release_version=PLAN3_RUNTIME_RELEASE_VERSION)
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


def _native_lock_binding(root: Path, sha: str) -> dict[str, Any]:
    lock = _lock_binding(root, sha, dependency_lock.current_profile_id())
    return {
        **lock,
        "observed_python": {"implementation": sys.implementation.name, "version": platform.python_version()},
    }


def _lock_binding(root: Path, sha: str, profile: str) -> dict[str, Any]:
    """Validate the repository's exact lock bytes even in a CRLF checkout."""
    index_bytes = _exact_checkout_bytes(root, sha, LOCK_INDEX)
    index = json.loads(index_bytes)
    _require(index.get("index_sha256") == LOCK_INDEX_SHA256, "native dependency lock index pin changed")
    lock_paths = [
        f"requirements/locks/{platform_id}/{filename}"
        for platform_id in sorted(dependency_lock.SUPPORTED_PROFILES)
        for filename in ("profile.json", "toolchain.lock.txt", "runtime.lock.txt", "dev.lock.txt")
    ]
    with TemporaryDirectory() as directory:
        staged = Path(directory)
        for relative in ("pyproject.toml", LOCK_INDEX, *lock_paths):
            path = staged / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(index_bytes if relative == LOCK_INDEX else _exact_checkout_bytes(root, sha, relative))
        verified = dependency_lock.validate_lock_index(root=staged, index_path=LOCK_INDEX)
        record = verified["profiles"][profile]
        profile_manifest = dependency_lock.validate_profile_manifest(
            root=staged, manifest_path=record["path"], enforce_current_platform=False,
        )
    return {
        "index_sha256": verified["index_sha256"],
        "profile_id": profile,
        "profile_manifest_sha256": profile_manifest["manifest_sha256"],
        "runtime_lock_sha256": profile_manifest["locks"]["runtime"]["sha256"],
        "expected_python": profile_manifest["python"],
    }


def inspect_native_launch(
    root: Path, release: Path, *, packet: dict[str, Any], expected_packet_sha256: str,
    checkpoint_root: Path, device: str,
) -> dict[str, Any]:
    """Observe the local environment and capacity floors without creating a training run."""
    checkout = _candidate_checkout_sha(root)
    lock = _native_lock_binding(root, checkout)
    verified = verify_candidate_packet(
        root, release, packet=packet, expected_packet_sha256=expected_packet_sha256,
        environment_lock_sha256=lock["runtime_lock_sha256"],
    )
    _require(verified["physical_cursor"]["next_batch_index"] == 0,
             "native launch inspection requires the initial candidate packet")
    model = ModelSpec.from_dict(verified["model_spec"])
    recipe = TrainingRecipe.from_dict(verified["run_manifest"]["training"]["recipe"])
    resources = observe_host_resources(checkpoint_root, device)
    envelope = assess_resource_envelope(
        model, resources, remaining_updates=recipe.trainer.max_steps, initial_checkpoint_needed=True,
    )
    denials = list(envelope["denials"])
    if any(lock["observed_python"][key] != lock["expected_python"][key]
           for key in ("implementation", "version")):
        denials.append("PYTHON_RUNTIME_LOCK_MISMATCH")
    report = {
        "schema": NATIVE_PREFLIGHT_SCHEMA,
        "decision": "NATIVE_PREFLIGHT_DENIED" if denials else "NATIVE_FLOORS_SCREENED_CAMPAIGN_PENDING",
        "candidate_packet_sha256": verified["packet_sha256"],
        "run_manifest_sha256": verified["run_manifest_sha256"],
        "source_git_sha": checkout,
        "environment_lock": lock,
        "resource_envelope": envelope,
        "denials": denials,
        "installed_distribution_provenance_verified": False,
        "target_profile_measured": False,
        "campaign_authorized": False,
        "training_executed": False,
        "paid_compute_used": False,
    }
    return {**report, "native_preflight_sha256": _sha(_canonical(report))}


def _gpu_resource_floors(model: ModelSpec, *, remaining_updates: int | None) -> dict[str, Any]:
    envelope = assess_resource_envelope(
        model, HostResources("cuda", None, None, 0),
        remaining_updates=1 if remaining_updates is None else remaining_updates,
        initial_checkpoint_needed=True,
    )
    return {
        "minimum_device_bytes_for_adamw": envelope["fp32_adamw_device_bytes"],
        "minimum_device_bytes_at_native_save": envelope["native_checkpoint_save_device_bytes"],
        "minimum_host_checkpoint_staging_bytes": envelope["checkpoint_payload_floor_bytes"],
        "minimum_initial_checkpoint_payload_bytes": envelope["serialized_model_weight_bytes"],
        "minimum_optimizer_checkpoint_payload_bytes": envelope["checkpoint_payload_floor_bytes"],
        "minimum_unpruned_checkpoint_volume_bytes": (
            None if remaining_updates is None else envelope["unpruned_checkpoint_volume_floor_bytes"]
        ),
    }


def plan_external_gpu_profiles(root: Path) -> dict[str, Any]:
    """Report the three exact native profile floors without selecting a device or starting a run."""
    checkout = _candidate_checkout_sha(root)
    recipe = TrainingRecipe.from_dict(json.loads(_exact_checkout_bytes(root, checkout, ONE_PASS_RECIPE_FILE)))
    _require(recipe.identity_sha256() == ONE_PASS_RECIPE_SHA256, "external GPU baseline recipe pin changed")
    profiles = []
    for profile, (source, source_sha256) in GPU_PROFILE_SOURCES.items():
        _exact_checkout_bytes(root, checkout, source)
        config, seed = _load_profile(root, profile)
        baseline = recipe if profile == PROFILE else None
        updates = None if baseline is None else baseline.trainer.max_steps
        profiles.append({
            "profile": profile,
            "profile_source_path": source,
            "profile_source_sha256": source_sha256,
            "model_spec_sha256": config.model.identity_sha256(),
            "init_spec_sha256": config.init.identity_sha256(),
            "parameters": config.model.parameter_count(),
            "max_context_tokens": config.model.max_seq_len,
            "initialization": seed,
            "general_baseline_recipe_sha256": None if baseline is None else baseline.identity_sha256(),
            "general_baseline_optimizer_updates": updates,
            "campaign_recipe_sha256": None,
            "campaign_data_release_sha256": None,
            "resource_floors": _gpu_resource_floors(config.model, remaining_updates=updates),
        })
    core = {
        "schema": GPU_PROFILE_PLAN_SCHEMA,
        "source_git_sha": checkout,
        "profile_preparation_order": list(GPU_PROFILE_SOURCES),
        "profiles": profiles,
        "resource_evidence": "ARITHMETIC_ONLY",
        "gpu_count": 1,
        "optimizer_state_layout": "ONE_FULL_FP32_ADAMW_REPLICA",
        "pending": [
            "qualify_scale_specific_cybersecurity_majority_data_and_independent_eval",
            "bind_scale_specific_campaign_recipe_and_update_horizon",
            "measure_selected_gpu_memory_checkpoint_storage_and_recovery",
            "verify_selected_runtime_and_obtain_campaign_authority",
        ],
        "target_profile_measured": False,
        "campaign_authorized": False,
        "training_executed": False,
        "paid_compute_used": False,
    }
    return {**core, "gpu_profile_plan_sha256": _sha(_canonical(core))}


def _cyber_linked(path: Path) -> bool:
    metadata = path.lstat()
    return path.is_symlink() or bool(getattr(metadata, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _audit_cyber_shards(candidate: Path, pack: dict[str, Any]) -> tuple[int, int, int]:
    shards = pack["shards"]
    shard_root = candidate / "shards"
    _require(not _cyber_linked(shard_root) and shard_root.is_dir(),
             "cyber shard directory is missing or linked")
    _require({p.name for p in shard_root.iterdir()} ==
             {f"shard-{i:06d}.json" for i in range(len(shards))}, "cyber shard inventory changed")
    blocks = targets = 0
    for i, shard in enumerate(shards):
        member = f"shards/shard-{i:06d}.json"
        _require(shard["path"] == member, "cyber shard manifest order changed")
        _require(not _cyber_linked(candidate / member), f"linked cyber shard: {member}")
        raw = _read(candidate / member)
        _require(len(raw) == shard["byte_count"] and _sha(raw) == shard["sha256"],
                 f"cyber shard bytes changed: {member}")
        value = json.loads(raw)
        _require(value["shard_index"] == i and value["block_size"] == pack["block_size"],
                 f"cyber shard identity changed: {member}")
        shard_targets = 0
        for block in value["blocks"]:
            mask = block["loss_mask"]
            _require(len(mask) == pack["block_size"] and all(type(bit) is int and bit in (0, 1) for bit in mask),
                     f"invalid causal mask: {member}")
            actual = sum(mask)
            _require(0 < actual == block["target_count"], f"cyber block target count changed: {member}")
            shard_targets += actual
        _require(len(value["blocks"]) == shard["block_count"] and shard_targets == shard["target_count"],
                 f"cyber shard accounting changed: {member}")
        blocks += len(value["blocks"])
        targets += shard_targets
    return len(shards), blocks, targets


def prepare_cyber_gpu_profiles(root: Path, candidate: Path) -> dict[str, Any]:
    """Audit frozen development shards and estimate profile exposure without creating launch authority."""
    plan = plan_external_gpu_profiles(root)
    checkout = plan["source_git_sha"]
    evidence_names = ("corpus", "holdout", "attack_eval")
    evidence = {
        key: _exact_checkout_bytes(root, checkout, f"{CYBER_EVIDENCE}/{key}_receipt.json")
        for key in evidence_names
    }
    pins = {key: _sha(raw) for key, raw in evidence.items()}
    corpus = json.loads(evidence["corpus"])
    holdout = json.loads(evidence["holdout"])
    independent = json.loads(evidence["attack_eval"])
    _require(corpus["scope"] == "SOURCE_VERIFIED_DEVELOPMENT_CANDIDATE"
             and corpus["public_redistribution_authorized"] is corpus["campaign_authorized"] is False
             and holdout["candidate_receipt_sha256"] == independent["candidate_receipt_sha256"] == pins["corpus"]
             and independent["verified_training_sha256"] == corpus["files"]["train.jsonl"]
             and independent["evaluation_only"] is True and independent["training_uses_evaluation"] is False,
             "cyber development receipts disagree")
    _require(not _cyber_linked(candidate) and candidate.is_dir(),
             "cyber candidate directory is missing or linked")
    receipt_path = candidate / "receipt.json"
    _require(not _cyber_linked(receipt_path) and _read(receipt_path) == evidence["corpus"],
             "cyber candidate receipt changed")
    for name, expected in corpus["files"].items():
        if name in {"validation.jsonl", "test.jsonl", "packing-manifest.json"}:
            continue
        _require(name in {"train.jsonl", "LICENSE-cisa.txt", "LICENSE-nist.txt", "general-rights/rights-attribution.json"}
                 or re.fullmatch(r"general-rights/notices/(?:[A-Za-z0-9_-][A-Za-z0-9_.-]*/)*"
                                 r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", name) is not None,
                 "unexpected cyber evidence member")
        member = candidate / name
        for parent in member.relative_to(candidate).parents:
            path = candidate / parent
            _require(not _cyber_linked(path), f"linked cyber member: {name}")
        _require(not _cyber_linked(member) and _sha(_read(member, limit=8_000_000)) == expected,
                 f"cyber member bytes changed: {name}")
    packing_path = candidate / "packing-manifest.json"
    _require(not _cyber_linked(packing_path), "linked cyber packing manifest")
    pack = _read_pinned_json(packing_path, corpus["files"]["packing-manifest.json"], limit=8_000_000)
    shards, blocks, targets = _audit_cyber_shards(candidate, pack)
    _require(blocks == pack["block_count"] == corpus["blocks"]
             and targets == pack["target_count"] == sum(corpus["causal_targets"].values())
             and pack["actual_loss_targets_by_stratum"] == corpus["causal_targets"], "cyber target accounting changed")
    blocks_per_microbatch = 3
    accumulation = 3
    blocks_per_update = blocks_per_microbatch * accumulation
    _require(blocks > 0 and blocks % blocks_per_update == 0, "example schedule does not divide cyber blocks exactly")
    profiles = []
    for row in plan["profiles"]:
        _require(pack["block_size"] <= row["max_context_tokens"], "cyber block exceeds profile context")
        profiles.append({
            "profile": row["profile"],
            "model_spec_sha256": row["model_spec_sha256"],
            "parameters": row["parameters"],
            "max_context_tokens": row["max_context_tokens"],
            "campaign_data_release_sha256": None,
            "campaign_recipe_sha256": None,
        })
    core = {
        "schema": CYBER_GPU_PREPARATION_SCHEMA,
        "scope": "DEVELOPMENT_PLANNING_ONLY",
        "source_git_sha": checkout,
        "gpu_profile_plan_sha256": plan["gpu_profile_plan_sha256"],
        "evidence_sha256": pins,
        "profiles": profiles,
        "verified_training_inventory": {
            "shards": shards, "blocks": blocks, "block_size": pack["block_size"],
            "causal_targets_by_stratum": corpus["causal_targets"], "cyber_share_ppm": corpus["cyber_share_ppm"],
        },
        "example_one_pass_arithmetic": {
            "blocks_per_microbatch": blocks_per_microbatch, "gradient_accumulation_microbatches": accumulation,
            "microbatches": blocks // blocks_per_microbatch, "optimizer_updates": blocks // blocks_per_update,
            "unique_causal_targets": targets, "recipe_approved": False,
        },
        "evaluation_payload_opened": False,
        "external_data_transfer_authorized": False,
        "campaign_authorized": False,
        "can_start_training": False,
        "training_executed": False,
    }
    return {**core, "preparation_sha256": _sha(_canonical(core))}


def build_external_gpu_packet(
    root: Path, release: Path, *, lock_profile: str, run_id: str,
) -> dict[str, Any]:
    """Prepare the exact local general-data baseline for an external single CUDA GPU."""
    _require(lock_profile in EXTERNAL_GPU_PROFILES, "external GPU baseline requires a committed Linux lock profile")
    checkout = _candidate_checkout_sha(root)
    lock = _lock_binding(root, checkout, lock_profile)
    recipe = TrainingRecipe.from_dict(json.loads(_exact_checkout_bytes(root, checkout, ONE_PASS_RECIPE_FILE)))
    _require(recipe.identity_sha256() == ONE_PASS_RECIPE_SHA256, "external GPU baseline recipe pin changed")
    budget = json.loads(_exact_checkout_bytes(root, checkout, ONE_PASS_BUDGET_FILE))
    budget_core = {key: value for key, value in budget.items() if key != "data_budget_sha256"}
    _require(budget.get("data_budget_sha256") == _sha(_canonical(budget_core)) == ONE_PASS_BUDGET_SHA256,
             "external GPU baseline data budget pin changed")
    candidate = build_candidate_packet(
        root, release, recipe=recipe, environment_lock_sha256=lock["runtime_lock_sha256"], run_id=run_id,
    )
    data = candidate["preparation"]["data"]
    _require(
        candidate["run_manifest"]["candidate"]["git_sha"] == checkout
        and budget["recipe_sha256"] == recipe.identity_sha256()
        and budget["release_manifest_sha256"] == data["release_manifest_sha256"]
        and budget["release_authority_sha256"] == data["release_authority_sha256"]
        and budget["nominal_optimizer_steps"] == recipe.trainer.max_steps
        and budget["planned_microbatches"] * recipe.microbatch_size == data["packed_block_count"]
        and budget["planned_microbatches"] == recipe.trainer.max_steps * recipe.trainer.gradient_accumulation_steps
        and budget["exact_one_pass_unique_loss_positions"] == data["ordered_causal_loss_positions"]
        and budget["exact_one_pass_total_exposures"] == recipe.maximum_total_exposures
        and budget["maximum_total_exposures"] == recipe.maximum_total_exposures
        and budget["campaign_authorized"] is False,
        "accepted data budget differs from the external GPU baseline candidate",
    )
    floors = _gpu_resource_floors(ModelSpec.from_dict(candidate["model_spec"]), remaining_updates=recipe.trainer.max_steps)
    core = {
        "schema": EXTERNAL_GPU_SCHEMA,
        "decision": "EXTERNAL_GPU_BASELINE_PREPARED_CAMPAIGN_PENDING",
        "profile_preparation_order": list(GPU_PROFILE_SOURCES),
        "candidate": candidate,
        "execution_requirements": {
            "placement": "EXTERNAL",
            "device": "cuda",
            "gpu_count": 1,
            "optimizer_state_layout": "ONE_FULL_REPLICA",
            "environment_lock": lock,
            "resource_evidence": "ARITHMETIC_ONLY",
            "minimum_device_bytes_for_adamw": floors["minimum_device_bytes_for_adamw"],
            "minimum_device_bytes_at_native_save": floors["minimum_device_bytes_at_native_save"],
            "minimum_host_checkpoint_staging_bytes": floors["minimum_host_checkpoint_staging_bytes"],
            "minimum_unpruned_checkpoint_volume_bytes": floors["minimum_unpruned_checkpoint_volume_bytes"],
        },
        "data": {
            "purpose": "GENERAL_BASELINE",
            "data_scope": budget["data_scope"],
            "one_pass_budget_sha256": ONE_PASS_BUDGET_SHA256,
            "cybersecurity_majority_verified": False,
        },
        "pending": [
            "qualify_cybersecurity_majority_physical_release_and_eval_firewall",
            "verify_external_target_cuda_driver_installed_wheels_and_measured_capacity",
            "verify_bounded_checkpoint_storage_transport_recovery_and_stop_policy",
            "obtain_explicit_campaign_and_compute_authority",
        ],
        "target_profile_measured": False,
        "campaign_authorized": False,
        "training_executed": False,
        "paid_compute_used": False,
    }
    return {**core, "external_gpu_packet_sha256": _sha(_canonical(core))}


def verify_external_gpu_packet(
    root: Path, release: Path, *, packet: dict[str, Any], expected_packet_sha256: str,
) -> dict[str, Any]:
    """Rebuild an external GPU baseline with the trusted checkout and physical release."""
    _require(type(packet) is dict and packet.get("external_gpu_packet_sha256") == expected_packet_sha256,
             "external GPU packet differs from the trusted identity")
    core = {key: value for key, value in packet.items() if key != "external_gpu_packet_sha256"}
    _require(_sha(_canonical(core)) == expected_packet_sha256, "external GPU packet digest mismatch")
    actual = build_external_gpu_packet(
        root, release, lock_profile=packet["execution_requirements"]["environment_lock"]["profile_id"],
        run_id=packet["candidate"]["run_manifest"]["run_id"],
    )
    _require(actual["external_gpu_packet_sha256"] == expected_packet_sha256,
             "external GPU packet differs from the verified baseline")
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
    parser.add_argument("--release", type=Path, help="Verified local data release for data-bound modes")
    parser.add_argument("--recipe", type=Path, help="Explicit candidate TrainingRecipe v1 JSON")
    parser.add_argument("--next-batch-index", type=int, help="Inspect the sealed stream cursor")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--build-packet", action="store_true", help="Emit a canonical candidate packet")
    modes.add_argument("--verify-packet", type=Path, help="Reopen a pinned candidate packet")
    modes.add_argument("--inspect-native-launch", type=Path, help="Check a pinned candidate on this local host")
    modes.add_argument("--build-external-gpu-packet", choices=EXTERNAL_GPU_PROFILES,
                       help="Prepare the general baseline for an external CUDA GPU without measuring a host")
    modes.add_argument("--plan-external-gpu-profiles", action="store_true",
                       help="Print 500M, 1B and 2B GPU floors without opening data or measuring a host")
    modes.add_argument("--prepare-cyber-gpu-profiles", type=Path,
                       help="Audit local cyber development shards against the ordered GPU profiles")
    modes.add_argument("--verify-external-gpu-packet", type=Path, help="Reopen a pinned external GPU baseline")
    modes.add_argument("--check-one-pass-budget", action="store_true", help="Check rights and exact data budget")
    parser.add_argument("--packet-sha256", help="Trusted packet digest for verification")
    parser.add_argument("--environment-lock", type=Path, help="Environment lock bytes to pin without host qualification")
    parser.add_argument("--run-id", help="Candidate run identity for packet construction")
    parser.add_argument("--optimizer-step", type=int, default=0, help="Candidate checkpoint position")
    parser.add_argument("--checkpoint-root", type=Path, help="Local checkpoint volume to inspect")
    parser.add_argument("--device", default="cpu", help="Local device to inspect for native launch")
    args = parser.parse_args()
    try:
        root = args.root.resolve(strict=True)
        if args.plan_external_gpu_profiles:
            _require(all(value is None for value in (
                args.release, args.recipe, args.next_batch_index, args.environment_lock,
                args.run_id, args.packet_sha256, args.checkpoint_root,
            )) and args.optimizer_step == 0 and args.device == "cpu",
                     "profile planning accepts --root only")
            sys.stdout.buffer.write(_canonical(plan_external_gpu_profiles(root)))
            return 0
        if args.prepare_cyber_gpu_profiles is not None:
            _require(all(value is None for value in (
                args.release, args.recipe, args.next_batch_index, args.environment_lock,
                args.run_id, args.packet_sha256, args.checkpoint_root,
            )) and args.optimizer_step == 0 and args.device == "cpu",
                     "cyber profile preparation accepts the candidate path and --root only")
            sys.stdout.buffer.write(_canonical(prepare_cyber_gpu_profiles(root, args.prepare_cyber_gpu_profiles)))
            return 0
        if args.release is None:
            raise LaunchPreparationDenied("data-bound modes require --release")
        recipe = None if args.recipe is None else TrainingRecipe.from_dict(json.loads(_read(args.recipe)))
        if args.build_external_gpu_packet is not None:
            if args.run_id is None:
                raise LaunchPreparationDenied("--build-external-gpu-packet requires --run-id")
            _require(args.recipe is None and args.environment_lock is None and args.next_batch_index is None
                     and args.optimizer_step == 0, "external GPU baseline requires the exact committed recipe and lock")
            receipt = build_external_gpu_packet(
                root, args.release, lock_profile=args.build_external_gpu_packet, run_id=args.run_id,
            )
            sys.stdout.buffer.write(_canonical(receipt))
            return 0
        if args.check_one_pass_budget:
            if recipe is None:
                raise LaunchPreparationDenied("--check-one-pass-budget requires --recipe")
            sys.stdout.buffer.write(_canonical(check_one_pass_data_budget(root, args.release, recipe=recipe)))
            return 0
        packet_path = args.verify_packet or args.inspect_native_launch or args.verify_external_gpu_packet
        if packet_path is not None:
            if args.packet_sha256 is None:
                raise LaunchPreparationDenied("packet readback requires a trusted --packet-sha256")
            packet_bytes = _read(packet_path)
            packet = json.loads(packet_bytes)
            _require(type(packet) is dict and _canonical(packet) == packet_bytes,
                     "candidate packet must use canonical JSON bytes")
        if args.verify_external_gpu_packet is not None:
            receipt = verify_external_gpu_packet(
                root, args.release, packet=packet, expected_packet_sha256=args.packet_sha256,
            )
            sys.stdout.buffer.write(_canonical(receipt))
            return 0
        if args.inspect_native_launch is not None:
            if args.checkpoint_root is None:
                raise LaunchPreparationDenied("--inspect-native-launch requires --checkpoint-root")
            receipt = inspect_native_launch(
                root, args.release, packet=packet, expected_packet_sha256=args.packet_sha256,
                checkpoint_root=args.checkpoint_root, device=args.device,
            )
            sys.stdout.buffer.write(_canonical(receipt))
            return 2 if receipt["denials"] else 0
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
