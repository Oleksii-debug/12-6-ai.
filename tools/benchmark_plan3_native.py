"""Observe native Plan 3 benchmark admission for one exact scale profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

from twelve_six.checkpoint import hash_json
from twelve_six.checkpoint.core import _strict_json_object
from twelve_six.model import BaseInitManifest, StageConfig
from twelve_six.training.config import TrainerConfig, TrainingRecipe
from twelve_six.training.resource_envelope import assess_resource_envelope, observe_host_resources

ROOT = Path(__file__).resolve().parents[1]
PROFILES = {
    f"SCALE-{name}": f"configs/stages/scale_{name.lower()}.json"
    for name in ("500M", "750M", "1B", "2B", "3B", "4B", "5B")
}
SCHEMA = "12-6.plan3-native-profile-benchmark-preflight.v1"
DEVICE = re.compile(r"cpu|cuda:[0-9]+")
STEPS = 4
MIDPOINT = 2
WIDTH = 32
SEED = 341
HOST_RESERVE_BYTES = 1024**3
DISK_RESERVE_BYTES = 128 * 1024**2


def _git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=root, timeout=10)


def _committed_file(root: Path, commit: str, relative: str) -> bytes:
    path = root / relative
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError(f"benchmark input path is unavailable: {relative}")
    raw = path.read_bytes().replace(b"\r\n", b"\n")
    expected = _git(root, "show", f"{commit}:{relative}")
    if raw != expected:
        raise ValueError(f"benchmark input differs from the source commit: {relative}")
    return raw


def _profile(root: Path, commit: str, profile: str) -> tuple[StageConfig, BaseInitManifest, str]:
    relative = PROFILES[profile]
    raw = _committed_file(root, commit, relative)
    metadata = _strict_json_object(raw, artifact=relative, invalid_message="invalid profile")
    config = StageConfig.from_dict(metadata)
    init = BaseInitManifest.from_dict(metadata["base_init_manifest"])
    if (
        metadata["profile_id"] != profile
        or config.stage != profile
        or config.canonical_base != "random_init"
        or config.model.parameter_count() != config.expected_parameters
        or config.model.identity_sha256() != config.expected_model_identity_sha256
        or config.init.identity_sha256() != config.expected_init_identity_sha256
        or init.model_spec_sha256 != config.model.identity_sha256()
        or init.init_spec_sha256 != config.init.identity_sha256()
    ):
        raise ValueError("selected profile identity differs")
    return config, init, hashlib.sha256(raw).hexdigest()


def observe_profile(root: Path, scratch: Path, profile: str, device: str) -> dict:
    """Build capacity evidence and a synthetic recipe without model allocation."""
    if DEVICE.fullmatch(device) is None:
        raise ValueError("select cpu or an explicit cuda:N device")
    root = root.resolve(strict=True)
    scratch = scratch.resolve(strict=True)
    if not scratch.is_dir():
        raise ValueError("scratch must be an existing directory")
    commit = _git(root, "rev-parse", "HEAD").decode().strip()
    config, init, profile_sha256 = _profile(root, commit, profile)
    if config.model.max_seq_len < WIDTH:
        raise ValueError("selected profile cannot hold the benchmark fixture")
    lock = _committed_file(root, commit, "requirements/locks/index.json")
    tokens = STEPS * (WIDTH - 1)
    recipe = TrainingRecipe(
        trainer=TrainerConfig(max_steps=STEPS, seed=SEED),
        microbatch_size=1,
        sequence_length=WIDTH,
        target_unique_loss_positions=tokens,
        maximum_total_exposures=tokens,
        max_exposures_per_unique_position=1,
    )
    observed = observe_host_resources(scratch, device)
    report = assess_resource_envelope(
        config.model, observed, remaining_updates=3, initial_checkpoint_needed=False,
    )
    # Separate checkpoint staging on the host from the live optimizer/model
    # device. The observation includes headroom for a fresh process to restore.
    payload = report["checkpoint_payload_floor_bytes"]
    weights = report["serialized_model_weight_bytes"]
    device_bytes = report["native_checkpoint_save_device_bytes"]
    host_bytes = 4 * payload + weights + HOST_RESERVE_BYTES
    if device == "cpu":
        host_bytes = max(host_bytes, device_bytes + HOST_RESERVE_BYTES)
        device_bytes = host_bytes
    else:
        device_bytes = max(device_bytes, 2 * payload + weights)
    disk_bytes = report["unpruned_checkpoint_volume_floor_bytes"] + DISK_RESERVE_BYTES
    denials = list(report["denials"])
    if (observed.host_available_bytes or 0) < host_bytes:
        denials.append("BENCHMARK_HOST_MEMORY_BELOW_RESTORE_ENVELOPE")
    if (observed.device_available_bytes or 0) < device_bytes:
        denials.append("BENCHMARK_DEVICE_MEMORY_BELOW_RESTORE_ENVELOPE")
    if observed.checkpoint_volume_free_bytes < disk_bytes:
        denials.append("BENCHMARK_DISK_BELOW_SCRATCH_ENVELOPE")
    receipt = {
        "schema": SCHEMA,
        "source_git_sha": commit,
        "source_tree_sha": _git(root, "rev-parse", f"{commit}^{{tree}}").decode().strip(),
        "profile": profile,
        "profile_config_sha256_lf": profile_sha256,
        "model_spec_sha256": config.model.identity_sha256(),
        "initialization": init.to_dict(),
        "environment_lock_sha256_lf": hashlib.sha256(lock).hexdigest(),
        "fixture": {
            "input": "DETERMINISTIC_SYNTHETIC_TOKEN_IDS_ONLY",
            "recipe": recipe.to_dict(),
            "recipe_sha256": recipe.identity_sha256(),
            "numpy_process_seed": SEED,
            "restart_boundary_step": MIDPOINT,
            "independent_baseline": True,
        },
        "capacity": {
            **report,
            "benchmark_host_required_bytes": host_bytes,
            "benchmark_device_required_bytes": device_bytes,
            "benchmark_disk_required_bytes": disk_bytes,
            "benchmark_capacity_denials": denials,
            "disposition": "CAPACITY_DENIED" if denials else "CAPACITY_SCREEN_PASSED",
        },
        "physical_measurement": "NOT_RUN",
        "model_allocated": False,
        "optimizer_steps_executed": 0,
        "checkpoint_files_written": 0,
        "campaign_authorized": False,
        "paid_compute_used": False,
    }
    return {**receipt, "receipt_sha256": hash_json(receipt)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=PROFILES)
    parser.add_argument("--scratch", type=Path, required=True, help="Existing checkpoint volume")
    parser.add_argument("--device", default="cpu", help="cpu or cuda:N")
    args = parser.parse_args(argv)
    try:
        receipt = observe_profile(ROOT, args.scratch, args.profile, args.device)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"benchmark observation refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
