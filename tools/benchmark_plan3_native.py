"""Screen or explicitly benchmark one exact Plan 3 profile on Linux CUDA."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import multiprocessing
import os
import re
import subprocess
import sys
import tempfile
import time
from importlib import metadata
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import torch

from tools.bootstrap_dependency_lock import _packages
from twelve_six.checkpoint import CheckpointIdentity, hash_json, load_trainer_checkpoint
from twelve_six.checkpoint.core import _strict_json_object
from twelve_six.integration import dependency_lock
from twelve_six.model import BaseInitManifest, StageConfig, build_canonical_base
from twelve_six.training import Trainer, runtime_release
from twelve_six.training.config import TrainerConfig, TrainingRecipe
from twelve_six.training.resource_envelope import (
    assess_resource_envelope,
    benchmark_optimizer_boundary,
    observe_host_resources,
    process_peak_working_set_bytes,
)

ROOT = Path(__file__).resolve().parents[1]
PROFILES = {
    f"SCALE-{name}": f"configs/stages/scale_{name.lower()}.json"
    for name in ("500M", "750M", "1B", "2B", "3B", "4B", "5B")
}
SCHEMA = "12-6.plan3-native-profile-benchmark-preflight.v1"
MEASUREMENT_SCHEMA = "12-6.plan3-native-profile-benchmark-measurement.v2"
RUNTIME_RELEASES = {
    1: (runtime_release.RELEASE_PATH, runtime_release.RELEASE_SHA256),
    2: (runtime_release.RELEASE_V2_PATH, runtime_release.RELEASE_V2_SHA256),
}
DEVICE = re.compile(r"cpu|cuda:[0-9]+")
SHA256 = re.compile(r"[0-9a-f]{64}")
GIT_SHA = re.compile(r"[0-9a-f]{40}")
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


def restore_memory_floors(report: dict, device: str) -> tuple[int, int]:
    """Conservative host and device arithmetic for a native checkpoint restore."""
    payload = report["checkpoint_payload_floor_bytes"]
    weights = report["serialized_model_weight_bytes"]
    device_bytes = report["native_checkpoint_save_device_bytes"]
    host_bytes = 4 * payload + weights + HOST_RESERVE_BYTES
    if device == "cpu":
        host_bytes = max(host_bytes, device_bytes + HOST_RESERVE_BYTES)
        return host_bytes, host_bytes
    return host_bytes, max(device_bytes, 2 * payload + weights)


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
    host_bytes, device_bytes = restore_memory_floors(report, device)
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


def _verify_measurement_inputs(
    root: Path, source_sha: str, expected_lock_sha: str, install_report: Path, report_sha: str,
    release_version: int,
) -> dict:
    if _git(root, "rev-parse", "HEAD").decode().strip() != source_sha:
        raise ValueError("benchmark checkout differs from the explicitly pinned source")
    if _git(root, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("benchmark requires clean tracked checkout files")
    for component in (
        observe_profile, _packages, dependency_lock.validate_lock_index,
        runtime_release.verify_runtime_release, build_canonical_base, Trainer,
        benchmark_optimizer_boundary, load_trainer_checkpoint,
    ):
        relative = Path(inspect.getfile(component)).resolve().relative_to(root).as_posix()
        _committed_file(root, source_sha, relative)
    release_path, release_sha = RUNTIME_RELEASES[release_version]
    runtime_release.verify_runtime_release(root, release_version=release_version)
    _committed_file(root, source_sha, release_path)
    dependency_lock.assert_exact_python()
    platform = dependency_lock.current_profile_id()
    index_path = "requirements/locks/index.json"
    index = dependency_lock.validate_lock_index(root=root, index_path=index_path)
    profile_path = index["profiles"][platform]["path"]
    lock = dependency_lock.validate_profile_manifest(root=root, manifest_path=profile_path)
    runtime = lock["locks"]["runtime"]
    runtime_bytes = _committed_file(root, source_sha, runtime["path"])
    if hashlib.sha256(runtime_bytes).hexdigest() != expected_lock_sha:
        raise ValueError("selected runtime lock differs from the explicitly pinned digest")
    _committed_file(root, source_sha, profile_path)
    _committed_file(root, source_sha, index_path)
    _committed_file(root, source_sha, "pyproject.toml")

    raw_report = install_report.read_bytes()
    if hashlib.sha256(raw_report).hexdigest() != report_sha:
        raise ValueError("runtime install report differs from the explicitly pinned digest")
    report = _strict_json_object(
        raw_report, artifact="runtime install report", invalid_message="invalid install report",
    )
    if report.get("version") != "1":
        raise ValueError("unsupported pip install report version")
    installed = _packages(report)
    installations = report["install"]
    if len(installations) != len(installed) or any(
        not urlsplit(item["download_info"]["url"]).path.lower().endswith(".whl")
        for item in installations
    ):
        raise ValueError("runtime install report must contain one wheel per distribution")
    expected = {}
    for line in runtime_bytes.decode().splitlines():
        if line and not line.startswith("#"):
            requirement, *hashes = line.split()
            name, version = requirement.split("==")
            prefix = "--hash=sha256:"
            digests = {value.removeprefix(prefix) for value in hashes}
            if (
                not digests or any(SHA256.fullmatch(digest) is None for digest in digests)
                or any(not value.startswith(prefix) for value in hashes) or name in expected
            ):
                raise ValueError(f"runtime lock requires exact unique hashed wheels: {name}")
            expected[name] = (version, digests)
    if len(expected) != runtime["package_count"]:
        raise ValueError("runtime lock distribution count differs from the committed profile")
    if set(installed) != set(expected):
        raise ValueError("runtime install report distribution set differs from the lock")
    for name, (version, archives) in installed.items():
        locked_version, allowed_archives = expected[name]
        if version != locked_version or not set(archives) <= allowed_archives:
            raise ValueError(f"runtime install report wheel differs from the lock: {name}")
        if metadata.version(name) != locked_version:
            raise ValueError(f"installed runtime version differs from the lock: {name}")
    return {
        "profile_id": platform,
        "runtime_lock_sha256": expected_lock_sha,
        "install_report_sha256": report_sha,
        "verified_distribution_versions": len(installed),
        "runtime_release_version": release_version,
        "runtime_release_sha256": release_sha,
    }


def _worker_entry(sender, operation, args) -> None:
    with sender:
        try:
            result = operation(*args)
        except (KeyError, OSError, RuntimeError, TypeError, ValueError, subprocess.SubprocessError) as exc:
            sender.send((False, f"{type(exc).__name__}: {exc}"))
        else:
            sender.send((True, result))


def _in_fresh_process(operation, *args):
    spawn = multiprocessing.get_context("spawn")
    receiver, sender = spawn.Pipe(duplex=False)
    worker = spawn.Process(target=_worker_entry, args=(sender, operation, args))
    with receiver, sender:
        worker.start()
        sender.close()
        try:
            success, report = receiver.recv()
        except EOFError as exc:
            worker.join()
            raise RuntimeError(f"benchmark worker exited without a report ({worker.exitcode})") from exc
        except KeyboardInterrupt:
            worker.terminate()
            worker.join()
            raise
        worker.join()
        if worker.exitcode or not success:
            raise RuntimeError(f"benchmark worker failed ({worker.exitcode}): {report}")
    return report


def _synthetic_identity(preflight: dict, config: StageConfig, lock_sha: str) -> CheckpointIdentity:
    fixture = preflight["fixture"]
    fixture_input = hash_json({
        "schema": "12-6.synthetic-integer-token-fixture.v1", "input_ids": list(range(WIDTH)),
    })
    return CheckpointIdentity(
        git_sha=preflight["source_git_sha"], model_spec=config.model.to_dict(),
        parameter_count=config.expected_parameters, tokenizer_hash=fixture_input,
        tokenizer_vocab_hash=fixture_input, dataset_manifest_hash=fixture_input,
        run_manifest_hash=hash_json({
            "profile_config_sha256": preflight["profile_config_sha256_lf"],
            "recipe_sha256": fixture["recipe_sha256"], "fixture_sha256": fixture_input,
        }),
        training_config=fixture["recipe"], seed=SEED, precision="fp32", step=0, tokens_seen=0,
        optimizer={"name": "AdamW"}, scheduler=None,
        environment_lock_hash=lock_sha,
    )


def _benchmark_trainer(
    config: StageConfig, init: BaseInitManifest, recipe: TrainingRecipe, device: str,
) -> Trainer:
    if torch.get_default_dtype() != torch.float32 or torch.get_default_device().type != "cpu":
        raise ValueError("benchmark requires default FP32 model allocation on CPU")
    np.random.seed(SEED)
    return Trainer(build_canonical_base(config.model, config.init, init), recipe.trainer, device=device)


def _worker_memory(device: str) -> dict:
    return {
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "process_peak_working_set_bytes": process_peak_working_set_bytes(),
    }


def _cuda_environment(workspace: str) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace
    if torch.version.cuda is None or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("benchmark requires exactly one visible NVIDIA CUDA device")


def cuda_checkpoint_context(scratch: Path, device: str) -> dict:
    """Read the selected physical GPU and the checkpoint filesystem identity."""
    properties = torch.cuda.get_device_properties(device)
    uuid = getattr(properties, "uuid", None)
    if not uuid or not str(uuid).strip():
        raise RuntimeError("CUDA device does not expose a stable GPU UUID")
    checkpoint_fs = os.statvfs(scratch)
    return {
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip(),
        "device": {
            "uuid": str(uuid), "name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "total_memory_bytes": properties.total_memory, "cuda_runtime": torch.version.cuda,
            "torch_version": str(torch.__version__),
        },
        "checkpoint_filesystem": {
            "st_dev": scratch.stat().st_dev,
            "f_fsid": getattr(checkpoint_fs, "f_fsid", None),
        },
    }


def _benchmark_preflight(root: Path, scratch: Path, profile: str, device: str, workspace: str):
    _cuda_environment(workspace)
    return observe_profile(root, scratch, profile, device)


def checkpoint_context_matches(expected: dict, observed: dict) -> bool:
    """Compare filesystem ID when provided, falling back to this boot's device ID."""
    expected_fs, observed_fs = expected["checkpoint_filesystem"], observed["checkpoint_filesystem"]
    if expected_fs["f_fsid"] and observed_fs["f_fsid"]:
        same_fs = expected_fs["f_fsid"] == observed_fs["f_fsid"]
    else:
        same_fs = expected["boot_id"] == observed["boot_id"] and expected_fs["st_dev"] == observed_fs["st_dev"]
    return expected["device"] == observed["device"] and same_fs


def _measurement_worker(
    root: Path, device: str, preflight: dict, lock_sha: str, checkpoint: Path, workspace: str,
):
    profile, source = preflight["profile"], preflight["source_git_sha"]
    live = _benchmark_preflight(root, checkpoint.parent, profile, device, workspace)
    _require_capacity(live)
    if (
        live["source_git_sha"] != source or live["fixture"] != preflight["fixture"]
        or live["profile_config_sha256_lf"] != preflight["profile_config_sha256_lf"]
    ):
        raise ValueError("benchmark inputs changed after capacity screening")
    config, init, _ = _profile(root, source, profile)
    identity = _synthetic_identity(preflight, config, lock_sha)
    recipe = TrainingRecipe.from_dict(preflight["fixture"]["recipe"])
    context = cuda_checkpoint_context(checkpoint.parent, device)
    torch.cuda.reset_peak_memory_stats(device)
    result = benchmark_optimizer_boundary(
        _benchmark_trainer(config, init, recipe, device),
        lambda: _benchmark_trainer(config, init, recipe, device),
        identity,
        [{"input_ids": torch.arange(WIDTH, dtype=torch.long).view(1, WIDTH)}],
        checkpoint,
        projection_tokens=recipe.target_unique_loss_positions,
    )
    if result["tokens_seen"] != WIDTH - 1:
        raise RuntimeError("benchmark did not consume the exact synthetic target count")
    return {
        "boundary": result, "memory": _worker_memory(device), "worker_pid": os.getpid(),
        "execution_context": context,
    }


def _restore_worker(
    root: Path, device: str, preflight: dict, measured: dict, checkpoint: Path, workspace: str,
):
    _cuda_environment(workspace)
    context = cuda_checkpoint_context(checkpoint.parent, device)
    observed = observe_host_resources(checkpoint.parent, device)
    limits = preflight["capacity"]
    if (
        (observed.host_available_bytes or 0) < limits["benchmark_host_required_bytes"]
        or (observed.device_available_bytes or 0) < limits["benchmark_device_required_bytes"]
    ):
        raise RuntimeError("fresh-process restore capacity no longer meets the benchmark floor")
    config, init, _ = _profile(root, preflight["source_git_sha"], preflight["profile"])
    recipe = TrainingRecipe.from_dict(preflight["fixture"]["recipe"])
    torch.cuda.reset_peak_memory_stats(device)
    trainer = _benchmark_trainer(config, init, recipe, device)
    torch.cuda.synchronize(device)
    start = time.perf_counter_ns()
    load_trainer_checkpoint(
        checkpoint, model=trainer.model, trainer=trainer,
        expected_checkpoint_id=measured["checkpoint_id"],
        expected_run_manifest_hash=measured["run_manifest_sha256"],
        expected_step=1, expected_tokens_seen=measured["tokens_seen"],
    )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter_ns() - start
    if trainer.optimizer_step != 1 or trainer.tokens_seen != measured["tokens_seen"]:
        raise RuntimeError("fresh process did not restore the measured checkpoint boundary")
    return {
        "checkpoint_id": measured["checkpoint_id"], "restore_ns": elapsed,
        "committed_step": trainer.optimizer_step, "tokens_seen": trainer.tokens_seen,
        "memory": _worker_memory(device),
        "worker_pid": os.getpid(),
        "execution_context": context,
    }


def _require_capacity(preflight: dict) -> None:
    capacity = preflight["capacity"]
    if capacity["disposition"] != "CAPACITY_SCREEN_PASSED" or capacity["benchmark_capacity_denials"]:
        raise RuntimeError("benchmark capacity denied: " + ", ".join(capacity["benchmark_capacity_denials"]))


def measure_profile(
    root: Path, scratch: Path, profile: str, device: str, *,
    source_sha: str, runtime_lock_sha: str, install_report: Path, install_report_sha: str,
    runtime_release_version: int,
) -> dict:
    """Run one synthetic optimizer boundary after explicit Linux CUDA admission."""
    if sys.platform != "linux" or device != "cuda:0":
        raise ValueError("physical benchmark requires Linux and explicit cuda:0 with one visible GPU")
    if type(runtime_release_version) is not int or runtime_release_version not in RUNTIME_RELEASES:
        raise ValueError("physical benchmark requires an explicit trusted runtime release version")
    if (
        GIT_SHA.fullmatch(source_sha) is None
        or SHA256.fullmatch(runtime_lock_sha) is None
        or SHA256.fullmatch(install_report_sha) is None
    ):
        raise ValueError("physical benchmark requires exact source, lock and install report digests")
    root, scratch = root.resolve(strict=True), scratch.resolve(strict=True)
    lock = _verify_measurement_inputs(
        root, source_sha, runtime_lock_sha, install_report, install_report_sha, runtime_release_version,
    )
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if workspace not in {":4096:8", ":16:8"}:
        raise ValueError("benchmark requires a deterministic CUBLAS_WORKSPACE_CONFIG")
    preflight = _in_fresh_process(_benchmark_preflight, root, scratch, profile, device, workspace)
    if preflight["source_git_sha"] != source_sha:
        raise ValueError("benchmark source changed during preflight")
    _require_capacity(preflight)
    with tempfile.TemporaryDirectory(prefix="plan3-cuda-benchmark-", dir=scratch) as directory:
        checkpoint = Path(directory) / "checkpoint"
        measured = _in_fresh_process(
            _measurement_worker, root, device, preflight, runtime_lock_sha, checkpoint, workspace,
        )
        restored = _in_fresh_process(
            _restore_worker, root, device, preflight, measured["boundary"], checkpoint, workspace,
        )
    if (
        restored["checkpoint_id"] != measured["boundary"]["checkpoint_id"]
        or restored["tokens_seen"] != measured["boundary"]["tokens_seen"]
        or restored["committed_step"] != 1
        or not checkpoint_context_matches(measured["execution_context"], restored["execution_context"])
    ):
        raise RuntimeError("fresh process receipt differs from the measured checkpoint")
    receipt = {
        "schema": MEASUREMENT_SCHEMA,
        "scope": "SYNTHETIC_SINGLE_OPTIMIZER_BOUNDARY_AND_FRESH_PROCESS_RESTORE",
        "profile": profile, "device": device, "source_git_sha": source_sha,
        "environment": {**lock, "cublas_workspace_config": workspace},
        "capacity_preflight": preflight, "measurement": measured, "fresh_process_restore": restored,
        "optimizer_steps_executed": 1, "campaign_authorized": False,
        "compute_billing_observed": False,
    }
    return {**receipt, "receipt_sha256": hash_json(receipt)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=PROFILES)
    parser.add_argument("--scratch", type=Path, required=True, help="Existing checkpoint volume")
    parser.add_argument("--device", default="cpu", help="cpu or cuda:N")
    parser.add_argument("--benchmark", action="store_true", help="Measure one step on Linux CUDA")
    parser.add_argument("--source-sha")
    parser.add_argument("--runtime-release-version", type=int, choices=RUNTIME_RELEASES)
    parser.add_argument("--runtime-lock-sha256")
    parser.add_argument("--install-report", type=Path)
    parser.add_argument("--install-report-sha256")
    args = parser.parse_args(argv)
    try:
        pins = (
            args.source_sha, args.runtime_lock_sha256, args.install_report,
            args.install_report_sha256, args.runtime_release_version,
        )
        if args.benchmark:
            if not all(pins):
                raise ValueError(
                    "benchmark mode requires a source SHA, trusted runtime release version, "
                    "runtime lock SHA and pinned install report"
                )
            receipt = measure_profile(
                ROOT, args.scratch, args.profile, args.device, source_sha=args.source_sha,
                runtime_lock_sha=args.runtime_lock_sha256, install_report=args.install_report,
                install_report_sha=args.install_report_sha256,
                runtime_release_version=args.runtime_release_version,
            )
        else:
            if any(pins):
                raise ValueError("physical benchmark inputs require --benchmark")
            receipt = observe_profile(ROOT, args.scratch, args.profile, args.device)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"benchmark observation refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
