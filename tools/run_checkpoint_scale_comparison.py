from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from twelve_six.checkpoint.core import (
    CheckpointIdentity,
    hash_json,
    load_checkpoint,
    save_checkpoint,
    sha256_file,
)
from twelve_six.distributed.checkpoint_scale_ops import checkpoint_scale_policy
from twelve_six.distributed.contracts import ParallelPlan
from twelve_six.distributed.dcp_checkpoint import (
    ResumeMode,
    ScaleCheckpointIdentity,
    load_scale_checkpoint,
    save_scale_checkpoint,
)
from twelve_six.model import TwelveSixDecoder, load_stage_config

SCHEMA = "12-6.checkpoint-scale-comparison.v1"
AUTHORITY = "LOCAL_FREE_CHECKPOINT_MECHANICS_NOT_TRAINING_OR_CAPABILITY_EVIDENCE"


def _model_fingerprint(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _parameter_bytes(model: torch.nn.Module) -> int:
    return sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())


def _optimizer_tensor_bytes(optimizer: torch.optim.Optimizer) -> int:
    return sum(
        value.numel() * value.element_size()
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )


def _populate_adamw(model: torch.nn.Module) -> torch.optim.AdamW:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return optimizer


def _directory_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _identities(
    *,
    source_sha: str,
    stage: Any,
    stage_path: Path,
    model: TwelveSixDecoder,
    repo_root: Path,
) -> tuple[CheckpointIdentity, ScaleCheckpointIdentity]:
    marker = {
        "schema": SCHEMA,
        "stage": stage.stage,
        "stage_config_sha256": sha256_file(stage_path),
        "scope": "checkpoint-mechanics-only-no-training",
    }
    tokenizer_config = hash_json({**marker, "identity": "synthetic-tokenizer-config"})
    tokenizer_vocab = hash_json({**marker, "identity": "synthetic-tokenizer-vocab"})
    data_manifest = hash_json({**marker, "identity": "no-training-data"})
    run_manifest = hash_json({**marker, "identity": "checkpoint-comparison-run"})
    packing = hash_json({**marker, "identity": "no-training-packing"})
    training_config = {
        "training_executed": False,
        "purpose": "checkpoint-v1-vs-d18-mechanics-comparison",
    }
    training_hash = hash_json(training_config)
    environment_lock = sha256_file(repo_root / "requirements/locks/linux-x86_64/runtime.lock.txt")
    optimizer_descriptor = {
        "name": "AdamW",
        "lr": 1e-3,
        "weight_decay": 0.0,
        "state_population": "zero-gradient-single-step-for-moments-only",
    }
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    v1 = CheckpointIdentity(
        git_sha=source_sha,
        model_spec=model.spec.to_dict(),
        parameter_count=parameter_count,
        tokenizer_hash=tokenizer_config,
        tokenizer_vocab_hash=tokenizer_vocab,
        dataset_manifest_hash=data_manifest,
        run_manifest_hash=run_manifest,
        training_config=training_config,
        seed=20260825,
        precision="fp32",
        step=0,
        tokens_seen=0,
        optimizer=optimizer_descriptor,
        scheduler=None,
        environment_lock_hash=environment_lock,
    )
    d18 = ScaleCheckpointIdentity(
        git_sha=source_sha,
        model_spec_sha256=stage.model.identity_sha256(),
        init_spec_sha256=stage.init.identity_sha256(),
        tokenizer_config_sha256=tokenizer_config,
        tokenizer_vocab_sha256=tokenizer_vocab,
        data_manifest_sha256=data_manifest,
        packing_sha256=packing,
        training_config_sha256=training_hash,
        environment_lock_sha256=environment_lock,
        seed=20260825,
        step=0,
        tokens_seen=0,
    )
    return v1, d18


def _fresh_target(stage: Any) -> tuple[TwelveSixDecoder, torch.optim.AdamW]:
    model = TwelveSixDecoder(stage.model, stage.init).cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    return model, optimizer


def _run_sample(
    *,
    source_sha: str,
    stage_path: Path,
    repo_root: Path,
    work_root: Path,
    repetition: int,
) -> dict[str, Any]:
    stage = load_stage_config(stage_path)
    torch.manual_seed(20260825 + repetition)
    model = TwelveSixDecoder(stage.model, stage.init).cpu()
    optimizer = _populate_adamw(model)
    source_fingerprint = _model_fingerprint(model)
    v1_identity, d18_identity = _identities(
        source_sha=source_sha,
        stage=stage,
        stage_path=stage_path,
        model=model,
        repo_root=repo_root,
    )
    parameter_bytes = _parameter_bytes(model)
    optimizer_bytes = _optimizer_tensor_bytes(optimizer)
    v1_path = work_root / f"{stage.stage.lower()}-rep-{repetition}-v1"
    d18_path = work_root / f"{stage.stage.lower()}-rep-{repetition}-d18"

    v1_started = time.perf_counter()
    v1_manifest = save_checkpoint(
        v1_path,
        model=model,
        optimizer=optimizer,
        identity=v1_identity,
        trainer_state={"probe_repetition": repetition},
    )
    v1_save_seconds = time.perf_counter() - v1_started
    v1_snapshot_bytes = sum(record["bytes"] for record in v1_manifest["files"].values())
    v1_total_bytes = _directory_bytes(v1_path)

    d18_started = time.perf_counter()
    d18_manifest = save_scale_checkpoint(
        d18_path,
        model=model,
        optimizer=optimizer,
        plan=ParallelPlan(),
        identity=d18_identity,
        metadata={"probe_repetition": repetition, "stage": stage.stage},
        rank_state={"probe_repetition": repetition},
    )
    d18_save_seconds = time.perf_counter() - d18_started
    d18_payload_bytes = sum(record["size_bytes"] for record in d18_manifest["artifacts"])
    d18_total_bytes = _directory_bytes(d18_path)

    del optimizer
    del model
    gc.collect()

    torch.manual_seed(9000 + repetition)
    v1_model, v1_optimizer = _fresh_target(stage)
    v1_started = time.perf_counter()
    v1_result = load_checkpoint(
        v1_path,
        model=v1_model,
        optimizer=v1_optimizer,
        restore_rng=False,
        expected_git_sha=source_sha,
        expected_model_spec_hash=stage.model.identity_sha256(),
        expected_tokenizer_hash=v1_identity.tokenizer_hash,
        expected_tokenizer_vocab_hash=v1_identity.tokenizer_vocab_hash,
        expected_dataset_manifest_hash=v1_identity.dataset_manifest_hash,
        expected_run_manifest_hash=v1_identity.run_manifest_hash,
    )
    v1_load_seconds = time.perf_counter() - v1_started
    v1_exact = _model_fingerprint(v1_model) == source_fingerprint
    v1_optimizer_entries = len(v1_optimizer.state)
    del v1_result
    del v1_optimizer
    del v1_model
    gc.collect()

    torch.manual_seed(12000 + repetition)
    d18_model, d18_optimizer = _fresh_target(stage)
    d18_started = time.perf_counter()
    d18_result = load_scale_checkpoint(
        d18_path,
        model=d18_model,
        optimizer=d18_optimizer,
        target_plan=ParallelPlan(),
        mode=ResumeMode.EXACT_TOPOLOGY,
        expected_identity_sha256=d18_identity.sha256,
    )
    d18_load_seconds = time.perf_counter() - d18_started
    d18_exact = _model_fingerprint(d18_model) == source_fingerprint
    d18_optimizer_entries = len(d18_optimizer.state)
    d18_rank_state_exact = d18_result.rank_state == {"probe_repetition": repetition}
    expected_optimizer_entries = len(list(d18_model.parameters()))

    shutil.rmtree(v1_path)
    shutil.rmtree(d18_path)
    return {
        "repetition": repetition,
        "parameters": v1_identity.parameter_count,
        "model_parameter_bytes": parameter_bytes,
        "optimizer_tensor_bytes": optimizer_bytes,
        "expected_optimizer_state_entries": expected_optimizer_entries,
        "v1": {
            "checkpoint_total_bytes": v1_total_bytes,
            "verified_snapshot_payload_bytes_retained_in_ram": v1_snapshot_bytes,
            "save_seconds": v1_save_seconds,
            "load_seconds": v1_load_seconds,
            "model_fingerprint_exact": v1_exact,
            "optimizer_state_entries": v1_optimizer_entries,
        },
        "d18": {
            "checkpoint_total_bytes": d18_total_bytes,
            "stream_verified_payload_bytes_not_retained_as_one_snapshot": d18_payload_bytes,
            "save_seconds": d18_save_seconds,
            "load_seconds": d18_load_seconds,
            "model_fingerprint_exact": d18_exact,
            "optimizer_state_entries": d18_optimizer_entries,
            "rank_state_exact": d18_rank_state_exact,
            "aggregate_checkpoint_sha256": d18_manifest["aggregate_checkpoint_sha256"],
        },
    }


def _median(samples: list[dict[str, Any]], branch: str, field: str) -> float:
    return float(statistics.median(float(sample[branch][field]) for sample in samples))


def build_report(
    *,
    source_sha: str,
    stage_path: Path,
    repo_root: Path,
    work_root: Path,
    repetitions: int,
) -> dict[str, Any]:
    stage = load_stage_config(stage_path)
    samples = [
        _run_sample(
            source_sha=source_sha,
            stage_path=stage_path,
            repo_root=repo_root,
            work_root=work_root,
            repetition=index,
        )
        for index in range(repetitions)
    ]
    for sample in samples:
        if not sample["v1"]["model_fingerprint_exact"]:
            raise RuntimeError("checkpoint-v1 model round-trip mismatch")
        if not sample["d18"]["model_fingerprint_exact"]:
            raise RuntimeError("D18 model round-trip mismatch")
        expected = sample["expected_optimizer_state_entries"]
        if sample["v1"]["optimizer_state_entries"] != expected:
            raise RuntimeError("checkpoint-v1 optimizer state round-trip mismatch")
        if sample["d18"]["optimizer_state_entries"] != expected:
            raise RuntimeError("D18 optimizer state round-trip mismatch")
        if not sample["d18"]["rank_state_exact"]:
            raise RuntimeError("D18 rank-state round-trip mismatch")

    v1_total = int(statistics.median(sample["v1"]["checkpoint_total_bytes"] for sample in samples))
    d18_total = int(
        statistics.median(sample["d18"]["checkpoint_total_bytes"] for sample in samples)
    )
    v1_snapshot = int(
        statistics.median(
            sample["v1"]["verified_snapshot_payload_bytes_retained_in_ram"] for sample in samples
        )
    )
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "source_sha": source_sha,
        "stage": stage.stage,
        "stage_config": stage_path.relative_to(repo_root).as_posix(),
        "stage_config_sha256": sha256_file(stage_path),
        "model_spec_sha256": stage.model.identity_sha256(),
        "init_spec_sha256": stage.init.identity_sha256(),
        "parameters": stage.model.parameter_count(),
        "expected_parameters": stage.expected_parameters,
        "repetitions": repetitions,
        "policy": checkpoint_scale_policy(stage.model.parameter_count()).__dict__,
        "samples": samples,
        "summary": {
            "v1_checkpoint_total_bytes_median": v1_total,
            "d18_checkpoint_total_bytes_median": d18_total,
            "d18_to_v1_storage_ratio": d18_total / v1_total,
            "v1_verified_snapshot_payload_bytes_median": v1_snapshot,
            "v1_snapshot_to_model_parameter_bytes_ratio": (
                v1_snapshot / samples[0]["model_parameter_bytes"]
            ),
            "v1_save_seconds_median": _median(samples, "v1", "save_seconds"),
            "v1_load_seconds_median": _median(samples, "v1", "load_seconds"),
            "d18_save_seconds_median": _median(samples, "d18", "save_seconds"),
            "d18_load_seconds_median": _median(samples, "d18", "load_seconds"),
        },
        "runtime": {
            "torch": torch.__version__,
            "cpu_threads": torch.get_num_threads(),
            "distributed_backend": dist.get_backend(),
            "world_size": dist.get_world_size(),
            "locked_runtime_sha256": sha256_file(
                repo_root / "requirements/locks/linux-x86_64/runtime.lock.txt"
            ),
        },
        "truth_boundary": {
            "training_executed": False,
            "optimizer_moments_populated": True,
            "paid_compute": False,
            "quality_or_capability_evidence": False,
            "single_rank_gloo": True,
            "fsdp_dtensor_shards": "NOT_TESTED_BY_THIS_PROBE",
            "async_save": "NOT_IMPLEMENTED_BY_D18_INCUMBENT",
            "object_storage": "NOT_IMPLEMENTED_BY_D18_INCUMBENT",
        },
    }
    report["report_sha256"] = hash_json(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--stage-config", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    if len(args.source_sha) != 40 or args.source_sha != args.source_sha.lower() or any(
        char not in "0123456789abcdef" for char in args.source_sha
    ):
        raise ValueError("--source-sha must be exact lowercase 40-hex")
    if args.repetitions < 1 or args.cpu_threads < 1:
        raise ValueError("repetitions and cpu-threads must be positive")

    repo_root = args.repo_root.resolve()
    stage_path = args.stage_config
    if not stage_path.is_absolute():
        stage_path = repo_root / stage_path
    stage_path = stage_path.resolve()
    output = args.output.resolve()
    work_root = output.parent / f".{output.stem}.checkpoint-work"
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True)
    torch.set_num_threads(args.cpu_threads)
    init_path = work_root / "gloo-init"
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_path}",
        rank=0,
        world_size=1,
    )
    try:
        report = build_report(
            source_sha=args.source_sha,
            stage_path=stage_path,
            repo_root=repo_root,
            work_root=work_root,
            repetitions=args.repetitions,
        )
    finally:
        dist.destroy_process_group()
        shutil.rmtree(work_root, ignore_errors=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(
        json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
        + b"\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
