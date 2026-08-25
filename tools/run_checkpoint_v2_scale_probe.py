from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import shutil
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from twelve_six.checkpoint.core import CheckpointIdentity, hash_json, sha256_file
from twelve_six.checkpoint.v2 import load_checkpoint_v2, save_checkpoint_v2
from twelve_six.model import TwelveSixDecoder, load_stage_config

SCHEMA = "12-6.checkpoint-v2-scale-probe.v1"
AUTHORITY = "LOCAL_FREE_CHECKPOINT_MECHANICS_EVIDENCE_NOT_STAGE_OR_CAPABILITY_EVIDENCE"


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
    total = 0
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                total += value.numel() * value.element_size()
    return total


def _populate_adamw_state(model: torch.nn.Module) -> torch.optim.AdamW:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return optimizer


def _identity(
    *,
    source_sha: str,
    stage_name: str,
    stage_config_path: Path,
    model: TwelveSixDecoder,
) -> CheckpointIdentity:
    marker = {
        "schema": SCHEMA,
        "scope": "checkpoint-mechanics-only",
        "stage": stage_name,
        "stage_config_sha256": sha256_file(stage_config_path),
    }
    return CheckpointIdentity(
        git_sha=source_sha,
        model_spec=model.spec.to_dict(),
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        tokenizer_hash=hash_json({**marker, "identity": "synthetic-tokenizer"}),
        tokenizer_vocab_hash=hash_json({**marker, "identity": "synthetic-vocab"}),
        dataset_manifest_hash=hash_json({**marker, "identity": "no-training-dataset"}),
        run_manifest_hash=hash_json({**marker, "identity": "checkpoint-scale-probe"}),
        training_config={
            "training_executed": False,
            "purpose": "checkpoint-v2-save-load-mechanics",
        },
        seed=20260825,
        precision="fp32",
        step=0,
        tokens_seen=0,
        optimizer={
            "name": "AdamW",
            "lr": 1e-3,
            "weight_decay": 0.0,
            "state_population": "zero-gradient-single-step-for-moments-only",
        },
        scheduler=None,
        environment_lock_hash=sha256_file(
            stage_config_path.parents[2] / "requirements/locks/linux-x86_64/runtime.lock.txt"
        ),
    )


def _directory_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _run_sample(
    *,
    stage_config_path: Path,
    source_sha: str,
    repetition: int,
    work_root: Path,
) -> dict[str, Any]:
    stage = load_stage_config(stage_config_path)
    torch.manual_seed(20260825 + repetition)
    model = TwelveSixDecoder(stage.model, stage.init).cpu()
    optimizer = _populate_adamw_state(model)
    identity = _identity(
        source_sha=source_sha,
        stage_name=stage.stage,
        stage_config_path=stage_config_path,
        model=model,
    )
    before = _model_fingerprint(model)
    parameter_bytes = _parameter_bytes(model)
    optimizer_bytes = _optimizer_tensor_bytes(optimizer)
    checkpoint = work_root / f"{stage.stage.lower()}-rep-{repetition}"

    started = time.perf_counter()
    manifest = save_checkpoint_v2(
        checkpoint,
        model=model,
        optimizer=optimizer,
        identity=identity,
        trainer_state={"probe_repetition": repetition},
    )
    save_seconds = time.perf_counter() - started
    payload_bytes = sum(record["bytes"] for record in manifest["files"])
    dcp_bytes = sum(
        record["bytes"] for record in manifest["files"] if record["path"].startswith("dcp/")
    )
    total_bytes = _directory_bytes(checkpoint)

    del optimizer
    del model
    gc.collect()

    torch.manual_seed(10101 + repetition)
    restored = TwelveSixDecoder(stage.model, stage.init).cpu()
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3, weight_decay=0.0)
    started = time.perf_counter()
    result = load_checkpoint_v2(
        checkpoint,
        model=restored,
        optimizer=restored_optimizer,
        expected_identity=identity,
    )
    load_seconds = time.perf_counter() - started
    after = _model_fingerprint(restored)
    optimizer_entries = len(restored_optimizer.state)
    expected_optimizer_entries = len(list(restored.parameters()))
    shutil.rmtree(checkpoint)

    return {
        "repetition": repetition,
        "parameters": identity.parameter_count,
        "model_parameter_bytes": parameter_bytes,
        "optimizer_tensor_bytes": optimizer_bytes,
        "checkpoint_payload_bytes": payload_bytes,
        "checkpoint_total_bytes": total_bytes,
        "dcp_bytes": dcp_bytes,
        "save_seconds": save_seconds,
        "load_seconds": load_seconds,
        "model_fingerprint_before": before,
        "model_fingerprint_after": after,
        "model_fingerprint_exact": before == after,
        "optimizer_state_entries": optimizer_entries,
        "expected_optimizer_state_entries": expected_optimizer_entries,
        "optimizer_state_populated": optimizer_entries == expected_optimizer_entries,
        "trainer_state_restored": result.trainer_state == {"probe_repetition": repetition},
        "checkpoint_id": manifest["checkpoint_id"],
        "rng_restored": result.rng_restored,
    }


def _median(samples: list[dict[str, Any]], key: str) -> float:
    return float(statistics.median(float(sample[key]) for sample in samples))


def build_report(
    *,
    stage_config_path: Path,
    source_sha: str,
    repetitions: int,
    work_root: Path,
) -> dict[str, Any]:
    stage = load_stage_config(stage_config_path)
    samples = [
        _run_sample(
            stage_config_path=stage_config_path,
            source_sha=source_sha,
            repetition=index,
            work_root=work_root,
        )
        for index in range(repetitions)
    ]
    if any(not sample["model_fingerprint_exact"] for sample in samples):
        raise RuntimeError("checkpoint-v2 model round-trip mismatch")
    if any(not sample["optimizer_state_populated"] for sample in samples):
        raise RuntimeError("checkpoint-v2 optimizer state did not restore completely")
    if any(not sample["trainer_state_restored"] for sample in samples):
        raise RuntimeError("checkpoint-v2 trainer state did not restore")

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "source_sha": source_sha,
        "stage": stage.stage,
        "stage_config": stage_config_path.as_posix(),
        "stage_config_sha256": sha256_file(stage_config_path),
        "model_spec_sha256": stage.model.identity_sha256(),
        "parameters": stage.model.parameter_count(),
        "expected_parameters": stage.expected_parameters,
        "repetitions": repetitions,
        "samples": samples,
        "summary": {
            "checkpoint_total_bytes_median": int(
                statistics.median(sample["checkpoint_total_bytes"] for sample in samples)
            ),
            "checkpoint_payload_bytes_median": int(
                statistics.median(sample["checkpoint_payload_bytes"] for sample in samples)
            ),
            "dcp_bytes_median": int(
                statistics.median(sample["dcp_bytes"] for sample in samples)
            ),
            "model_parameter_bytes_median": int(
                statistics.median(sample["model_parameter_bytes"] for sample in samples)
            ),
            "optimizer_tensor_bytes_median": int(
                statistics.median(sample["optimizer_tensor_bytes"] for sample in samples)
            ),
            "save_seconds_median": _median(samples, "save_seconds"),
            "load_seconds_median": _median(samples, "load_seconds"),
            "all_model_round_trips_exact": all(
                sample["model_fingerprint_exact"] for sample in samples
            ),
            "all_optimizer_states_populated": all(
                sample["optimizer_state_populated"] for sample in samples
            ),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cpu_threads": torch.get_num_threads(),
            "locked_runtime_sha256": sha256_file(
                stage_config_path.parents[2]
                / "requirements/locks/linux-x86_64/runtime.lock.txt"
            ),
        },
        "truth_boundary": {
            "training_executed": False,
            "quality_or_capability_evidence": False,
            "paid_compute": False,
            "single_process_scale_probe": True,
            "distributed_reshard": "NOT_TESTED_BY_THIS_SINGLE_PROCESS_PROBE",
            "object_storage": "NOT_IMPLEMENTED_REQUIRES_STORE_ADAPTER",
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
    if len(args.source_sha) != 40:
        raise ValueError("--source-sha must be an exact 40-hex Git SHA")
    if args.source_sha != args.source_sha.lower() or any(
        char not in "0123456789abcdef" for char in args.source_sha
    ):
        raise ValueError("--source-sha must be an exact lowercase 40-hex Git SHA")
    if args.repetitions < 1:
        raise ValueError("--repetitions must be positive")
    if args.cpu_threads < 1:
        raise ValueError("--cpu-threads must be positive")

    repo_root = args.repo_root.resolve()
    stage_path = args.stage_config
    if not stage_path.is_absolute():
        stage_path = repo_root / stage_path
    stage_path = stage_path.resolve()
    torch.set_num_threads(args.cpu_threads)
    work_root = args.output.resolve().parent / f".{args.output.stem}.checkpoints"
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True)
    try:
        report = build_report(
            stage_config_path=stage_path,
            source_sha=args.source_sha,
            repetitions=args.repetitions,
            work_root=work_root,
        )
    finally:
        shutil.rmtree(work_root, ignore_errors=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(
        json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
        + b"\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
