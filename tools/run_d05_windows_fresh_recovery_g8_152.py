#!/usr/bin/env python3
"""Exact-head Windows fresh-process D05 checkpoint/recovery mechanics qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

D05_RUNTIME_SHA = os.environ.get("D05_RUNTIME_SHA", "")
QUALIFICATION_CARRIER_SHA = os.environ.get("QUALIFICATION_CARRIER_SHA", "")
QUALIFICATION_RUNNER_BLOB = os.environ.get("QUALIFICATION_RUNNER_BLOB", "")
QUALIFICATION_ROOT = Path(__file__).resolve().parents[1]
QUALIFICATION_RUNNER_PATH = "tools/run_d05_windows_fresh_recovery_g8_152.py"
PRODUCT_PR = 2778
EXPECTED_BLOBS = {
    "src/twelve_six/training/trainer.py": "5c13f182e55ade72f743089e8bda59bef1a419f7",
    "src/twelve_six/checkpoint/core.py": "915eec641d90ed4f22e6e52dc5fda6f75969876e",
    "src/twelve_six/checkpoint/trainer_adapter.py": "9d47a7b86a2f883f8d09709e820c411dffcb388f",
    "src/twelve_six/checkpoint/progress_trainer.py": "917d5b108e83105383e21e167609dac0bea93498",
    "src/twelve_six/checkpoint/__init__.py": "ba6d91821e36f3ee359bd652a989d57ffa4c94d7",
}
SEED = 8152
VOCAB_SIZE = 32
HIDDEN_SIZE = 8
TREE_HASH_SCHEME = "sha256:length-prefixed-structural-v2"
SYNTHETIC_TOKENIZER_HASH = hashlib.sha256(
    b"d05-windows-tiny-tokenizer-mechanics-v1"
).hexdigest()
SYNTHETIC_VOCAB_HASH = hashlib.sha256(
    b"d05-windows-tiny-vocab-mechanics-v1"
).hexdigest()
SYNTHETIC_DATASET_HASH = hashlib.sha256(
    b"d05-windows-tiny-dataset-mechanics-v1"
).hexdigest()
SYNTHETIC_RUN_HASH = hashlib.sha256(
    b"d05-windows-fresh-recovery-mechanics-v1"
).hexdigest()


def _require_sha(value: str, *, field: str) -> str:
    if (
        len(value) != 40
        or value != value.lower()
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise RuntimeError(f"{field} must be exact lowercase 40-hex")
    return value


_require_sha(D05_RUNTIME_SHA, field="D05_RUNTIME_SHA")
_require_sha(QUALIFICATION_CARRIER_SHA, field="QUALIFICATION_CARRIER_SHA")
_require_sha(QUALIFICATION_RUNNER_BLOB, field="QUALIFICATION_RUNNER_BLOB")


def _git_head(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
        timeout=15,
    ).strip()


def _git_blob(root: Path, relative: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "hash-object", relative],
        text=True,
        timeout=15,
    ).strip()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _json_write(path: Path, value: Any) -> None:
    path.write_bytes(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _tree_hash(value: Any) -> str:
    digest = hashlib.sha256()

    def emit(tag: bytes, payload: bytes = b"") -> None:
        digest.update(len(tag).to_bytes(2, "big"))
        digest.update(tag)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    def walk(obj: Any) -> None:
        if isinstance(obj, torch.Tensor):
            tensor = obj.detach().cpu().contiguous()
            emit(b"tensor.dtype", str(tensor.dtype).encode("utf-8"))
            emit(b"tensor.shape", _canonical_json_bytes(list(tensor.shape)))
            if tensor.dtype == torch.bfloat16:
                payload = tensor.view(torch.uint16).numpy().tobytes()
            else:
                payload = tensor.numpy().tobytes()
            emit(b"tensor.data", payload)
            return
        if isinstance(obj, np.ndarray):
            array = np.ascontiguousarray(obj)
            emit(b"numpy.dtype", str(array.dtype).encode("utf-8"))
            emit(b"numpy.shape", _canonical_json_bytes(list(array.shape)))
            emit(b"numpy.data", array.tobytes())
            return
        if isinstance(obj, np.generic):
            emit(
                b"numpy.scalar",
                str(obj.dtype).encode("utf-8") + b"\0" + obj.tobytes(),
            )
            return
        if isinstance(obj, Mapping):
            emit(b"mapping", len(obj).to_bytes(8, "big"))
            keys = sorted(
                obj,
                key=lambda item: (
                    type(item).__module__,
                    type(item).__qualname__,
                    _tree_hash(item),
                ),
            )
            for key in keys:
                walk(key)
                walk(obj[key])
            return
        if isinstance(obj, list):
            emit(b"list", len(obj).to_bytes(8, "big"))
            for item in obj:
                walk(item)
            return
        if isinstance(obj, tuple):
            emit(b"tuple", len(obj).to_bytes(8, "big"))
            for item in obj:
                walk(item)
            return
        if isinstance(obj, bytes):
            emit(b"bytes", obj)
            return
        emit(
            (
                "scalar:"
                + type(obj).__module__
                + "."
                + type(obj).__qualname__
            ).encode("utf-8"),
            repr(obj).encode("utf-8"),
        )

    walk(value)
    return digest.hexdigest()


def _assert_hash_domain_separation() -> None:
    pairs = [
        ([[], []], [[[]]]),
        (((), ()), (((),),)),
        ([(), []], [([],)]),
    ]
    for left, right in pairs:
        if _tree_hash(left) == _tree_hash(right):
            raise AssertionError("structural hash domain separation failed")


class TinyCausalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
        self.output = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.output(self.embedding(input_ids))


def _runtime() -> dict[str, Any]:
    from twelve_six.checkpoint import (
        CheckpointCompatibilityError,
        CheckpointIdentity,
        capture_rng_state,
        load_trainer_checkpoint,
        save_trainer_checkpoint,
    )
    from twelve_six.training.config import TrainerConfig
    from twelve_six.training.trainer import Trainer

    return {
        "CheckpointCompatibilityError": CheckpointCompatibilityError,
        "CheckpointIdentity": CheckpointIdentity,
        "capture_rng_state": capture_rng_state,
        "load_trainer_checkpoint": load_trainer_checkpoint,
        "save_trainer_checkpoint": save_trainer_checkpoint,
        "TrainerConfig": TrainerConfig,
        "Trainer": Trainer,
    }


def _assert_checkout() -> None:
    if sys.platform != "win32":
        raise AssertionError(f"Windows qualification requires win32, got {sys.platform!r}")
    if torch.cuda.is_available():
        raise AssertionError("Windows recovery mechanics lane must remain CPU-only")
    if _git_head(QUALIFICATION_ROOT) != QUALIFICATION_CARRIER_SHA:
        raise AssertionError("qualification carrier checkout SHA drifted")
    if (
        _git_blob(QUALIFICATION_ROOT, QUALIFICATION_RUNNER_PATH)
        != QUALIFICATION_RUNNER_BLOB
    ):
        raise AssertionError("qualification runner blob drifted")
    for path, expected in EXPECTED_BLOBS.items():
        if _git_blob(QUALIFICATION_ROOT, path) != expected:
            raise AssertionError(f"Product source blob drifted: {path}")


def _configure_cpu_runtime() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def _config(rt: dict[str, Any]) -> Any:
    return rt["TrainerConfig"](
        learning_rate=1e-3,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        eps=1e-8,
        max_steps=2,
        warmup_steps=0,
        scheduler="constant",
        gradient_accumulation_steps=1,
        gradient_clip_norm=1.0,
        precision="fp32",
        seed=SEED,
        deterministic_algorithms=True,
        deterministic_warn_only=False,
    )


def _new_trainer(rt: dict[str, Any], *, init_seed: int) -> Any:
    random.seed(init_seed)
    np.random.seed(init_seed)
    torch.manual_seed(init_seed)
    model = TinyCausalModel()
    return rt["Trainer"](model, _config(rt), device="cpu")


def _batch() -> dict[str, torch.Tensor]:
    return {"input_ids": torch.tensor([[3, 5, 7]], dtype=torch.long)}


def _rng_probe_from_state(state: Mapping[str, Any]) -> dict[str, Any]:
    py = random.Random()
    py.setstate(state["python"])
    np_probe = np.random.RandomState()
    np_probe.set_state(state["numpy"])
    torch_probe = torch.Generator(device="cpu")
    torch_probe.set_state(state["torch"]["cpu"].cpu())
    return {
        "python": [py.random(), py.random()],
        "numpy": np_probe.random_sample(2).tolist(),
        "torch_cpu": torch.rand(2, generator=torch_probe).tolist(),
    }


def _trainer_state_mapping(trainer: Any) -> dict[str, Any]:
    state = trainer.state_dict()
    return {
        "micro_step": state.micro_step,
        "optimizer_step": state.optimizer_step,
        "tokens_seen": state.tokens_seen,
        "optimizer": state.optimizer,
        "scheduler": state.scheduler,
        "scaler": state.scaler,
        "config": state.config,
    }


def _summary(
    trainer: Any,
    metrics: Any,
    *,
    rng_probe: dict[str, Any],
    rng_state_sha256: str,
) -> dict[str, Any]:
    scheduler_state = (
        None if trainer.scheduler is None else trainer.scheduler.state_dict()
    )
    scaler_state = None if trainer.scaler is None else trainer.scaler.state_dict()
    return {
        "loss": metrics.loss,
        "update_loss": metrics.update_loss,
        "optimizer_step": trainer.optimizer_step,
        "micro_step": trainer.micro_step,
        "tokens_seen": trainer.tokens_seen,
        "model_state_sha256": _tree_hash(trainer.model.state_dict()),
        "optimizer_state_sha256": _tree_hash(trainer.optimizer.state_dict()),
        "scheduler_state_sha256": _tree_hash(scheduler_state),
        "scaler_state_sha256": _tree_hash(scaler_state),
        "trainer_state_sha256": _tree_hash(_trainer_state_mapping(trainer)),
        "rng_probe_before_step": rng_probe,
        "rng_state_sha256": rng_state_sha256,
    }


def _fresh_target_snapshot(trainer: Any, rt: dict[str, Any]) -> dict[str, Any]:
    scheduler_state = (
        None if trainer.scheduler is None else trainer.scheduler.state_dict()
    )
    scaler_state = None if trainer.scaler is None else trainer.scaler.state_dict()
    return {
        "model": _tree_hash(trainer.model.state_dict()),
        "optimizer": _tree_hash(trainer.optimizer.state_dict()),
        "scheduler": _tree_hash(scheduler_state),
        "scaler": _tree_hash(scaler_state),
        "config": _tree_hash(asdict(trainer.config)),
        "counters": [
            trainer.micro_step,
            trainer.optimizer_step,
            trainer.tokens_seen,
        ],
        "pending_tokens": trainer._pending_tokens,
        "pending_loss_sum": trainer._pending_loss_sum,
        "failure_reason": trainer._failure_reason,
        "update_incomplete": trainer._update_incomplete,
        "model_training": trainer.model.training,
        "deterministic_policy": [
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
        ],
        "rng": _tree_hash(rt["capture_rng_state"]()),
    }


def _identity(rt: dict[str, Any], trainer: Any) -> Any:
    return rt["CheckpointIdentity"](
        git_sha=D05_RUNTIME_SHA,
        model_spec={
            "kind": "d05-windows-tiny-causal-v1",
            "vocab_size": VOCAB_SIZE,
            "hidden_size": HIDDEN_SIZE,
        },
        parameter_count=sum(
            parameter.numel() for parameter in trainer.model.parameters()
        ),
        tokenizer_hash=SYNTHETIC_TOKENIZER_HASH,
        tokenizer_vocab_hash=SYNTHETIC_VOCAB_HASH,
        dataset_manifest_hash=SYNTHETIC_DATASET_HASH,
        run_manifest_hash=SYNTHETIC_RUN_HASH,
        training_config=asdict(trainer.config),
        seed=SEED,
        precision="fp32",
        step=trainer.optimizer_step,
        tokens_seen=trainer.tokens_seen,
        optimizer={
            "name": "AdamW",
            "scope": "synthetic_windows_checkpoint_mechanics_only",
        },
        scheduler=None,
    )


def _load_expected(
    *,
    rt: dict[str, Any],
    checkpoint: Path,
    trainer: Any,
    checkpoint_id: str,
    manifest_sha256: str,
    expected_step: int,
    expected_tokens_seen: int,
    run_manifest_hash: str,
) -> Any:
    return rt["load_trainer_checkpoint"](
        checkpoint,
        model=trainer.model,
        trainer=trainer,
        strict_model=True,
        restore_rng=True,
        expected_checkpoint_id=checkpoint_id,
        expected_manifest_sha256=manifest_sha256,
        expected_git_sha=D05_RUNTIME_SHA,
        expected_tokenizer_hash=SYNTHETIC_TOKENIZER_HASH,
        expected_tokenizer_vocab_hash=SYNTHETIC_VOCAB_HASH,
        expected_dataset_manifest_hash=SYNTHETIC_DATASET_HASH,
        expected_run_manifest_hash=run_manifest_hash,
        expected_seed=SEED,
        expected_step=expected_step,
        expected_tokens_seen=expected_tokens_seen,
    )


def _run_child(args: argparse.Namespace) -> int:
    _assert_hash_domain_separation()
    _assert_checkout()
    rt = _runtime()
    trainer = _new_trainer(rt, init_seed=SEED + 999)

    checkpoint = Path(args.checkpoint).resolve()
    before = _fresh_target_snapshot(trainer, rt)

    failed_closed = False
    try:
        _load_expected(
            rt=rt,
            checkpoint=checkpoint,
            trainer=trainer,
            checkpoint_id=args.expected_checkpoint_id,
            manifest_sha256=args.expected_manifest_sha256,
            expected_step=args.expected_step,
            expected_tokens_seen=args.expected_tokens_seen,
            run_manifest_hash="f" * 64,
        )
    except rt["CheckpointCompatibilityError"]:
        failed_closed = True
    if not failed_closed:
        raise AssertionError("wrong run-manifest binding did not fail closed")
    after = _fresh_target_snapshot(trainer, rt)
    if after != before:
        raise AssertionError("wrong-binding preflight mutated fresh target or RNG")

    loaded = _load_expected(
        rt=rt,
        checkpoint=checkpoint,
        trainer=trainer,
        checkpoint_id=args.expected_checkpoint_id,
        manifest_sha256=args.expected_manifest_sha256,
        expected_step=args.expected_step,
        expected_tokens_seen=args.expected_tokens_seen,
        run_manifest_hash=SYNTHETIC_RUN_HASH,
    )
    if (
        trainer.optimizer_step != args.expected_step
        or trainer.micro_step != args.expected_step
        or trainer.tokens_seen != args.expected_tokens_seen
    ):
        raise AssertionError("restored trainer counters differ from checkpoint")
    if loaded.manifest.get("checkpoint_id") != args.expected_checkpoint_id:
        raise AssertionError("loaded checkpoint id differs from expected identity")

    restored_rng = rt["capture_rng_state"]()
    rng_sha = _tree_hash(restored_rng)
    probe = _rng_probe_from_state(restored_rng)
    metrics = trainer.train_microbatch(_batch())
    child = _summary(
        trainer,
        metrics,
        rng_probe=probe,
        rng_state_sha256=rng_sha,
    )
    child.update(
        {
            "pid": os.getpid(),
            "parent_pid": os.getppid(),
            "binding_mismatch_failed_closed": True,
            "binding_mismatch_retry_same_target": True,
            "binding_mismatch_unchanged_scope": sorted(before),
            "restored_rng_scope": {
                "python": "python" in loaded.rng_state,
                "numpy": "numpy" in loaded.rng_state,
                "torch_cpu": (
                    "torch" in loaded.rng_state
                    and "cpu" in loaded.rng_state["torch"]
                ),
            },
        }
    )
    _json_write(Path(args.child_output), child)
    return 0


def _scientific_credit() -> dict[str, Any]:
    return {
        "real_corpus_used": False,
        "training_authorized_corpus_used": False,
        "optimized_target_exposure": 0,
        "model_training_credit": False,
        "learned_weights_created": False,
        "final_test_read": False,
        "paid_compute_used": False,
        "foreign_pretrained_weights": False,
        "tokenizer_fit_authorized": False,
        "scale_promotion_authorized": False,
    }


def _limits() -> dict[str, Any]:
    return {
        "maximum_optimizer_updates": 3,
        "long_campaign": False,
        "selection_or_recipe_tuning": False,
        "learned20m_terminal_claim": False,
        "windows_recovery_mechanics_only": True,
        "model341_resource_claim": False,
        "scale_200m_1b_claim": False,
    }


def _validate_report_contract(report: Mapping[str, Any]) -> None:
    if report.get("schema") != "d05.windows-fresh-recovery-qualification.v1":
        raise AssertionError("qualification schema differs")
    if report.get("verdict") != "PASS_D05_WINDOWS_FRESH_RECOVERY_MECHANICS":
        raise AssertionError("qualification verdict is not PASS")

    runtime = report["runtime"]
    if runtime["product_pr"] != PRODUCT_PR or runtime["git_sha"] != D05_RUNTIME_SHA:
        raise AssertionError("Product runtime binding is contradictory")
    if runtime["qualification_git_sha"] != QUALIFICATION_CARRIER_SHA:
        raise AssertionError("qualification carrier binding is contradictory")
    if runtime["qualification_runner_blob_sha"] != QUALIFICATION_RUNNER_BLOB:
        raise AssertionError("qualification runner binding is contradictory")
    if runtime["platform"] != "win32":
        raise AssertionError("qualification did not run on Windows")
    if runtime["torch_cuda_available"] is not False:
        raise AssertionError("qualification is not CPU-only")
    if runtime["public_save_api"] != "save_trainer_checkpoint":
        raise AssertionError("save API binding is contradictory")
    if runtime["public_restore_api"] != "load_trainer_checkpoint":
        raise AssertionError("restore API binding is contradictory")

    execution = report["execution"]
    required_true = (
        "local_free",
        "synthetic_mechanics_only",
        "fresh_process_distinct",
        "same_next_step_equal",
        "binding_mismatch_failed_closed",
        "binding_mismatch_retry_same_target",
        "exact_checkpoint_id_bound",
        "exact_manifest_sha256_bound",
        "exact_full_rng_state_equal",
        "structural_hash_self_test",
    )
    if not all(execution[name] is True for name in required_true):
        raise AssertionError("execution flags contradict PASS")
    if execution["state_hash_scheme"] != TREE_HASH_SCHEME:
        raise AssertionError("state-hash scheme differs")
    if (
        execution["optimizer_updates_total"] != 3
        or execution["parent_updates"] != 2
        or execution["fresh_process_resumed_updates"] != 1
    ):
        raise AssertionError("optimizer-update boundary differs")
    if execution["rng_scope_restored"] != {
        "numpy": True,
        "python": True,
        "torch_cpu": True,
    }:
        raise AssertionError("RNG restore scope differs")

    if report["baseline_step2"] != report["resumed_step2"]:
        raise AssertionError("baseline and resumed next-step state differ")
    rng_sha = report["baseline_step2"].get("rng_state_sha256")
    if not isinstance(rng_sha, str) or len(rng_sha) != 64:
        raise AssertionError("exact RNG identity missing")
    if execution["full_rng_state_sha256"] != rng_sha:
        raise AssertionError("execution RNG identity differs")

    checkpoint = report["checkpoint"]
    if checkpoint["identity"]["git_sha"] != D05_RUNTIME_SHA:
        raise AssertionError("checkpoint Product identity differs")
    checkpoint_bytes = sum(item["bytes"] for item in checkpoint["files"].values())
    resources = report["resource_observation"]
    if resources["checkpoint_total_bytes"] != checkpoint_bytes:
        raise AssertionError("checkpoint byte accounting differs")
    if resources != {
        "checkpoint_total_bytes": checkpoint_bytes,
        "memory_measured": False,
        "resource_envelope_claimed": False,
        "model341_resource_claim": False,
        "scale_200m_1b_claim": False,
        "scope": (
            "Windows fresh-process recovery mechanics only; no memory, throughput, "
            "MODEL-341, 20M, 200M or 1B resource envelope is claimed"
        ),
    }:
        raise AssertionError("resource nonclaim boundary differs")

    if report["scientific_credit"] != _scientific_credit():
        raise AssertionError("scientific-credit boundary widened")
    if report["limits"] != _limits():
        raise AssertionError("qualification limits widened")

    identity = report.get("identity_sha256")
    if not isinstance(identity, str) or len(identity) != 64:
        raise AssertionError("report identity missing")
    preimage = dict(report)
    del preimage["identity_sha256"]
    if hashlib.sha256(_canonical_json_bytes(preimage)).hexdigest() != identity:
        raise AssertionError("report identity does not match canonical preimage")


def _run_parent(args: argparse.Namespace) -> int:
    _assert_hash_domain_separation()
    _assert_checkout()
    rt = _runtime()

    with tempfile.TemporaryDirectory(prefix="d05-windows-fresh-recovery-") as tmp:
        work = Path(tmp)
        checkpoint = work / "step-000001"
        child_output = work / "child.json"

        trainer = _new_trainer(rt, init_seed=SEED)
        first = trainer.train_microbatch(_batch())
        if not first.optimizer_stepped or trainer.optimizer_step != 1:
            raise AssertionError("parent step 1 did not commit one optimizer update")
        step1_tokens = trainer.tokens_seen

        manifest = rt["save_trainer_checkpoint"](
            checkpoint,
            model=trainer.model,
            trainer=trainer,
            identity=_identity(rt, trainer),
        )
        checkpoint_id = manifest["checkpoint_id"]
        manifest_sha256 = _sha_file(checkpoint / "manifest.json")

        parent_rng = rt["capture_rng_state"]()
        rng_sha = _tree_hash(parent_rng)
        probe = _rng_probe_from_state(parent_rng)
        second = trainer.train_microbatch(_batch())
        baseline = _summary(
            trainer,
            second,
            rng_probe=probe,
            rng_state_sha256=rng_sha,
        )

        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            "--checkpoint",
            str(checkpoint),
            "--expected-checkpoint-id",
            checkpoint_id,
            "--expected-manifest-sha256",
            manifest_sha256,
            "--expected-step",
            "1",
            "--expected-tokens-seen",
            str(step1_tokens),
            "--child-output",
            str(child_output),
        ]
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"fresh Windows child recovery failed: exit={completed.returncode}"
            )
        child = json.loads(child_output.read_text(encoding="utf-8"))

        comparison_keys = [
            "loss",
            "update_loss",
            "optimizer_step",
            "micro_step",
            "tokens_seen",
            "model_state_sha256",
            "optimizer_state_sha256",
            "scheduler_state_sha256",
            "scaler_state_sha256",
            "trainer_state_sha256",
            "rng_probe_before_step",
            "rng_state_sha256",
        ]
        resumed = {key: child[key] for key in comparison_keys}
        baseline_cmp = {key: baseline[key] for key in comparison_keys}
        if baseline_cmp != resumed:
            mismatches = {
                key: {"baseline": baseline_cmp[key], "resumed": resumed[key]}
                for key in comparison_keys
                if baseline_cmp[key] != resumed[key]
            }
            raise AssertionError(
                "same-next-step continuation mismatch: "
                + json.dumps(mismatches, sort_keys=True)
            )

        checkpoint_files = {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": _sha_file(path),
            }
            for path in sorted(checkpoint.iterdir())
            if path.is_file()
        }
        checkpoint_total_bytes = sum(
            item["bytes"] for item in checkpoint_files.values()
        )
        report: dict[str, Any] = {
            "schema": "d05.windows-fresh-recovery-qualification.v1",
            "verdict": "PASS_D05_WINDOWS_FRESH_RECOVERY_MECHANICS",
            "runtime": {
                "product_pr": PRODUCT_PR,
                "git_sha": D05_RUNTIME_SHA,
                "qualification_git_sha": QUALIFICATION_CARRIER_SHA,
                "qualification_runner_blob_sha": QUALIFICATION_RUNNER_BLOB,
                "platform": sys.platform,
                "platform_detail": platform.platform(),
                "python": sys.version,
                "torch_version": torch.__version__,
                "numpy_version": np.__version__,
                "torch_cuda_available": torch.cuda.is_available(),
                "public_save_api": "save_trainer_checkpoint",
                "public_restore_api": "load_trainer_checkpoint",
                "product_source_blobs": dict(EXPECTED_BLOBS),
            },
            "execution": {
                "local_free": True,
                "synthetic_mechanics_only": True,
                "fresh_process_distinct": child["pid"] != os.getpid(),
                "same_next_step_equal": True,
                "binding_mismatch_failed_closed": child[
                    "binding_mismatch_failed_closed"
                ],
                "binding_mismatch_retry_same_target": child[
                    "binding_mismatch_retry_same_target"
                ],
                "binding_mismatch_unchanged_scope": child[
                    "binding_mismatch_unchanged_scope"
                ],
                "exact_checkpoint_id_bound": (
                    manifest["checkpoint_id"] == checkpoint_id
                ),
                "exact_manifest_sha256_bound": (
                    _sha_file(checkpoint / "manifest.json") == manifest_sha256
                ),
                "exact_full_rng_state_equal": (
                    baseline["rng_state_sha256"] == child["rng_state_sha256"]
                ),
                "full_rng_state_sha256": baseline["rng_state_sha256"],
                "rng_scope_restored": child["restored_rng_scope"],
                "state_hash_scheme": TREE_HASH_SCHEME,
                "structural_hash_self_test": True,
                "optimizer_updates_total": 3,
                "parent_updates": 2,
                "fresh_process_resumed_updates": 1,
            },
            "baseline_step2": baseline_cmp,
            "resumed_step2": resumed,
            "checkpoint": {
                "checkpoint_id": checkpoint_id,
                "manifest_sha256": manifest_sha256,
                "identity": manifest["identity"],
                "files": checkpoint_files,
            },
            "resource_observation": {
                "checkpoint_total_bytes": checkpoint_total_bytes,
                "memory_measured": False,
                "resource_envelope_claimed": False,
                "model341_resource_claim": False,
                "scale_200m_1b_claim": False,
                "scope": (
                    "Windows fresh-process recovery mechanics only; no memory, "
                    "throughput, MODEL-341, 20M, 200M or 1B resource envelope is claimed"
                ),
            },
            "scientific_credit": _scientific_credit(),
            "limits": _limits(),
        }
        report["identity_sha256"] = hashlib.sha256(
            _canonical_json_bytes(report)
        ).hexdigest()
        _validate_report_contract(report)
        _json_write(Path(args.output), report)
    return 0


def _validate_file(path: Path) -> int:
    report = json.loads(path.read_text(encoding="utf-8"))
    _validate_report_contract(report)
    print("D05_WINDOWS_FRESH_RECOVERY=PASS_ZERO_CREDIT")
    print("RESULT_IDENTITY_SHA256=" + report["identity_sha256"])
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    parser.add_argument("--validate-report")
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--expected-checkpoint-id")
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--expected-step", type=int)
    parser.add_argument("--expected-tokens-seen", type=int)
    parser.add_argument("--child-output")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.validate_report:
        return _validate_file(Path(args.validate_report))
    _configure_cpu_runtime()
    if args.child:
        required = {
            "checkpoint": args.checkpoint,
            "expected_checkpoint_id": args.expected_checkpoint_id,
            "expected_manifest_sha256": args.expected_manifest_sha256,
            "expected_step": args.expected_step,
            "expected_tokens_seen": args.expected_tokens_seen,
            "child_output": args.child_output,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"missing child arguments: {missing}")
        return _run_child(args)
    if not args.output:
        raise ValueError("--output is required")
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
