#!/usr/bin/env python3
"""CHECKPOINT-346 current-D05 bounded MODEL-341 recovery qualification.

This is checkpoint/recovery mechanics only. It executes exactly three optimizer
updates over deterministic synthetic token IDs:
1) parent step 1, then public D05 save_trainer_checkpoint();
2) parent step 2 as uninterrupted reference;
3) a distinct Python child process uses public D05 load_trainer_checkpoint()
   to restore step 1 and repeats step 2.

The canonical MODEL-341 JSON configuration is read from a separately pinned
MODEL-341 carrier checkout. All executable model/trainer/checkpoint code is
imported from a separately pinned current D05 Product checkout.

This grants no corpus, tokenizer-fit, model-training, learned-weight, final-test,
paid-compute, or scale-promotion credit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

D05_RUNTIME_SHA = "3e62f28addbfbc25708742801c14d2eb226baea1"
MODEL341_CARRIER_SHA = "f151c77a8ef8721f0f568509147b1a5f961ae6c7"
MODEL341_CANDIDATE_BLOB = "69e3cbd5f5c83c9d3d529a2a6376db3055979c40"
EXPECTED_PARAMETERS = 20_613_440
EXPECTED_MODEL_ID = "fbff24d561a2818453554d58ca23fc6ace3303b078f1935a8576c4565bd92441"
EXPECTED_INIT_ID = "86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5"
SEED = 346_341
SYNTHETIC_DATASET_HASH = hashlib.sha256(
    b"checkpoint346-current-d05-synthetic-mechanics-only-dataset-v1"
).hexdigest()
SYNTHETIC_RUN_HASH = hashlib.sha256(
    b"checkpoint346-current-d05-synthetic-mechanics-only-run-v1"
).hexdigest()
SYNTHETIC_TOKENIZER_HASH = hashlib.sha256(
    b"checkpoint346-byte-tokenizer-mechanics-identity-v1"
).hexdigest()
SYNTHETIC_VOCAB_HASH = hashlib.sha256(
    b"checkpoint346-byte-vocab-0-255-mechanics-v1"
).hexdigest()


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
        timeout=10,
    ).strip()


def _git_blob(root: Path, relative: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "hash-object", relative],
        text=True,
        timeout=10,
    ).strip()


def _peak_rss_bytes() -> int | None:
    try:
        import resource
    except ImportError:
        return None
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform.startswith("linux"):
        return int(value) * 1024
    if sys.platform == "darwin":
        return int(value)
    return None


def _tree_hash(value: Any) -> str:
    """Stable structural hash for tensors plus optimizer/trainer/RNG state."""
    digest = hashlib.sha256()

    def walk(obj: Any) -> None:
        if isinstance(obj, torch.Tensor):
            tensor = obj.detach().cpu().contiguous()
            digest.update(b"T")
            digest.update(str(tensor.dtype).encode())
            digest.update(json.dumps(list(tensor.shape)).encode())
            if tensor.dtype == torch.bfloat16:
                digest.update(tensor.view(torch.uint16).numpy().tobytes())
            else:
                digest.update(tensor.numpy().tobytes())
            return
        if isinstance(obj, np.ndarray):
            array = np.ascontiguousarray(obj)
            digest.update(b"N")
            digest.update(str(array.dtype).encode())
            digest.update(json.dumps(list(array.shape)).encode())
            digest.update(array.tobytes())
            return
        if isinstance(obj, dict):
            digest.update(b"D")
            for key in sorted(obj, key=lambda item: repr(item)):
                walk(key)
                walk(obj[key])
            return
        if isinstance(obj, (list, tuple)):
            digest.update(b"L" if isinstance(obj, list) else b"U")
            for item in obj:
                walk(item)
            return
        digest.update(b"S")
        digest.update(type(obj).__name__.encode())
        digest.update(repr(obj).encode())

    walk(value)
    return digest.hexdigest()


def _rng_probe_from_state(state: dict[str, Any]) -> dict[str, Any]:
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


def _install_runtime(runtime_root: Path) -> None:
    src = runtime_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _load_runtime(runtime_root: Path) -> dict[str, Any]:
    _install_runtime(runtime_root)
    from twelve_six.checkpoint import (
        CheckpointCompatibilityError,
        CheckpointIdentity,
        capture_rng_state,
        load_trainer_checkpoint,
        save_trainer_checkpoint,
    )
    from twelve_six.model import TwelveSixDecoder, count_trainable_parameters, load_stage_config
    from twelve_six.training.config import TrainerConfig
    from twelve_six.training.trainer import Trainer

    return {
        "CheckpointCompatibilityError": CheckpointCompatibilityError,
        "CheckpointIdentity": CheckpointIdentity,
        "capture_rng_state": capture_rng_state,
        "load_trainer_checkpoint": load_trainer_checkpoint,
        "save_trainer_checkpoint": save_trainer_checkpoint,
        "TwelveSixDecoder": TwelveSixDecoder,
        "count_trainable_parameters": count_trainable_parameters,
        "load_stage_config": load_stage_config,
        "TrainerConfig": TrainerConfig,
        "Trainer": Trainer,
    }


def _assert_checkout_roots(runtime_root: Path, model341_root: Path) -> None:
    if _git_head(runtime_root) != D05_RUNTIME_SHA:
        raise AssertionError("D05 runtime checkout SHA drifted")
    if _git_head(model341_root) != MODEL341_CARRIER_SHA:
        raise AssertionError("MODEL-341 carrier checkout SHA drifted")
    if (
        _git_blob(
            model341_root,
            "configs/candidates/model341_20m_candidate_a.json",
        )
        != MODEL341_CANDIDATE_BLOB
    ):
        raise AssertionError("MODEL-341 candidate config blob drifted")


def _stage_and_config(
    model341_root: Path,
    rt: dict[str, Any],
) -> tuple[Any, Any, Path]:
    candidate = model341_root / "configs/candidates/model341_20m_candidate_a.json"
    stage = rt["load_stage_config"](candidate)
    assert stage.canonical_base == "random_init"
    assert stage.expected_parameters == EXPECTED_PARAMETERS
    assert stage.model.parameter_count() == EXPECTED_PARAMETERS
    assert stage.model.identity_sha256() == EXPECTED_MODEL_ID
    assert stage.init.identity_sha256() == EXPECTED_INIT_ID
    config = rt["TrainerConfig"](
        learning_rate=1e-4,
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
    return stage, config, candidate


def _new_trainer(
    stage: Any,
    config: Any,
    rt: dict[str, Any],
    init_seed: int,
) -> Any:
    random.seed(init_seed)
    np.random.seed(init_seed)
    torch.manual_seed(init_seed)
    model = rt["TwelveSixDecoder"](stage.model, stage.init)
    assert rt["count_trainable_parameters"](model) == EXPECTED_PARAMETERS
    return rt["Trainer"](model, config, device="cpu")


def _batch() -> dict[str, torch.Tensor]:
    # Sequence length 2 minimizes mechanics cost while retaining one causal target.
    return {"input_ids": torch.tensor([[17, 91]], dtype=torch.long)}


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
    rng_probe: dict[str, Any],
) -> dict[str, Any]:
    return {
        "loss": metrics.loss,
        "update_loss": metrics.update_loss,
        "optimizer_step": trainer.optimizer_step,
        "micro_step": trainer.micro_step,
        "tokens_seen": trainer.tokens_seen,
        "model_state_sha256": _tree_hash(trainer.model.state_dict()),
        "optimizer_state_sha256": _tree_hash(trainer.optimizer.state_dict()),
        "trainer_state_sha256": _tree_hash(_trainer_state_mapping(trainer)),
        "rng_probe_before_step": rng_probe,
    }


def _identity(stage: Any, config: Any, trainer: Any, rt: dict[str, Any]) -> Any:
    return rt["CheckpointIdentity"](
        git_sha=D05_RUNTIME_SHA,
        model_spec=stage.model.to_dict(),
        parameter_count=EXPECTED_PARAMETERS,
        tokenizer_hash=SYNTHETIC_TOKENIZER_HASH,
        tokenizer_vocab_hash=SYNTHETIC_VOCAB_HASH,
        dataset_manifest_hash=SYNTHETIC_DATASET_HASH,
        run_manifest_hash=SYNTHETIC_RUN_HASH,
        training_config=asdict(config),
        seed=SEED,
        precision="fp32",
        step=trainer.optimizer_step,
        tokens_seen=trainer.tokens_seen,
        optimizer={
            "name": "AdamW",
            "scope": "synthetic_checkpoint_mechanics_only",
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
        expected_model_spec_hash=EXPECTED_MODEL_ID,
        expected_tokenizer_hash=SYNTHETIC_TOKENIZER_HASH,
        expected_tokenizer_vocab_hash=SYNTHETIC_VOCAB_HASH,
        expected_dataset_manifest_hash=SYNTHETIC_DATASET_HASH,
        expected_run_manifest_hash=run_manifest_hash,
        expected_seed=SEED,
        expected_step=1,
        expected_tokens_seen=1,
    )


def child_main(args: argparse.Namespace) -> int:
    runtime_root = Path(args.runtime_root).resolve()
    model341_root = Path(args.model341_root).resolve()
    _assert_checkout_roots(runtime_root, model341_root)
    rt = _load_runtime(runtime_root)
    stage, config, _ = _stage_and_config(model341_root, rt)
    trainer = _new_trainer(stage, config, rt, init_seed=SEED + 999)

    checkpoint = Path(args.checkpoint).resolve()
    before_wrong_binding = {
        "model": _tree_hash(trainer.model.state_dict()),
        "optimizer": _tree_hash(trainer.optimizer.state_dict()),
        "counters": [
            trainer.micro_step,
            trainer.optimizer_step,
            trainer.tokens_seen,
        ],
        "rng": _tree_hash(rt["capture_rng_state"]()),
    }

    binding_fail_closed = False
    try:
        _load_expected(
            rt=rt,
            checkpoint=checkpoint,
            trainer=trainer,
            checkpoint_id=args.expected_checkpoint_id,
            manifest_sha256=args.expected_manifest_sha256,
            run_manifest_hash="f" * 64,
        )
    except rt["CheckpointCompatibilityError"]:
        binding_fail_closed = True
    if not binding_fail_closed:
        raise AssertionError("wrong run-manifest binding did not fail closed")

    after_wrong_binding = {
        "model": _tree_hash(trainer.model.state_dict()),
        "optimizer": _tree_hash(trainer.optimizer.state_dict()),
        "counters": [
            trainer.micro_step,
            trainer.optimizer_step,
            trainer.tokens_seen,
        ],
        "rng": _tree_hash(rt["capture_rng_state"]()),
    }
    if after_wrong_binding != before_wrong_binding:
        raise AssertionError(
            "wrong run-manifest preflight mutated fresh restore target or RNG"
        )

    load_started = time.perf_counter()
    loaded = _load_expected(
        rt=rt,
        checkpoint=checkpoint,
        trainer=trainer,
        checkpoint_id=args.expected_checkpoint_id,
        manifest_sha256=args.expected_manifest_sha256,
        run_manifest_hash=SYNTHETIC_RUN_HASH,
    )
    load_seconds = time.perf_counter() - load_started

    if trainer.optimizer_step != 1 or trainer.micro_step != 1 or trainer.tokens_seen != 1:
        raise AssertionError("restored canonical trainer counters differ from step-1 checkpoint")

    restored_rng = rt["capture_rng_state"]()
    probe = _rng_probe_from_state(restored_rng)
    metrics = trainer.train_microbatch(_batch())
    child = _summary(trainer, metrics, probe)
    child.update(
        {
            "pid": os.getpid(),
            "parent_pid": os.getppid(),
            "binding_mismatch_failed_closed": binding_fail_closed,
            "binding_mismatch_retry_same_target": True,
            "restored_rng_scope": {
                "python": "python" in loaded.rng_state,
                "numpy": "numpy" in loaded.rng_state,
                "torch_cpu": (
                    "torch" in loaded.rng_state
                    and "cpu" in loaded.rng_state["torch"]
                ),
            },
            "load_trainer_checkpoint_seconds": load_seconds,
            "peak_rss_bytes": _peak_rss_bytes(),
        }
    )
    _json_write(Path(args.child_output), child)
    return 0


def parent_main(args: argparse.Namespace) -> int:
    runtime_root = Path(args.runtime_root).resolve()
    model341_root = Path(args.model341_root).resolve()
    if not runtime_root.is_dir():
        raise FileNotFoundError(f"D05 runtime root missing: {runtime_root}")
    if not model341_root.is_dir():
        raise FileNotFoundError(f"MODEL-341 root missing: {model341_root}")
    _assert_checkout_roots(runtime_root, model341_root)

    rt = _load_runtime(runtime_root)
    stage, config, _candidate = _stage_and_config(model341_root, rt)

    with tempfile.TemporaryDirectory(prefix="checkpoint346-current-d05-model341-") as temp:
        work = Path(temp)
        checkpoint = work / "step-000001"
        child_output = work / "child.json"

        trainer = _new_trainer(stage, config, rt, init_seed=SEED)
        first = trainer.train_microbatch(_batch())
        if first.optimizer_step != 1:
            raise AssertionError("parent step 1 did not commit exactly one optimizer update")

        save_started = time.perf_counter()
        checkpoint_manifest = rt["save_trainer_checkpoint"](
            checkpoint,
            model=trainer.model,
            trainer=trainer,
            identity=_identity(stage, config, trainer, rt),
        )
        save_seconds = time.perf_counter() - save_started
        checkpoint_id = checkpoint_manifest["checkpoint_id"]
        manifest_path = checkpoint / "manifest.json"
        manifest_sha256 = _sha_file(manifest_path)

        parent_rng = rt["capture_rng_state"]()
        rng_probe = _rng_probe_from_state(parent_rng)
        baseline_step_started = time.perf_counter()
        second = trainer.train_microbatch(_batch())
        baseline_step_seconds = time.perf_counter() - baseline_step_started
        baseline = _summary(trainer, second, rng_probe)
        parent_peak_rss_bytes = _peak_rss_bytes()

        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            "--runtime-root",
            str(runtime_root),
            "--model341-root",
            str(model341_root),
            "--checkpoint",
            str(checkpoint),
            "--expected-checkpoint-id",
            checkpoint_id,
            "--expected-manifest-sha256",
            manifest_sha256,
            "--child-output",
            str(child_output),
        ]
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"fresh child public-D05 restore failed with exit code "
                f"{completed.returncode}"
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
            "trainer_state_sha256",
            "rng_probe_before_step",
        ]
        mismatches = {
            key: {"baseline": baseline[key], "resumed": child[key]}
            for key in comparison_keys
            if baseline[key] != child[key]
        }
        if mismatches:
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
        parent_peak = parent_peak_rss_bytes
        child_peak = child["peak_rss_bytes"]
        concurrent_upper = (
            parent_peak + child_peak
            if isinstance(parent_peak, int) and isinstance(child_peak, int)
            else None
        )

        report = {
            "schema": "checkpoint346.model341-current-d05-recovery-qualification.v1",
            "verdict": "PASS_MODEL341_CURRENT_D05_RECOVERY_MECHANICS",
            "runtime": {
                "product_pr": 2778,
                "git_sha": D05_RUNTIME_SHA,
                "model_blob_sha": _git_blob(
                    runtime_root,
                    "src/twelve_six/model.py",
                ),
                "trainer_blob_sha": _git_blob(
                    runtime_root,
                    "src/twelve_six/training/trainer.py",
                ),
                "checkpoint_core_blob_sha": _git_blob(
                    runtime_root,
                    "src/twelve_six/checkpoint/core.py",
                ),
                "trainer_adapter_blob_sha": _git_blob(
                    runtime_root,
                    "src/twelve_six/checkpoint/trainer_adapter.py",
                ),
                "progress_trainer_blob_sha": _git_blob(
                    runtime_root,
                    "src/twelve_six/checkpoint/progress_trainer.py",
                ),
                "public_save_api": "save_trainer_checkpoint",
                "public_restore_api": "load_trainer_checkpoint",
            },
            "model341": {
                "carrier_pr": 802,
                "carrier_git_sha": MODEL341_CARRIER_SHA,
                "candidate_path": (
                    "configs/candidates/model341_20m_candidate_a.json"
                ),
                "candidate_blob_sha": MODEL341_CANDIDATE_BLOB,
                "model_identity_sha256": stage.model.identity_sha256(),
                "init_identity_sha256": stage.init.identity_sha256(),
                "parameter_count": EXPECTED_PARAMETERS,
                "canonical_base": stage.canonical_base,
                "execution_model_code_source": "D05_RUNTIME",
            },
            "execution": {
                "local_free": True,
                "device": "cpu",
                "precision": "fp32",
                "synthetic_mechanics_only": True,
                "optimizer_updates_total": 3,
                "parent_updates": 2,
                "fresh_process_resumed_updates": 1,
                "parent_pid": os.getpid(),
                "fresh_child_pid": child["pid"],
                "fresh_process_distinct": child["pid"] != os.getpid(),
                "checkpoint_step": 1,
                "final_step": 2,
                "same_next_step_equal": True,
                "binding_mismatch_failed_closed": child[
                    "binding_mismatch_failed_closed"
                ],
                "binding_mismatch_retry_same_target": child[
                    "binding_mismatch_retry_same_target"
                ],
                "exact_checkpoint_id_bound": True,
                "exact_manifest_sha256_bound": True,
                "rng_scope_restored": child["restored_rng_scope"],
            },
            "resource_observation": {
                "platform": sys.platform,
                "checkpoint_total_bytes": checkpoint_total_bytes,
                "parent_peak_rss_bytes": parent_peak,
                "fresh_child_peak_rss_bytes": child_peak,
                "concurrent_process_peak_rss_upper_bound_bytes": concurrent_upper,
                "save_trainer_checkpoint_seconds": save_seconds,
                "load_trainer_checkpoint_seconds": child[
                    "load_trainer_checkpoint_seconds"
                ],
                "baseline_step2_seconds": baseline_step_seconds,
                "measurement_scope": (
                    "single Ubuntu runner observation; per-process ru_maxrss, "
                    "not a 200M/1B feasibility claim"
                ),
            },
            "baseline_step2": baseline,
            "resumed_step2": {
                key: child[key] for key in comparison_keys
            },
            "checkpoint": {
                "checkpoint_id": checkpoint_id,
                "manifest_sha256": manifest_sha256,
                "identity": checkpoint_manifest["identity"],
                "files": checkpoint_files,
            },
            "scientific_credit": {
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
            },
            "limits": {
                "maximum_optimizer_updates": 3,
                "long_campaign": False,
                "selection_or_recipe_tuning": False,
                "learned20m_terminal_claim": False,
                "model341_current_d05_recovery_mechanics_only": True,
            },
        }
        report_bytes = json.dumps(
            report,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        report["identity_sha256"] = hashlib.sha256(report_bytes).hexdigest()
        _json_write(Path(args.output), report)
        print(
            json.dumps(
                {
                    "verdict": report["verdict"],
                    "identity_sha256": report["identity_sha256"],
                    "optimizer_updates_total": 3,
                    "same_next_step_equal": True,
                    "binding_mismatch_failed_closed": True,
                    "public_d05_save_restore": True,
                    "checkpoint_total_bytes": checkpoint_total_bytes,
                    "parent_peak_rss_bytes": parent_peak,
                    "fresh_child_peak_rss_bytes": child_peak,
                },
                sort_keys=True,
            )
        )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--model341-root", required=True)
    parser.add_argument("--output")
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--expected-checkpoint-id")
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--child-output")
    args = parser.parse_args(argv)
    if args.child:
        missing = [
            name
            for name in (
                "checkpoint",
                "expected_checkpoint_id",
                "expected_manifest_sha256",
                "child_output",
            )
            if not getattr(args, name)
        ]
        if missing:
            parser.error(
                "--child requires --checkpoint, --expected-checkpoint-id, "
                "--expected-manifest-sha256 and --child-output"
            )
    elif not args.output:
        parser.error("parent mode requires --output")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    raise SystemExit(child_main(parsed) if parsed.child else parent_main(parsed))
