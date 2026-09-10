#!/usr/bin/env python3
"""CHECKPOINT-346 bounded MODEL-341 recovery qualification.

This is checkpoint/recovery mechanics only. It executes exactly three optimizer
updates over deterministic synthetic tokens:
1) parent step 1, then publish checkpoint;
2) parent step 2 as uninterrupted reference;
3) fresh child process restores step 1 and repeats step 2.

It grants no corpus, tokenizer, training, learned-weight, or final-test credit.
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
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

CARRIER_SHA = "82c43005bb5db153482ae5b20a31d59240faaebc"
EXPECTED_PARAMETERS = 20_613_440
EXPECTED_MODEL_ID = "fbff24d561a2818453554d58ca23fc6ace3303b078f1935a8576c4565bd92441"
EXPECTED_INIT_ID = "86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5"
SEED = 346_341
SYNTHETIC_DATASET_HASH = hashlib.sha256(b"checkpoint346-synthetic-mechanics-only-dataset-v1").hexdigest()
SYNTHETIC_RUN_HASH = hashlib.sha256(b"checkpoint346-synthetic-mechanics-only-run-v1").hexdigest()
SYNTHETIC_TOKENIZER_HASH = hashlib.sha256(b"checkpoint346-byte-tokenizer-mechanics-identity-v1").hexdigest()
SYNTHETIC_VOCAB_HASH = hashlib.sha256(b"checkpoint346-byte-vocab-0-255-mechanics-v1").hexdigest()


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _tree_hash(value: Any) -> str:
    """Stable structural hash for tensors plus optimizer/trainer state."""
    h = hashlib.sha256()

    def walk(obj: Any) -> None:
        if isinstance(obj, torch.Tensor):
            tensor = obj.detach().cpu().contiguous()
            h.update(b"T")
            h.update(str(tensor.dtype).encode())
            h.update(json.dumps(list(tensor.shape)).encode())
            if tensor.dtype == torch.bfloat16:
                h.update(tensor.view(torch.uint16).numpy().tobytes())
            else:
                h.update(tensor.numpy().tobytes())
            return
        if isinstance(obj, np.ndarray):
            array = np.ascontiguousarray(obj)
            h.update(b"N")
            h.update(str(array.dtype).encode())
            h.update(json.dumps(list(array.shape)).encode())
            h.update(array.tobytes())
            return
        if isinstance(obj, dict):
            h.update(b"D")
            for key in sorted(obj, key=lambda item: repr(item)):
                walk(key)
                walk(obj[key])
            return
        if isinstance(obj, (list, tuple)):
            h.update(b"L" if isinstance(obj, list) else b"U")
            for item in obj:
                walk(item)
            return
        h.update(b"S")
        h.update(type(obj).__name__.encode())
        h.update(repr(obj).encode())

    walk(value)
    return h.hexdigest()


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


def _install_carrier(carrier_root: Path) -> None:
    src = carrier_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _load_runtime(carrier_root: Path):
    _install_carrier(carrier_root)
    from twelve_six.checkpoint.core import (
        CheckpointCompatibilityError,
        CheckpointIdentity,
        assert_identity,
        capture_rng_state,
        load_checkpoint,
        prepare_checkpoint_load,
        save_checkpoint,
    )
    from twelve_six.model import TwelveSixDecoder, count_trainable_parameters, load_stage_config
    from twelve_six.training.config import TrainerConfig
    from twelve_six.training.trainer import Trainer

    return {
        "CheckpointCompatibilityError": CheckpointCompatibilityError,
        "CheckpointIdentity": CheckpointIdentity,
        "capture_rng_state": capture_rng_state,
        "load_checkpoint": load_checkpoint,
        "prepare_checkpoint_load": prepare_checkpoint_load,
        "assert_identity": assert_identity,
        "save_checkpoint": save_checkpoint,
        "TwelveSixDecoder": TwelveSixDecoder,
        "count_trainable_parameters": count_trainable_parameters,
        "load_stage_config": load_stage_config,
        "TrainerConfig": TrainerConfig,
        "Trainer": Trainer,
    }


def _stage_and_config(carrier_root: Path, rt: dict[str, Any]):
    candidate = carrier_root / "configs/candidates/model341_20m_candidate_a.json"
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


def _new_trainer(stage: Any, config: Any, rt: dict[str, Any], init_seed: int):
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


def _summary(trainer: Any, metrics: Any, rng_probe: dict[str, Any]) -> dict[str, Any]:
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


def _identity(stage: Any, config: Any, trainer: Any, rt: dict[str, Any]):
    return rt["CheckpointIdentity"](
        git_sha=CARRIER_SHA,
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
        optimizer={"name": "AdamW", "scope": "synthetic_checkpoint_mechanics_only"},
        scheduler=None,
    )


def child_main(args: argparse.Namespace) -> int:
    carrier_root = Path(args.carrier_root).resolve()
    rt = _load_runtime(carrier_root)
    stage, config, _ = _stage_and_config(carrier_root, rt)
    trainer = _new_trainer(stage, config, rt, init_seed=SEED + 999)

    checkpoint = Path(args.checkpoint).resolve()
    verified = rt["prepare_checkpoint_load"](checkpoint)
    binding_fail_closed = False
    try:
        rt["assert_identity"](verified.manifest, run_manifest_hash="f" * 64)
    except rt["CheckpointCompatibilityError"]:
        binding_fail_closed = True
    if not binding_fail_closed:
        raise AssertionError("wrong run-manifest binding did not fail closed")

    loaded = rt["load_checkpoint"](
        checkpoint,
        model=trainer.model,
        restore_rng=True,
        expected_git_sha=CARRIER_SHA,
        expected_model_spec_hash=EXPECTED_MODEL_ID,
        expected_tokenizer_hash=SYNTHETIC_TOKENIZER_HASH,
        expected_tokenizer_vocab_hash=SYNTHETIC_VOCAB_HASH,
        expected_dataset_manifest_hash=SYNTHETIC_DATASET_HASH,
        expected_run_manifest_hash=SYNTHETIC_RUN_HASH,
    )
    trainer.load_state_dict(loaded.trainer_state)
    assert trainer.optimizer_step == 1
    assert trainer.micro_step == 1
    assert trainer.tokens_seen == 1

    restored_rng = rt["capture_rng_state"]()
    probe = _rng_probe_from_state(restored_rng)
    metrics = trainer.train_microbatch(_batch())
    child = _summary(trainer, metrics, probe)
    child.update(
        {
            "pid": os.getpid(),
            "parent_pid": os.getppid(),
            "binding_mismatch_failed_closed": binding_fail_closed,
            "restored_rng_scope": {
                "python": "python" in loaded.rng_state,
                "numpy": "numpy" in loaded.rng_state,
                "torch_cpu": "torch" in loaded.rng_state and "cpu" in loaded.rng_state["torch"],
            },
        }
    )
    _json_write(Path(args.child_output), child)
    return 0


def parent_main(args: argparse.Namespace) -> int:
    carrier_root = Path(args.carrier_root).resolve()
    if not carrier_root.is_dir():
        raise FileNotFoundError(f"carrier root missing: {carrier_root}")
    rt = _load_runtime(carrier_root)
    stage, config, _candidate = _stage_and_config(carrier_root, rt)

    with tempfile.TemporaryDirectory(prefix="checkpoint346-model341-") as temp:
        work = Path(temp)
        checkpoint = work / "step-000001"
        child_output = work / "child.json"

        trainer = _new_trainer(stage, config, rt, init_seed=SEED)
        first = trainer.train_microbatch(_batch())
        assert first.optimizer_step == 1
        state_after_step1 = _trainer_state_mapping(trainer)
        checkpoint_manifest = rt["save_checkpoint"](
            checkpoint,
            model=trainer.model,
            identity=_identity(stage, config, trainer, rt),
            trainer_state=state_after_step1,
        )

        parent_rng = rt["capture_rng_state"]()
        rng_probe = _rng_probe_from_state(parent_rng)
        second = trainer.train_microbatch(_batch())
        baseline = _summary(trainer, second, rng_probe)

        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            "--carrier-root",
            str(carrier_root),
            "--checkpoint",
            str(checkpoint),
            "--child-output",
            str(child_output),
        ]
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"fresh child restore failed with exit code {completed.returncode}")
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
            raise AssertionError("same-next-step continuation mismatch: " + json.dumps(mismatches, sort_keys=True))

        checkpoint_files = {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha_file(path)}
            for path in sorted(checkpoint.iterdir())
            if path.is_file()
        }
        report = {
            "schema": "checkpoint346.model341-recovery-qualification.v1",
            "verdict": "PASS_MODEL341_RECOVERY_MECHANICS",
            "carrier": {
                "pr": 802,
                "git_sha": CARRIER_SHA,
                "candidate_path": "configs/candidates/model341_20m_candidate_a.json",
                "candidate_blob_sha": "69e3cbd5f5c83c9d3d529a2a6376db3055979c40",
                "model_identity_sha256": stage.model.identity_sha256(),
                "init_identity_sha256": stage.init.identity_sha256(),
                "parameter_count": EXPECTED_PARAMETERS,
                "canonical_base": stage.canonical_base,
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
                "binding_mismatch_failed_closed": child["binding_mismatch_failed_closed"],
                "rng_scope_restored": child["restored_rng_scope"],
            },
            "baseline_step2": baseline,
            "resumed_step2": {key: child[key] for key in comparison_keys},
            "checkpoint": {
                "checkpoint_id": checkpoint_manifest["checkpoint_id"],
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
            },
            "limits": {
                "maximum_optimizer_updates": 3,
                "long_campaign": False,
                "selection_or_recipe_tuning": False,
            },
        }
        report_bytes = json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        report["identity_sha256"] = hashlib.sha256(report_bytes).hexdigest()
        _json_write(Path(args.output), report)
        print(json.dumps({
            "verdict": report["verdict"],
            "identity_sha256": report["identity_sha256"],
            "optimizer_updates_total": 3,
            "same_next_step_equal": True,
            "binding_mismatch_failed_closed": True,
        }, sort_keys=True))
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--carrier-root", required=True)
    parser.add_argument("--output")
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--child-output")
    args = parser.parse_args(argv)
    if args.child:
        if not args.checkpoint or not args.child_output:
            parser.error("--child requires --checkpoint and --child-output")
    elif not args.output:
        parser.error("parent mode requires --output")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    raise SystemExit(child_main(parsed) if parsed.child else parent_main(parsed))
