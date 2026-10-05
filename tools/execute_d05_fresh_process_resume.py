#!/usr/bin/env python3
"""Execution-only fresh-process D02/D05 checkpoint replay acceptance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from twelve_six.checkpoint import CheckpointIdentity
from twelve_six.checkpoint import core, progress_trainer, trainer_adapter
from twelve_six.training import Trainer, TrainerConfig

PRODUCT_HEAD = "819fcc7eb968b9ae52734e27b30bc7778154fe0f"
SEED = 703
_BATCH = {
    "input_ids": torch.tensor([[0, 1]], dtype=torch.long),
    "target_ids": torch.tensor([[1, 2]], dtype=torch.long),
}


class TinyLogits(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.1, -0.2, 0.3]))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.weight.reshape(1, 1, 3).expand(*input_ids.shape, 3)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def write_create_only(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical(value)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def tensor_record(value: torch.Tensor) -> dict[str, Any]:
    tensor = value.detach().cpu().contiguous()
    raw = tensor.view(torch.uint8).numpy().tobytes()
    return {
        "kind": "torch_tensor",
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "sha256": sha256(raw),
        "bytes": len(raw),
    }


def numpy_record(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    raw = array.tobytes()
    return {
        "kind": "numpy_array",
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "sha256": sha256(raw),
        "bytes": len(raw),
    }


def normalize(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return tensor_record(value)
    if isinstance(value, np.ndarray):
        return numpy_record(value)
    if isinstance(value, np.generic):
        return normalize(value.item())
    if type(value) is float:
        return {"kind": "float", "hex": value.hex()}
    if type(value) in (str, int, bool) or value is None:
        return value
    if isinstance(value, dict):
        return {
            "kind": "mapping",
            "items": [
                [normalize(key), normalize(item)]
                for key, item in sorted(value.items(), key=lambda row: repr(row[0]))
            ],
        }
    if isinstance(value, (list, tuple)):
        return {
            "kind": "tuple" if isinstance(value, tuple) else "list",
            "items": [normalize(item) for item in value],
        }
    raise TypeError(f"unsupported evidence value type: {type(value)!r}")


def reset_rng(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_trainer() -> Trainer:
    return Trainer(
        TinyLogits(),
        TrainerConfig(seed=SEED, max_steps=3, scheduler="cosine"),
        device="cpu",
    )


def identity(product_head: str) -> CheckpointIdentity:
    require(product_head == PRODUCT_HEAD, "execution Product head drift")
    return CheckpointIdentity(
        git_sha=product_head,
        model_spec={"kind": "d05-fresh-process-execution", "width": 3},
        parameter_count=3,
        tokenizer_hash="b" * 64,
        tokenizer_vocab_hash="c" * 64,
        dataset_manifest_hash="d" * 64,
        run_manifest_hash="e" * 64,
        training_config={"steps": 3},
        seed=SEED,
        precision="fp32",
        step=1,
        tokens_seen=2,
        optimizer={"name": "AdamW"},
        scheduler={"name": "cosine"},
        environment_lock_hash="f" * 64,
    )


def rng_record() -> dict[str, Any]:
    return {
        "python": sha256(canonical(normalize(random.getstate()))),
        "numpy": sha256(canonical(normalize(np.random.get_state()))),
        "torch": tensor_record(torch.get_rng_state())["sha256"],
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": (
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
    }


def trainer_record(trainer: Trainer) -> dict[str, Any]:
    scheduler = None if trainer.scheduler is None else trainer.scheduler.state_dict()
    return {
        "model": normalize(trainer.model.state_dict()),
        "optimizer": normalize(trainer.optimizer.state_dict()),
        "scheduler": normalize(scheduler),
        "counters": {
            "micro_step": trainer.micro_step,
            "optimizer_step": trainer.optimizer_step,
            "tokens_seen": trainer.tokens_seen,
        },
        "failure_reason": trainer._failure_reason,
        "update_incomplete": trainer._update_incomplete,
        "rng": rng_record(),
    }


def load_checkpoint(
    loader_name: str,
    checkpoint: Path,
    trainer: Trainer,
    product_head: str,
) -> None:
    kwargs = {
        "model": trainer.model,
        "trainer": trainer,
        "strict_model": True,
        "restore_rng": True,
        "expected_git_sha": product_head,
    }
    if loader_name == "adapter":
        trainer_adapter.load_trainer_checkpoint(checkpoint, **kwargs)
        return
    if loader_name == "progress":
        progress_trainer.load_trainer_checkpoint(
            checkpoint,
            expected_step=1,
            expected_tokens_seen=2,
            **kwargs,
        )
        return
    raise ValueError(f"unsupported loader: {loader_name}")


def produce(
    checkpoint: Path,
    baseline: Path,
    metadata: Path,
    product_head: str,
) -> None:
    require(product_head == PRODUCT_HEAD, "execution Product head drift")
    reset_rng(SEED)
    trainer = make_trainer()
    first = trainer.train_microbatch(_BATCH)
    require(first.optimizer_stepped, "first optimizer step was not committed")
    require(
        (trainer.micro_step, trainer.optimizer_step, trainer.tokens_seen) == (1, 1, 2),
        "first committed progress drift",
    )
    manifest = trainer_adapter.save_trainer_checkpoint(
        checkpoint,
        model=trainer.model,
        trainer=trainer,
        identity=identity(product_head),
    )
    verified = core.verify_checkpoint(checkpoint)
    require(
        verified["checkpoint_id"] == manifest["checkpoint_id"],
        "checkpoint verification identity drift",
    )
    second = trainer.train_microbatch(_BATCH)
    require(second.optimizer_stepped, "baseline second optimizer step not committed")
    require(
        (trainer.micro_step, trainer.optimizer_step, trainer.tokens_seen) == (2, 2, 4),
        "baseline second committed progress drift",
    )
    baseline_value = trainer_record(trainer)
    write_create_only(baseline, baseline_value)
    manifest_bytes = (checkpoint / core.MANIFEST_NAME).read_bytes()
    metadata_value = {
        "product_head": product_head,
        "checkpoint_id": manifest["checkpoint_id"],
        "manifest_sha256": sha256(manifest_bytes),
        "checkpoint_step": manifest["identity"]["step"],
        "checkpoint_tokens_seen": manifest["identity"]["tokens_seen"],
        "baseline_state_sha256": sha256(canonical(baseline_value)),
        "truth_boundary": {
            "real_learned20m_checkpoint": False,
            "synthetic_physical_resume_only": True,
            "authorized_optimized_target_exposure": 0,
            "training_executed_on_admitted_targets": False,
            "learned_weights_created": False,
            "final_test_outcomes_read": False,
            "paid_compute_used": False,
        },
    }
    write_create_only(metadata, metadata_value)


def resume(
    loader_name: str,
    checkpoint: Path,
    output: Path,
    product_head: str,
) -> None:
    require(product_head == PRODUCT_HEAD, "execution Product head drift")
    reset_rng(SEED + 999)
    trainer = make_trainer()
    load_checkpoint(loader_name, checkpoint, trainer, product_head)
    require(
        (trainer.micro_step, trainer.optimizer_step, trainer.tokens_seen) == (1, 1, 2),
        f"{loader_name}: restored progress drift",
    )
    second = trainer.train_microbatch(_BATCH)
    require(second.optimizer_stepped, f"{loader_name}: second optimizer step not committed")
    require(
        (trainer.micro_step, trainer.optimizer_step, trainer.tokens_seen) == (2, 2, 4),
        f"{loader_name}: resumed second progress drift",
    )
    write_create_only(output, trainer_record(trainer))


def corrupt_copy(source: Path, destination: Path) -> None:
    require(not destination.exists(), "corrupt destination already exists")
    shutil.copytree(source, destination)
    payload = destination / core.STATE_TENSORS_NAME
    raw = bytearray(payload.read_bytes())
    require(len(raw) > 32, "state tensor payload unexpectedly small")
    index = len(raw) // 2
    raw[index] ^= 0x01
    payload.write_bytes(raw)


def reject_corrupt(
    loader_name: str,
    checkpoint: Path,
    output: Path,
    product_head: str,
) -> None:
    require(product_head == PRODUCT_HEAD, "execution Product head drift")
    reset_rng(SEED + 1234)
    trainer = make_trainer()
    before = trainer_record(trainer)
    exception_type = ""
    try:
        load_checkpoint(loader_name, checkpoint, trainer, product_head)
    except Exception as exc:
        exception_type = type(exc).__name__
    else:
        raise RuntimeError(f"{loader_name}: corrupt checkpoint unexpectedly loaded")
    after = trainer_record(trainer)
    require(before == after, f"{loader_name}: corrupt rejection mutated fresh target")
    write_create_only(
        output,
        {
            "loader": loader_name,
            "rejected": True,
            "unchanged": True,
            "exception_type": exception_type,
            "before_sha256": sha256(canonical(before)),
            "after_sha256": sha256(canonical(after)),
        },
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(allow_abbrev=False)
    sub = root.add_subparsers(dest="operation", required=True)

    produce_parser = sub.add_parser("produce", allow_abbrev=False)
    produce_parser.add_argument("--checkpoint", type=Path, required=True)
    produce_parser.add_argument("--baseline", type=Path, required=True)
    produce_parser.add_argument("--metadata", type=Path, required=True)
    produce_parser.add_argument("--product-head", required=True)

    resume_parser = sub.add_parser("resume", allow_abbrev=False)
    resume_parser.add_argument("--loader", choices=("adapter", "progress"), required=True)
    resume_parser.add_argument("--checkpoint", type=Path, required=True)
    resume_parser.add_argument("--output", type=Path, required=True)
    resume_parser.add_argument("--product-head", required=True)

    corrupt_parser = sub.add_parser("corrupt-copy", allow_abbrev=False)
    corrupt_parser.add_argument("--source", type=Path, required=True)
    corrupt_parser.add_argument("--destination", type=Path, required=True)

    reject_parser = sub.add_parser("reject-corrupt", allow_abbrev=False)
    reject_parser.add_argument("--loader", choices=("adapter", "progress"), required=True)
    reject_parser.add_argument("--checkpoint", type=Path, required=True)
    reject_parser.add_argument("--output", type=Path, required=True)
    reject_parser.add_argument("--product-head", required=True)

    return root


def main() -> int:
    args = parser().parse_args()
    if args.operation == "produce":
        produce(
            args.checkpoint,
            args.baseline,
            args.metadata,
            args.product_head,
        )
    elif args.operation == "resume":
        resume(args.loader, args.checkpoint, args.output, args.product_head)
    elif args.operation == "corrupt-copy":
        corrupt_copy(args.source, args.destination)
    elif args.operation == "reject-corrupt":
        reject_corrupt(args.loader, args.checkpoint, args.output, args.product_head)
    else:
        raise AssertionError(args.operation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
