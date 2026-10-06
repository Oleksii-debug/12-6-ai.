"""Fresh-process D05 optimizer named-slot qualification with zero scientific credit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from twelve_six.checkpoint import (
    CheckpointCompatibilityError,
    CheckpointIdentity,
    trainer_adapter,
)
from twelve_six.training import Trainer, TrainerConfig


class _TwoNamedParameters(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left = torch.nn.Parameter(torch.ones(3))
        self.right = torch.nn.Parameter(torch.full((3,), 2.0))


def _trainer(
    model: _TwoNamedParameters,
    config: TrainerConfig,
    *,
    reverse: bool = False,
    multiple_groups: bool = False,
    use_default: bool = False,
) -> Trainer:
    if use_default:
        if reverse or multiple_groups:
            raise ValueError("default optimizer mode cannot reorder or split groups")
        return Trainer(model, config, device="cpu")
    if multiple_groups:
        groups = [
            {"params": [model.right if reverse else model.left]},
            {"params": [model.left if reverse else model.right]},
        ]
        optimizer = torch.optim.AdamW(groups, lr=config.learning_rate)
    else:
        params = [model.right, model.left] if reverse else [model.left, model.right]
        optimizer = torch.optim.AdamW(params, lr=config.learning_rate)
    return Trainer(model, config, optimizer=optimizer, device="cpu")


def _identity() -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha="a" * 40,
        model_spec={"kind": "optimizer-named-order-fresh-process", "width": 3},
        parameter_count=6,
        tokenizer_hash="b" * 64,
        tokenizer_vocab_hash="c" * 64,
        dataset_manifest_hash="d" * 64,
        run_manifest_hash="e" * 64,
        training_config={"steps": 3},
        seed=703,
        precision="fp32",
        step=1,
        tokens_seen=3,
        optimizer={"name": "AdamW"},
        scheduler=None,
        environment_lock_hash="f" * 64,
    )


def _tensor_sha256(value: torch.Tensor) -> str:
    payload = value.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _rng_hashes() -> dict[str, str]:
    python_payload = pickle.dumps(
        random.getstate(),
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    numpy_state = np.random.get_state()
    numpy_payload = pickle.dumps(
        (
            numpy_state[0],
            numpy_state[1].tobytes(),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        ),
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    return {
        "python": hashlib.sha256(python_payload).hexdigest(),
        "numpy": hashlib.sha256(numpy_payload).hexdigest(),
        "torch_cpu": _tensor_sha256(torch.random.get_rng_state()),
    }


def _model_hashes(model: _TwoNamedParameters) -> dict[str, str]:
    return {
        name: _tensor_sha256(value)
        for name, value in sorted(model.state_dict().items())
    }


def _optimizer_hashes(
    trainer: Trainer,
    model: _TwoNamedParameters,
) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for name in ("left", "right"):
        parameter = getattr(model, name)
        slot = trainer.optimizer.state[parameter]
        result[name] = {
            key: _tensor_sha256(value)
            for key, value in sorted(slot.items())
            if isinstance(value, torch.Tensor)
        }
    return result


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"expected JSON object: {path}")
    return data


def _producer(
    root: Path,
    *,
    multiple_groups: bool,
    use_default: bool = False,
) -> None:
    checkpoint = root / "checkpoint"
    config = TrainerConfig(seed=703, max_steps=3)
    model = _TwoNamedParameters()
    source = _trainer(
        model,
        config,
        multiple_groups=multiple_groups,
        use_default=use_default,
    )
    model.left.grad = torch.ones(3)
    model.right.grad = torch.full((3,), 9.0)
    source.optimizer.step()
    source.optimizer.zero_grad(set_to_none=True)
    source.micro_step = 1
    source.optimizer_step = 1
    source.tokens_seen = 3
    source.assert_checkpoint_safe()

    sealed = source.state_dict()
    expected_names = (
        [["left"], ["right"]]
        if multiple_groups
        else [["left", "right"]]
    )
    actual_names = [
        group["param_names"] for group in sealed.optimizer["param_groups"]
    ]
    if actual_names != expected_names:
        raise RuntimeError(
            f"source optimizer named-slot identity drifted: {actual_names!r}"
        )

    trainer_adapter.save_trainer_checkpoint(
        checkpoint,
        model=model,
        trainer=source,
        identity=_identity(),
    )
    before_next = _optimizer_hashes(source, model)

    model.left.grad = torch.full_like(model.left, 0.25)
    model.right.grad = torch.full_like(model.right, 0.5)
    source.optimizer.step()
    expected_next = _model_hashes(model)

    _write_json(
        root / "producer.json",
        {
            "pid": os.getpid(),
            "optimizer_hashes": before_next,
            "next_model_hashes": expected_next,
            "expected_names": expected_names,
        },
    )


def _consumer_valid(
    root: Path,
    *,
    multiple_groups: bool,
    use_default: bool = False,
) -> None:
    expected = _read_json(root / "producer.json")
    config = TrainerConfig(seed=703, max_steps=3)
    model = _TwoNamedParameters()
    target = _trainer(
        model,
        config,
        multiple_groups=multiple_groups,
        use_default=use_default,
    )

    trainer_adapter.load_trainer_checkpoint(
        root / "checkpoint",
        model=model,
        trainer=target,
        strict_model=False,
        restore_rng=False,
    )
    if (target.micro_step, target.optimizer_step, target.tokens_seen) != (1, 1, 3):
        raise RuntimeError("fresh-process progress counters did not restore exactly")
    if any("param_names" in group for group in target.optimizer.param_groups):
        raise RuntimeError("checkpoint-only param_names leaked into live optimizer")

    loaded_hashes = _optimizer_hashes(target, model)
    if loaded_hashes != expected["optimizer_hashes"]:
        raise RuntimeError("fresh-process AdamW moments differ from producer")

    model.left.grad = torch.full_like(model.left, 0.25)
    model.right.grad = torch.full_like(model.right, 0.5)
    target.optimizer.step()
    next_hashes = _model_hashes(model)
    if next_hashes != expected["next_model_hashes"]:
        raise RuntimeError("fresh-process next optimizer update diverged")

    _write_json(
        root / "consumer-valid.json",
        {
            "pid": os.getpid(),
            "same_optimizer_moments": True,
            "same_next_update": True,
        },
    )


def _consumer_reversed(root: Path, *, multiple_groups: bool) -> None:
    config = TrainerConfig(seed=703, max_steps=3)
    model = _TwoNamedParameters()
    target = _trainer(
        model,
        config,
        reverse=True,
        multiple_groups=multiple_groups,
    )
    random.seed(913)
    np.random.seed(917)
    torch.manual_seed(919)
    before_rng = _rng_hashes()
    before_model = _model_hashes(model)
    try:
        trainer_adapter.load_trainer_checkpoint(
            root / "checkpoint",
            model=model,
            trainer=target,
            strict_model=False,
            restore_rng=True,
        )
    except CheckpointCompatibilityError as exc:
        if "optimizer parameter order" not in str(exc):
            raise RuntimeError(
                f"unexpected reversed-order diagnostic: {exc}"
            ) from exc
    else:
        raise RuntimeError("reversed same-shape optimizer order was accepted")

    if _rng_hashes() != before_rng:
        raise RuntimeError("reversed-order refusal mutated ambient RNG state")
    if _model_hashes(model) != before_model:
        raise RuntimeError("reversed-order refusal mutated model weights")
    if target.optimizer.state:
        raise RuntimeError("reversed-order refusal mutated optimizer state")
    if (target.micro_step, target.optimizer_step, target.tokens_seen) != (0, 0, 0):
        raise RuntimeError("reversed-order refusal mutated progress counters")
    if target._failure_reason is not None or target._update_incomplete:
        raise RuntimeError("preapply reversed-order refusal poisoned clean target")

    _write_json(
        root / "consumer-reversed.json",
        {
            "pid": os.getpid(),
            "reversed_order_failed_closed": True,
            "rng_unchanged": True,
            "target_unchanged": True,
        },
    )


def _run_child(
    root: Path,
    *,
    phase: str,
    multiple_groups: bool,
    use_default: bool = False,
) -> None:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--phase",
        phase,
        "--root",
        str(root),
        "--groups",
        "default" if use_default else ("two" if multiple_groups else "one"),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"child phase {phase!r} failed with {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )


def _orchestrate(output: Path) -> None:
    modes: dict[str, dict[str, Any]] = {}
    child_pids: list[int] = []
    with tempfile.TemporaryDirectory(prefix="d05 optimizer named order ") as raw:
        base = Path(raw)

        default_root = base / "default-one-group"
        default_root.mkdir()
        for phase in ("producer", "consumer-valid"):
            _run_child(
                default_root,
                phase=phase,
                multiple_groups=False,
                use_default=True,
            )
        default_producer = _read_json(default_root / "producer.json")
        default_valid = _read_json(default_root / "consumer-valid.json")
        default_pids = [
            int(default_producer["pid"]),
            int(default_valid["pid"]),
        ]
        child_pids.extend(default_pids)
        modes["default-one-group"] = {
            "fresh_process_pids": default_pids,
            "same_optimizer_moments": default_valid["same_optimizer_moments"],
            "same_next_update": default_valid["same_next_update"],
            "expected_names": default_producer["expected_names"],
        }

        for mode, multiple_groups in (
            ("one-group", False),
            ("two-groups", True),
        ):
            root = base / mode
            root.mkdir()
            for phase in ("producer", "consumer-valid", "consumer-reversed"):
                _run_child(
                    root,
                    phase=phase,
                    multiple_groups=multiple_groups,
                )

            producer = _read_json(root / "producer.json")
            valid = _read_json(root / "consumer-valid.json")
            reversed_result = _read_json(root / "consumer-reversed.json")
            pids = [
                int(producer["pid"]),
                int(valid["pid"]),
                int(reversed_result["pid"]),
            ]
            child_pids.extend(pids)
            modes[mode] = {
                "fresh_process_pids": pids,
                "same_optimizer_moments": valid["same_optimizer_moments"],
                "same_next_update": valid["same_next_update"],
                "reversed_order_failed_closed": (
                    reversed_result["reversed_order_failed_closed"]
                ),
                "reversed_rng_unchanged": reversed_result["rng_unchanged"],
                "reversed_target_unchanged": reversed_result["target_unchanged"],
                "expected_names": producer["expected_names"],
            }

    if len(set(child_pids)) != len(child_pids) or os.getpid() in child_pids:
        raise RuntimeError("qualification phases did not execute in distinct processes")

    report = {
        "schema_version": 1,
        "verdict": "PASS_D05_OPTIMIZER_NAMED_ORDER_FRESH_PROCESS_ZERO_CREDIT",
        "qualification_pid": os.getpid(),
        "source_sha": os.environ.get("SOURCE_SHA"),
        "qualification_carrier_sha": os.environ.get(
            "QUALIFICATION_CARRIER_SHA"
        ),
        "modes": modes,
        "scientific_credit": {
            "real_corpus_used": False,
            "tokenizer_fit_authorized": False,
            "optimized_target_exposure": 0,
            "real_target_optimizer_updates": 0,
            "model_training_credit": False,
            "learned_weights_created": False,
            "final_test_read": False,
            "paid_compute_used": False,
            "scale_promotion_authorized": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, report)
    print(report["verdict"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--phase",
        choices=("producer", "consumer-valid", "consumer-reversed"),
    )
    parser.add_argument("--root", type=Path)
    parser.add_argument("--groups", choices=("default", "one", "two"))
    args = parser.parse_args()

    if args.phase is None:
        if args.output is None:
            parser.error("--output is required for orchestration")
        _orchestrate(args.output)
        return

    if args.root is None or args.groups is None:
        parser.error("--root and --groups are required for child phases")
    use_default = args.groups == "default"
    multiple_groups = args.groups == "two"
    if args.phase == "producer":
        _producer(
            args.root,
            multiple_groups=multiple_groups,
            use_default=use_default,
        )
    elif args.phase == "consumer-valid":
        _consumer_valid(
            args.root,
            multiple_groups=multiple_groups,
            use_default=use_default,
        )
    else:
        _consumer_reversed(args.root, multiple_groups=multiple_groups)


if __name__ == "__main__":
    main()
