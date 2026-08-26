"""Deterministic local-vs-locked runtime parity evidence for 12-6 research."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch

from .checkpoint import CheckpointIdentity, load_trainer_checkpoint, save_trainer_checkpoint, verify_checkpoint
from .model import InitSpec, ModelSpec, TwelveSixDecoder
from .training import Trainer, TrainerConfig
from .training.loss import causal_lm_loss

SCHEMA_TRACE = "12-6.environment-parity-trace.v1"
SCHEMA_COMPARE = "12-6.environment-parity-comparison.v1"
SCHEMA_BOOTSTRAP = "12-6.environment-bootstrap.v1"
EXPECTED_PYTHON = "3.11.16"
EXPECTED_TORCH = "2.13.0"
DEFAULT_ATOL = 1e-6
DEFAULT_RTOL = 1e-5
ZERO64 = "0" * 64


class EnvironmentParityError(RuntimeError):
    """Fail-closed ENV-160 parity error."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def hash_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_head(repo: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _lock_fingerprints(repo: Path) -> dict[str, str | None]:
    base = repo / "requirements" / "locks" / "linux-x86_64"
    result: dict[str, str | None] = {}
    for name in ("toolchain.lock.txt", "runtime.lock.txt", "dev.lock.txt"):
        path = base / name
        result[name] = _sha256_file(path) if path.is_file() else None
    return result


def environment_fingerprint(repo: Path, *, source_sha: str | None = None) -> dict[str, Any]:
    locks = _lock_fingerprints(repo)
    value: dict[str, Any] = {
        "schema": "12-6.environment-fingerprint.v1",
        "source_sha": source_sha or _git_head(repo),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "torch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
        "locks": locks,
        "expected_locked_runtime": {
            "python": EXPECTED_PYTHON,
            "torch": EXPECTED_TORCH,
            "profile": "linux-x86_64",
        },
        "exact_locked_runtime": (
            platform.python_version() == EXPECTED_PYTHON
            and torch.__version__.split("+", 1)[0] == EXPECTED_TORCH
            and all(value is not None for value in locks.values())
        ),
        "local_free": True,
    }
    value["fingerprint_sha256"] = hash_json(value)
    return value


def bootstrap_environment(
    repo: Path,
    *,
    source_sha: str | None,
    require_locked: bool,
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    head = _git_head(repo)
    checks["git_available_and_repo"] = head is not None
    checks["source_head_matches"] = source_sha is None or head == source_sha
    checks["python_import_torch"] = True
    checks["project_model_import"] = TwelveSixDecoder is not None
    checks["project_trainer_import"] = Trainer is not None
    checks["project_checkpoint_import"] = save_trainer_checkpoint is not None
    checks["lock_files_present"] = all(
        value is not None for value in _lock_fingerprints(repo).values()
    )
    fingerprint = environment_fingerprint(repo, source_sha=source_sha)
    checks["exact_locked_runtime"] = fingerprint["exact_locked_runtime"]
    required = [
        "git_available_and_repo",
        "source_head_matches",
        "python_import_torch",
        "project_model_import",
        "project_trainer_import",
        "project_checkpoint_import",
        "lock_files_present",
    ]
    if require_locked:
        required.append("exact_locked_runtime")
    passed = all(checks[name] is True for name in required)
    report = {
        "schema": SCHEMA_BOOTSTRAP,
        "status": "PASS" if passed else "FAIL",
        "require_locked": require_locked,
        "checks": checks,
        "environment_fingerprint": fingerprint,
        "authority": "EXACT_LOCKED" if fingerprint["exact_locked_runtime"] else "SOURCE_EQUIVALENT_DEBUG_ONLY",
    }
    report["report_sha256"] = hash_json(report)
    if not passed:
        failed = [name for name in required if checks[name] is not True]
        raise EnvironmentParityError(f"environment bootstrap failed: {failed}")
    return report


def _tiny_spec() -> ModelSpec:
    return ModelSpec(
        schema_version=1,
        vocab_size=32,
        max_seq_len=16,
        d_model=16,
        n_layers=1,
        n_heads=2,
        n_kv_heads=2,
        head_dim=8,
        d_ff=32,
        rope_rotary_dim=8,
    )


def _trainer_config() -> TrainerConfig:
    return TrainerConfig(
        learning_rate=1e-3,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        eps=1e-8,
        max_steps=3,
        warmup_steps=0,
        scheduler="constant",
        gradient_accumulation_steps=1,
        gradient_clip_norm=1.0,
        precision="fp32",
        seed=160,
        deterministic_algorithms=True,
        deterministic_warn_only=False,
    )


def _batches() -> list[dict[str, torch.Tensor]]:
    rows = [
        [1, 2, 3, 4, 5, 6, 7, 8],
        [8, 7, 6, 5, 4, 3, 2, 1],
        [2, 4, 6, 8, 10, 12, 14, 16],
        [3, 6, 9, 12, 15, 18, 21, 24],
        [5, 10, 15, 20, 25, 30, 3, 8],
        [31, 29, 27, 25, 23, 21, 19, 17],
    ]
    out: list[dict[str, torch.Tensor]] = []
    for i in range(0, len(rows), 2):
        ids = torch.tensor(rows[i : i + 2], dtype=torch.long)
        labels = ids.clone()
        labels[0, -1] = -100
        out.append({"input_ids": ids, "labels": labels})
    return out


def _heldout_batch() -> dict[str, torch.Tensor]:
    ids = torch.tensor(
        [[4, 8, 12, 16, 20, 24, 28, 0], [0, 3, 6, 9, 12, 15, 18, 21]],
        dtype=torch.long,
    )
    labels = ids.clone()
    labels[1, -1] = -100
    return {"input_ids": ids, "labels": labels}


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).reshape(-1).tolist()
    return bytes(raw)


def _tensor_record(tensor: torch.Tensor, *, include_values: bool = True) -> dict[str, Any]:
    value = tensor.detach().cpu().contiguous()
    record: dict[str, Any] = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(_tensor_bytes(value)).hexdigest(),
    }
    if include_values:
        record["values"] = [float(item) for item in value.float().reshape(-1).tolist()]
    return record


def _state_record(model: torch.nn.Module) -> dict[str, Any]:
    tensors = {name: _tensor_record(tensor) for name, tensor in sorted(model.state_dict().items())}
    return {"tensors": tensors, "state_sha256": hash_json({k: v["sha256"] for k, v in tensors.items()})}


def _gradient_record(model: torch.nn.Module) -> dict[str, Any]:
    tensors: dict[str, Any] = {}
    for name, parameter in sorted(model.named_parameters()):
        if parameter.grad is None:
            tensors[name] = None
        else:
            tensors[name] = _tensor_record(parameter.grad)
    return {
        "tensors": tensors,
        "gradient_sha256": hash_json(
            {name: None if value is None else value["sha256"] for name, value in tensors.items()}
        ),
    }


def _extract_logits(output: Any) -> torch.Tensor:
    logits = getattr(output, "logits", None)
    if not isinstance(logits, torch.Tensor):
        raise TypeError("expected project model output with logits tensor")
    return logits


def _eval_non_mutating(model: torch.nn.Module, batch: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    before = _state_record(model)["state_sha256"]
    training = model.training
    with torch.no_grad():
        output = model(batch["input_ids"])
        loss = causal_lm_loss(_extract_logits(output), batch["labels"])
    after = _state_record(model)["state_sha256"]
    if before != after or model.training != training:
        raise EnvironmentParityError("held-out evaluation mutated model state or module mode")
    return {
        "loss": float(loss.item()),
        "state_sha256_before": before,
        "state_sha256_after": after,
        "module_training_before": training,
        "module_training_after": model.training,
        "non_mutation_passed": True,
    }


def _fixed_hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _checkpoint_identity(
    *,
    source_sha: str,
    spec: ModelSpec,
    init: InitSpec,
    cfg: TrainerConfig,
    trainer: Trainer,
    env_lock_hash: str,
) -> CheckpointIdentity:
    training_config = {
        "trainer": json.loads(json.dumps(asdict(cfg))),
        "init_spec_sha256": init.identity_sha256(),
        "data": {
            "split_identity": _fixed_hash("env160-split-v1"),
            "packing_sha256": _fixed_hash("env160-packing-v1"),
            "packing_version": "env160-fixed-trace-v1",
        },
    }
    return CheckpointIdentity(
        git_sha=source_sha,
        model_spec=spec.to_dict(),
        parameter_count=spec.parameter_count(),
        tokenizer_hash=_fixed_hash("env160-tokenizer-config-v1"),
        tokenizer_vocab_hash=_fixed_hash("env160-tokenizer-vocab-v1"),
        dataset_manifest_hash=_fixed_hash("env160-dataset-v1"),
        run_manifest_hash=_fixed_hash("env160-run-v1"),
        training_config=training_config,
        seed=cfg.seed,
        precision=cfg.precision,
        step=trainer.optimizer_step,
        tokens_seen=trainer.tokens_seen,
        optimizer={
            "name": "AdamW",
            "learning_rate": cfg.learning_rate,
            "betas": list(cfg.betas),
            "eps": cfg.eps,
            "weight_decay": cfg.weight_decay,
        },
        scheduler=None,
        environment_lock_hash=env_lock_hash,
    )


def _checkpoint_semantics(manifest: Mapping[str, Any]) -> dict[str, Any]:
    identity = dict(manifest["identity"])
    return {
        "checkpoint_id": manifest["checkpoint_id"],
        "identity": identity,
        "semantic_identity_sha256": hash_json(identity),
    }


def capture_trace(repo: Path, *, source_sha: str, require_locked: bool) -> dict[str, Any]:
    bootstrap = bootstrap_environment(repo, source_sha=source_sha, require_locked=require_locked)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(160)
    spec = _tiny_spec()
    init = InitSpec()
    cfg = _trainer_config()
    fingerprint = environment_fingerprint(repo, source_sha=source_sha)
    lock_values = fingerprint["locks"]
    env_lock_hash = hash_json(lock_values)

    probe = TwelveSixDecoder(spec, init)
    initial_state = _state_record(probe)
    first_batch = _batches()[0]
    first_logits = _extract_logits(probe(first_batch["input_ids"]))
    first_loss = causal_lm_loss(first_logits, first_batch["labels"])
    first_loss.backward()
    initial_gradients = _gradient_record(probe)
    probe.zero_grad(set_to_none=True)

    torch.manual_seed(160)
    model = TwelveSixDecoder(spec, init)
    if _state_record(model)["state_sha256"] != initial_state["state_sha256"]:
        raise EnvironmentParityError("reconstructed initial state is not deterministic within environment")
    trainer = Trainer(model, cfg, device="cpu")
    step_metrics: list[dict[str, Any]] = []
    checkpoints: dict[str, Any] = {}
    state_after_step_1: dict[str, Any] | None = None
    with tempfile.TemporaryDirectory(prefix="env160-parity-") as temp_name:
        temp = Path(temp_name)
        for index, batch in enumerate(_batches(), start=1):
            metrics = trainer.train_microbatch(batch)
            step_metrics.append(asdict(metrics))
            if index in (1, 3):
                identity = _checkpoint_identity(
                    source_sha=source_sha,
                    spec=spec,
                    init=init,
                    cfg=cfg,
                    trainer=trainer,
                    env_lock_hash=env_lock_hash,
                )
                directory = temp / f"checkpoint-{index}"
                save_trainer_checkpoint(
                    directory,
                    model=model,
                    trainer=trainer,
                    identity=identity,
                    overwrite=True,
                )
                verified = verify_checkpoint(directory)
                checkpoints[f"step_{index}"] = _checkpoint_semantics(verified)
                if index == 1:
                    state_after_step_1 = _state_record(model)
                    torch.manual_seed(999)
                    fresh_model = TwelveSixDecoder(spec, init)
                    fresh_trainer = Trainer(fresh_model, cfg, device="cpu")
                    loaded = load_trainer_checkpoint(
                        directory,
                        model=fresh_model,
                        trainer=fresh_trainer,
                        restore_rng=True,
                        expected_git_sha=source_sha,
                        expected_model_spec_hash=spec.identity_sha256(),
                    )
                    if loaded.manifest["checkpoint_id"] != verified["checkpoint_id"]:
                        raise EnvironmentParityError("checkpoint load changed checkpoint identity")
                    if _state_record(fresh_model)["state_sha256"] != state_after_step_1["state_sha256"]:
                        raise EnvironmentParityError("checkpoint reload changed model state")
                    if (
                        fresh_trainer.optimizer_step != trainer.optimizer_step
                        or fresh_trainer.tokens_seen != trainer.tokens_seen
                    ):
                        raise EnvironmentParityError("checkpoint reload changed trainer counters")
        final_state = _state_record(model)
        heldout = _eval_non_mutating(model, _heldout_batch())

    if state_after_step_1 is None:
        raise AssertionError("step-one state missing")
    trace = {
        "schema": SCHEMA_TRACE,
        "source_sha": source_sha,
        "authority": "EXACT_LOCKED" if fingerprint["exact_locked_runtime"] else "SOURCE_EQUIVALENT_DEBUG_ONLY",
        "environment_fingerprint": fingerprint,
        "bootstrap_report_sha256": bootstrap["report_sha256"],
        "model": {
            "spec": spec.to_dict(),
            "model_spec_sha256": spec.identity_sha256(),
            "parameter_count": spec.parameter_count(),
            "init_spec": init.to_dict(),
            "init_spec_sha256": init.identity_sha256(),
        },
        "optimizer": {
            "config": json.loads(json.dumps(asdict(cfg))),
            "name": "AdamW",
        },
        "inputs": {
            "train_batches": [
                {
                    "input_ids": batch["input_ids"].tolist(),
                    "labels": batch["labels"].tolist(),
                }
                for batch in _batches()
            ],
            "heldout": {
                "input_ids": _heldout_batch()["input_ids"].tolist(),
                "labels": _heldout_batch()["labels"].tolist(),
            },
        },
        "initial": {
            "weights": initial_state,
            "logits": _tensor_record(first_logits),
            "loss": float(first_loss.item()),
            "gradients": initial_gradients,
        },
        "updates": {
            "step_metrics": step_metrics,
            "state_after_step_1": state_after_step_1,
            "state_after_step_3": final_state,
        },
        "checkpoints": checkpoints,
        "heldout_evaluation": heldout,
        "token_counters": {
            "micro_step": trainer.micro_step,
            "optimizer_step": trainer.optimizer_step,
            "tokens_seen": trainer.tokens_seen,
            "per_step_tokens": [row["tokens"] for row in step_metrics],
        },
        "truth_boundary": {
            "local_free": True,
            "foreign_pretrained_weights": False,
            "scientific_authority_requires_exact_locked_runtime": True,
        },
    }
    trace["trace_sha256"] = hash_json(trace)
    return trace


def _numeric_pairs(left: Any, right: Any, prefix: str = "") -> list[tuple[str, float, float]]:
    pairs: list[tuple[str, float, float]] = []
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) & set(right)):
            child = f"{prefix}.{key}" if prefix else key
            pairs.extend(_numeric_pairs(left[key], right[key], child))
    elif isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        if left and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in left + right):
            for index, (a, b) in enumerate(zip(left, right)):
                pairs.append((f"{prefix}[{index}]", float(a), float(b)))
        else:
            for index, (a, b) in enumerate(zip(left, right)):
                pairs.extend(_numeric_pairs(a, b, f"{prefix}[{index}]"))
    elif (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        pairs.append((prefix, float(left), float(right)))
    return pairs


def _numeric_view(trace: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "initial_logits": trace["initial"]["logits"]["values"],
        "initial_loss": trace["initial"]["loss"],
        "initial_gradients": {
            name: None if value is None else value["values"]
            for name, value in trace["initial"]["gradients"]["tensors"].items()
        },
        "state_after_step_1": {
            name: value["values"]
            for name, value in trace["updates"]["state_after_step_1"]["tensors"].items()
        },
        "state_after_step_3": {
            name: value["values"]
            for name, value in trace["updates"]["state_after_step_3"]["tensors"].items()
        },
        "heldout_loss": trace["heldout_evaluation"]["loss"],
        "step_metrics": trace["updates"]["step_metrics"],
    }


def compare_traces(
    canonical: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> dict[str, Any]:
    semantic_checks = {
        "source_sha": canonical["source_sha"] == candidate["source_sha"],
        "model_spec": canonical["model"]["model_spec_sha256"] == candidate["model"]["model_spec_sha256"],
        "parameter_count": canonical["model"]["parameter_count"] == candidate["model"]["parameter_count"],
        "init_spec": canonical["model"]["init_spec_sha256"] == candidate["model"]["init_spec_sha256"],
        "optimizer_config": canonical["optimizer"] == candidate["optimizer"],
        "inputs": canonical["inputs"] == candidate["inputs"],
        "token_counters": canonical["token_counters"] == candidate["token_counters"],
        "checkpoint_step_1_semantics": (
            canonical["checkpoints"]["step_1"]["identity"]
            == candidate["checkpoints"]["step_1"]["identity"]
        ),
        "checkpoint_step_3_semantics": (
            canonical["checkpoints"]["step_3"]["identity"]
            == candidate["checkpoints"]["step_3"]["identity"]
        ),
        "heldout_non_mutation": (
            canonical["heldout_evaluation"]["non_mutation_passed"] is True
            and candidate["heldout_evaluation"]["non_mutation_passed"] is True
        ),
    }
    semantic_pass = all(semantic_checks.values())
    pairs = _numeric_pairs(_numeric_view(canonical), _numeric_view(candidate))
    max_abs = 0.0
    max_rel = 0.0
    worst = None
    numeric_pass = True
    for path, left, right in pairs:
        absolute = abs(left - right)
        denominator = max(abs(left), abs(right), 1e-30)
        relative = absolute / denominator
        if absolute > max_abs:
            max_abs = absolute
            worst = {"path": path, "canonical": left, "candidate": right, "absolute": absolute}
        max_rel = max(max_rel, relative)
        if not math.isclose(left, right, rel_tol=rtol, abs_tol=atol):
            numeric_pass = False
    bitwise_checks = {
        "initial_weights": canonical["initial"]["weights"]["state_sha256"] == candidate["initial"]["weights"]["state_sha256"],
        "initial_logits": canonical["initial"]["logits"]["sha256"] == candidate["initial"]["logits"]["sha256"],
        "initial_gradients": canonical["initial"]["gradients"]["gradient_sha256"] == candidate["initial"]["gradients"]["gradient_sha256"],
        "state_after_step_1": canonical["updates"]["state_after_step_1"]["state_sha256"] == candidate["updates"]["state_after_step_1"]["state_sha256"],
        "state_after_step_3": canonical["updates"]["state_after_step_3"]["state_sha256"] == candidate["updates"]["state_after_step_3"]["state_sha256"],
        "checkpoint_id_step_1": canonical["checkpoints"]["step_1"]["checkpoint_id"] == candidate["checkpoints"]["step_1"]["checkpoint_id"],
        "checkpoint_id_step_3": canonical["checkpoints"]["step_3"]["checkpoint_id"] == candidate["checkpoints"]["step_3"]["checkpoint_id"],
    }
    if not semantic_pass:
        classification = "SEMANTIC_DRIFT"
    elif not numeric_pass:
        classification = "NUMERIC_DRIFT_REQUIRES_EXACT_HEAD"
    elif all(bitwise_checks.values()):
        classification = "PASS_BITWISE"
    else:
        classification = "PASS_NUMERIC_TOLERANCE"
    canonical_locked = canonical["environment_fingerprint"]["exact_locked_runtime"] is True
    candidate_locked = candidate["environment_fingerprint"]["exact_locked_runtime"] is True
    scientific_authority = semantic_pass and numeric_pass and canonical_locked and candidate_locked
    report = {
        "schema": SCHEMA_COMPARE,
        "classification": classification,
        "tolerances": {"atol": atol, "rtol": rtol},
        "semantic_checks": semantic_checks,
        "semantic_pass": semantic_pass,
        "numeric": {
            "compared_scalar_count": len(pairs),
            "pass": numeric_pass,
            "max_absolute_difference": max_abs,
            "max_relative_difference": max_rel,
            "worst_absolute_difference": worst,
        },
        "bitwise_checks": bitwise_checks,
        "environment_differences_expected_and_nonsemantic": {
            "python_version": [
                canonical["environment_fingerprint"]["python"]["version"],
                candidate["environment_fingerprint"]["python"]["version"],
            ],
            "torch_version": [
                canonical["environment_fingerprint"]["torch"]["version"],
                candidate["environment_fingerprint"]["torch"]["version"],
            ],
            "platform_release": [
                canonical["environment_fingerprint"]["platform"]["release"],
                candidate["environment_fingerprint"]["platform"]["release"],
            ],
        },
        "scientific_authority": scientific_authority,
        "truth_boundary": (
            "Cross-version bitwise equality is not required. Semantic identities and token/checkpoint "
            "semantics must match exactly; fp32 numeric tensors must satisfy explicit tolerances. "
            "A source-equivalent candidate remains debugging evidence even when numerically equivalent."
        ),
        "decision_policy": decision_policy(),
    }
    report["report_sha256"] = hash_json(report)
    return report


def decision_policy() -> dict[str, Any]:
    return {
        "source_equivalent_debugging_allowed_for": [
            "failure localization",
            "syntax and invariant debugging",
            "deterministic trace development",
            "rough local performance diagnostics explicitly labeled non-authoritative",
        ],
        "exact_head_locked_rerun_mandatory_for": [
            "published held-out quality or learned-result numbers",
            "checkpoint winner selection",
            "cross-scale quality/efficiency/scaling ranking",
            "architecture tokenizer optimizer or schedule scientific decisions",
            "stage promotion freeze or reproducibility claims",
            "any result used as canonical evidence",
        ],
        "source_equivalent_never_upgrades_itself_to_authority": True,
        "cross_version_bitwise_parity_required": False,
        "semantic_identity_parity_required": True,
        "numeric_tolerance_required": {"fp32_atol": DEFAULT_ATOL, "fp32_rtol": DEFAULT_RTOL},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bootstrap")
    b.add_argument("--repo-root", type=Path, default=Path("."))
    b.add_argument("--source-sha", required=True)
    b.add_argument("--require-locked", action="store_true")
    b.add_argument("--output", type=Path, required=True)

    c = sub.add_parser("capture")
    c.add_argument("--repo-root", type=Path, default=Path("."))
    c.add_argument("--source-sha", required=True)
    c.add_argument("--require-locked", action="store_true")
    c.add_argument("--output", type=Path, required=True)

    q = sub.add_parser("compare")
    q.add_argument("canonical", type=Path)
    q.add_argument("candidate", type=Path)
    q.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    q.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    q.add_argument("--output", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.cmd == "bootstrap":
        value = bootstrap_environment(
            args.repo_root.resolve(), source_sha=args.source_sha, require_locked=args.require_locked
        )
    elif args.cmd == "capture":
        value = capture_trace(
            args.repo_root.resolve(), source_sha=args.source_sha, require_locked=args.require_locked
        )
    else:
        value = compare_traces(
            _read_json(args.canonical), _read_json(args.candidate), atol=args.atol, rtol=args.rtol
        )
    _write_json(args.output, value)
    print(json.dumps({"status": value.get("status", value.get("classification", "PASS")), "report": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
