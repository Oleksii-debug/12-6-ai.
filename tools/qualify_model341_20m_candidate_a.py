from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from twelve_six import TwelveSixDecoder, count_trainable_parameters, load_stage_config
from twelve_six.checkpoint import CheckpointIdentity, load_checkpoint, save_checkpoint
from twelve_six.inference.static_kv import (
    allocate_static_kv_cache,
    decode_one_with_static_kv_cache,
    prefill_static_kv_cache,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "candidates" / "model341_20m_candidate_a.json"
EXPECTED_PARAMETERS = 20_613_440
SEED = 341


def _all_finite(tensors: list[torch.Tensor]) -> bool:
    return all(bool(torch.isfinite(tensor).all()) for tensor in tensors)


def _identity(stage, *, step: int, tokens_seen: int) -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha="f" * 40,
        model_spec=stage.model.to_dict(),
        parameter_count=stage.expected_parameters,
        tokenizer_hash="1" * 64,
        tokenizer_vocab_hash="2" * 64,
        dataset_manifest_hash="3" * 64,
        run_manifest_hash="4" * 64,
        training_config={"worker": "MODEL-341-20M-CANDIDATE-A", "long_training": False},
        seed=SEED,
        precision="float32",
        step=step,
        tokens_seen=tokens_seen,
        optimizer={"name": "SGD", "lr": 0.001, "momentum": 0.9},
        scheduler=None,
    )


def qualify() -> dict[str, object]:
    torch.manual_seed(SEED)
    stage = load_stage_config(CONFIG)
    checks: dict[str, dict[str, object]] = {}

    started = time.perf_counter()
    model = TwelveSixDecoder(stage.model, stage.init)
    actual_parameters = count_trainable_parameters(model)
    checks["parameter_count"] = {
        "pass": actual_parameters == EXPECTED_PARAMETERS == stage.model.parameter_count(),
        "actual": actual_parameters,
        "expected": EXPECTED_PARAMETERS,
        "seconds": time.perf_counter() - started,
    }

    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    input_ids = torch.randint(0, stage.model.vocab_size, (1, 32), dtype=torch.long)
    targets = torch.randint(0, stage.model.vocab_size, (1, 32), dtype=torch.long)
    before = model.token_embedding.weight.detach().clone()
    started = time.perf_counter()
    logits = model(input_ids).logits
    loss = F.cross_entropy(logits.reshape(-1, stage.model.vocab_size), targets.reshape(-1))
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    gradient_norm = math.sqrt(
        sum(float(gradient.float().pow(2).sum()) for gradient in gradients)
    )
    optimizer.step()
    forward_backward_ok = (
        bool(torch.isfinite(logits).all())
        and bool(torch.isfinite(loss))
        and _all_finite(gradients)
        and not torch.equal(before, model.token_embedding.weight.detach())
        and _all_finite([parameter.detach() for parameter in model.parameters()])
    )
    checks["forward_backward_update"] = {
        "pass": forward_backward_ok,
        "logits_shape": list(logits.shape),
        "loss": float(loss.detach()),
        "grad_norm_l2": gradient_norm,
        "embedding_changed": not torch.equal(before, model.token_embedding.weight.detach()),
        "seconds": time.perf_counter() - started,
    }

    model.eval()
    checkpoint_probe = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
    expected_state = {
        name: tensor.detach().clone() for name, tensor in model.state_dict().items()
    }
    expected_logits = model(checkpoint_probe).logits.detach().clone()
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="model341-d05-") as temporary:
        checkpoint_dir = Path(temporary) / "checkpoint"
        manifest = save_checkpoint(
            checkpoint_dir,
            model=model,
            optimizer=optimizer,
            trainer_state={"worker": "MODEL-341-20M-CANDIDATE-A", "probe_only": True},
            identity=_identity(stage, step=1, tokens_seen=32),
        )
        torch.manual_seed(999)
        restored = TwelveSixDecoder(stage.model, stage.init)
        restored_optimizer = torch.optim.SGD(restored.parameters(), lr=9.0, momentum=0.0)
        result = load_checkpoint(
            checkpoint_dir,
            model=restored,
            optimizer=restored_optimizer,
            restore_rng=False,
            expected_model_spec_hash=stage.model.identity_sha256(),
        )
        restored.eval()
        restored_logits = restored(checkpoint_probe).logits
        state_equal = all(
            torch.equal(restored.state_dict()[name], tensor)
            for name, tensor in expected_state.items()
        )
        optimizer_groups_equal = (
            restored_optimizer.state_dict()["param_groups"]
            == optimizer.state_dict()["param_groups"]
        )
        checkpoint_bytes = sum(path.stat().st_size for path in checkpoint_dir.iterdir())
        d05_ok = (
            state_equal
            and torch.equal(restored_logits, expected_logits)
            and optimizer_groups_equal
            and manifest["serialization"]["pickle"] is False
            and result.manifest["checkpoint_id"] == manifest["checkpoint_id"]
        )
        checks["d05_save_load"] = {
            "pass": d05_ok,
            "checkpoint_id": manifest["checkpoint_id"],
            "state_equal": state_equal,
            "logits_bitwise_equal": torch.equal(restored_logits, expected_logits),
            "optimizer_param_groups_equal": optimizer_groups_equal,
            "pickle": manifest["serialization"]["pickle"],
            "checkpoint_bytes": checkpoint_bytes,
            "seconds": time.perf_counter() - started,
        }

    model.eval()
    prompt = torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17]], dtype=torch.long)
    next_token = torch.tensor([[18]], dtype=torch.long)
    cache = allocate_static_kv_cache(model, batch_size=1)
    storage = cache.storage_signature
    allocated_bytes = cache.allocated_bytes
    started = time.perf_counter()
    cached_prompt = prefill_static_kv_cache(model, prompt, cache).logits
    stateless_prompt = model(prompt).logits
    cached_next = decode_one_with_static_kv_cache(model, next_token, cache).logits
    stateless_next = model(torch.cat((prompt, next_token), dim=1)).logits[:, -1:, :]
    prefill_matches = torch.allclose(cached_prompt, stateless_prompt, rtol=1e-5, atol=1e-5)
    decode_matches = torch.allclose(cached_next, stateless_next, rtol=1e-5, atol=1e-5)
    storage_stable = storage == cache.storage_signature and allocated_bytes == cache.allocated_bytes
    static_finite = bool(torch.isfinite(cached_prompt).all() and torch.isfinite(cached_next).all())
    checks["static_kv"] = {
        "pass": prefill_matches and decode_matches and storage_stable and static_finite,
        "prefill_matches_stateless": prefill_matches,
        "decode_matches_stateless": decode_matches,
        "storage_stable": storage_stable,
        "cache_shape": [
            1,
            stage.model.n_kv_heads,
            stage.model.max_seq_len,
            stage.model.head_dim,
        ],
        "allocated_bytes": allocated_bytes,
        "valid_length": cache.valid_lengths[0],
        "finite_logits": static_finite,
        "seconds": time.perf_counter() - started,
    }

    started = time.perf_counter()
    over_limit_rejected = False
    try:
        model(torch.zeros((1, stage.model.max_seq_len + 1), dtype=torch.long))
    except ValueError as error:
        over_limit_rejected = "exceeds max_seq_len" in str(error)
    boundary = torch.arange(stage.model.max_seq_len, dtype=torch.long)
    boundary = boundary.remainder(stage.model.vocab_size).unsqueeze(0)
    with torch.no_grad():
        boundary_logits = model(boundary).logits
    boundary_finite = bool(torch.isfinite(boundary_logits).all())
    boundary_shape_ok = boundary_logits.shape == (
        1,
        stage.model.max_seq_len,
        stage.model.vocab_size,
    )
    checks["context_bounds_and_finite_logits"] = {
        "pass": over_limit_rejected and boundary_finite and boundary_shape_ok,
        "1025_rejected": over_limit_rejected,
        "1024_shape": list(boundary_logits.shape),
        "1024_finite_logits": boundary_finite,
        "seconds": time.perf_counter() - started,
    }

    passed = all(bool(check["pass"]) for check in checks.values())
    return {
        "worker": "MODEL-341-20M-CANDIDATE-A",
        "qualification": "PASS" if passed else "FAIL",
        "pass": passed,
        "random_init_only": True,
        "long_training_performed": False,
        "device": str(next(model.parameters()).device),
        "torch": torch.__version__,
        "model_spec_sha256": stage.model.identity_sha256(),
        "init_spec_sha256": stage.init.identity_sha256(),
        "parameter_breakdown": stage.model.parameter_breakdown(),
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    report = qualify()
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
