"""Independent, fail-closed verification for MODEL-341 20M candidate A.

The verifier deliberately recomputes candidate identity and parameter arithmetic
without calling ModelSpec.parameter_count(). Runtime checks then instantiate the
repository's TwelveSixDecoder and inspect actual tensors/geometry.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
from typing import Any

EXPECTED_STAGE = "MODEL-341-20M-CANDIDATE-A"
EXPECTED_PARAMETERS = 20_613_440
EXPECTED_MODEL_SHA256 = "fbff24d561a2818453554d58ca23fc6ace3303b078f1935a8576c4565bd92441"
EXPECTED_INIT_SHA256 = "86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5"
EXPECTED_MODEL_FIELDS = {
    "schema_version": 1,
    "vocab_size": 256,
    "max_seq_len": 1024,
    "d_model": 320,
    "n_layers": 16,
    "n_heads": 10,
    "n_kv_heads": 2,
    "head_dim": 32,
    "d_ff": 1080,
    "activation": "swiglu",
    "norm_kind": "rmsnorm",
    "norm_placement": "pre",
    "norm_eps": 1e-5,
    "position_embedding": "rope",
    "rope_theta": 10000.0,
    "rope_rotary_dim": 32,
    "attention_bias": False,
    "mlp_bias": False,
    "attention_dropout": 0.0,
    "final_norm": True,
    "tie_word_embeddings": True,
    "lm_head_bias": False,
}
EXPECTED_INIT_FIELDS = {
    "schema_version": 1,
    "family": "normal",
    "std": 0.02,
    "residual_branch_scale": "sqrt_2_layers",
}


def canonical_sha256(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def independent_parameter_breakdown(model: dict[str, Any]) -> dict[str, int]:
    d_model = int(model["d_model"])
    q_dim = int(model["n_heads"]) * int(model["head_dim"])
    kv_dim = int(model["n_kv_heads"]) * int(model["head_dim"])
    embedding = int(model["vocab_size"]) * d_model
    attn_weights = 2 * d_model * (q_dim + kv_dim)
    attn_biases = q_dim + 2 * kv_dim + d_model if model["attention_bias"] else 0
    mlp_weights = 3 * d_model * int(model["d_ff"])
    mlp_biases = 2 * int(model["d_ff"]) + d_model if model["mlp_bias"] else 0
    norms = 2 * d_model
    block = attn_weights + attn_biases + mlp_weights + mlp_biases + norms
    final_norm = d_model if model["final_norm"] else 0
    lm_head_extra = 0 if model["tie_word_embeddings"] else int(model["vocab_size"]) * d_model
    if model["lm_head_bias"]:
        lm_head_extra += int(model["vocab_size"])
    total = embedding + int(model["n_layers"]) * block + final_norm + lm_head_extra
    return {
        "q_dim": q_dim,
        "kv_dim": kv_dim,
        "token_embedding": embedding,
        "attention_per_layer": attn_weights + attn_biases,
        "mlp_per_layer": mlp_weights + mlp_biases,
        "norms_per_layer": norms,
        "block_per_layer": block,
        "blocks_total": int(model["n_layers"]) * block,
        "final_norm": final_norm,
        "lm_head_extra": lm_head_extra,
        "total": total,
    }


def validate_candidate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    model = payload.get("model")
    init = payload.get("init")
    if not isinstance(model, dict) or not isinstance(init, dict):
        raise TypeError("candidate must contain object-valued model and init")

    if payload.get("stage") != EXPECTED_STAGE:
        failures.append("stage_identity")
    if payload.get("canonical_base") != "random_init":
        failures.append("canonical_base_random_init")
    if payload.get("expected_parameters") != EXPECTED_PARAMETERS:
        failures.append("expected_parameter_identity")
    if model != EXPECTED_MODEL_FIELDS:
        failures.append("exact_model_fields")
    if init != EXPECTED_INIT_FIELDS:
        failures.append("exact_init_fields")

    model_sha = canonical_sha256(model)
    init_sha = canonical_sha256(init)
    if model_sha != EXPECTED_MODEL_SHA256 or payload.get("expected_model_identity_sha256") != model_sha:
        failures.append("model_identity_sha256")
    if init_sha != EXPECTED_INIT_SHA256 or payload.get("expected_init_identity_sha256") != init_sha:
        failures.append("init_identity_sha256")

    if int(model["n_heads"]) % int(model["n_kv_heads"]) != 0:
        failures.append("gqa_divisibility")
    if int(model["n_heads"]) * int(model["head_dim"]) != int(model["d_model"]):
        failures.append("q_projection_width")
    if int(model["rope_rotary_dim"]) > int(model["head_dim"]) or int(model["rope_rotary_dim"]) % 2:
        failures.append("rope_geometry")

    breakdown = independent_parameter_breakdown(model)
    if breakdown["total"] != EXPECTED_PARAMETERS:
        failures.append("independent_parameter_total")

    if failures:
        raise ValueError("candidate verification failed: " + ", ".join(sorted(set(failures))))
    return {"model_sha256": model_sha, "init_sha256": init_sha, "breakdown": breakdown}


def _state_fingerprint(model: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        cpu = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(cpu.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(cpu.shape)).encode("ascii") + b"\0")
        digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def runtime_checks(payload: dict[str, Any], seed: int) -> dict[str, Any]:
    import torch

    from twelve_six.model import InitSpec, ModelSpec, TwelveSixDecoder, count_trainable_parameters

    spec = ModelSpec.from_dict(payload["model"])
    init_spec = InitSpec.from_dict(payload["init"])

    def build_and_fingerprint(one_seed: int) -> tuple[str, dict[str, Any]]:
        torch.manual_seed(one_seed)
        decoder = TwelveSixDecoder(spec, init_spec)
        count = count_trainable_parameters(decoder)
        block0 = decoder.blocks[0]
        facts = {
            "parameter_count": count,
            "tied_embedding_object_identity": decoder.lm_head.weight is decoder.token_embedding.weight,
            "tied_embedding_storage_identity": decoder.lm_head.weight.data_ptr() == decoder.token_embedding.weight.data_ptr(),
            "q_projection_shape": list(block0.attn.q_proj.weight.shape),
            "k_projection_shape": list(block0.attn.k_proj.weight.shape),
            "v_projection_shape": list(block0.attn.v_proj.weight.shape),
            "out_projection_shape": list(block0.attn.out_proj.weight.shape),
            "rope_inv_freq_shape": list(block0.attn.rope.inv_freq.shape),
        }
        fingerprint = _state_fingerprint(decoder)
        del decoder
        gc.collect()
        return fingerprint, facts

    fp_a, facts = build_and_fingerprint(seed)
    fp_b, _ = build_and_fingerprint(seed)
    fp_c, _ = build_and_fingerprint(seed + 1)

    expected_shapes = {
        "q_projection_shape": [320, 320],
        "k_projection_shape": [64, 320],
        "v_projection_shape": [64, 320],
        "out_projection_shape": [320, 320],
        "rope_inv_freq_shape": [16],
    }
    failures = []
    if facts["parameter_count"] != EXPECTED_PARAMETERS:
        failures.append("instantiated_parameter_count")
    if not facts["tied_embedding_object_identity"] or not facts["tied_embedding_storage_identity"]:
        failures.append("tied_embedding_runtime_identity")
    for name, expected in expected_shapes.items():
        if facts[name] != expected:
            failures.append(name)
    if fp_a != fp_b:
        failures.append("same_seed_reproducibility")
    if fp_a == fp_c:
        failures.append("different_seed_negative_control")

    torch.manual_seed(seed)
    decoder = TwelveSixDecoder(spec, init_spec)
    oversized_rejected = False
    try:
        decoder(torch.zeros((1, spec.max_seq_len + 1), dtype=torch.long))
    except ValueError:
        oversized_rejected = True
    if not oversized_rejected:
        failures.append("max_seq_len_fail_closed")
    del decoder
    gc.collect()

    if failures:
        raise RuntimeError("runtime verification failed: " + ", ".join(sorted(set(failures))))
    return {
        **facts,
        "seed": seed,
        "same_seed_state_sha256": fp_a,
        "same_seed_reproduced": fp_a == fp_b,
        "different_seed_state_sha256": fp_c,
        "different_seed_changed": fp_a != fp_c,
        "oversized_context_rejected": oversized_rejected,
        "torch_version": torch.__version__,
        "checkpoint_loaded": False,
    }


def adversarial_self_checks(payload: dict[str, Any]) -> dict[str, bool]:
    mutations: dict[str, Any] = {
        "parameter_count_drift": ("expected_parameters", EXPECTED_PARAMETERS + 1),
        "model_hash_drift": ("expected_model_identity_sha256", "0" * 64),
        "canonical_base_drift": ("canonical_base", "pretrained"),
    }
    results: dict[str, bool] = {}
    for name, (key, value) in mutations.items():
        mutated = copy.deepcopy(payload)
        mutated[key] = value
        try:
            validate_candidate_payload(mutated)
        except ValueError:
            results[name] = True
        else:
            results[name] = False

    nested = {
        "tie_embedding_drift": ("tie_word_embeddings", False),
        "gqa_drift": ("n_kv_heads", 4),
        "context_drift": ("max_seq_len", 2048),
        "rope_drift": ("rope_rotary_dim", 30),
    }
    for name, (key, value) in nested.items():
        mutated = copy.deepcopy(payload)
        mutated["model"][key] = value
        try:
            validate_candidate_payload(mutated)
        except ValueError:
            results[name] = True
        else:
            results[name] = False
    if not all(results.values()):
        raise RuntimeError("adversarial self-check failed open")
    return results


def verify(config_path: Path, seed: int) -> dict[str, Any]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    static = validate_candidate_payload(payload)
    adversarial = adversarial_self_checks(payload)
    runtime = runtime_checks(payload, seed)
    return {
        "schema_version": 1,
        "worker": "SWARM-735",
        "candidate": EXPECTED_STAGE,
        "verdict": "PASS",
        "pass": True,
        "truth_boundary": {
            "local_free_cpu_only": True,
            "random_init_only": True,
            "training_performed": False,
            "checkpoint_loaded": False,
            "learned_quality_claimed": False,
        },
        "config_path": str(config_path),
        "static_identity": static,
        "runtime": runtime,
        "adversarial_fail_closed": adversarial,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/candidates/model341_20m_candidate_a.json"))
    parser.add_argument("--seed", type=int, default=341)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify(args.config, args.seed)
    rendered = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
