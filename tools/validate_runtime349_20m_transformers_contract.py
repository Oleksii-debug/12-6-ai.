from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence/runtime349/20m_transformers_contract_v1.json"

EXPECTED_WORKER = "RUNTIME-349-20M-TRANSFORMERS-CONTRACT"
EXPECTED_STATUS = "BLOCKED_NO_PUBLISHED_PRIMARY_20M_MODELSPEC"
EXPECTED_BASE_SHA = "0eb3c017a778eab30fd44ec23b84785ea5866e9d"
EXPECTED_PATHS = {
    "hf_exporter": "src/twelve_six/checkpoint/hf_export.py",
    "interop": "src/twelve_six/inference/transformers_llama.py",
    "standard_llama_materializer": "src/twelve_six/inference/llama_runtime_export.py",
    "runtime": "src/twelve_six/inference/transformers_llama_runtime.py",
}
SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA64 = re.compile(r"^[0-9a-f]{64}$")


def _canonical_sha256(payload: dict[str, Any]) -> str:
    stripped = dict(payload)
    stripped.pop("evidence_sha256", None)
    encoded = json.dumps(
        stripped,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate(payload: dict[str, Any]) -> None:
    assert payload["schema"] == "12-6.runtime349.20m-transformers-contract.v1"
    assert payload["worker_id"] == EXPECTED_WORKER
    assert payload["status"] == EXPECTED_STATUS
    assert payload["execution_policy"] == "LOCAL_FREE"

    repo = payload["repository"]
    assert repo["resolved_name"] == "Oleksii-debug/12-6-ai."
    assert repo["base_branch"] == "runtime225/transformers-learned-10m-v2-20260826"
    assert SHA40.fullmatch(repo["base_sha"])
    assert repo["base_sha"] == EXPECTED_BASE_SHA

    path = payload["maintained_standard_llama_path"]
    for key, value in EXPECTED_PATHS.items():
        assert path[key] == value
    assert path["target_architecture"] == "LlamaForCausalLM"
    assert path["transformers_version"] == "5.15.1"
    assert path["second_exporter_added"] is False

    primary = payload["primary_20m"]
    assert primary["modelspec_status"] == "MISSING"
    assert primary["modelspec_identity_sha256"] is None
    assert primary["parameter_count"] is None
    assert primary["learned_checkpoint_required_for_mechanics"] is False
    assert primary["mechanics_weight_policy"] == "RANDOM_INIT_AFTER_EXACT_MODELSPEC_EXISTS"

    discovery = payload["live_discovery"]
    assert discovery["research339_branch_found"] is False
    assert discovery["research339_pr_found"] is False
    assert discovery["model341_branch_found"] is False
    assert discovery["model341_pr_found"] is False
    assert discovery["latest_observed_pr_number"] == 421

    gate = payload["llama_exact_representability_gate"]
    assert gate == {
        "schema_version": 1,
        "activation": "swiglu",
        "norm_kind": "rmsnorm",
        "norm_placement": "pre",
        "position_embedding": "rope",
        "full_rope_required": "rope_rotary_dim == head_dim",
        "hidden_head_geometry_required": "n_heads * head_dim == d_model",
        "attention_bias": False,
        "mlp_bias": False,
        "lm_head_bias": False,
        "final_norm": True,
        "gqa_supported": "n_heads % n_kv_heads == 0",
        "rope_basis_conversion": "PAIRWISE_INTERLEAVED_TO_LLAMA_HALF_SPLIT",
        "unsupported_semantics": "FAIL_CLOSED",
    }

    incumbent = payload["incumbent_mechanics_evidence"]
    assert incumbent["strict_state_dict_load"] is True
    assert incumbent["complete_logits_comparison"] is True
    assert incumbent["atol"] == 1e-5
    assert incumbent["rtol"] == 1e-5
    assert incumbent["greedy_argmax_exact"] is True

    required = payload["required_primary_20m_parity"]
    assert required["status"] == "NOT_RUN"
    assert required["complete_logits_required"] is True
    assert required["strict_state_dict_load_required"] is True
    assert required["rope_conversion_layerwise_required"] is True
    assert required["context_boundary_required"] is True
    assert required["over_context_rejection_required"] is True
    assert required["random_init_allowed_if_no_learned_20m"] is True
    assert required["foreign_pretrained_weights_allowed"] is False

    verdict = payload["verdict"]
    assert verdict["exactly_representable"] is None
    assert verdict["complete_logits_parity"] is None
    assert verdict["decision"] == EXPECTED_STATUS

    evidence_sha = payload["evidence_sha256"]
    assert SHA64.fullmatch(evidence_sha)
    assert _canonical_sha256(payload) == evidence_sha


def main() -> None:
    with EVIDENCE.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    validate(payload)
    print(f"RUNTIME349_STATUS={payload['status']}")
    print(f"RUNTIME349_EVIDENCE_SHA={payload['evidence_sha256']}")


if __name__ == "__main__":
    main()
