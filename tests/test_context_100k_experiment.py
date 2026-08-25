from __future__ import annotations

import json
from pathlib import Path

from twelve_six.context_100k_experiment import compare_context_conditions
from twelve_six.context_scaling import ContextPackingSpec, context_probe_spec
from twelve_six.model import load_stage_config
from twelve_six.packing import PACKING_CONFIG_HASH, packing_config_hash
from twelve_six.tokenization import ByteTokenizer


CONFIG = Path("configs/experiments/model17_context_100k.json")


def test_context_candidates_preserve_parameters_and_distinguish_identity() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    stage = load_stage_config(payload["stage_config"])
    spec128 = context_probe_spec(stage.model, max_seq_len=128)
    spec256 = context_probe_spec(stage.model, max_seq_len=256)
    assert spec128.parameter_count() == spec256.parameter_count() == 107_856
    assert spec128.identity_sha256() != spec256.identity_sha256()
    assert spec128.max_seq_len == 128
    assert spec256.max_seq_len == 256


def test_context_packing_identities_differ_without_touching_s0() -> None:
    tokenizer = ByteTokenizer()
    p128 = ContextPackingSpec(sequence_length=128)
    p256 = ContextPackingSpec(sequence_length=256)
    assert p128.identity_sha256(
        tokenizer_config_sha256=tokenizer.identity.config_sha256
    ) != p256.identity_sha256(
        tokenizer_config_sha256=tokenizer.identity.config_sha256
    )
    assert packing_config_hash() == PACKING_CONFIG_HASH


def test_comparison_fails_closed_on_initial_weight_drift() -> None:
    base = {
        "schema": "12-6.context-100k-experiment.v1",
        "source_sha": "a" * 40,
        "parameters": 107856,
        "initial_state_sha256": "b" * 64,
        "dataset_manifest_sha256": "c" * 64,
        "dataset_identity_sha256": "d" * 64,
        "context_length": 128,
        "training": {
            "optimized_tokens": 100,
            "optimizer_updates": 1,
            "seconds_per_optimized_token": 1.0,
        },
        "evaluation": {
            "final_native": {"bpb": 1.0},
            "final_common_128": {"bpb": 1.0},
        },
        "train_packing": {"causal_pair_utilization": 0.5},
    }
    other = json.loads(json.dumps(base))
    other["context_length"] = 256
    other["initial_state_sha256"] = "e" * 64
    try:
        compare_context_conditions(base, other)
    except RuntimeError as exc:
        assert "initial_state_sha256" in str(exc)
    else:
        raise AssertionError("comparison accepted different initial parameters")
