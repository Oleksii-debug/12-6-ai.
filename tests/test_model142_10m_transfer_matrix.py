from __future__ import annotations

import copy
import json
from pathlib import Path

from twelve_six.architecture_transfer_10m import (
    INCUMBENT_MODEL_SHA,
    load_experiment_config,
    summarize_matrix,
    validate_summary,
)
from twelve_six.model import ModelSpec


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/model142_10m_transfer_matrix.v1.json"


def test_model142_candidate_family_is_narrow_and_parameter_matched() -> None:
    config = load_experiment_config(CONFIG)
    assert [candidate["id"] for candidate in config["candidates"]] == [
        "incumbent_gqa_8q2kv",
        "transfer_gqa_8q4kv",
        "mha_8q8kv",
    ]
    specs = {
        candidate["id"]: ModelSpec.from_dict(candidate["model"])
        for candidate in config["candidates"]
    }
    assert specs["incumbent_gqa_8q2kv"].identity_sha256() == INCUMBENT_MODEL_SHA
    assert specs["incumbent_gqa_8q2kv"].parameter_count() == 10_000_640
    assert specs["transfer_gqa_8q4kv"].parameter_count() == 9_997_568
    assert specs["mha_8q8kv"].parameter_count() == 10_000_640
    assert abs(9_997_568 - 10_000_640) / 10_000_640 < 0.001
    assert {spec.d_model for spec in specs.values()} == {256}
    assert {spec.n_layers for spec in specs.values()} == {12}
    assert {spec.n_heads for spec in specs.values()} == {8}
    assert {spec.head_dim for spec in specs.values()} == {32}
    assert [spec.n_kv_heads for spec in specs.values()] == [2, 4, 8]


def test_model142_does_not_reopen_unsupported_dimensions() -> None:
    config = load_experiment_config(CONFIG)
    intake = config["evidence_intake"]
    assert intake["gqa_mha"]["status"] == "TRANSFER_AXIS"
    assert intake["depth_width"]["status"] == "NOT_TRANSFERRED"
    assert intake["ffn_ratio"]["status"] == "NOT_TRANSFERRED_AS_WINNER"
    assert intake["head_count"]["status"] == "NOT_TRANSFERRED_AS_QUALITY_WINNER"
    assert intake["tokenizer_allocation"]["status"] == "HELD_FIXED"
    assert intake["initialization"]["status"] == "HELD_FIXED"
    controls = config["controls"]
    assert controls["dataset_id"] == "s0-tiny-controlled-v1"
    assert controls["tokenizer"] == "s0-byte-v1"
    assert controls["optimized_causal_tokens"] == 8_064
    assert controls["seeds"] == [1515, 1516, 1517]


def _fake_run(candidate: str, seed: int, bpb: float, *, kv_bytes: int) -> dict:
    parameters = {
        "incumbent_gqa_8q2kv": 10_000_640,
        "transfer_gqa_8q4kv": 9_997_568,
        "mha_8q8kv": 10_000_640,
    }[candidate]
    return {
        "candidate": candidate,
        "seed": seed,
        "model": {"parameters": parameters},
        "kv_cache": {"bf16_bytes": kv_bytes},
        "metrics": {
            "final_heldout_bpb": bpb,
            "final_train_bpb": bpb - 0.2,
            "optimized_tokens_per_s": 1000.0,
            "clip_rate": 0.5,
            "global_gradient_health": {"grad_nonzero_fraction": 1.0},
            "global_update": {"changed_fraction": 1.0},
        },
    }


def test_model142_summary_supports_transfer_only_on_all_paired_wins() -> None:
    config = load_experiment_config(CONFIG)
    runs: list[dict] = []
    for seed, incumbent, transfer, mha in (
        (1515, 4.60, 4.50, 4.58),
        (1516, 4.65, 4.49, 4.60),
        (1517, 4.62, 4.48, 4.59),
    ):
        runs.append(_fake_run("incumbent_gqa_8q2kv", seed, incumbent, kv_bytes=3 * 1024 * 1024))
        runs.append(_fake_run("transfer_gqa_8q4kv", seed, transfer, kv_bytes=6 * 1024 * 1024))
        runs.append(_fake_run("mha_8q8kv", seed, mha, kv_bytes=12 * 1024 * 1024))
    summary = summarize_matrix(config, runs)
    validate_summary(summary)
    assert summary["decision"] == "TRANSFER_SUPPORTED_BOUNDED"
    assert all(
        delta < 0
        for delta in summary["paired_deltas_bpb"][
            "transfer_gqa_8q4kv_minus_incumbent_gqa_8q2kv"
        ]
    )

    rejected = copy.deepcopy(runs)
    for run in rejected:
        if run["candidate"] == "transfer_gqa_8q4kv" and run["seed"] == 1516:
            run["metrics"]["final_heldout_bpb"] = 4.70
    fallback = summarize_matrix(config, rejected)
    validate_summary(fallback)
    assert fallback["decision"] == "KEEP_10M_INCUMBENT"


def test_model142_config_is_not_a_freeze_or_paid_compute_authority() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    truth = payload["truth_boundary"]
    assert truth["architecture_freeze"] is False
    assert truth["stage_promotion"] is False
    assert truth["representative_corpus_quality_claim"] is False
    assert truth["paid_compute"] is False
    assert truth["foreign_pretrained_weights"] is False
