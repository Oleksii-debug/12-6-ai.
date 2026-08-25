from __future__ import annotations

import json
from pathlib import Path


MANIFEST = Path("configs/runs/compute32_eur10k_campaign.experimental.json")


def _manifest() -> dict[str, object]:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_campaign_manifest_never_embeds_compute_authorization() -> None:
    manifest = _manifest()
    assert manifest["status"] == "PLANNING_ONLY_NOT_COMPUTE_AUTHORIZATION"
    assert manifest["paid_execution"] == "FORBIDDEN_UNLESS_EXTERNAL_COMPUTE_AUTHORIZED"

    main_run = manifest["main_run"]
    assert isinstance(main_run, dict)
    cost_gate = main_run["cost_gate"]
    assert isinstance(cost_gate, dict)
    assert "authorization" not in cost_gate
    assert cost_gate["source_sha"] == "NOT_FROZEN"
    assert cost_gate["tokenizer_status"] == "NOT_FROZEN"
    assert cost_gate["corpus_status"] == "NOT_FROZEN"


def test_main_token_budget_is_exactly_twenty_thousand_global_updates() -> None:
    manifest = _manifest()
    main_run = manifest["main_run"]
    assert isinstance(main_run, dict)

    assert main_run["parameters"] == 999_106_560
    assert main_run["optimizer_steps"] == 20_000
    assert main_run["global_batch_tokens"] == 1_048_576
    assert main_run["target_tokens"] == (
        main_run["optimizer_steps"] * main_run["global_batch_tokens"]
    )


def test_qualification_uses_the_same_s6_shape_and_exact_batch_algebra() -> None:
    manifest = _manifest()
    qualification = manifest["qualification_experiment"]
    main_run = manifest["main_run"]
    assert isinstance(qualification, dict)
    assert isinstance(main_run, dict)

    assert qualification["parameters"] == main_run["parameters"]
    assert qualification["architecture_identity"] == main_run["architecture_identity"]
    assert qualification["context"] == main_run["context"]
    assert qualification["precision"] == main_run["precision"]
    assert qualification["target_tokens"] == (
        qualification["optimizer_steps"] * qualification["global_batch_tokens"]
    )
    assert qualification["forced_checkpoint_after_step"] < qualification["optimizer_steps"]


def test_single_gpu_primary_plan_does_not_depend_on_unproven_distributed_runtime() -> None:
    manifest = _manifest()
    main_run = manifest["main_run"]
    assert isinstance(main_run, dict)
    parallelism = main_run["parallelism"]
    assert isinstance(parallelism, dict)

    assert main_run["gpu_count"] == 1
    assert parallelism["data_parallel"] == 1
    assert parallelism["tensor_parallel"] == 1
    assert parallelism["pipeline_parallel"] == 1
    assert parallelism["context_parallel"] == 1
    assert parallelism["expert_parallel"] == 1
