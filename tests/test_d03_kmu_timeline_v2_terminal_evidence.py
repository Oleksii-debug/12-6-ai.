from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = ROOT / "evidence/d03/kmu_timeline_v2_terminal.json"

EXPECTED_EXECUTION_HEAD = "f3d4b969e84e134b6cf8be5ec42168449456ec42"
EXPECTED_ARTIFACT_DIGEST = (
    "sha256:358e24ec247d31c92f35401c8da0a89293095611e23ba90e88ca29ac7afaff3e"
)
EXPECTED_PROBE_IDENTITY = "f6b695b5a6f86b3a736016a2d643ac6c197e3f35ce37b399d44ff6584f76237d"


def _load() -> dict[str, object]:
    return json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))


def test_terminal_evidence_binds_exact_network_execution() -> None:
    evidence = _load()
    assert evidence["execution_head_sha"] == EXPECTED_EXECUTION_HEAD
    assert evidence["workflow_run_id"] == 34156356548
    assert evidence["workflow_conclusion"] == "success"
    artifact = evidence["artifact"]
    assert artifact["id"] == 10031215186
    assert artifact["digest"] == EXPECTED_ARTIFACT_DIGEST
    assert artifact["retained_report_bytes"] == 1818
    assert artifact["retained_probe_identity_sha256"] == EXPECTED_PROBE_IDENTITY


def test_live_timeline_is_terminally_discovery_blocked() -> None:
    evidence = _load()
    discovery = evidence["discovery"]
    assert discovery["status"] == "BLOCKED_CLIENT_RENDERED_TIMELINE_NO_ARTICLE_URLS"
    assert discovery["client_rendered_shell_detected"] is True
    assert discovery["pages_visited"] == 1
    assert discovery["news_urls"] == 0
    assert discovery["snapshot_bytes"] == 248_990
    assert discovery["snapshot_sha256"] == (
        "8c1e44c8bb3dc86e029aeb9fe7270dff8be8d015948089e42e96d3431117d478"
    )
    assert evidence["verdict"] == "PROBE_DISCOVERY_BLOCKED_NO_ARTICLE_URLS"


def test_terminal_result_grants_no_capacity_or_training_credit() -> None:
    evidence = _load()
    sample = evidence["sample"]
    assert sample["attempted_pages"] == 0
    assert sample["eligible_pages"] == 0
    assert sample["observed_unique_probe_bytes"] == 0
    boundary = evidence["claim_boundary"]
    assert boundary["canonical_registry_mutated"] is False
    assert boundary["family_count_credit_added"] == 0
    assert boundary["training_authorized_bytes"] == 0
    assert boundary["tokenizer_fit_executed"] is False
    assert boundary["optimizer_updates"] == 0
    assert boundary["final_test_accessed"] is False
    assert boundary["paid_compute_used"] is False


def test_same_http_probe_is_not_repeated_without_new_discovery_mechanism() -> None:
    evidence = _load()
    terminal = evidence["terminal_interpretation"]
    assert terminal["same_http_timeline_probe_is_terminal"] is True
    assert terminal["repeat_without_discovery_mechanism_change_required"] is False
    assert terminal["capacity_credit_added"] == 0
    assert terminal["successor_requires_new_server_or_api_discovery_evidence"] is True
