from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = ROOT / "evidence/d03/kmu_timeline_v2_terminal.json"

EXPECTED_EXECUTION_HEAD = "0464618d0099e9792c34128f0727b14ad77b293f"
EXPECTED_ARTIFACT_DIGEST = (
    "sha256:b7bd7f0f78d5e829c2e9931a154ec6deaae638bdd2e1f945d25bf8c7d2998f75"
)
EXPECTED_REPORT_SHA256 = "7e2c71d31524254642f39b204133283bbe248e08a44641ed32af3c5fa57ccef0"
EXPECTED_PROBE_IDENTITY = "4246345a4b9ebed8df08ce3d5915e66ef4d54c1dc1e3754d107621bc56204993"
EXPECTED_DISCOVERY_IDENTITY = (
    "866b991fa22c90ef089217a6b797e09922765f5875448bbdcba58c2cba59d8fc"
)
EXPECTED_SNAPSHOTS = [
    (
        "https://www.kmu.gov.ua/timeline?category_id=3&type=posts",
        248_990,
        "7df8b2c505020fae5c85d420604bde1dbef5cccc01a9f41068cfedbd9002eebf",
    ),
    (
        "https://www.kmu.gov.ua/timeline?category_id=3&type=posts&page=2",
        249_001,
        "48755c6fcdfae622bf33e9307b79c30b5f6c1e9fc47092f5af101868fd4a99dc",
    ),
    (
        "https://www.kmu.gov.ua/timeline?category_id=3&type=posts&page=3",
        249_001,
        "0d8df0d2a336a6de72a84289442dc4ef8f491fffaad31d1f17fe32d34d3f6c12",
    ),
    (
        "https://www.kmu.gov.ua/timeline?category_id=3&type=posts&page=4",
        249_001,
        "5e14bf3382bea290f7abc41cd1e64f7082f087d0744818023e08ff14c5f5a792",
    ),
    (
        "https://www.kmu.gov.ua/timeline?category_id=3&type=posts&page=5",
        249_001,
        "783c8d3e286b87baffbd2819eb4f7f2baf763c694f3e39cee9c72d0f4050168c",
    ),
]


def _load() -> dict[str, object]:
    return json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))


def test_terminal_evidence_binds_exact_five_page_network_execution() -> None:
    evidence = _load()
    assert evidence["execution_head_sha"] == EXPECTED_EXECUTION_HEAD
    assert evidence["execution_head_sha"] != "f3d4b969e84e134b6cf8be5ec42168449456ec42"
    assert evidence["workflow_run_id"] == 34159249597
    assert evidence["workflow_job_id"] == 101857434610
    assert evidence["workflow_conclusion"] == "success"
    artifact = evidence["artifact"]
    assert artifact["id"] == 10032495499
    assert artifact["digest"] == EXPECTED_ARTIFACT_DIGEST
    assert artifact["retained_report_bytes"] == 2654
    assert artifact["retained_report_sha256"] == EXPECTED_REPORT_SHA256
    assert artifact["retained_probe_identity_sha256"] == EXPECTED_PROBE_IDENTITY
    assert artifact["retained_discovery_identity_sha256"] == EXPECTED_DISCOVERY_IDENTITY


def test_live_timeline_terminal_claim_binds_all_five_explicit_pages() -> None:
    evidence = _load()
    discovery = evidence["discovery"]
    assert discovery["status"] == "BLOCKED_CLIENT_RENDERED_TIMELINE_NO_ARTICLE_URLS"
    assert discovery["client_rendered_shell_detected"] is True
    assert discovery["pages_visited"] == 5
    assert discovery["news_urls"] == 0
    snapshots = discovery["snapshots"]
    assert [
        (row["url"], row["bytes"], row["sha256"])
        for row in snapshots
    ] == EXPECTED_SNAPSHOTS
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
    assert evidence["discovery"]["pages_visited"] == 5
    assert terminal["same_http_timeline_probe_is_terminal"] is True
    assert terminal["repeat_without_discovery_mechanism_change_required"] is False
    assert terminal["capacity_credit_added"] == 0
    assert terminal["successor_requires_new_server_or_api_discovery_evidence"] is True
