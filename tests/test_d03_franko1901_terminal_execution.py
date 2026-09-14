from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = (
    ROOT / "configs" / "data" / "d03_franko1901_terminal_execution_v1.json"
)


def git_blob_sha1(path: Path) -> str:
    payload = path.read_bytes()
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def evidence() -> dict[str, object]:
    return json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))


def test_terminal_execution_evidence_is_exact_and_zero_credit() -> None:
    value = evidence()
    assert value["schema_version"] == "12-6.d03-franko1901-terminal-execution.v1"
    assert value["decision"] == "CANDIDATE_MATERIALIZED_ZERO_CREDIT"

    execution = value["execution"]
    assert execution == {
        "evidence_pr": 855,
        "execution_head_sha": "181a54e6f7527ed66e59f72d42290a4e0013330f",
        "workflow_run_id": 34479584975,
        "job_id": 102878704552,
        "execution_profile": "LOCAL_FREE",
        "python_runtime": "CPython 3.11.16",
        "focused_tests_passed": 16,
        "independent_materializations": 2,
        "byte_identical_materializations": True,
        "artifact_id": 10152982171,
        "artifact_retention_days": 30,
        "artifact_zip_sha256": (
            "2c09571dcfc633f52abf9038b3087c870ce0f80801b1e5dcc049b6c20ab37519"
        ),
    }

    source = value["source"]
    assert source["source_git_blob_sha1"] == "45f33ac620907e1d1ed727524975b3a0fd1a0994"
    assert source["source_sha256"] == (
        "10b0db2e6451b49d63f931a607a6b27fb10c11288e65802dcd1e6793a7a2586d"
    )
    assert source["license_git_blob_sha1"] == "0e259d42c996742e9e3cba14c677129b2c1b6311"
    assert source["license_sha256"] == (
        "a2010f343487d3f7618affe54f789f5487602331c0a8d03f49e9a7c547cf0499"
    )

    result = value["materialization"]
    assert result["rows_seen"] == 30936
    assert result["accepted_rows"] == 30660
    assert result["rejected_rows"] == 276
    assert result["accepted_text_utf8_bytes"] == 1762005
    assert result["accepted_payload_jsonl_bytes"] == 10685784
    assert result["accepted_payload_jsonl_sha256"] == (
        "9b00da2a2a5110cc4e384711fffbbbdaa917904462ddd133e7178e2d07dcb5ec"
    )
    assert result["record_inventory_identity_sha256"] == (
        "bdedac35f8d3ef1ba6faca6f79f002d1a67da1b13b5fade2a341c526d830a4c7"
    )
    assert result["report_identity_sha256"] == (
        "fc7efd35d30e739619503d946112cf4b684f1394d4b07bf4d3cc287f62a01e1b"
    )
    assert result["rejected_by_reason"] == {
        "empty": 30,
        "exact_duplicate": 244,
        "low_cyrillic_share": 2,
    }

    truth = value["truth_boundary"]
    assert truth["candidate_only"] is True
    assert truth["canonical_capacity_credit_bytes"] == 0
    assert truth["family_credit_authorized"] is False
    assert truth["corpus_admitted"] is False
    assert truth["global_dedup"] == "NOT_RUN"
    assert truth["evaluation_decontamination"] == "NOT_RUN"
    assert truth["post_composition_quality_privacy"] == "NOT_RUN"
    assert truth["balance_family_caps"] == "NOT_RUN_FOR_THIS_ADDITION"
    assert truth["tokenizer_fit_authorized"] is False
    assert truth["training_authorized_bytes"] == 0
    assert truth["authorized_unique_loss_positions"] == 0
    assert truth["model_training_executed"] is False
    assert truth["optimizer_updates"] == 0
    assert truth["final_test_accessed"] is False
    assert truth["paid_compute_used"] is False
    assert truth["learned_20m_promoted"] is False


def test_terminal_execution_evidence_is_scoped_to_old_exact_product_blobs() -> None:
    identity = evidence()["executed_product_identity"]
    expected_old = {
        "configs/data/d03_franko1901_exact_materialization_v1.json": (
            "90d6f1f56ad572bbd81f1e3cf87098e2304f7784"
        ),
        "tools/materialize_d03_franko1901.py": (
            "0817f1ca353b22b8de1cbf3049a4aa19c6c2dd46"
        ),
        "tests/test_d03_franko1901_materialization.py": (
            "7916ff6ccf9c7609041b20c570c8f916f5eb12a3"
        ),
    }
    assert identity["config_path"] in expected_old
    assert identity["materializer_path"] in expected_old
    assert identity["focused_test_path"] in expected_old
    assert identity["temporary_evidence_workflow_git_blob_sha1"] == (
        "bd09988b7789db6fdca07f63a41710e5b30dfa2f"
    )

    assert identity["config_git_blob_sha1"] == expected_old[identity["config_path"]]
    assert (
        identity["materializer_git_blob_sha1"]
        == expected_old[identity["materializer_path"]]
    )
    assert (
        identity["focused_test_git_blob_sha1"]
        == expected_old[identity["focused_test_path"]]
    )

    # AUDIT1025 repair changes Product validation semantics. The historical
    # execution remains valid only for the exact old blobs above and must not
    # silently certify the repaired materializer/tests.
    assert git_blob_sha1(ROOT / identity["materializer_path"]) != (
        identity["materializer_git_blob_sha1"]
    )
    assert git_blob_sha1(ROOT / identity["focused_test_path"]) != (
        identity["focused_test_git_blob_sha1"]
    )
