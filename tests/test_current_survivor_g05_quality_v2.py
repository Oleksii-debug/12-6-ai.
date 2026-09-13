from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools import run_current_survivor_g05_quality_v2 as runner


def test_canonical_checkout_assets_are_the_pinned_trust_roots() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    bindings = runner._verify_checkout_assets(repo_root)

    assert bindings == {
        "g05_module_git_blob_sha1": runner.CANONICAL_G05_MODULE_BLOB_SHA1,
        "survivor_evidence_git_blob_sha1": runner.CANONICAL_SURVIVOR_EVIDENCE_BLOB_SHA1,
    }
    evidence = runner._validate_survivor_evidence(
        repo_root / runner.CANONICAL_SURVIVOR_EVIDENCE_PATH
    )
    assert evidence["evidence_identity_sha256"] == runner.CANONICAL_SURVIVOR_EVIDENCE_IDENTITY
    assert evidence["result"]["record_payload_jsonl_sha256"] == (
        runner.CANONICAL_SURVIVOR_JSONL_SHA256
    )


def test_zero_credit_int_fields_reject_bool_aliases() -> None:
    with pytest.raises(ValueError, match="exact integer"):
        runner._require_exact_int(False, "authorized_optimized_target_exposure")
    with pytest.raises(ValueError, match="exact integer"):
        runner._require_exact_int(True, "optimizer_updates_executed")


def test_tampered_survivor_authority_fails_even_if_json_is_well_formed(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source = repo_root / runner.CANONICAL_SURVIVOR_EVIDENCE_PATH
    value = json.loads(source.read_text(encoding="utf-8"))
    tampered = copy.deepcopy(value)
    tampered["result"]["record_payload_jsonl_sha256"] = "a" * 64
    target = tmp_path / "tampered.json"
    target.write_text(json.dumps(tampered, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="Git blob drift"):
        runner._validate_survivor_evidence(target)


def test_complete_g05_row_identity_is_order_independent_and_payload_bound() -> None:
    rows = [
        {
            "record_id": "b",
            "source_id": "source-b",
            "family": "fixture",
            "modality": "en",
            "normalized_payload": "second payload",
        },
        {
            "record_id": "a",
            "source_id": "source-a",
            "family": "fixture",
            "modality": "uk",
            "normalized_payload": "перший текст",
        },
    ]
    expected = runner._g05_row_root(rows)
    assert runner._g05_row_root(list(reversed(rows))) == expected

    substituted = copy.deepcopy(rows)
    substituted[0]["normalized_payload"] += "!"
    assert runner._g05_row_root(substituted) != expected


def test_runner_exposes_no_caller_authority_head_or_manifest_switches() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    forbidden = (
        "--g05-product-head",
        "--survivor-product-head",
        "--execution-head-sha",
        "--input-manifest-sha256",
        "--expected-survivor-jsonl-sha256",
        "--expected-survivor-inventory-root",
        "--expected-survivor-payload-root",
    )
    for switch in forbidden:
        assert switch not in source
