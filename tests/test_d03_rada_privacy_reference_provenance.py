from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/materialize_d03_rada_trees_quality_windows_authoritative.py"
SPEC = importlib.util.spec_from_file_location("d03_rada_privacy_provenance", TOOL)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_privacy_reference_is_explicit_feature_branch_sync_not_main_merge() -> None:
    assert (
        MODULE.PRIVACY_REPAIR_FEATURE_BRANCH_SYNC_MERGE_SHA
        == "c5b9d0922cd63ade1a90c7f0163325dcf35d0b41"
    )
    assert not hasattr(MODULE, "PRIVACY_REPAIR_REFERENCE_MERGE_SHA")

    source = TOOL.read_text(encoding="utf-8")
    assert '"privacy_repair_reference_merge_sha"' not in source
    assert (
        '"privacy_repair_feature_branch_sync_merge_sha": '
        "PRIVACY_REPAIR_FEATURE_BRANCH_SYNC_MERGE_SHA"
    ) in source


def test_open_branch_report_does_not_claim_canonical_privacy_integration(
    monkeypatch, tmp_path: Path
) -> None:
    candidate_sha = "a" * 64
    upstream_sha = "b" * 64
    privacy_blob = MODULE.EXPECTED_PRIVACY_IMPLEMENTATION_GIT_BLOB_SHA

    monkeypatch.setattr(
        MODULE,
        "_load_bound_mechanics",
        lambda: ((object(), object(), object(), object()), privacy_blob),
    )
    monkeypatch.setattr(
        MODULE,
        "_validate_upstream_report",
        lambda _path, _expected: (upstream_sha, candidate_sha),
    )
    monkeypatch.setattr(
        MODULE,
        "_candidate_identity",
        lambda _path: (candidate_sha, MODULE.EXPECTED_RECORDS, MODULE.EXPECTED_SOURCE_BYTES),
    )

    def fake_materialize(
        _candidate: Path,
        output: Path,
        _mechanics_report: Path,
        *,
        expected_candidate_sha256: str,
    ) -> dict[str, object]:
        assert expected_candidate_sha256 == candidate_sha
        output.write_bytes(b"fixture")
        core: dict[str, object] = {
            "input_candidate_jsonl_sha256": candidate_sha,
            "claim_boundary": {
                "candidate_jsonl_is_canonical_corpus": False,
                "training_authorized_bytes": 0,
                "optimizer_updates": 0,
                "model_training_executed": False,
            },
        }
        return {
            **core,
            "report_sha256": MODULE.base.sha256_bytes(MODULE.base.canonical_bytes(core)),
        }

    monkeypatch.setattr(MODULE.base, "materialize", fake_materialize)

    report = MODULE.materialize_authoritative(
        tmp_path / "candidate.jsonl",
        tmp_path / "upstream.json",
        tmp_path / "output.jsonl",
        tmp_path / "report.json",
        expected_upstream_report_sha256="c" * 64,
    )

    boundary = report["claim_boundary"]
    assert isinstance(boundary, dict)
    assert boundary["privacy_filter_v3_exact_implementation_bound"] is True
    assert boundary["canonical_privacy_repair_bound"] is False
    assert report["authority_binding"]["privacy_filter_v3_git_blob_sha"] == privacy_blob
    assert (
        report["authority_binding"]["privacy_repair_feature_branch_sync_merge_sha"]
        == MODULE.PRIVACY_REPAIR_FEATURE_BRANCH_SYNC_MERGE_SHA
    )
