from __future__ import annotations

import json
from pathlib import Path

from twelve_six.data.privacy_filter import (
    assert_no_secret_values_in_manifest,
    scan_record,
)
from twelve_six.data.privacy_reporting import build_corpus_scan_summary


def test_compact_corpus_summary_binds_identity_without_record_text() -> None:
    rows = [
        scan_record(
            record_id="natural-1",
            source_id="source:natural",
            source_version="v1",
            modality="natural",
            text="No sensitive material appears in this ordinary project-authored sentence.",
        ),
        scan_record(
            record_id="code-1",
            source_id="source:code",
            source_version="v1",
            modality="code",
            text="def safe_fixture():\n    return 'ordinary value'\n",
        ),
    ]
    summary = build_corpus_scan_summary(rows, corpus_identity_sha256="b" * 64)
    assert summary["records_total"] == 2
    assert summary["by_modality"]["natural"]["ALLOW"] == 1
    assert summary["by_modality"]["code"]["ALLOW"] == 1
    assert len(summary["records_evidence_sha256"]) == 64
    assert "records" not in summary
    assert_no_secret_values_in_manifest(summary)


def test_retained_corpus_v01_scan_is_bound_to_current_corpus_manifest() -> None:
    corpus = json.loads(Path("data/corpus/v0.1/manifest.json").read_text(encoding="utf-8"))
    privacy = json.loads(
        Path("reports/data33/pii_secrets_scan_corpus_v01_20260825.json").read_text(
            encoding="utf-8"
        )
    )
    assert privacy["corpus_identity_sha256"] == corpus["corpus_identity_sha256"]
    assert privacy["records_total"] == sum(
        item["documents"] for item in corpus["by_modality"].values()
    )
    assert privacy["by_modality"]["natural"]["ALLOW"] == corpus["by_modality"]["natural"][
        "documents"
    ]
    assert privacy["by_modality"]["code"]["ALLOW"] == corpus["by_modality"]["code"][
        "documents"
    ]
    for source_id, stats in corpus["by_source"].items():
        assert privacy["by_source"][source_id]["ALLOW"] == stats["documents"]
    assert privacy["records_train_eligible_after_privacy"] == privacy["records_total"]
    assert_no_secret_values_in_manifest(privacy)
