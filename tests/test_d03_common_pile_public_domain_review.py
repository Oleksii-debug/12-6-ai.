from __future__ import annotations

import copy
import gzip
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
SPEC = importlib.util.spec_from_file_location(
    "pdr",
    TOOLS / "materialize_d03_common_pile_public_domain_review.py",
)
assert SPEC and SPEC.loader
pdr = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pdr
SPEC.loader.exec_module(pdr)


def _base_config() -> dict:
    return {
        "schema_version": pdr.CONFIG_SCHEMA,
        "worker": pdr.EXPECTED_WORKER,
        "base_main_sha": pdr.EXPECTED_BASE_MAIN_SHA,
        "execution_profile": "LOCAL_FREE",
        "common_pile_rights_authority": {
            "merge_commit_sha": pdr.EXPECTED_RIGHTS_MERGE_SHA,
            "registry_id": pdr.EXPECTED_RIGHTS_REGISTRY_ID,
            "registry_identity_sha256": pdr.EXPECTED_RIGHTS_REGISTRY_SHA256,
            "source_key": "public_domain_review",
            "project_review_status": "REVIEW_REQUIRED",
        },
        "source": {
            "dataset": "common-pile/public_domain_review",
            "revision": pdr.EXPECTED_SOURCE_REVISION,
            "family_id": pdr.EXPECTED_FAMILY_ID,
            "source_value": pdr.EXPECTED_SOURCE_VALUE,
            "origin_host": pdr.EXPECTED_ORIGIN_HOST,
            "expected_record_license": pdr.EXPECTED_RECORD_LICENSE,
            "attribution_required": True,
        },
        "files": [],
        "selection": dict(pdr.EXPECTED_SELECTION),
        "truth_boundary": {
            "candidate_snapshot_only": True,
            **pdr.REQUIRED_ZERO_FIELDS,
        },
    }


def _record(record_id: str, text: str, **overrides: object) -> dict:
    row = {
        "id": record_id,
        "text": text,
        "source": "public-domain-review",
        "metadata": {
            "license": (
                "Creative Commons - Attribution Share-Alike - "
                "https://creativecommons.org/licenses/by-sa/4.0/"
            ),
            "url": f"https://publicdomainreview.org/essay/{record_id}/",
        },
        "author": "Excluded Author",
        "date": "2024-01-01",
    }
    row.update(overrides)
    return row


def _payload(rows: list[dict]) -> bytes:
    body = b"".join(
        json.dumps(row, ensure_ascii=False).encode("utf-8") + b"\n"
        for row in rows
    )
    return gzip.compress(body, mtime=0)


def _install_file(
    cfg: dict,
    tmp_path: Path,
    rows: list[dict],
    *,
    path: str = "v0/00000_essays.jsonl.gz",
) -> None:
    payload = _payload(rows)
    target = tmp_path / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    spec = {
        "path": path,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "max_compressed_bytes": len(payload) + 10,
        "max_uncompressed_bytes": 100_000,
    }
    cfg["files"] = [spec]
    pdr.EXPECTED_FILES = {
        path: {
            "sha256": spec["sha256"],
            "max_compressed_bytes": spec["max_compressed_bytes"],
            "max_uncompressed_bytes": spec["max_uncompressed_bytes"],
        }
    }


def _good_text(seed: str) -> str:
    return (
        f"The public domain review article {seed} is a carefully written essay "
        "about the history of art and literature, and it is presented with context "
        "for readers who are interested in the cultural record."
    )


def test_materializes_deterministically_with_zero_training_credit(tmp_path: Path) -> None:
    cfg = _base_config()
    rows = [_record("b", _good_text("beta")), _record("a", _good_text("alpha"))]
    _install_file(cfg, tmp_path, rows)

    first_rows, first_report, first_manifest = pdr.materialize(cfg, input_dir=tmp_path)
    second_rows, second_report, second_manifest = pdr.materialize(
        copy.deepcopy(cfg), input_dir=tmp_path
    )

    assert first_rows == second_rows
    assert first_report == second_report
    assert first_manifest == second_manifest
    assert [row["record_id"] for row in first_rows] == ["a", "b"]
    assert all(row["training_eligible"] is False for row in first_rows)
    assert all(row["evaluation_eligible"] is False for row in first_rows)
    boundary = first_report["truth_boundary"]
    assert boundary["training_authorized_bytes"] == 0
    assert boundary["canonical_corpus_admitted"] is False
    assert boundary["global_dedup_complete"] is False
    assert first_manifest["training_authorized_bytes"] == 0


def test_report_and_manifest_are_text_free(tmp_path: Path) -> None:
    cfg = _base_config()
    secret_phrase = "The unusual marmalade telescope is in the public domain article."
    _install_file(cfg, tmp_path, [_record("a", _good_text(secret_phrase))])

    rows, report, manifest = pdr.materialize(cfg, input_dir=tmp_path)
    assert rows
    durable = json.dumps({"report": report, "manifest": manifest}, ensure_ascii=False)
    assert "marmalade telescope" not in durable
    assert "Excluded Author" not in durable


def test_rejects_source_payload_hash_substitution(tmp_path: Path) -> None:
    cfg = _base_config()
    _install_file(cfg, tmp_path, [_record("a", _good_text("alpha"))])
    cfg["files"][0]["sha256"] = "f" * 64

    with pytest.raises(pdr.PublicDomainReviewError, match="source file policy drift"):
        pdr.materialize(cfg, input_dir=tmp_path)


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ({"source": "other-source"}, "source_drift"),
        (
            {
                "metadata": {
                    "license": "All Rights Reserved",
                    "url": "https://publicdomainreview.org/essay/a/",
                }
            },
            "license_drift",
        ),
        (
            {
                "metadata": {
                    "license": (
                        "Creative Commons - Attribution Share-Alike - "
                        "https://creativecommons.org/licenses/by-sa/4.0/"
                    ),
                    "url": "https://example.com/essay/a/",
                }
            },
            "origin_drift",
        ),
    ],
)
def test_record_authority_drift_is_excluded(
    tmp_path: Path,
    mutation: dict,
    reason: str,
) -> None:
    cfg = _base_config()
    row = _record("a", _good_text("alpha"))
    row.update(mutation)
    _install_file(cfg, tmp_path, [row])

    rows, report, _ = pdr.materialize(cfg, input_dir=tmp_path)
    assert rows == []
    assert report["rejected"][reason] == 1


def test_contact_or_secret_text_is_excluded(tmp_path: Path) -> None:
    cfg = _base_config()
    rows = [
        _record("email", _good_text("contact") + " editor@example.com"),
        _record("token", _good_text("token") + " API_KEY=abcdefghijklmnop"),
        _record("ok", _good_text("clean")),
    ]
    _install_file(cfg, tmp_path, rows)

    selected, report, _ = pdr.materialize(cfg, input_dir=tmp_path)
    assert [row["record_id"] for row in selected] == ["ok"]
    assert report["rejected"]["privacy_or_secret"] == 2


def test_exact_duplicate_is_counted_once(tmp_path: Path) -> None:
    cfg = _base_config()
    duplicate = _good_text("same")
    _install_file(
        cfg,
        tmp_path,
        [_record("a", duplicate), _record("b", duplicate)],
    )

    rows, report, _ = pdr.materialize(cfg, input_dir=tmp_path)
    assert len(rows) == 1
    assert report["rejected"]["exact_duplicate"] == 1


def test_family_byte_cap_cannot_create_training_credit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _base_config()
    first = _good_text("one")
    second = _good_text("two")
    cfg["selection"]["max_normalized_utf8_bytes"] = len(first.encode("utf-8")) + 5
    monkeypatch.setattr(pdr, "EXPECTED_SELECTION", dict(cfg["selection"]))
    _install_file(cfg, tmp_path, [_record("a", first), _record("b", second)])

    rows, report, _ = pdr.materialize(cfg, input_dir=tmp_path)
    assert len(rows) == 1
    assert report["rejected"]["family_byte_cap"] == 1
    assert report["truth_boundary"]["training_authorized_bytes"] == 0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            ("common_pile_rights_authority", "registry_identity_sha256", "f" * 64),
            "rights registry identity drift",
        ),
        (("source", "revision", "2" * 40), "source revision drift"),
        (("selection", "max_normalized_utf8_bytes", 9_999_999), "selection policy drift"),
    ],
)
def test_exact_authority_or_policy_substitution_fails_closed(
    tmp_path: Path,
    mutation: tuple[str, str, object],
    message: str,
) -> None:
    cfg = _base_config()
    _install_file(cfg, tmp_path, [_record("a", _good_text("alpha"))])
    section, key, value = mutation
    cfg[section][key] = value

    with pytest.raises(pdr.PublicDomainReviewError, match=message):
        pdr.materialize(cfg, input_dir=tmp_path)


def test_truth_boundary_promotion_fails_closed(tmp_path: Path) -> None:
    cfg = _base_config()
    _install_file(cfg, tmp_path, [_record("a", _good_text("alpha"))])
    cfg["truth_boundary"]["training_authorized_bytes"] = 1

    with pytest.raises(pdr.PublicDomainReviewError, match="truth boundary weakened"):
        pdr.materialize(cfg, input_dir=tmp_path)
