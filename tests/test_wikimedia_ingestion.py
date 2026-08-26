from __future__ import annotations

import bz2
import hashlib
import json
from pathlib import Path

import pytest

from twelve_six.data.wikimedia_ingestion import (
    WikipediaDumpPlan,
    WikimediaIngestionError,
    materialize_wikimedia_jsonl,
)

XML = b"""<?xml version="1.0" encoding="utf-8"?>
<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">
  <page>
    <title>Alpha</title><ns>0</ns><id>10</id>
    <revision>
      <id>101</id><timestamp>2026-08-20T00:00:00Z</timestamp>
      <sha1>alpha-upstream</sha1>
      <text xml:space="preserve">'''Alpha''' is a test.</text>
    </revision>
  </page>
  <page>
    <title>Redirect</title><ns>0</ns><id>11</id><redirect title="Alpha" />
    <revision>
      <id>102</id><timestamp>2026-08-20T00:00:01Z</timestamp>
      <text xml:space="preserve">#REDIRECT [[Alpha]]</text>
    </revision>
  </page>
  <page>
    <title>Talk:Alpha</title><ns>1</ns><id>12</id>
    <revision>
      <id>103</id><timestamp>2026-08-20T00:00:02Z</timestamp>
      <text xml:space="preserve">Discussion</text>
    </revision>
  </page>
  <page>
    <title>Beta</title><ns>0</ns><id>13</id>
    <revision>
      <id>104</id><timestamp>2026-08-20T00:00:03Z</timestamp>
      <text xml:space="preserve">Beta body</text>
    </revision>
  </page>
</mediawiki>
"""


def _write_snapshot(tmp_path: Path) -> tuple[Path, str]:
    payload = bz2.compress(XML)
    path = tmp_path / "ukwiki-20260820-pages-articles.xml.bz2"
    path.write_bytes(payload)
    return path, hashlib.sha256(payload).hexdigest()


def _plan(snapshot_sha256: str) -> WikipediaDumpPlan:
    filename = "ukwiki-20260820-pages-articles.xml.bz2"
    return WikipediaDumpPlan(
        source_id="wikimedia.ukwiki.pages-articles.20260820",
        language="uk",
        dump_date="20260820",
        dump_url=f"https://dumps.wikimedia.org/ukwiki/20260820/{filename}",
        dump_filename=filename,
        snapshot_sha256=snapshot_sha256,
        rights_authority_id="TEST-WIKIMEDIA-RIGHTS",
    )


def test_materialization_is_deterministic_and_fail_closed(tmp_path: Path) -> None:
    snapshot, digest = _write_snapshot(tmp_path)
    plan = _plan(digest)
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"

    result_a = materialize_wikimedia_jsonl(snapshot, first, plan)
    result_b = materialize_wikimedia_jsonl(snapshot, second, plan)

    assert result_a == result_b
    assert result_a.record_count == 2
    assert result_a.training_authorized is False
    assert first.read_bytes() == second.read_bytes()

    rows = [json.loads(line) for line in first.read_text(encoding="utf-8").splitlines()]
    assert [row["title"] for row in rows] == ["Alpha", "Beta"]
    assert [row["revision_id"] for row in rows] == [101, 104]
    assert all(row["content_state"] == "RAW_WIKITEXT_STAGING_ONLY" for row in rows)
    assert all(row["training_authorized"] is False for row in rows)
    assert rows[0]["attribution_url"].endswith("oldid=101")


def test_snapshot_hash_mismatch_does_not_publish_output(tmp_path: Path) -> None:
    snapshot, _digest = _write_snapshot(tmp_path)
    plan = _plan("0" * 64)
    output = tmp_path / "out.jsonl"

    with pytest.raises(WikimediaIngestionError, match="snapshot SHA-256 mismatch"):
        materialize_wikimedia_jsonl(snapshot, output, plan)

    assert not output.exists()


def test_bounded_materialization_stops_at_record_boundary(tmp_path: Path) -> None:
    snapshot, digest = _write_snapshot(tmp_path)
    plan = _plan(digest)
    output = tmp_path / "bounded.jsonl"

    result = materialize_wikimedia_jsonl(snapshot, output, plan, max_records=1)

    assert result.record_count == 1
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["title"] == "Alpha"


def test_zero_record_bound_fails_without_publishing(tmp_path: Path) -> None:
    snapshot, digest = _write_snapshot(tmp_path)
    plan = _plan(digest)
    output = tmp_path / "empty.jsonl"

    with pytest.raises(WikimediaIngestionError, match="zero eligible records"):
        materialize_wikimedia_jsonl(snapshot, output, plan, max_text_bytes=1)

    assert not output.exists()


def test_plan_rejects_mismatched_or_mutable_dump_url() -> None:
    with pytest.raises(WikimediaIngestionError, match="exactly match immutable dump identity"):
        WikipediaDumpPlan(
            source_id="x",
            language="uk",
            dump_date="20260820",
            dump_url=(
                "https://dumps.wikimedia.org/ukwiki/latest/"
                "ukwiki-latest-pages-articles.xml.bz2"
            ),
            dump_filename="ukwiki-20260820-pages-articles.xml.bz2",
            snapshot_sha256="a" * 64,
            rights_authority_id="rights",
        )
