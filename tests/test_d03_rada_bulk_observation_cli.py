from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import probe_d03_rada_bulk_source as probe  # noqa: E402


def _archive() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("d1.htm", b"<html>one</html>")
        zf.writestr("README.txt", b"ignored")
    return buffer.getvalue()


def test_accept_current_upstream_retains_below_minimum_observation(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "observation.json"
    monkeypatch.setattr(
        probe,
        "_parse_args",
        lambda: SimpleNamespace(
            config=probe.DEFAULT_CONFIG,
            archive=None,
            output=output,
            accept_current_upstream=True,
        ),
    )
    monkeypatch.setattr(
        probe,
        "_download",
        lambda url, max_bytes: (_archive(), {"etag": "fixture"}),
    )

    probe.main()

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["inventory"]["canonical_entry_count"] == 1
    assert report["gates"]["safe_zip_inventory"] == "PASS"
    assert report["gates"]["discovery_capacity_threshold"] == "FAIL_BELOW_MINIMUM"
    assert report["gates"]["exact_archive_identity"] == "OBSERVED_UNPINNED"
    assert report["safe_result"] == (
        "CURRENT_UPSTREAM_OBSERVED_BELOW_DISCOVERY_MINIMUM_SUCCESSOR_TRIAGE_REQUIRED"
    )
    assert report["training_authorized_bytes"] == 0
    assert report["corpus_admitted"] is False
    assert report["discovery_observation_revalidated"] is False
