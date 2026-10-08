from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from tools.plan2_source_inventory_v1 import (
    SourceInventoryError,
    build_inventory,
    diff_inventories,
    require_known_inputs,
    verify_inventory,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/data/plan2_source_inventory_v1.json"


def source(status="accepted"):
    return {
        "source_id": "fixture.local.hello", "source_family": "fixture.local",
        "location": "repo://tests/fixtures/hello.txt",
        "acquisition_method": "local_fixture", "language": "en",
        "modality": "text", "update_cadence": "pinned", "status": status,
        "provenance": {
            "authority_path": "tests/test_plan2_source_inventory_v1.py",
            "upstream_revision": "fixture-v1",
            "content_sha256": hashlib.sha256(b"hello").hexdigest(),
        },
    }


def test_real_incumbent_source_inventory_is_reproducible_and_zero_training():
    data = json.loads(CONFIG.read_text(encoding="utf-8"))
    observed = build_inventory(data["revision"], data["sources"])
    assert observed == build_inventory(data["revision"], list(reversed(data["sources"])))
    assert verify_inventory(observed) == observed
    assert observed["training_authorized"] is False
    assert observed["status_counts"] == {"accepted": 0, "candidate": 1, "rejected": 0}
    manifest = json.loads((ROOT / "data/external/snapshots/data324-kubernetes-ua-v1/manifest.json").read_text(encoding="utf-8"))
    source = data["sources"][0]
    assert source["source_id"] == manifest["objects"][0]["source_id"]
    assert source["provenance"]["content_sha256"] == manifest["objects"][0]["normalized_sha256"]
    normalized = (ROOT / manifest["objects"][0]["normalized_snapshot_path"]).read_bytes()
    assert hashlib.sha256(normalized).hexdigest() == source["provenance"]["content_sha256"]


def test_reordering_inputs_cannot_change_inventory_identity():
    a, b = source(), copy.deepcopy(source())
    b["source_id"] = "fixture.local.world"
    assert build_inventory("r1", [a, b]) == build_inventory("r1", [b, a])


def test_diff_records_added_changed_removed_and_rejects_revision_reuse():
    a = build_inventory("r1", [source()])
    s = source("candidate")
    b = build_inventory("r2", [s])
    diff = diff_inventories(a, b)
    assert diff["changed"] == [s["source_id"]]
    assert diff["added"] == diff["removed"] == []
    assert diff_inventories(a, build_inventory("r3", []))["removed"] == [s["source_id"]]
    with pytest.raises(SourceInventoryError):
        diff_inventories(a, build_inventory("r1", [s]))


@pytest.mark.parametrize("mutate", [
    lambda s: s.update(source_id="not a stable id"),
    lambda s: s.update(status="train"),
    lambda s: s.update(language=""),
    lambda s: s.update(location="file:///private/secret"),
    lambda s: s["provenance"].update(content_sha256="0" * 63),
    lambda s: s["provenance"].pop("authority_path"),
])
def test_malformed_or_unattributed_sources_fail_closed(mutate):
    s = source()
    mutate(s)
    with pytest.raises(SourceInventoryError):
        build_inventory("r1", [s])


def test_duplicate_ids_and_inventory_tampering_fail_closed():
    with pytest.raises(SourceInventoryError):
        build_inventory("r1", [source(), source()])
    i = build_inventory("r1", [source()])
    i["status_counts"]["accepted"] = 42
    with pytest.raises(SourceInventoryError):
        verify_inventory(i)


def test_unknown_candidate_rejected_and_hash_drift_never_bind():
    accepted = source()
    inv = build_inventory("r1", [accepted])
    bound = {"source_id": accepted["source_id"],
             "content_sha256": accepted["provenance"]["content_sha256"]}
    assert require_known_inputs(inv, [bound]) == [bound]
    for faulty in [
        {"source_id": "unknown", "content_sha256": bound["content_sha256"]},
        {"source_id": bound["source_id"], "content_sha256": "0" * 64},
    ]:
        with pytest.raises(SourceInventoryError):
            require_known_inputs(inv, [faulty])
    with pytest.raises(SourceInventoryError):
        require_known_inputs(inv, [bound, bound])
    for state in ("candidate", "rejected"):
        with pytest.raises(SourceInventoryError):
            require_known_inputs(build_inventory("r1", [source(state)]), [bound])
