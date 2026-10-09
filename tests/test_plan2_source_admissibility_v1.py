from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from tools.plan2_source_admissibility_v1 import (
    SourceRightsError,
    build_catalog,
    materialization_receipt,
    removal_plan,
    verify_catalog,
    verify_receipt,
)
from tools.plan2_source_inventory_v1 import build_inventory

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = b"fixture own public material explicit permission\n"


def fixture():
    snapshot = hashlib.sha256(b"fixture snapshot\n").hexdigest()
    source = {
        "source_id": "fixture.public.alpha", "source_family": "fixture.public",
        "location": "repo://tests/fixtures/sample", "acquisition_method": "local_fixture",
        "language": "en", "modality": "text", "update_cadence": "pinned",
        "status": "accepted", "provenance": {
            "authority_path": "tests/test_plan2_source_admissibility_v1.py",
            "upstream_revision": "fixture-v1", "content_sha256": snapshot,
        },
    }
    grant = {
        "source_id": source["source_id"], "rights_class": "public_licensed",
        "license_id": "CC0-1.0",
        "terms_ref": "https://creativecommons.org/publicdomain/zero/1.0/",
        "permission_ref": "repo://tests/fixtures/local-permission",
        "legal_basis": "explicit_permission",
        "allowed_uses": ["training", "research"],
        "evidence_path": "tests/fixtures/fixture-rights",
        "evidence_sha256": hashlib.sha256(EVIDENCE).hexdigest(), "revoked": False,
    }
    inv = build_inventory("fixture-revision", [source])
    evidence = {"tests/fixtures/fixture-rights": EVIDENCE}
    cat = build_catalog(inv, [grant], evidence)
    rows = [{
        "record_id": f"record-{i}", "member_id": f"chapter{i}",
        "source_id": source["source_id"], "source_snapshot_sha256": snapshot,
        "origin_member_ref": f"book/chapter{i}", "text": f"record text {i}"
    } for i in (1, 2)]
    return inv, cat, evidence, rows


def test_real_data324_has_pinned_license_but_training_candidate_is_blocked():
    cfg = json.loads((ROOT / "configs/data/plan2_source_inventory_v1.json").read_text())
    rights = json.loads((ROOT / "configs/data/plan2_rights_admissibility_v1.json").read_text())
    incumbent = json.loads((ROOT / "configs/data/data324_kubernetes_ua_current_main_v1.json").read_text())
    inv = build_inventory(cfg["revision"], cfg["sources"])
    grant = rights["grants"][0]
    assert grant["license_id"] == incumbent["rights_authority"]["license_id"]
    assert grant["evidence_sha256"] == incumbent["rights_authority"]["license_sha256"]
    assert incumbent["current_main_truth_boundary"]["corpus_admitted"] is False
    evidence = {grant["evidence_path"]: (ROOT / grant["evidence_path"]).read_bytes()}
    cat = build_catalog(inv, rights["grants"], evidence)
    assert verify_catalog(inv, cat, evidence) == cat
    assert cat["training_corpus_authorized"] is False
    source = inv["sources"][0]
    row = {"record_id": "candidate1", "member_id": "doc",
           "source_id": source["source_id"], "origin_member_ref": "doc",
           "source_snapshot_sha256": source["provenance"]["content_sha256"],
           "text": "candidate"}
    with pytest.raises(SourceRightsError):
        materialization_receipt(inv, cat, evidence, [row], "training")


def test_restart_order_and_membership_are_deterministic_without_payload_spillage():
    inv, cat, evidence, rows = fixture()
    a = materialization_receipt(inv, cat, evidence, rows, "training")
    b = materialization_receipt(inv, cat, evidence, list(reversed(rows)), "training")
    assert a == b
    assert a["record_count"] == 2
    assert not a["raw_payload_persisted"] and not a["training_corpus_authorized"]
    assert "record text 1" not in json.dumps(a)
    assert a["members"][0]["payload_sha256"] == hashlib.sha256(b"record text 1").hexdigest()
    assert verify_receipt(inv, cat, evidence, rows, "training", a) == a


def test_source_level_removal_produces_exact_member_rebuild_plan():
    inv, cat, evidence, rows = fixture()
    receipt = materialization_receipt(inv, cat, evidence, rows, "research")
    plan = removal_plan(receipt, "fixture.public.alpha")
    assert plan["affected_records"] == 2 and plan["downstream_rebuild_required"] is True
    assert {r["record_id"] for r in plan["affected"]} == {"record-1", "record-2"}
    damaged = copy.deepcopy(receipt)
    damaged["members"][0]["record_id"] = "forged"
    with pytest.raises(SourceRightsError):
        removal_plan(damaged, "fixture.public.alpha")


@pytest.mark.parametrize(("key", "value"), [
    ("rights_class", "research_only"), ("rights_class", "private"),
    ("rights_class", "unknown"), ("revoked", True),
    ("evidence_sha256", "0" * 64), ("legal_basis", "unknown"),
    ("license_id", "UNKNOWN"),
])
def test_untrusted_rights_cannot_become_training_grants(key, value):
    inv, cat, evidence, rows = fixture()
    changed = copy.deepcopy(cat["grants"][0])
    changed[key] = value
    with pytest.raises(SourceRightsError):
        catalog = build_catalog(inv, [changed], evidence)
        materialization_receipt(inv, catalog, evidence, rows, "training")


def test_research_only_allowed_for_research_never_training_or_release():
    inv, cat, evidence, rows = fixture()
    limited = copy.deepcopy(cat["grants"][0])
    limited.update(rights_class="research_only", legal_basis="research_terms",
                   allowed_uses=["research"])
    c = build_catalog(inv, [limited], evidence)
    assert materialization_receipt(inv, c, evidence, rows, "research")["record_count"] == 2
    for purpose in ("training", "release"):
        with pytest.raises(SourceRightsError):
            materialization_receipt(inv, c, evidence, rows, purpose)


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(source_id="foreign"),
    lambda r: r.update(source_snapshot_sha256="0" * 64),
    lambda r: r.update(record_id="record-2"),
    lambda r: r.update(origin_member_ref=""),
    lambda r: r.update(text=""),
    lambda r: r.update(text="\x00"),
    lambda r: r.pop("member_id"),
])
def test_member_lineage_fail_closed_on_missing_or_forged_components(mutation):
    inv, cat, evidence, rows = fixture()
    damaged = copy.deepcopy(rows)
    mutation(damaged[0])
    with pytest.raises(SourceRightsError):
        materialization_receipt(inv, cat, evidence, damaged, "training")


def test_missing_or_replaced_license_evidence_cannot_pass():
    inv, cat, evidence, _ = fixture()
    with pytest.raises(SourceRightsError):
        verify_catalog(inv, cat, {})
    with pytest.raises(SourceRightsError):
        verify_catalog(inv, cat, {next(iter(evidence)): b"forged"})


def test_unknown_rights_source_cannot_hide_beside_known_source():
    inv, cat, evidence, _ = fixture()
    foreign = copy.deepcopy(cat["grants"][0])
    foreign["source_id"] = "fixture.foreign"
    with pytest.raises(SourceRightsError):
        build_catalog(inv, cat["grants"] + [foreign], evidence)


def test_receipt_tampering_or_purpose_change_fails_restart_validation():
    inv, cat, evidence, rows = fixture()
    good = materialization_receipt(inv, cat, evidence, rows, "training")
    wrong = copy.deepcopy(good)
    wrong["members"][0]["payload_sha256"] = "0" * 64
    with pytest.raises(SourceRightsError):
        verify_receipt(inv, cat, evidence, rows, "training", wrong)
    with pytest.raises(SourceRightsError):
        verify_receipt(inv, cat, evidence, rows, "research", good)
