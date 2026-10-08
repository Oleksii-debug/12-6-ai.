from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from tools.plan2_rights_provenance_v1 import (
    Plan2AdmissibilityError, attest_data324_source,
    require_training_materialization, verify_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "configs/data/plan2_source_inventory_v1.json"
MANIFEST = ROOT / "data/external/snapshots/data324-kubernetes-ua-v1/manifest.json"


def seed():
    return json.loads(SEED.read_text(encoding="utf-8"))


def test_incumbent_rights_provenance_verified_but_training_still_blocked():
    receipt = attest_data324_source(ROOT, seed())
    assert receipt["source_level_rights_verified"] is True
    assert receipt["corpus_training_authorized"] is False
    assert receipt["evaluation_authorized"] is False
    assert receipt["rights_basis"]["license_id"] == "CC-BY-4.0"
    verify_receipt(receipt, attest_data324_source(ROOT, seed()))
    with pytest.raises(Plan2AdmissibilityError):
        require_training_materialization(receipt)


@pytest.mark.parametrize("tamper", [
    lambda s: s["sources"][0].update(status="accepted"),
    lambda s: s["sources"][0]["provenance"].update(content_sha256="0" * 64),
    lambda s: s["sources"][0]["provenance"].update(upstream_revision="0" * 40),
    lambda s: s["sources"][0].update(source_family="wrong.family"),
    lambda s: s["sources"][0].update(location="repo://another/location.txt"),
])
def test_admission_rejects_seed_rebinding(tamper):
    altered = seed()
    tamper(altered)
    with pytest.raises(Plan2AdmissibilityError):
        attest_data324_source(ROOT, altered)


def test_forged_self_consistent_receipt_has_no_independent_authority():
    expected = attest_data324_source(ROOT, seed())
    altered = copy.deepcopy(expected)
    altered["corpus_training_authorized"] = True
    altered["tokenizer_fit_authorized"] = True
    with pytest.raises(Plan2AdmissibilityError):
        verify_receipt(altered, expected)


def test_duplicate_or_missing_seed_fields_fail_closed():
    altered = seed()
    altered["extra_rights"] = "ALLOWED"
    with pytest.raises(Plan2AdmissibilityError):
        attest_data324_source(ROOT, altered)


def test_materialized_license_tampering_is_rejected(tmp_path):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    paths = [
        "configs/data/data324_kubernetes_ua_current_main_v1.json",
        "configs/data/data324_kubernetes_ua_recovery_v1.json",
        "data/external/snapshots/data324-kubernetes-ua-v1/manifest.json",
        "reports/data324/kubernetes-ua-recovery-v1.json",
        manifest["objects"][0]["raw_snapshot_path"],
        manifest["objects"][0]["normalized_snapshot_path"],
        manifest["license_evidence"]["materialized_path"],
        manifest["attribution_path"],
    ]
    for path in paths:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / path, target)
    license_path = tmp_path / manifest["license_evidence"]["materialized_path"]
    license_path.write_bytes(license_path.read_bytes() + b"\nforged permission")
    with pytest.raises((Plan2AdmissibilityError, ValueError)):
        attest_data324_source(tmp_path, seed())


def test_symlinked_member_payload_cannot_be_trusted(tmp_path):
    with pytest.raises(Plan2AdmissibilityError):
        from tools.plan2_rights_provenance_v1 import _strict_relative
        link = tmp_path / "payload"
        link.symlink_to(SEED)
        _strict_relative(tmp_path, "payload")


def test_record_bound_to_normalized_snapshot_exact_hash():
    receipt = attest_data324_source(ROOT, seed())
    payload = (ROOT / receipt["record"]["member_path"]).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == receipt["record"]["payload_sha256"]
    assert len(payload) == receipt["record"]["payload_bytes"]
