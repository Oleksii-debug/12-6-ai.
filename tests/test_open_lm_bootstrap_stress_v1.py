import json
from pathlib import Path

from tools.validate_open_lm_bootstrap_stress_v1 import canonical_hash, validate

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/research/open_lm_bootstrap_stress_v1.json"


def mutated_manifest(tmp_path, mutate):
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    mutate(payload)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_canonical_manifest_passes():
    assert validate(MANIFEST)["status"] == "PASS"


def test_manifest_identity_is_deterministic():
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert canonical_hash(payload) == canonical_hash(json.loads(json.dumps(payload)))


def test_base_sha_drift_fails_closed(tmp_path):
    path = mutated_manifest(tmp_path, lambda p: p.update(project_base_sha="0" * 40))
    try:
        validate(path)
    except ValueError as error:
        assert str(error) == "project base SHA drift"
    else:
        raise AssertionError("base SHA drift was accepted")


def test_upstream_commit_drift_fails_closed(tmp_path):
    path = mutated_manifest(
        tmp_path,
        lambda p: p["upstream"].update(immutable_commit="1" * 40),
    )
    try:
        validate(path)
    except ValueError as error:
        assert str(error) == "upstream commit drift"
    else:
        raise AssertionError("upstream commit drift was accepted")


def test_unverified_tag_fails_closed(tmp_path):
    path = mutated_manifest(
        tmp_path,
        lambda p: p["upstream"].update(tag_or_release="v9.9.9"),
    )
    try:
        validate(path)
    except ValueError as error:
        assert str(error) == "unexpected unverified tag/release binding"
    else:
        raise AssertionError("unverified tag was accepted")


def test_floating_requirement_inventory_is_required(tmp_path):
    path = mutated_manifest(
        tmp_path,
        lambda p: p["upstream_requirements"].update(floating_or_lower_bound_entries=[]),
    )
    try:
        validate(path)
    except ValueError as error:
        assert str(error) == "floating dependency inventory unexpectedly empty"
    else:
        raise AssertionError("floating dependency inventory loss was accepted")


def test_exact_requirement_is_required(tmp_path):
    def remove_pin(payload):
        payload["upstream_requirements"]["entries"] = [
            item
            for item in payload["upstream_requirements"]["entries"]
            if item != "pandas==2.1.4"
        ]

    path = mutated_manifest(tmp_path, remove_pin)
    try:
        validate(path)
    except ValueError as error:
        assert str(error) == "missing exact upstream pandas pin"
    else:
        raise AssertionError("exact upstream pin loss was accepted")


def test_fabricated_artifact_hash_fails_closed(tmp_path):
    path = mutated_manifest(
        tmp_path,
        lambda p: p["installation_attempt"].update(artifact_sha256="a" * 64),
    )
    try:
        validate(path)
    except ValueError as error:
        assert str(error) == "unavailable artifact must not have a fabricated hash"
    else:
        raise AssertionError("fabricated artifact hash was accepted")


def test_runtime_pass_claim_fails_closed(tmp_path):
    path = mutated_manifest(
        tmp_path,
        lambda p: p["runtime"].update(execution_status="PASS"),
    )
    try:
        validate(path)
    except ValueError as error:
        assert str(error) == "runtime cannot be promoted without execution"
    else:
        raise AssertionError("runtime PASS claim was accepted")


def test_parity_claim_fails_closed(tmp_path):
    path = mutated_manifest(
        tmp_path,
        lambda p: p["runtime"].update(parity_proven=True),
    )
    try:
        validate(path)
    except ValueError as error:
        assert str(error) == "parity cannot be true without runtime evidence"
    else:
        raise AssertionError("parity claim was accepted")


def test_foreign_weight_flag_fails_closed(tmp_path):
    path = mutated_manifest(
        tmp_path,
        lambda p: p["canonical_base_safety"].update(foreign_pretrained_weights_used=True),
    )
    try:
        validate(path)
    except ValueError as error:
        assert str(error) == "canonical Base safety violation: foreign_pretrained_weights_used"
    else:
        raise AssertionError("foreign-weight flag was accepted")


def test_evidence_identity_changes_after_tamper():
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    before = canonical_hash(payload)
    payload["network"]["pypi_reachable"] = True
    assert before != canonical_hash(payload)
