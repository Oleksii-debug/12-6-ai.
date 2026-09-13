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


def expect_error(path, expected):
    try:
        validate(path)
    except ValueError as error:
        assert str(error) == expected
    else:
        raise AssertionError(f"expected {expected}")


def test_canonical_manifest_passes():
    assert validate(MANIFEST)["status"] == "PASS"


def test_manifest_identity_is_deterministic():
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert canonical_hash(payload) == canonical_hash(json.loads(json.dumps(payload)))


def test_base_sha_drift_fails_closed(tmp_path):
    path = mutated_manifest(tmp_path, lambda p: p.update(project_base_sha="0" * 40))
    expect_error(path, "project base SHA drift")


def test_upstream_commit_drift_fails_closed(tmp_path):
    path = mutated_manifest(
        tmp_path,
        lambda p: p["upstream"].update(immutable_commit="1" * 40),
    )
    expect_error(path, "upstream commit drift")


def test_package_version_drift_fails_closed(tmp_path):
    path = mutated_manifest(
        tmp_path,
        lambda p: p["upstream"].update(package_version_at_commit="0.0.35"),
    )
    expect_error(path, "upstream package version drift")


def test_unverified_tag_fails_closed(tmp_path):
    path = mutated_manifest(
        tmp_path,
        lambda p: p["upstream"].update(tag_or_release="v9.9.9"),
    )
    expect_error(path, "unexpected unverified tag/release binding")


def test_floating_requirement_inventory_is_required(tmp_path):
    path = mutated_manifest(
        tmp_path,
        lambda p: p["upstream_requirements"].update(floating_or_lower_bound_entries=[]),
    )
    expect_error(path, "floating dependency inventory unexpectedly empty")


def test_exact_requirement_is_required(tmp_path):
    def remove_pin(payload):
        payload["upstream_requirements"]["entries"] = [
            item
            for item in payload["upstream_requirements"]["entries"]
            if item != "pandas==2.1.4"
        ]

    path = mutated_manifest(tmp_path, remove_pin)
    expect_error(path, "missing exact upstream pandas pin")


def test_fabricated_artifact_hash_fails_closed(tmp_path):
    path = mutated_manifest(
        tmp_path,
        lambda p: p["installation_attempt"].update(artifact_sha256="a" * 64),
    )
    expect_error(path, "unavailable artifact must not have a fabricated hash")


def test_runtime_pass_claim_fails_closed(tmp_path):
    path = mutated_manifest(
        tmp_path,
        lambda p: p["runtime"].update(execution_status="PASS"),
    )
    expect_error(path, "runtime cannot be promoted without execution")


def test_parity_claim_fails_closed(tmp_path):
    path = mutated_manifest(
        tmp_path,
        lambda p: p["runtime"].update(parity_proven=True),
    )
    expect_error(path, "parity cannot be true without runtime evidence")


def test_foreign_weight_flag_fails_closed(tmp_path):
    path = mutated_manifest(
        tmp_path,
        lambda p: p["canonical_base_safety"].update(foreign_pretrained_weights_used=True),
    )
    expect_error(path, "canonical Base safety violation: foreign_pretrained_weights_used")


def test_evidence_identity_changes_after_tamper():
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    before = canonical_hash(payload)
    payload["network"]["pypi_reachable"] = True
    assert before != canonical_hash(payload)
