from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = REPO_ROOT / "tools" / "validate_data324_current_main_port.py"
VALIDATOR_SPEC = importlib.util.spec_from_file_location(
    "data324_current_main_port_validator", VALIDATOR_PATH
)
if VALIDATOR_SPEC is None or VALIDATOR_SPEC.loader is None:
    raise RuntimeError("cannot load DATA-324 current-main validator")
VALIDATOR_MODULE = importlib.util.module_from_spec(VALIDATOR_SPEC)
VALIDATOR_SPEC.loader.exec_module(VALIDATOR_MODULE)
PortValidationError = VALIDATOR_MODULE.PortValidationError
validate_snapshot = VALIDATOR_MODULE.validate_snapshot

PORT_CONFIG = Path("configs/data/data324_kubernetes_ua_current_main_v1.json")
REQUIRED = (
    "configs/data/data324_kubernetes_ua_current_main_v1.json",
    "configs/data/data324_kubernetes_ua_recovery_v1.json",
    "data/external/snapshots/data324-kubernetes-ua-v1/manifest.json",
    "reports/data324/kubernetes-ua-recovery-v1.json",
    "data/external/snapshots/data324-kubernetes-ua-v1/raw/what-is-kubernetes.md",
    "data/external/snapshots/data324-kubernetes-ua-v1/normalized/what-is-kubernetes.uk.txt",
    "data/external/rights-evidence/data324/kubernetes-website-cc-by-4.0-25f3dcb.txt",
    "data/external/snapshots/data324-kubernetes-ua-v1/ATTRIBUTION.txt",
)


def _copy_fixture(tmp_path: Path) -> Path:
    repo = Path(__file__).resolve().parents[1]
    for relative in REQUIRED:
        source = repo / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return tmp_path


def _mutate_port(root: Path, mutate: object) -> None:
    path = root / PORT_CONFIG
    config = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(config, dict)
    assert callable(mutate)
    mutate(config)
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def test_frozen_current_main_port_revalidates() -> None:
    result = validate_snapshot(Path(__file__).resolve().parents[1])
    assert result["decision"] == "PASS_SOURCE_LEVEL_CURRENT_MAIN_PORT"
    assert result["normalized_utf8_bytes"] == 17415
    assert result["current_global_dedup"] == "NOT_RUN_REQUIRE_CURRENT_CANONICAL_OWNER"
    assert result["family_credit_authorized"] is False
    assert result["training_authorized_bytes"] == 0


def test_reserved_sha_identities_must_be_exact(tmp_path: Path) -> None:
    root = _copy_fixture(tmp_path)

    def mutate(config: dict[str, object]) -> None:
        firewall = config["evaluation_firewall"]
        assert isinstance(firewall, dict)
        reserved = firewall["reserved_raw_sha256"]
        assert isinstance(reserved, list)
        reserved[0] = "0" * 63

    _mutate_port(root, mutate)
    with pytest.raises(PortValidationError, match="evaluation firewall drift"):
        validate_snapshot(root)


def test_well_formed_reservation_substitution_fails(tmp_path: Path) -> None:
    root = _copy_fixture(tmp_path)

    def mutate(config: dict[str, object]) -> None:
        firewall = config["evaluation_firewall"]
        assert isinstance(firewall, dict)
        firewall["reserved_raw_sha256"] = ["f" * 64, "e" * 64]

    _mutate_port(root, mutate)
    with pytest.raises(PortValidationError, match="evaluation firewall drift"):
        validate_snapshot(root)


def test_evaluation_authority_substitution_fails(tmp_path: Path) -> None:
    root = _copy_fixture(tmp_path)

    def mutate(config: dict[str, object]) -> None:
        firewall = config["evaluation_firewall"]
        assert isinstance(firewall, dict)
        firewall["authority"] = "EVAL-290-SUBSTITUTED"

    _mutate_port(root, mutate)
    with pytest.raises(PortValidationError, match="evaluation firewall drift"):
        validate_snapshot(root)


def test_reservation_commit_substitution_fails(tmp_path: Path) -> None:
    root = _copy_fixture(tmp_path)

    def mutate(config: dict[str, object]) -> None:
        firewall = config["evaluation_firewall"]
        assert isinstance(firewall, dict)
        firewall["reservation_commit_sha"] = "0" * 40

    _mutate_port(root, mutate)
    with pytest.raises(PortValidationError, match="evaluation firewall drift"):
        validate_snapshot(root)


def test_ported_from_head_substitution_fails(tmp_path: Path) -> None:
    root = _copy_fixture(tmp_path)

    def mutate(config: dict[str, object]) -> None:
        ported_from = config["ported_from"]
        assert isinstance(ported_from, dict)
        ported_from["head_sha"] = "0" * 40

    _mutate_port(root, mutate)
    with pytest.raises(PortValidationError, match="ported-from provenance drift"):
        validate_snapshot(root)


def test_source_authority_substitution_fails(tmp_path: Path) -> None:
    root = _copy_fixture(tmp_path)

    def mutate(config: dict[str, object]) -> None:
        source = config["source_authority"]
        assert isinstance(source, dict)
        source["upstream_revision"] = "0" * 40

    _mutate_port(root, mutate)
    with pytest.raises(PortValidationError, match="source authority drift"):
        validate_snapshot(root)


def test_rights_authority_substitution_fails(tmp_path: Path) -> None:
    root = _copy_fixture(tmp_path)

    def mutate(config: dict[str, object]) -> None:
        rights = config["rights_authority"]
        assert isinstance(rights, dict)
        rights["license_sha256"] = "0" * 64

    _mutate_port(root, mutate)
    with pytest.raises(PortValidationError, match="rights authority drift"):
        validate_snapshot(root)


def test_unknown_port_field_fails_closed(tmp_path: Path) -> None:
    root = _copy_fixture(tmp_path)

    def mutate(config: dict[str, object]) -> None:
        config["unexpected_authority"] = {"status": "PASS"}

    _mutate_port(root, mutate)
    with pytest.raises(PortValidationError, match="top-level schema drift"):
        validate_snapshot(root)


def test_normalized_payload_tamper_fails(tmp_path: Path) -> None:
    root = _copy_fixture(tmp_path)
    path = root / (
        "data/external/snapshots/data324-kubernetes-ua-v1/normalized/"
        "what-is-kubernetes.uk.txt"
    )
    path.write_bytes(path.read_bytes() + b"tamper\n")
    with pytest.raises(PortValidationError, match="normalized snapshot identity mismatch"):
        validate_snapshot(root)


def test_current_main_family_credit_promotion_fails(tmp_path: Path) -> None:
    root = _copy_fixture(tmp_path)

    def mutate(config: dict[str, object]) -> None:
        truth = config["current_main_truth_boundary"]
        assert isinstance(truth, dict)
        truth["family_credit_authorized"] = True

    _mutate_port(root, mutate)
    with pytest.raises(PortValidationError, match="truth boundary drift"):
        validate_snapshot(root)
