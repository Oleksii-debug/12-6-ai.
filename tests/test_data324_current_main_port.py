from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from tools.validate_data324_current_main_port import PortValidationError, validate_snapshot

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


def test_frozen_current_main_port_revalidates() -> None:
    result = validate_snapshot(Path(__file__).resolve().parents[1])
    assert result["decision"] == "PASS_SOURCE_LEVEL_CURRENT_MAIN_PORT"
    assert result["normalized_utf8_bytes"] == 17415
    assert result["current_global_dedup"] == "NOT_RUN_REQUIRE_CURRENT_CANONICAL_OWNER"
    assert result["family_credit_authorized"] is False
    assert result["training_authorized_bytes"] == 0


def test_reserved_sha_identities_must_be_exact(tmp_path: Path) -> None:
    root = _copy_fixture(tmp_path)
    path = root / "configs/data/data324_kubernetes_ua_current_main_v1.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["evaluation_firewall"]["reserved_raw_sha256"][0] = "0" * 63
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(PortValidationError, match="reservation identity malformed"):
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
    path = root / "configs/data/data324_kubernetes_ua_current_main_v1.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["current_main_truth_boundary"]["family_credit_authorized"] = True
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(PortValidationError, match="truth boundary drift"):
        validate_snapshot(root)
