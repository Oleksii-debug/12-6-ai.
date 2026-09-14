from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools/validate_next100_063_terminal_source_registry_v6.py"
SPEC = importlib.util.spec_from_file_location("registry_v6", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
registry_v6 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(registry_v6)

CONFIG = json.loads(
    (ROOT / "configs/data/next100_063_terminal_source_registry_v6.json").read_text(
        encoding="utf-8"
    )
)
RAW_INTAKE = (
    ROOT / "evidence/data_bulk_code1/permissive_python_bundle_v1_terminal.json"
).read_bytes()
INTAKE = json.loads(RAW_INTAKE.decode("utf-8"))
INTAKE_BLOB = registry_v6.git_blob_sha1(RAW_INTAKE)


def test_v6_accepts_exact_terminal_intake_without_capacity_promotion() -> None:
    registry_v6.validate(CONFIG, INTAKE, intake_blob_sha1=INTAKE_BLOB)


def test_rejects_automatic_capacity_credit() -> None:
    config = copy.deepcopy(CONFIG)
    config["terminal_intake_addition"]["canonical_capacity_credit_bytes"] = 3_880_009
    config["registry_identity_sha256"] = registry_v6.canonical_identity(config)
    try:
        registry_v6.validate(config, INTAKE, intake_blob_sha1=INTAKE_BLOB)
    except registry_v6.RegistryV6Error:
        return
    raise AssertionError("premature capacity credit was accepted")


def test_rejects_fabricated_post_dedup_capacity() -> None:
    config = copy.deepcopy(CONFIG)
    config["truth_boundary"]["post_global_dedup_capacity_bytes"] = 6_095_624
    config["registry_identity_sha256"] = registry_v6.canonical_identity(config)
    try:
        registry_v6.validate(config, INTAKE, intake_blob_sha1=INTAKE_BLOB)
    except registry_v6.RegistryV6Error:
        return
    raise AssertionError("fabricated post-dedup capacity was accepted")


def test_rejects_family_overlap() -> None:
    config = copy.deepcopy(CONFIG)
    config["terminal_intake_addition"]["families"][0] = "github:psf/requests"
    config["registry_identity_sha256"] = registry_v6.canonical_identity(config)
    try:
        registry_v6.validate(config, INTAKE, intake_blob_sha1=INTAKE_BLOB)
    except registry_v6.RegistryV6Error:
        return
    raise AssertionError("overlapping canonical family was accepted")


def test_rejects_nonterminal_intake() -> None:
    intake = copy.deepcopy(INTAKE)
    intake["workflow_conclusion"] = "failure"
    try:
        registry_v6.validate(CONFIG, intake, intake_blob_sha1=INTAKE_BLOB)
    except registry_v6.RegistryV6Error:
        return
    raise AssertionError("nonterminal intake was accepted")


def test_rejects_intake_blob_drift() -> None:
    wrong_blob = hashlib.sha1(b"not-the-evidence").hexdigest()
    try:
        registry_v6.validate(CONFIG, INTAKE, intake_blob_sha1=wrong_blob)
    except registry_v6.RegistryV6Error:
        return
    raise AssertionError("drifted intake blob was accepted")


def test_rejects_training_exposure_promotion() -> None:
    config = copy.deepcopy(CONFIG)
    config["truth_boundary"]["authorized_balanced_no_replay_loss_positions"] = 1
    config["registry_identity_sha256"] = registry_v6.canonical_identity(config)
    try:
        registry_v6.validate(config, INTAKE, intake_blob_sha1=INTAKE_BLOB)
    except registry_v6.RegistryV6Error:
        return
    raise AssertionError("training exposure promotion was accepted")
