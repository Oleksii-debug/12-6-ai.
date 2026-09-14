from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_IMPL_PATH = Path(__file__).with_name("_materialize_d03_ua_nbu_pdftotext_v1_impl.py")
_SPEC = importlib.util.spec_from_file_location("_twelve_six_nbu_pdftotext_impl", _IMPL_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load NBU pdftotext implementation from {_IMPL_PATH}")
_impl = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _impl
_SPEC.loader.exec_module(_impl)

NbuTextMaterializationError = _impl.NbuTextMaterializationError
DEFAULT_CONFIG = _impl.DEFAULT_CONFIG
_legacy_validate_config = _impl.validate_config

_FROZEN_CONFIG: dict[str, Any] = {
    "schema": "12-6.d03-ua-nbu-pdftotext-materialization.v1",
    "status": "PREPARED_TEXT_MATERIALIZATION_ZERO_CREDIT",
    "base_authority": {
        "repository": "Oleksii-debug/12-6-ai.",
        "parent_pr": 912,
        "parent_head_sha": "ea9436a84960e31282ccbddd7b3167be0e7c2362",
        "parent_pdf_pin_contract_identity_sha256": (
            "0532eea5d03014473a30cf78d19da5670530759d4cd4b15dda2c9fb7ac30912f"
        ),
        "discovery_pr": 898,
        "discovery_head_sha": "86a8a2a6f91eb4093bee414aaa8e8bbd3d895761",
        "discovery_contract_identity_sha256": (
            "2e04335a2e6e665167b63f392d09756ae52e4f65f5253d9a037c48f5d191d9ec"
        ),
        "lane": "D03",
        "control_issue": 548,
    },
    "source": {
        "source_id": "ua.nbu.official-resolutions",
        "family_id": "ua.nbu.official-resolutions",
        "allowed_origin": "https://bank.gov.ua",
        "document_path_regex": r"^/ua/legislation/Resolution_[0-9]{8}_[0-9A-Za-z-]+$",
        "official_pdf_path_regex": r"^/admin_uploads/law/[0-9A-Za-z_.-]+\.pdf$",
    },
    "extractor": {
        "backend": "poppler-pdftotext",
        "executable": "pdftotext",
        "required_version": "25.06.0",
        "arguments": ["-enc", "UTF-8", "-nopgbrk"],
        "ocr_allowed": False,
        "shell_allowed": False,
        "network_allowed_by_extractor": False,
        "double_extract_required": True,
        "exact_output_bytes_required": True,
    },
    "materialization": {
        "class": "LOCAL_FREE",
        "one_record_per_pdf": True,
        "hard_max_records": 240,
        "min_text_bytes": 32,
        "hard_max_text_bytes_per_record": 10_000_000,
        "hard_max_total_text_bytes": 100_000_000,
        "source_fetch_timeout_seconds": 30,
        "utf8_strict": True,
        "nul_forbidden": True,
        "durable_evidence_body_free": True,
        "text_artifact_is_training_authority": False,
    },
    "required_downstream_gates": [
        "DOCUMENT_LEVEL_RIGHTS_CONFIRMATION",
        "QUALITY_AND_PRIVACY",
        "GLOBAL_EXACT_NEAR_FRAGMENT_LINEAGE_DEDUP",
        "RESERVED_EVALUATION_DECONTAMINATION",
        "POST_COMPOSITION_BALANCE_FAMILY_CAPS",
        "CLUSTER_SAFE_SPLIT",
        "DETERMINISTIC_TOKENIZER_PACKING_DOUBLE_BUILD",
        "POSITIVE_EXACT_UNIQUE_CAUSAL_LOSS_LEDGER",
    ],
    "claims": {
        "canonical_capacity_credit_bytes": 0,
        "training_authorized_bytes": 0,
        "authorized_unique_loss_positions": 0,
        "text_materialization_execution_authorized": True,
        "tokenizer_fit_authorized": False,
        "model_training_executed": False,
        "optimizer_updates": 0,
        "final_test_accessed": False,
        "paid_compute_authorized": False,
        "learned_20m_claim": False,
    },
    "contract_identity_sha256": (
        "c7e3222fc609cbe22d255a1732bd8130ce21d50185a9ef48bbb92f1d5ab288cf"
    ),
}


def _require_exact(actual: object, expected: object, path: str) -> None:
    if type(actual) is not type(expected):
        raise NbuTextMaterializationError(f"config type drift at {path}")
    if isinstance(expected, dict):
        actual_dict = actual
        if set(actual_dict) != set(expected):
            raise NbuTextMaterializationError(f"config key-set drift at {path}")
        for key, expected_value in expected.items():
            _require_exact(actual_dict[key], expected_value, f"{path}.{key}")
        return
    if isinstance(expected, list):
        actual_list = actual
        if len(actual_list) != len(expected):
            raise NbuTextMaterializationError(f"config list-length drift at {path}")
        for index, (actual_value, expected_value) in enumerate(zip(actual_list, expected)):
            _require_exact(actual_value, expected_value, f"{path}[{index}]")
        return
    if actual != expected:
        raise NbuTextMaterializationError(f"config value drift at {path}")


def validate_config(config: Mapping[str, Any]) -> None:
    _require_exact(config, _FROZEN_CONFIG, "$")
    _legacy_validate_config(config)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise NbuTextMaterializationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
    )
    if not isinstance(value, dict):
        raise NbuTextMaterializationError("config root must be object")
    validate_config(value)
    return value


_impl.validate_config = validate_config
_impl.load_config = load_config


def __getattr__(name: str) -> Any:
    return getattr(_impl, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_impl)))


if __name__ == "__main__":
    raise SystemExit(_impl.main())
