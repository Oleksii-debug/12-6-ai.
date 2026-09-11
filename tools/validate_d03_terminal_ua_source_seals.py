#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

SHA64 = re.compile(r"^[0-9a-f]{64}$")
AUTHORITY_IDENTITY = "dc891fd113d617269d4af7e6bcbe21ea849075423dbed7224cb707b99ba1f540"
TOP_LEVEL_KEYS = {
    "schema_version",
    "worker_id",
    "local_free_only",
    "authority_state",
    "sources",
    "aggregate",
    "current_main_truth_boundary",
    "required_downstream_gates",
    "notes",
    "authority_identity_sha256",
}
EXPECTED_SOURCES: dict[str, dict[str, Any]] = {
    "next100-028-php-doc-uk": {
        "source_id": "next100-028-php-doc-uk",
        "historical_pr": 454,
        "historical_head": "a8fde2cb31db85b31a422e094f0c45ebd2658d39",
        "terminal_workflow_run": 33006932756,
        "terminal_workflow_conclusion": "success",
        "terminal_artifact_id": 9623621921,
        "terminal_artifact_digest": (
            "sha256:f13ff28bd9bd998bbb8ebfb4a50893778f7edaea3f09ea052771bd1ba8b2c4e6"
        ),
        "terminal_evidence_identity_sha256": (
            "a591e8af0aa6aa9e3040c002087ea316485f38ed7d43ef30e127346498c02204"
        ),
        "terminal_verdict": "ADMIT",
        "repository": "https://github.com/php/doc-uk",
        "source_commit": "c165db75cc6f81cfdabf754656e73a68940de46c",
        "family": "php.manual.documentation",
        "language": "uk",
        "modality": "technical_documentation",
        "selected_objects": 10,
        "raw_bytes": 59986,
        "raw_bundle_sha256": (
            "8b6e7235083095de65398eba4923aaf37a40808341ac04f8accfdd800e83a5d6"
        ),
        "normalized_bytes": 30510,
        "normalized_bundle_sha256": (
            "8802e3aa3d608305f8612efa9d90fdef65a76cdf518596fb52e3e7e5201db2e1"
        ),
        "license_ids": ["CC-BY-3.0-or-later"],
        "license_sha256": [
            "6ecc1c14303ea6a60706fa2495e7dc0ee9654b9a996f98ccb3ff0a597571aace"
        ],
        "model_training_rights": "ALLOWED",
        "redistribution": "ALLOWED_WITH_ATTRIBUTION",
        "evaluation": "NOT_SEPARATELY_ADMITTED",
        "historical_gates_all_passed": True,
    },
    "next100-030-rustbook-ua-oer": {
        "source_id": "next100-030-rustbook-ua-oer",
        "historical_pr": 478,
        "historical_head": "0bccb897ff72c577535d640e102009958a3d833d",
        "terminal_workflow_run": 33006780092,
        "terminal_workflow_conclusion": "success",
        "terminal_artifact_id": 9623318910,
        "terminal_artifact_digest": (
            "sha256:87b9d3299f863450bff915f780de9f43f41428e6937147a2522cb7dfa7371e4d"
        ),
        "terminal_manifest_sha256": (
            "df9ab2943e140b8735c692eb4ee40c3f10dd03ca843ec9800e0bdf4d7ff664af"
        ),
        "terminal_verdict": "ADMIT",
        "repository": "https://github.com/rust-lang-ua/rustbook_ukrainian",
        "source_commit": "ca2d2e4f4434c661836926017af23bdd40ad4e3d",
        "family": "rust-book.documentation.uk-translation",
        "language": "uk",
        "modality": "open_educational_textbook",
        "selected_objects": 2,
        "raw_bytes": 20198,
        "raw_bundle_sha256": (
            "a51faa914faae31a9c0254577667b1c5e62bf2ac2cde3d8db1115b5f563bc6bb"
        ),
        "normalized_bytes": 18165,
        "normalized_bundle_sha256": (
            "f2c8c39c2edef995e90e8c0be6a8abc1b278ce98b534adf6e912c0c717a9cb1c"
        ),
        "license_ids": ["MIT", "Apache-2.0"],
        "license_sha256": [
            "9ebd13f30b03c699a767424d6e90b7e1e2a4e60ffd6b0d9c6f9fa7c0d3a78b80",
            "ed4a8f25c2867753376d2ab335ffbe28e600be79dc837859b6d648e5e576cd1c",
        ],
        "model_training_rights": "ALLOWED",
        "redistribution": "ALLOWED_WITH_NOTICES",
        "evaluation": "NOT_SEPARATELY_ADMITTED",
        "historical_gates_all_passed": True,
    },
}
EXPECTED_AGGREGATE = {
    "source_count": 2,
    "independent_family_count": 2,
    "normalized_source_bytes": 48675,
    "canonical_capacity_credit_bytes": 0,
    "canonical_family_credit": 0,
    "training_authorized_bytes": 0,
    "authorized_unique_loss_positions": 0,
}
EXPECTED_TRUTH_BOUNDARY = {
    "canonical_corpus_admitted": False,
    "current_global_dedup_executed_for_these_sources": False,
    "fresh_reserved_eval_decontamination_executed": False,
    "post_composition_quality_privacy_executed": False,
    "split_packed_two_clean_builds": False,
    "positive_unique_loss_ledger": False,
    "tokenizer_fit_authorized": False,
    "optimizer_updates": 0,
    "model_training_executed": False,
    "final_test_accessed": False,
    "paid_compute_used": False,
    "learned_20m_promoted": False,
}
REQUIRED_GATES = [
    "GLOBAL_EXACT_NEAR_FRAGMENT_LINEAGE_DEDUP",
    "FRESH_RESERVED_EVALUATION_DECONTAMINATION",
    "POST_COMPOSITION_QUALITY_PRIVACY",
    "BALANCE_AND_FAMILY_CAPS",
    "CLUSTER_SAFE_SPLIT",
    "DETERMINISTIC_PACKING_TWO_CLEAN_BUILDS",
    "POSITIVE_EXACT_UNIQUE_CAUSAL_LOSS_LEDGER",
]
EXPECTED_NOTES = [
    "Historical terminal admission is source-level authority only; it does not transfer old global-dedup or evaluation-decontamination state to current main.",
    "The two source families are distinct from each other but receive zero current canonical family credit until the current composed pipeline consumes them.",
    "No dedicated workflow is added; repository shared CI is the current-main engineering gate.",
]


def canonical_without_identity(obj: dict[str, Any]) -> bytes:
    clone = dict(obj)
    clone.pop("authority_identity_sha256", None)
    rendered = json.dumps(
        clone, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return (rendered + "\n").encode()


def _canonical_value(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("authority contains non-canonical JSON value") from exc


def _require_exact(actual: Any, expected: Any, label: str) -> None:
    if _canonical_value(actual) != _canonical_value(expected):
        raise ValueError(f"{label} drift")


def validate(obj: dict[str, Any]) -> None:
    if type(obj) is not dict or set(obj) != TOP_LEVEL_KEYS:
        raise ValueError("top-level authority schema drift")
    if obj["schema_version"] != "12-6.d03-terminal-ua-source-seals.v1":
        raise ValueError("schema mismatch")
    if obj["worker_id"] != "D03-UA-TERMINAL-SOURCE-SEALS-CURRENT-MAIN":
        raise ValueError("worker mismatch")
    if obj["local_free_only"] is not True:
        raise ValueError("LOCAL_FREE boundary weakened")
    if obj["authority_state"] != "TERMINAL_SOURCE_AUTHORITIES_ZERO_CANONICAL_CREDIT":
        raise ValueError("authority state mismatch")

    identity = obj["authority_identity_sha256"]
    if (
        type(identity) is not str
        or not SHA64.fullmatch(identity)
        or identity != AUTHORITY_IDENTITY
    ):
        raise ValueError("authority identity drift")
    if hashlib.sha256(canonical_without_identity(obj)).hexdigest() != identity:
        raise ValueError("authority self-identity mismatch")

    sources = obj["sources"]
    if type(sources) is not list or len(sources) != 2:
        raise ValueError("expected exactly two terminal sources")
    by_id: dict[str, dict[str, Any]] = {}
    for source in sources:
        if type(source) is not dict or type(source.get("source_id")) is not str:
            raise ValueError("source object schema drift")
        sid = source["source_id"]
        if sid in by_id:
            raise ValueError("duplicate source id")
        by_id[sid] = source
    if set(by_id) != set(EXPECTED_SOURCES):
        raise ValueError("source set drift")
    for sid, expected in EXPECTED_SOURCES.items():
        _require_exact(by_id[sid], expected, f"{sid}: source authority")

    _require_exact(obj["aggregate"], EXPECTED_AGGREGATE, "aggregate or zero-credit boundary")
    _require_exact(
        obj["current_main_truth_boundary"],
        EXPECTED_TRUTH_BOUNDARY,
        "current-main truth boundary",
    )
    _require_exact(obj["required_downstream_gates"], REQUIRED_GATES, "downstream gate order")
    _require_exact(obj["notes"], EXPECTED_NOTES, "authority notes")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config", default="configs/data/d03_terminal_ua_source_seals_v1.json"
    )
    args = ap.parse_args()
    obj = json.loads(Path(args.config).read_text(encoding="utf-8"))
    validate(obj)
    print("D03_UA_TERMINAL_SOURCE_SEALS=PASS")
    print("D03_UA_TERMINAL_SOURCE_BYTES=48675")
    print("D03_UA_TERMINAL_SOURCE_FAMILIES=2")
    print("D03_CURRENT_CANONICAL_CREDIT_BYTES=0")
    print("D03_TRAINING_AUTHORIZED_BYTES=0")


if __name__ == "__main__":
    main()
