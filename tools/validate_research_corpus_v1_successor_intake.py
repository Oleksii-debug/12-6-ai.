"""Fail-closed validator for the Research Corpus V1 successor intake authority."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

DEFAULT_PATH = Path("configs/data/research_corpus_v1_successor_intake_v1.json")

EXPECTED_BASE = {
    "head_sha": "b0523ccbc4b957615aac849d476cfa851be87578",
    "registry_identity_sha256": "917e9bc31b2fa040d25e807ae3c01aa2cce32420752a891caacfb6c830e6632c",
    "source_object_count": 5,
    "independent_family_count": 4,
    "normalized_bytes": 183061,
    "by_stratum_normalized_bytes": {"uk": 88565, "en": 84793, "code": 9703},
}
BASE_FAMILY_COUNTS = {"uk": 1, "en": 1, "code": 2}
EXPECTED_ADDITIONS = {
    "NEXT100-022-DATA-UA-WIKISOURCE": {
        "pr": 455,
        "head_sha": "84c51e42b6daa51796fd20d793b5ef1ff01cc9d2",
        "family_id": "ua.literature.lesia-ukrainka.na-krylah-pisen.1892-lviv",
        "stratum": "uk",
        "normalized_bytes": 1479,
        "record_count": 1,
        "evidence_kind": "authority_identity_sha256",
        "evidence_sha256": "6b443faa7fef777214022028d5fdb356dae0ab1a9b71822b4e16bea8f92cd0d6",
    },
    "NEXT100-026-DATA-UA-CABINET-MINISTRY": {
        "pr": 449,
        "head_sha": "40950a950b60921fd856af2719e1ae2486d9e892",
        "family_id": "ua.kmu.portal.secretariat-news",
        "stratum": "uk",
        "normalized_bytes": 9153,
        "record_count": 6,
        "evidence_kind": "source_manifest_identity_sha256",
        "evidence_sha256": "1f068e6cc5ce3fc4a51d8477acee31fab5a0178e15f49225b57de94c5178f7d9",
    },
    "NEXT100-037-DATA-EN-PYTHON-DOCS": {
        "pr": 467,
        "head_sha": "5a6a495a24bce449334cbc5126d0114f61a9f57c",
        "family_id": "python.cpython.documentation",
        "stratum": "en",
        "normalized_bytes": 17901,
        "record_count": 14,
        "evidence_kind": "authority_identity_sha256",
        "evidence_sha256": "46a00dc70db690ae2b3c4495a75283e7e752bdccb1047d4318c2ebadfa392f0d",
    },
    "NEXT100-034-DATA-EN-NIST": {
        "pr": 472,
        "head_sha": "b7491745b34ac8679baaf69cb96cd609dcbe0a16",
        "family_id": "en.usgov.nist.technical-series",
        "stratum": "en",
        "normalized_bytes": 59358,
        "record_count": 3,
        "evidence_kind": "terminal_payload_sha256",
        "evidence_sha256": "3ffba0fcd08ab42e940b2db12ffafb6f7234ad0bae6f7fe523071497485b9d1c",
    },
    "NEXT100-045-CODE-STARLETTE": {
        "pr": 458,
        "head_sha": "c6756b5ebb6eb1d3bf3de2499167833d99d99a72",
        "family_id": "github:Kludex/starlette",
        "stratum": "code",
        "normalized_bytes": 5274,
        "record_count": 2,
        "evidence_kind": "report_sha256_without_self",
        "evidence_sha256": "c6b210c8977cce4441134ef048ed7dbea1a1e74b295ee96ce70ce5d612962722",
    },
}
_ALLOWED_TRAINING_RIGHTS = {
    "ALLOW",
    "ALLOWED",
    "ALLOWED_PRETRAINING",
    "ALLOWED_WITH_NIST_SOURCE_PROVENANCE",
}


class IntakeValidationError(ValueError):
    """Raised when the frozen intake authority fails closed."""


def _canonical_sha256(payload: dict[str, Any]) -> str:
    value = copy.deepcopy(payload)
    value.pop("intake_identity_sha256", None)
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise IntakeValidationError(message)


def validate_payload(payload: dict[str, Any]) -> None:
    _require(
        payload.get("schema") == "12-6.research-corpus-v1-successor-intake.v1",
        "unexpected schema",
    )
    _require(payload.get("execution_profile") == "LOCAL_FREE", "execution profile drift")
    _require(
        payload.get("intake_identity_sha256") == _canonical_sha256(payload),
        "intake identity mismatch",
    )

    base = payload.get("base_registry")
    _require(isinstance(base, dict), "base_registry must be a mapping")
    for key, expected in EXPECTED_BASE.items():
        _require(base.get(key) == expected, f"base registry drift: {key}")

    additions = payload.get("terminal_additions")
    _require(isinstance(additions, list), "terminal_additions must be a list")
    _require(len(additions) == len(EXPECTED_ADDITIONS), "terminal authority count drift")
    indexed: dict[str, dict[str, Any]] = {}
    for item in additions:
        _require(isinstance(item, dict), "terminal authority row must be a mapping")
        authority = item.get("authority")
        _require(isinstance(authority, str), "terminal authority id missing")
        _require(authority not in indexed, f"duplicate terminal authority: {authority}")
        indexed[authority] = item
    _require(set(indexed) == set(EXPECTED_ADDITIONS), "terminal authority vector drift")

    base_families = set(base.get("families", []))
    added_families: set[str] = set()
    by_stratum = dict(EXPECTED_BASE["by_stratum_normalized_bytes"])
    family_counts = dict(BASE_FAMILY_COUNTS)
    record_count = int(EXPECTED_BASE["source_object_count"])

    for authority, expected in EXPECTED_ADDITIONS.items():
        item = indexed[authority]
        for field in ("pr", "head_sha", "family_id", "stratum", "normalized_bytes", "record_count"):
            _require(item.get(field) == expected[field], f"{authority} drift: {field}")
        evidence = item.get("evidence_binding")
        _require(isinstance(evidence, dict), f"{authority} evidence binding missing")
        _require(evidence.get("kind") == expected["evidence_kind"], f"{authority} evidence kind drift")
        _require(evidence.get("sha256") == expected["evidence_sha256"], f"{authority} evidence hash drift")
        _require(str(item.get("terminal_verdict", "")).startswith("ADMIT"), f"{authority} is not terminal ADMIT")
        _require(item.get("training_rights") in _ALLOWED_TRAINING_RIGHTS, f"{authority} lacks explicit training rights")
        _require(
            item.get("evaluation_rights") in {"NOT_SEPARATELY_ADMITTED", "NOT_AUTHORIZED_BY_THIS_AUTHORITY"},
            f"{authority} evaluation firewall drift",
        )
        _require(item.get("pre_decontamination_only") is True, f"{authority} scope drift")
        family_id = str(item["family_id"])
        _require(family_id not in base_families, f"{authority} duplicates base family")
        _require(family_id not in added_families, f"duplicate added family: {family_id}")
        added_families.add(family_id)
        stratum = str(item["stratum"])
        _require(stratum in by_stratum, f"unsupported stratum: {stratum}")
        by_stratum[stratum] += int(item["normalized_bytes"])
        family_counts[stratum] += 1
        record_count += int(item["record_count"])

    projection = payload.get("pre_decontamination_projection")
    _require(isinstance(projection, dict), "projection must be a mapping")
    expected_total = sum(by_stratum.values())
    expected_family_count = int(EXPECTED_BASE["independent_family_count"]) + len(added_families)
    _require(projection.get("candidate_record_count") == record_count, "record count drift")
    _require(projection.get("normalized_bytes") == expected_total, "normalized-byte total drift")
    _require(projection.get("by_stratum_normalized_bytes") == by_stratum, "stratum byte totals drift")
    _require(projection.get("independent_family_count") == expected_family_count, "family total drift")
    _require(projection.get("family_count_by_stratum") == family_counts, "stratum family counts drift")
    _require(min(family_counts.values()) >= 2, "pre-decontamination family minimum fails")
    target = projection.get("target_unique_normalized_bytes")
    _require(target == 20_000_000, "target capacity drift")
    _require(projection.get("normalized_byte_shortfall") == target - expected_total, "capacity shortfall drift")
    _require(projection.get("hard_two_family_minimum") == "PASS_PRE_DECONTAMINATION_ONLY", "family gate claim drift")

    gates = payload.get("gates")
    _require(isinstance(gates, dict), "gates must be a mapping")
    _require(gates.get("candidate_authority_inventory_frozen") is True, "inventory not frozen")
    _require(gates.get("corpus_terminal") is False, "corpus must remain nonterminal")
    _require(gates.get("long_training_authorized") is False, "long training must stay blocked")
    _require(gates.get("training_authorized_loss_positions") == 0, "training exposure must remain zero before materialization")
    _require(gates.get("exact_candidate_materialization") == "BLOCKED_NOT_MATERIALIZED", "materialization boundary drift")

    truth = payload.get("truth_boundary")
    _require(isinstance(truth, dict), "truth_boundary must be a mapping")
    for field in ("corpus_admission_claimed", "training_executed", "paid_compute_used"):
        _require(truth.get(field) is False, f"truth boundary violated: {field}")
    _require(truth.get("model_results_used_to_select_sources") is False, "model-result-guided source selection is forbidden")
    _require(truth.get("terminal_source_authority_does_not_equal_corpus_admission") is True, "source/corpus authority boundary removed")


def validate_file(path: Path = DEFAULT_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise IntakeValidationError("top-level payload must be a mapping")
    validate_payload(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args()
    payload = validate_file(args.path)
    projection = payload["pre_decontamination_projection"]
    print(
        "PASS",
        f"identity={payload['intake_identity_sha256']}",
        f"records={projection['candidate_record_count']}",
        f"families={projection['independent_family_count']}",
        f"normalized_bytes={projection['normalized_bytes']}",
        "training_authorized_loss_positions=0",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
