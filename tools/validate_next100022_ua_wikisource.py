#!/usr/bin/env python3
"""Validate NEXT100-022 Ukrainian Wikisource rights/snapshot authority."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/data/next100022_ua_wikisource_candidate_v1.json"


class QualificationError(ValueError):
    pass


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise QualificationError(f"{path} must contain a JSON object")
    return value


def validate(config_path: Path = CONFIG) -> dict[str, Any]:
    cfg = _load_json(config_path)

    if cfg.get("schema_version") != "12-6.next100022-ua-wikisource-qualification.v1":
        raise QualificationError("unexpected qualification schema")
    if cfg.get("worker_id") != "NEXT100-022-DATA-UA-WIKISOURCE":
        raise QualificationError("worker identity drift")
    if cfg.get("execution_class") != "LOCAL_FREE" or cfg.get("training_executed") is not False:
        raise QualificationError("LOCAL_FREE/no-training boundary violated")
    if cfg.get("verdict") != "ADMIT_BOUNDED_PD_EDITION_SNAPSHOT":
        raise QualificationError("terminal verdict drift")
    if cfg.get("canonical_registry_mutated") is not False:
        raise QualificationError("source qualification must not silently mutate canonical registry")

    expected_identity = cfg.get("authority_identity_sha256")
    tmp = copy.deepcopy(cfg)
    tmp.pop("authority_identity_sha256", None)
    if _sha256(_canonical_json_bytes(tmp)) != expected_identity:
        raise QualificationError("qualification authority identity mismatch")

    snapshot = cfg["snapshot"]
    payload_path = ROOT / snapshot["path"]
    payload = payload_path.read_bytes()
    if len(payload) != snapshot["bytes"] or _sha256(payload) != snapshot["sha256"]:
        raise QualificationError("immutable snapshot byte identity mismatch")
    text = payload.decode("utf-8")
    if "\r" in text or not text.endswith("\n") or text.endswith("\n\n"):
        raise QualificationError("snapshot LF/final-newline contract violated")
    if unicodedata.normalize("NFC", text) != text:
        raise QualificationError("snapshot must be NFC")
    if "\u00a0" in text:
        raise QualificationError("NBSP stanza markers must be normalized out")

    attr = cfg["attribution"]
    notice = (ROOT / attr["notice_path"]).read_bytes()
    if _sha256(notice) != attr["notice_sha256"]:
        raise QualificationError("attribution notice identity mismatch")
    if not attr["must_preserve_on_redistribution"]:
        raise QualificationError("redistribution attribution must remain mandatory")

    source = cfg["source"]
    if source["page_revision_id"] != 560107 or source["index_revision_id"] != 729499:
        raise QualificationError("Wikisource permanent revision drift")
    if source["proofread_status"] != "APPROVED_PAGE_WITH_FULLY_VERIFIED_INDEX":
        raise QualificationError("proofread/verified source status drift")

    edition = cfg["underlying_edition"]
    if edition["publication_year"] != 1892 or edition["author_death_date"] != "1913-08-01":
        raise QualificationError("underlying edition rights facts drift")
    if edition["commons_file_sha1"] != "4ae5ba96e7e76d7fb26b37d5277a2e82c1443407":
        raise QualificationError("Commons scan identity drift")

    rights = cfg["rights"]
    if rights["underlying_work"]["status"] != "PUBLIC_DOMAIN":
        raise QualificationError("underlying work must be explicitly public domain")
    uses = rights["uses"]
    for field in ("acquisition", "storage", "analysis", "model_training_permission"):
        if uses[field] != "ALLOWED":
            raise QualificationError(f"{field} must be ALLOWED")
    if uses["redistribution"] != "ALLOWED_WITH_ATTRIBUTION_AND_LICENSE_NOTICE":
        raise QualificationError("redistribution conditions weakened")
    if uses["evaluation"] != "NOT_SEPARATELY_ADMITTED":
        raise QualificationError("evaluation may not be inferred from training permission")
    if rights["model_output_license_inference"] != "NONE":
        raise QualificationError("model-output licensing may not be inferred here")
    scope = rights["scope_boundary"]
    for required in ("Generic Ukrainian Wikisource", "modern/original translations", "CC-BY-SA-only"):
        if required not in scope:
            raise QualificationError("generic licensed-content fail-close boundary weakened")

    family = cfg["family_lineage"]
    if family["wikimedia_platform_is_not_family_identity"] is not True:
        raise QualificationError("hosting platform must not define source-family identity")
    if family["family_basis"] != "CANONICAL_UNDERLYING_EDITION_DOCUMENT_LINEAGE_NOT_HOSTING_DOMAIN":
        raise QualificationError("family-lineage policy drift")

    registry_parent = cfg["registry_parent"]
    registry = _load_json(ROOT / registry_parent["path"])
    if registry.get("registry_identity_sha256") != registry_parent["registry_identity_sha256"]:
        raise QualificationError("live-parent registry identity changed; RETEST required")
    if registry.get("source_count") != registry_parent["source_count"]:
        raise QualificationError("live-parent source count changed; RETEST required")
    if registry.get("independent_source_family_count") != registry_parent["independent_source_family_count"]:
        raise QualificationError("live-parent family count changed; RETEST required")
    current_families = {row["key"] for row in registry["family_deduplication"]["family_rows"]}
    if family["family_id"] in current_families:
        raise QualificationError("candidate family already exists in current registry")
    if family["independent_from_registry_parent"] is not True:
        raise QualificationError("family independence flag must match checked parent registry")

    evaluation = cfg["evaluation_exclusion"]
    nfkc_ws = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()
    if _sha256(nfkc_ws.encode("utf-8")) != evaluation["data25_nfkc_whitespace_collapse_sha256"]:
        raise QualificationError("evaluation-normalized candidate hash drift")
    if evaluation["exact_hash_collision_observed"] is not False:
        raise QualificationError("known exact evaluation collision blocks snapshot qualification")
    if evaluation["task_reserved_new_evaluation_material"] is not False:
        raise QualificationError("this worker may not reserve evaluation material")
    if evaluation["near_match_scan"] != "NOT_RUN_EVALUATION_PAYLOAD_ACCESS_FORBIDDEN_TO_THIS_WORKER":
        raise QualificationError("evaluation access boundary drift")
    if evaluation["corpus_training_selection"] != "BLOCKED_UNTIL_STANDARD_DATA232_DATA299_NEAR_MATCH_DECONTAMINATION":
        raise QualificationError("corpus training must fail closed until standard decontamination")

    privacy = cfg["privacy"]
    if privacy["contributor_usernames_copied"] is not False:
        raise QualificationError("contributor usernames must not be copied into the snapshot")

    return {
        "status": "PASS",
        "verdict": cfg["verdict"],
        "authority_identity_sha256": expected_identity,
        "snapshot_sha256": snapshot["sha256"],
        "snapshot_bytes": snapshot["bytes"],
        "family_id": family["family_id"],
        "rights_training_permission": uses["model_training_permission"],
        "redistribution": uses["redistribution"],
        "evaluation": uses["evaluation"],
        "corpus_training_selection": evaluation["corpus_training_selection"],
    }


def main() -> int:
    try:
        result = validate()
    except (OSError, json.JSONDecodeError, UnicodeError, QualificationError, KeyError, TypeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
