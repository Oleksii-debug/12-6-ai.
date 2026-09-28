from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
SEAL_PATH = ROOT / "configs/data/next100_034_nist_terminal_authority_v2.json"
DETAIL_PATH = ROOT / "configs/data/next100_034_nist_technical_series_authority_v1.json"
RIGHTS_PATH = (
    ROOT
    / "data/external/rights-evidence/next100-034/nist-technical-series-rights-20260826.txt"
)

EXPECTED_SEAL_SHA256 = "3ffba0fcd08ab42e940b2db12ffafb6f7234ad0bae6f7fe523071497485b9d1c"
EXPECTED_DETAIL_BLOB = "5341f043dc9b98da530a58ce70bc2af530d9be12"
EXPECTED_RIGHTS_SHA256 = "aef587d83640eab2c0ac8a8c226657237e96becd6d0cf12804fd56e03bdce7e1"
EXPECTED_FAMILY = "en.usgov.nist.technical-series"
EXPECTED_IDS = ("NIST.SP.800-204", "NIST.SP.800-204C", "NIST.SP.800-215")
EXPECTED_NORMALIZED_BYTES = 59358
EXPECTED_RAW_BYTES = 2620454


class NistAuthorityError(ValueError):
    pass


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise NistAuthorityError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def _reject_constant(value: str) -> None:
    raise NistAuthorityError(f"non-finite JSON number: {value}")


def _strict_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise NistAuthorityError(f"non-finite JSON number: {value}")
    return parsed


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
            parse_float=_strict_float,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise NistAuthorityError(f"cannot read {path}: {exc}") from exc
    if type(value) is not dict:
        raise NistAuthorityError("authority root must be an object")
    return value


def _git_blob_sha1(data: bytes) -> str:
    prefix = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(prefix + data).hexdigest()


def _is_int(value: Any) -> bool:
    return type(value) is int


def validate_nist_source_authority() -> dict[str, Any]:
    seal = _load(SEAL_PATH)
    detail_bytes = DETAIL_PATH.read_bytes()
    detail = _load(DETAIL_PATH)
    rights_bytes = RIGHTS_PATH.read_bytes()

    if seal.get("schema_version") != "12-6.next100-034-nist-terminal-authority.v2":
        raise NistAuthorityError("seal schema drift")
    if seal.get("terminal_status") != "ADMIT" or seal.get("local_free_only") is not True:
        raise NistAuthorityError("terminal/local-free boundary drift")
    seal_core = dict(seal)
    embedded_identity = seal_core.pop("terminal_payload_sha256", None)
    actual_identity = hashlib.sha256(
        json.dumps(seal_core, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if embedded_identity != EXPECTED_SEAL_SHA256 or actual_identity != embedded_identity:
        raise NistAuthorityError("terminal authority identity drift")
    if _git_blob_sha1(detail_bytes) != EXPECTED_DETAIL_BLOB:
        raise NistAuthorityError("supporting authority blob drift")
    if hashlib.sha256(rights_bytes).hexdigest() != EXPECTED_RIGHTS_SHA256:
        raise NistAuthorityError("rights evidence drift")

    family = seal.get("family")
    if type(family) is not dict:
        raise NistAuthorityError("family object missing")
    if family.get("family_id") != EXPECTED_FAMILY:
        raise NistAuthorityError("family identity drift")
    if family.get("independent_family_count") != 1 or type(
        family.get("independent_family_count")
    ) is not int:
        raise NistAuthorityError("family count drift")
    if family.get("publication_count") != 3 or type(family.get("publication_count")) is not int:
        raise NistAuthorityError("publication count drift")

    rows = seal.get("admit")
    if type(rows) is not list or len(rows) != 3:
        raise NistAuthorityError("admitted object cardinality drift")
    ids = tuple(row.get("publication_id") for row in rows if type(row) is dict)
    if ids != EXPECTED_IDS:
        raise NistAuthorityError("admitted object identity/order drift")

    norm_total = 0
    raw_total = 0
    raw_hashes: set[str] = set()
    normalized_hashes: set[str] = set()
    for row in rows:
        if type(row) is not dict:
            raise NistAuthorityError("admitted row must be an object")
        nb = row.get("normalized_utf8_bytes")
        rb = row.get("raw_bytes")
        if not _is_int(nb) or not _is_int(rb) or nb <= 0 or rb <= 0:
            raise NistAuthorityError("byte counters must be positive strict integers")
        norm_total += nb
        raw_total += rb
        raw_hashes.add(str(row.get("raw_sha256")))
        normalized_hashes.add(str(row.get("normalized_sha256")))
    if norm_total != EXPECTED_NORMALIZED_BYTES or raw_total != EXPECTED_RAW_BYTES:
        raise NistAuthorityError("byte totals drift")
    if len(raw_hashes) != 3 or len(normalized_hashes) != 3:
        raise NistAuthorityError("duplicate source identity")

    detail_rows = detail.get("admit")
    if type(detail_rows) is not list:
        raise NistAuthorityError("supporting authority rows missing")
    detail_by_id = {r.get("publication_id"): r for r in detail_rows if type(r) is dict}
    for row in rows:
        source = detail_by_id.get(row["publication_id"])
        if type(source) is not dict:
            raise NistAuthorityError("supporting source row missing")
        for key in ("raw_bytes", "raw_sha256", "normalized_utf8_bytes", "normalized_sha256"):
            if row.get(key) != source.get(key) or type(row.get(key)) is not type(source.get(key)):
                raise NistAuthorityError("supporting identity drift")

    rights = seal.get("rights")
    if type(rights) is not dict:
        raise NistAuthorityError("rights boundary missing")
    if rights.get("model_training") != "ALLOWED_WITH_NIST_SOURCE_PROVENANCE":
        raise NistAuthorityError("training rights drift")
    if rights.get("evaluation") != "NOT_SEPARATELY_ADMITTED":
        raise NistAuthorityError("evaluation purpose drift")
    if rights.get("third_party_nist_publications") != "RETEST_DOCUMENT_SPECIFICALLY":
        raise NistAuthorityError("third-party caveat drift")
    if rights.get("standard_reference_data") != "RETEST_LICENSE_SPECIFICALLY":
        raise NistAuthorityError("SRD caveat drift")

    if seal.get("corpus_integration") != (
        "NOT_INTEGRATED_REQUIRES_SUCCESSOR_CORPUS_CONTRACT_AND_CANONICAL_DEDUP"
    ):
        raise NistAuthorityError("corpus integration overclaim")

    return {
        "schema": "12-6.nist-source-authority-current-main.v1",
        "historical_terminal_authority_sha256": EXPECTED_SEAL_SHA256,
        "family_id": EXPECTED_FAMILY,
        "publication_count": 3,
        "normalized_source_bytes": EXPECTED_NORMALIZED_BYTES,
        "raw_source_bytes": EXPECTED_RAW_BYTES,
        "canonical_capacity_credit_bytes": 0,
        "authorized_optimized_target_exposure": 0,
        "tokenizer_fit_authorized": False,
        "optimizer_updates_executed_on_real_targets": 0,
        "training_executed": False,
        "learned_weights_created": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
        "foreign_pretrained_weights": False,
        "current_corpus_external_llm_free_claimed_by_this_authority": False,
        "downstream_global_dedup_required": True,
        "evaluation_separately_admitted": False,
    }
