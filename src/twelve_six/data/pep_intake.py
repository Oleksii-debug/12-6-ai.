from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PEP_PATH_RE = re.compile(r"^peps/pep-\d{4}\.rst$")
_HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_ADORNMENT = set("=-`:'\"~^_*+#<>")
_OPL_MARKERS = (
    "open publication license",
    "https://spdx.org/licenses/opubl-1.0.html",
    "http://www.opencontent.org/openpub/",
)
_PUBLIC_DOMAIN_RE = re.compile(
    r"\bthis document (?:has been |is )?placed in the public domain\b"
    r"|\bthis document is in the public domain\b",
    re.IGNORECASE,
)
_EXPECTED_UPSTREAM = {
    "repository": "python/peps",
    "revision": "24419b92ae550bf2878716f57c257cba00d3c1a1",
    "tree_sha1": "fa6b95d941e6fbefc473fbcc50ca93458cdf0857",
}
_EXPECTED_AUDIT = {
    "registry_path": "configs/data/common_pile_source_rights_v1.json",
    "registry_blob_sha1": "7b4d6828288672bf25c551e85a5d7f7399e8ef0f",
    "audited_code_revision": "9457f04a14cb2355ab00023420369d46ffd4a395",
    "collector_path": "sources/pep/to_dolma.py",
    "collector_blob_sha1": "6bbb677559d11fb7b3473b936157ff3ce3afafda",
    "source_key": "python_enhancement_proposals",
}
_EXPECTED_SENTINELS = (
    "peps/pep-0437.rst",
    "peps/pep-0483.rst",
    "peps/pep-0551.rst",
    "peps/pep-0578.rst",
    "peps/pep-3145.rst",
)
_TOP_LEVEL_KEYS = {
    "authorized_unique_loss_positions",
    "canonical_capacity_credit_bytes",
    "common_pile_audit",
    "contract_identity_sha256",
    "corpus_admitted",
    "execution_profile",
    "known_opl_sentinels",
    "model_training_permitted",
    "paid_compute_authorized",
    "required_downstream_gates",
    "rights_policy",
    "schema_version",
    "selection_policy",
    "source_family",
    "tokenizer_fit_permitted",
    "training_authorized_bytes",
    "upstream",
}
_SELECTION_LIMITS = {
    "max_documents": 128,
    "max_examined_documents": 192,
    "max_total_source_bytes": 5_000_000,
}


class PepIntakeError(ValueError):
    """Fail-closed PEP intake contract violation."""


@dataclass(frozen=True)
class PepTreeEntry:
    path: str
    blob_sha1: str
    size: int


@dataclass(frozen=True)
class LicenseDecision:
    status: str
    reason: str


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob_sha1(data: bytes) -> str:
    prefix = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(prefix + data).hexdigest()


def self_identity(value: Mapping[str, Any], field: str) -> str:
    clone = deepcopy(dict(value))
    clone.pop(field, None)
    return sha256_bytes((canonical_json(clone) + "\n").encode("utf-8"))


def _require_hex(value: object, length: int, field: str) -> str:
    if not isinstance(value, str):
        raise PepIntakeError(f"{field} must be a string")
    pattern = _HEX40_RE if length == 40 else _HEX64_RE
    if not pattern.fullmatch(value):
        raise PepIntakeError(f"{field} must be lowercase {length}-hex")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    actual = set(value)
    if actual != expected:
        raise PepIntakeError(f"{field} keys drift")


def validate_config(config: Mapping[str, Any]) -> None:
    _require_exact_keys(config, _TOP_LEVEL_KEYS, "config")
    expected = {
        "schema_version": "12-6.d03-pep-public-domain-intake.v1",
        "source_family": "en.python.peps.public-domain",
        "execution_profile": "LOCAL_FREE",
        "training_authorized_bytes": 0,
        "authorized_unique_loss_positions": 0,
        "canonical_capacity_credit_bytes": 0,
        "corpus_admitted": False,
        "tokenizer_fit_permitted": False,
        "model_training_permitted": False,
        "paid_compute_authorized": False,
    }
    for field, value in expected.items():
        if config.get(field) != value or type(config.get(field)) is not type(value):
            raise PepIntakeError(f"{field} drift")

    upstream = config.get("upstream")
    if not isinstance(upstream, Mapping):
        raise PepIntakeError("upstream missing")
    _require_exact_keys(
        upstream,
        {"raw_url_template", "repository", "revision", "tree_sha1", "tree_url"},
        "upstream",
    )
    for field, expected_value in _EXPECTED_UPSTREAM.items():
        if upstream.get(field) != expected_value:
            raise PepIntakeError(f"upstream.{field} drift")
    _require_hex(upstream.get("revision"), 40, "upstream.revision")
    _require_hex(upstream.get("tree_sha1"), 40, "upstream.tree_sha1")
    if upstream.get("tree_url") != (
        "https://api.github.com/repos/python/peps/git/trees/"
        f"{_EXPECTED_UPSTREAM['tree_sha1']}?recursive=1"
    ):
        raise PepIntakeError("upstream.tree_url drift")
    if upstream.get("raw_url_template") != (
        "https://raw.githubusercontent.com/python/peps/"
        f"{_EXPECTED_UPSTREAM['revision']}/{{path}}"
    ):
        raise PepIntakeError("upstream.raw_url_template drift")

    audit = config.get("common_pile_audit")
    if not isinstance(audit, Mapping):
        raise PepIntakeError("common_pile_audit missing")
    _require_exact_keys(audit, set(_EXPECTED_AUDIT), "common_pile_audit")
    for field, expected_value in _EXPECTED_AUDIT.items():
        if audit.get(field) != expected_value:
            raise PepIntakeError(f"common_pile_audit.{field} drift")
    for field in ("registry_blob_sha1", "audited_code_revision", "collector_blob_sha1"):
        _require_hex(audit.get(field), 40, f"common_pile_audit.{field}")

    policy = config.get("selection_policy")
    if not isinstance(policy, Mapping):
        raise PepIntakeError("selection_policy missing")
    _require_exact_keys(
        policy,
        {
            "max_documents",
            "max_examined_documents",
            "max_total_source_bytes",
            "ordering",
            "path_regex",
        },
        "selection_policy",
    )
    if policy.get("path_regex") != PEP_PATH_RE.pattern:
        raise PepIntakeError("selection_policy.path_regex drift")
    for field, ceiling in _SELECTION_LIMITS.items():
        value = policy.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or not 0 < value <= ceiling:
            raise PepIntakeError(f"invalid {field}")
    if policy["max_examined_documents"] < policy["max_documents"]:
        raise PepIntakeError("max_examined_documents below max_documents")
    if policy.get("ordering") != "DESC_SIZE_THEN_PATH":
        raise PepIntakeError("selection ordering drift")

    sentinels = config.get("known_opl_sentinels")
    if sentinels != list(_EXPECTED_SENTINELS):
        raise PepIntakeError("known_opl_sentinels drift")

    rights = config.get("rights_policy")
    if not isinstance(rights, Mapping):
        raise PepIntakeError("rights_policy missing")
    _require_exact_keys(
        rights,
        {
            "ambiguous_copyright_section_action",
            "basis",
            "blanket_repository_rights_claimed",
            "legal_conclusion_claimed",
            "opl_markers",
            "require_explicit_public_domain_in_copyright_section",
        },
        "rights_policy",
    )
    rights_expected = {
        "ambiguous_copyright_section_action": "QUARANTINE",
        "basis": "DOCUMENT_LICENSE_FILTER",
        "blanket_repository_rights_claimed": False,
        "legal_conclusion_claimed": False,
        "opl_markers": list(_OPL_MARKERS),
        "require_explicit_public_domain_in_copyright_section": True,
    }
    for field, expected_value in rights_expected.items():
        if rights.get(field) != expected_value or type(rights.get(field)) is not type(expected_value):
            raise PepIntakeError(f"rights_policy.{field} drift")

    required = config.get("required_downstream_gates")
    if required != [
        "privacy",
        "quality",
        "global_exact_near_fragment_lineage_dedup",
        "reserved_evaluation_decontamination",
        "balance_family_caps",
        "cluster_safe_split",
        "deterministic_pack_two_clean_builds",
        "positive_unique_loss_ledger",
    ]:
        raise PepIntakeError("required downstream gates drift")

    identity = _require_hex(config.get("contract_identity_sha256"), 64, "contract identity")
    if self_identity(config, "contract_identity_sha256") != identity:
        raise PepIntakeError("contract identity mismatch")


def parse_tree(config: Mapping[str, Any], payload: Mapping[str, Any]) -> list[PepTreeEntry]:
    validate_config(config)
    upstream = config["upstream"]
    if payload.get("sha") != upstream["tree_sha1"]:
        raise PepIntakeError("upstream tree identity mismatch")
    if payload.get("truncated") is not False:
        raise PepIntakeError("upstream tree must be complete")
    raw_entries = payload.get("tree")
    if not isinstance(raw_entries, list):
        raise PepIntakeError("upstream tree entries missing")

    entries: list[PepTreeEntry] = []
    seen: set[str] = set()
    for item in raw_entries:
        if not isinstance(item, Mapping):
            raise PepIntakeError("malformed tree entry")
        path = item.get("path")
        if not isinstance(path, str) or not PEP_PATH_RE.fullmatch(path):
            continue
        if path in seen:
            raise PepIntakeError(f"duplicate tree path: {path}")
        seen.add(path)
        if item.get("type") != "blob" or item.get("mode") != "100644":
            raise PepIntakeError(f"PEP is not a regular blob: {path}")
        blob_sha = _require_hex(item.get("sha"), 40, f"{path}.sha")
        size = item.get("size")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise PepIntakeError(f"invalid PEP size: {path}")
        entries.append(PepTreeEntry(path, blob_sha, size))

    if not entries:
        raise PepIntakeError("no PEP documents discovered")
    sentinels = set(config["known_opl_sentinels"])
    if not sentinels.issubset(seen):
        missing = sorted(sentinels - seen)
        raise PepIntakeError(f"missing OPL sentinels: {missing}")
    return sorted(entries, key=lambda item: (-item.size, item.path))


def _is_adornment(line: str, title: str) -> bool:
    stripped = line.strip()
    return (
        len(stripped) >= max(3, len(title.strip()))
        and len(set(stripped)) == 1
        and stripped[0] in _ADORNMENT
    )


def copyright_sections(text: str) -> list[str]:
    lines = text.splitlines()
    headings: list[tuple[int, str]] = []
    for index in range(len(lines) - 1):
        title = lines[index].strip()
        if title and _is_adornment(lines[index + 1], title):
            headings.append((index, title.casefold()))
    copyright_indices = [idx for idx, title in headings if title == "copyright"]
    sections: list[str] = []
    for start in copyright_indices:
        next_headings = [idx for idx, _ in headings if idx > start]
        end = min(next_headings) if next_headings else len(lines)
        sections.append("\n".join(lines[start + 2 : end]).strip())
    return sections


def classify_license(text: str) -> LicenseDecision:
    sections = copyright_sections(text)
    if len(sections) != 1:
        return LicenseDecision("QUARANTINE", "copyright_section_missing_or_ambiguous")
    section = sections[0].casefold()
    if any(marker in section for marker in _OPL_MARKERS):
        return LicenseDecision("EXCLUDE", "open_publication_license")
    if not _PUBLIC_DOMAIN_RE.search(section):
        return LicenseDecision("QUARANTINE", "explicit_public_domain_statement_missing")
    return LicenseDecision("ACCEPT", "explicit_public_domain_copyright_section")


def normalize_text(raw: bytes) -> str:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PepIntakeError("PEP payload is not strict UTF-8") from exc
    if "\x00" in text:
        raise PepIntakeError("PEP payload contains NUL")
    return unicodedata.normalize("NFC", text).strip()


def verify_payload(entry: PepTreeEntry, raw: bytes) -> str:
    if len(raw) != entry.size:
        raise PepIntakeError(f"size mismatch for {entry.path}")
    if git_blob_sha1(raw) != entry.blob_sha1:
        raise PepIntakeError(f"Git blob mismatch for {entry.path}")
    return normalize_text(raw)


def _inventory_identity(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = canonical_json(list(rows)) + "\n"
    return sha256_bytes(payload.encode("utf-8"))


def materialize(
    config: Mapping[str, Any],
    tree_payload: Mapping[str, Any],
    fetch_raw: Callable[[str], bytes],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    entries = parse_tree(config, tree_payload)
    by_path = {entry.path: entry for entry in entries}
    sentinels = set(config["known_opl_sentinels"])
    policy = config["selection_policy"]
    max_docs = policy["max_documents"]
    max_examined = policy["max_examined_documents"]
    max_bytes = policy["max_total_source_bytes"]
    template = config["upstream"]["raw_url_template"]

    candidates: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    accepted_bytes = 0
    examined = 0
    sentinel_status: dict[str, str] = {}

    ordered_paths = [entry.path for entry in entries]
    for sentinel in sorted(sentinels):
        ordered_paths.remove(sentinel)
    ordered_paths = sorted(sentinels) + ordered_paths

    for path in ordered_paths:
        if examined >= max_examined and path not in sentinels:
            break
        if len(candidates) >= max_docs and sentinels.issubset(sentinel_status):
            break
        entry = by_path[path]
        raw = fetch_raw(template.format(path=path))
        text = verify_payload(entry, raw)
        decision = classify_license(text)
        text_bytes = text.encode("utf-8")
        row = {
            "path": path,
            "git_blob_sha1": entry.blob_sha1,
            "source_bytes": entry.size,
            "normalized_utf8_bytes": len(text_bytes),
            "normalized_sha256": sha256_bytes(text_bytes),
            "license_status": decision.status,
            "license_reason": decision.reason,
        }
        evidence.append(row)
        examined += 1
        if path in sentinels:
            sentinel_status[path] = decision.status
            if decision.status != "EXCLUDE":
                raise PepIntakeError(f"OPL sentinel did not fail closed: {path}")
            continue
        if decision.status != "ACCEPT":
            continue
        if accepted_bytes + len(text_bytes) > max_bytes:
            continue
        candidates.append(
            {
                "record_id": f"pep:{path}",
                "source_family": config["source_family"],
                "source_revision": config["upstream"]["revision"],
                "source_path": path,
                "source_git_blob_sha1": entry.blob_sha1,
                "text": text,
                "text_sha256": row["normalized_sha256"],
                "utf8_bytes": row["normalized_utf8_bytes"],
                "training_eligible": False,
            }
        )
        accepted_bytes += len(text_bytes)

    if not sentinels.issubset(sentinel_status):
        raise PepIntakeError("not all OPL sentinels were verified")
    if not candidates:
        raise PepIntakeError("bounded materialization retained zero candidates")

    inventory_rows = [
        {
            "record_id": row["record_id"],
            "source_git_blob_sha1": row["source_git_blob_sha1"],
            "text_sha256": row["text_sha256"],
            "utf8_bytes": row["utf8_bytes"],
        }
        for row in candidates
    ]
    evidence_rows = sorted(evidence, key=lambda row: row["path"])
    report = {
        "schema_version": "12-6.d03-pep-public-domain-materialization.v1",
        "status": "CANDIDATE_MATERIALIZED_ZERO_CREDIT",
        "source_family": config["source_family"],
        "upstream_revision": config["upstream"]["revision"],
        "upstream_tree_sha1": config["upstream"]["tree_sha1"],
        "contract_identity_sha256": config["contract_identity_sha256"],
        "examined_documents": examined,
        "accepted_documents": len(candidates),
        "accepted_normalized_utf8_bytes": accepted_bytes,
        "candidate_inventory_identity_sha256": _inventory_identity(inventory_rows),
        "rights_evidence_identity_sha256": _inventory_identity(evidence_rows),
        "known_opl_sentinels_verified_excluded": sorted(sentinel_status),
        "training_authorized_bytes": 0,
        "authorized_unique_loss_positions": 0,
        "corpus_admitted": False,
        "privacy_gate": "NOT_RUN",
        "quality_gate": "NOT_RUN",
        "global_dedup_gate": "NOT_RUN",
        "reserved_evaluation_decontamination_gate": "NOT_RUN",
        "split_gate": "NOT_RUN",
        "packing_gate": "NOT_RUN",
        "model_training_executed": False,
        "paid_compute_used": False,
    }
    return candidates, report


def write_materialization(
    output_dir: Path,
    candidates: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "pep-candidate.jsonl"
    report_path = output_dir / "pep-report.json"
    rows = [canonical_json(dict(row)) for row in candidates]
    candidate_path.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
