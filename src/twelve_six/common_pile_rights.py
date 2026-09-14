"""Fail-closed validation for the Common Pile v0.1 source-rights audit."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

REGISTRY_ID = "COMMON-PILE-SOURCE-RIGHTS-V1"
EXPECTED_STATUS = "CANDIDATE_SOURCE_RIGHTS_AUDIT"
EXPECTED_SOURCE_KEYS = frozenset(
    {
        "caselaw_access_project",
        "arxiv_abstracts",
        "arxiv_papers",
        "biodiversity_heritage_library",
        "cccc",
        "data_provenance_initiative",
        "doab",
        "foodista",
        "github_archive",
        "library_of_congress",
        "libretexts",
        "news",
        "oercommons",
        "pes2o",
        "pre_1929_books",
        "pressbooks",
        "project_gutenberg",
        "public_domain_review",
        "pubmed",
        "python_enhancement_proposals",
        "regulations",
        "stackexchange",
        "stackv2",
        "ubuntu_irc",
        "uk_hansard",
        "usgpo",
        "uspto",
        "wikimedia",
        "wikiteam",
        "youtube",
    }
)
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class CommonPileRightsError(ValueError):
    """Raised when the audit contract stops being fail-closed."""


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def registry_identity(payload: dict[str, Any]) -> str:
    """Return the registry identity excluding the identity field itself."""
    body = dict(payload)
    body.pop("registry_identity_sha256", None)
    return hashlib.sha256(_canonical_bytes(body)).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CommonPileRightsError(message)


def validate_registry(payload: dict[str, Any]) -> str:
    """Validate the non-authorizing Common Pile source-rights registry."""
    _require(payload.get("schema_version") == 1, "schema_version must be 1")
    _require(payload.get("registry_id") == REGISTRY_ID, "unexpected registry_id")
    _require(payload.get("status") == EXPECTED_STATUS, "registry may not claim terminality")

    project = payload.get("project_authority")
    _require(isinstance(project, dict), "project_authority must be an object")
    _require(project.get("swarm_protocol") == "SWARM-300-V2", "wrong swarm protocol")
    _require(project.get("control_issue") == 723, "wrong control issue")
    _require(project.get("claim_issue") == 746, "wrong claim issue")
    _require(project.get("parent_issue") == 720, "wrong parent issue")
    _require(bool(HEX40.fullmatch(str(project.get("base_sha", "")))), "mutable project base")

    upstream = payload.get("upstream_authority")
    _require(isinstance(upstream, dict), "upstream_authority must be an object")
    _require(upstream.get("release_name") == "Common Pile v0.1", "wrong release")
    _require(upstream.get("raw_collection_item_count") == 30, "wrong source-family count")
    _require(upstream.get("paper_arxiv_id") == "2506.05209", "wrong paper identity")
    _require(
        upstream.get("code_repository") == "https://github.com/r-three/common-pile",
        "wrong upstream repository",
    )
    _require(upstream.get("code_license") == "MIT", "wrong audited code license")
    _require(
        bool(HEX40.fullmatch(str(upstream.get("audited_code_commit", "")))),
        "audited code ref must be an immutable commit",
    )
    _require(
        upstream.get("code_ref_role") == "CURRENT_AUDIT_SNAPSHOT_NOT_ASSERTED_AS_V01_RELEASE_CUT",
        "code snapshot may not be relabeled as the v0.1 release cut",
    )

    policy = payload.get("global_policy")
    _require(isinstance(policy, dict), "global_policy must be an object")
    for field in (
        "dataset_package_license_is_training_authority",
        "repository_code_license_is_dataset_license",
        "canonical_training_authorized",
        "bulk_ingestion_authorized",
        "final_test_payload_accessed",
        "paid_compute_authorized",
        "legal_conclusion_claimed",
    ):
        _require(policy.get(field) is False, f"{field} must remain false")
    for field in ("source_level_rights_and_provenance_required", "per_source_review_required"):
        _require(policy.get(field) is True, f"{field} must remain true")
    _require(policy.get("corpus_credit_bytes") == 0, "corpus credit must remain zero")
    _require(policy.get("authorized_loss_positions") == 0, "loss-position credit must remain zero")

    source_rows = payload.get("sources")
    _require(isinstance(source_rows, list), "sources must be a list")
    _require(len(source_rows) == 30, "registry must contain exactly 30 v0.1 source families")
    keys = [row.get("key") for row in source_rows if isinstance(row, dict)]
    _require(len(keys) == len(source_rows), "each source must be an object")
    _require(len(set(keys)) == len(keys), "source keys must be unique")
    _require(set(keys) == EXPECTED_SOURCE_KEYS, "source-family set drifted")

    hf_ids: list[str] = []
    for row in source_rows:
        key = row["key"]
        hf_dataset = row.get("hf_dataset")
        _require(
            isinstance(hf_dataset, str) and hf_dataset.startswith("common-pile/"),
            f"{key}: invalid Hugging Face dataset identity",
        )
        hf_ids.append(hf_dataset)
        for field in (
            "display_name",
            "domain",
            "rights_basis_class",
            "upstream_rights_claim",
            "provenance_summary",
        ):
            _require(
                isinstance(row.get(field), str) and bool(row[field].strip()),
                f"{key}: missing {field}",
            )
        signals = row.get("license_or_status_signals")
        _require(isinstance(signals, list) and bool(signals), f"{key}: missing rights signals")
        _require(
            row.get("project_review_status") == "REVIEW_REQUIRED",
            f"{key}: source review may not be bypassed",
        )
        _require(
            row.get("canonical_training_authorized") is False,
            f"{key}: training authorization is forbidden in this audit",
        )
        _require(row.get("credited_bytes") == 0, f"{key}: credited bytes must remain zero")
        _require(
            row.get("authorized_loss_positions") == 0,
            f"{key}: loss-position credit must remain zero",
        )
        _require(
            row.get("evaluation_role") == "TRAINING_CANDIDATE_ONLY",
            f"{key}: evaluation role drifted",
        )
        _require(row.get("final_test_excluded") is True, f"{key}: final-test firewall weakened")
        collector_status = row.get("collector_path_status")
        _require(
            collector_status
            in {
                "PRESENT_AT_AUDITED_CODE_COMMIT",
                "NOT_PRESENT_AT_AUDITED_CODE_COMMIT",
            },
            f"{key}: invalid collector status",
        )
        if collector_status == "PRESENT_AT_AUDITED_CODE_COMMIT":
            path = row.get("audited_collector_path")
            _require(
                isinstance(path, str) and path.startswith("sources/"),
                f"{key}: collector path is missing",
            )
        else:
            _require(row.get("audited_collector_path") is None, f"{key}: collector mismatch")

    _require(len(set(hf_ids)) == len(hf_ids), "Hugging Face dataset identities must be unique")

    identity = payload.get("registry_identity_sha256")
    _require(bool(HEX64.fullmatch(str(identity or ""))), "registry identity must be sha256")
    expected = registry_identity(payload)
    _require(identity == expected, "registry identity mismatch")
    return expected


def load_and_validate(path: str | Path) -> dict[str, Any]:
    """Load JSON from path and return it after validation."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "registry root must be an object")
    validate_registry(payload)
    return payload
