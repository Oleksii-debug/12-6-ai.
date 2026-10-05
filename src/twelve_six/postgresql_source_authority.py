"""Validate PostgreSQL terminal source evidence without granting corpus capacity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("configs/data/next100_040_postgresql_source_authority_v2.json")
SCHEMA_VERSION = "12-6.next100-040-postgresql-source-authority.v2"
RECEIPT_SCHEMA = "12-6.next100-040.postgresql-docs-source-authority.v1"
RECEIPT_AUTHORITY_ID = (
    "51588291c95bd4f1e88654ad935d5cc22975d852f8d0c363219cb47a18df73a5"
)
HISTORICAL_HEAD = "fad19a6932f644dc7355d71d10156a72b2fd3c01"
RECEIPT_GIT_BLOB_SHA1 = "706ce8d60cdc676830f70879cbb2ff741709c089"

SOURCE = {
    "repository": "postgres/postgres",
    "repository_url": "https://github.com/postgres/postgres",
    "version": "18.6",
    "tag": "REL_18_6",
    "commit": "724edf9bde9d356724ad384a2e196edc3c9f80f7",
    "source_family": "github:postgres/postgres:documentation",
    "canonical_form": "source SGML only",
    "selected_document_count": 10,
    "raw_bytes": 958397,
    "normalized_bytes": 624335,
    "raw_manifest_sha256": (
        "e4e732794b01177998ce3a13933cde767541d841df848de7c483bf114596a071"
    ),
    "normalized_manifest_sha256": (
        "cf7766effee9528339e80ae8e9c9ba4c3f78f511bfb4f2af618c9bbac17f9942"
    ),
    "normalization_id": "POSTGRES_SGML_TEXT_V1",
}
LICENSE = {
    "name": "PostgreSQL License",
    "path": "COPYRIGHT",
    "git_blob_sha1": "0a397648dcd3c2177acc58bd7daecd11ad64be62",
    "sha256": "3d6af92ff8a4c2cdf69afb1cf44edea727922f5cd0cf8b5f72b11cdecac8fdfd",
    "rights_identity_sha256": (
        "a9d9d8895d476b1efd946fdf451996880892b086ba4cd0fe87118196382061a3"
    ),
    "model_training": "ALLOWED",
    "redistribution": "ALLOWED_WITH_NOTICE",
    "evaluation": "NOT_SEPARATELY_ADMITTED",
}
EXPECTED_DOCUMENTS = {
    "doc/src/sgml/syntax.sgml": (
        "916189a7d68ce9079be51e3f41b6f394fd3aac80",
        "449b0c500fccca068d0f8a1db2a5ebb5430eb83e33f12c6cb310a01a7437b635",
        105808,
        "4b07df6d22bd4f405dc77d32ffde871ada20f21178bcae557dcf1640f7b3d2a4",
        66750,
    ),
    "doc/src/sgml/ddl.sgml": (
        "ddd7096f6b2acfae4187626badc21e1eebed460d",
        "ce1919d9236f2e71672660e1a347146472e966e4d19b77fde5ae345dd1db6ec7",
        212910,
        "fdfb7eae6077a8a1b7372a6f30e0c9702e0571e9ecef007dffcf800d6df44105",
        146547,
    ),
    "doc/src/sgml/dml.sgml": (
        "458aee788b7fbe5e90ad1b9f3cff4e35f4669c33",
        "7d63ea98c09b08908ec9e038a8a2ec3d5d34ee66f9e79a9c10c82dbe0c321cf7",
        14495,
        "1c21a04f2b85ec067a62d8c6239d591f6a602cea67be65e0ffb22e958ecb4599",
        9942,
    ),
    "doc/src/sgml/queries.sgml": (
        "a326960ff4dfb714daa30644a94f1a1774601a8d",
        "21406147cb4723805167c9f1f3291b5e9cd4250fc821bf16148fe85de2044ce5",
        104636,
        "33cd1005a0b9ed1b1a68339e2b3d1807f0d5fa1bbcb042e4c099b9496e78e916",
        66238,
    ),
    "doc/src/sgml/datatype.sgml": (
        "833922207365d315b0d89a04d8d117f25b475d9e",
        "86328daa77e20d81d222376ec0306d9841a17104e436c08d138eb763aefb3700",
        196805,
        "3e7f617fff537d1abd29de83e25be59e81e082fe0c9d74c73c254924bad24d5a",
        103205,
    ),
    "doc/src/sgml/typeconv.sgml": (
        "2874874248668830fe49a9e91d3bbe8768737974",
        "53d244d87f9d9a283ee955ccb4f77f1f48053f5a0159a567ea21689abcdd4099",
        43118,
        "a86d3a4ad036c87ce4741b5c907911e1b04cf665af025ef47f88701e704cd068",
        31965,
    ),
    "doc/src/sgml/indices.sgml": (
        "9c4f76abf0dcd737252f2d52b6f50d815a7508a6",
        "b2d3d4be58abced7e88b6dd4feaca47585ef862f6c909efb089624c6ed1f422a",
        70918,
        "eafb0ddecc78eb4d243f0b3d5a675a142dda3bdb5d52886f6a2f4d12213d9dfa",
        51679,
    ),
    "doc/src/sgml/mvcc.sgml": (
        "049ee75a4ba3bba507f36b980c44d3684bb0864a",
        "4b7bf77ee0af4b86052330ff2421d567df3604b75fe7d4f522bec84fdea87340",
        82103,
        "546aa17b3cf60f3556d0af7e64c1ba32ab3e6de3c5efd2e6816ba674144c45cd",
        53290,
    ),
    "doc/src/sgml/perform.sgml": (
        "106583fb2965d57a7ec36c3e070d4f1a9adad772",
        "2de79e1797c6bb34cd71de600db358bc7a91f9b6ab2fa161f43c124bbcd2d9ea",
        100542,
        "e471d2804192c73f5ec473870a9673ecf2726ff1a2d153f9cc0f2ae0f59b9c63",
        75048,
    ),
    "doc/src/sgml/parallel.sgml": (
        "1ce9abf86f52514b43f772e194f6868432d42df2",
        "bdb0ae12a6817cc777b2514e86486e32d578b11670971898ea0c0d50017de0d5",
        27062,
        "ec513b502ae029454f933b74bc9398e0c6b9d19bb08abb15e509630750335711",
        19671,
    ),
}
ZERO_INT_FIELDS = {
    "authorized_optimized_target_exposure",
    "optimizer_updates_executed_on_real_targets",
}
FALSE_BOOL_FIELDS = {
    "current_retained_corpus_launch_authoritative",
    "tokenizer_fit_authorized",
    "training_executed",
    "learned_weights_created",
    "final_test_outcomes_read",
    "paid_compute_used",
    "foreign_pretrained_weights",
}


def _canonical_json_line(value: Any) -> bytes:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return (text + "\n").encode()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def _exact_int(value: Any, expected: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value == expected


def _zero_secret_counts(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and bool(value)
        and all(_exact_int(count, 0) for count in value.values())
    )


def _validate_receipt(receipt_bytes: bytes) -> list[str]:
    errors: list[str] = []
    if _git_blob_sha1(receipt_bytes) != RECEIPT_GIT_BLOB_SHA1:
        errors.append("historical_receipt_git_blob_sha1_mismatch")
    try:
        receipt = json.loads(receipt_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return errors + ["historical_receipt_json_invalid"]
    if not isinstance(receipt, dict):
        return errors + ["historical_receipt_root_must_be_object"]

    expected_keys = {
        "schema_version",
        "worker_id",
        "local_free_only",
        "verdict",
        "scope",
        "license",
        "family",
        "normalization",
        "documents",
        "aggregate",
        "privacy",
        "dedup",
        "registry_concurrency_binding",
        "gates",
        "claim_boundary",
        "authority_identity_sha256",
    }
    if set(receipt) != expected_keys:
        errors.append("historical_receipt_top_level_keys_mismatch")
    claimed = receipt.get("authority_identity_sha256")
    core = dict(receipt)
    core.pop("authority_identity_sha256", None)
    if claimed != RECEIPT_AUTHORITY_ID or _sha256(_canonical_json_line(core)) != claimed:
        errors.append("historical_receipt_authority_identity_mismatch")
    if receipt.get("schema_version") != RECEIPT_SCHEMA:
        errors.append("historical_receipt_schema_mismatch")
    if receipt.get("verdict") != "ADMIT" or receipt.get("local_free_only") is not True:
        errors.append("historical_receipt_verdict_or_profile_mismatch")

    scope = receipt.get("scope")
    expected_scope = {
        "source_kind": "official English technical documentation",
        "upstream_repository": SOURCE["repository_url"],
        "version": SOURCE["version"],
        "tag": SOURCE["tag"],
        "commit": SOURCE["commit"],
        "canonical_form": SOURCE["canonical_form"],
        "selected_document_count": SOURCE["selected_document_count"],
        "generated_html_included": False,
        "generated_pdf_included": False,
        "code_examples": (
            "retained only where embedded in selected documentation; same license; "
            "no separate code-family credit"
        ),
    }
    if scope != expected_scope:
        errors.append("historical_receipt_scope_mismatch")

    license_row = receipt.get("license")
    expected_receipt_license = {
        "name": LICENSE["name"],
        "path": LICENSE["path"],
        "git_blob_sha1": LICENSE["git_blob_sha1"],
        "raw_sha256": LICENSE["sha256"],
        "rights_identity_sha256": LICENSE["rights_identity_sha256"],
        "model_training": LICENSE["model_training"],
        "redistribution": LICENSE["redistribution"],
        "evaluation": LICENSE["evaluation"],
    }
    if not isinstance(license_row, dict) or any(
        license_row.get(key) != expected
        for key, expected in expected_receipt_license.items()
    ):
        errors.append("historical_receipt_license_mismatch")

    family = receipt.get("family")
    if not isinstance(family, dict) or any(
        family.get(key) != expected
        for key, expected in {
            "canonical_upstream_repository": SOURCE["repository_url"],
            "family_id": SOURCE["source_family"],
            "family_identity_sha256": (
                "0ed12da1597929911aa0130157532d88338c65315bbfd008eab2de036559b590"
            ),
            "independent_family_credit": 1,
            "version_invariant": True,
            "rendered_html_pdf_alias_same_family": True,
            "document_count_does_not_equal_family_count": True,
        }.items()
    ):
        errors.append("historical_receipt_family_mismatch")

    normalization = receipt.get("normalization")
    if not isinstance(normalization, dict) or (
        normalization.get("normalization_id") != SOURCE["normalization_id"]
        or normalization.get("source_only_no_rendered_duplicate") is not True
    ):
        errors.append("historical_receipt_normalization_mismatch")

    aggregate = receipt.get("aggregate")
    expected_aggregate = {
        "raw_bytes": SOURCE["raw_bytes"],
        "normalized_bytes": SOURCE["normalized_bytes"],
        "raw_manifest_sha256": SOURCE["raw_manifest_sha256"],
        "normalized_manifest_sha256": SOURCE["normalized_manifest_sha256"],
    }
    if aggregate != expected_aggregate:
        errors.append("historical_receipt_aggregate_mismatch")

    rows = receipt.get("documents")
    if not isinstance(rows, list) or len(rows) != len(EXPECTED_DOCUMENTS):
        errors.append("historical_receipt_document_count_mismatch")
    else:
        seen: set[str] = set()
        for row in rows:
            if not isinstance(row, dict):
                errors.append("historical_receipt_document_row_invalid")
                continue
            path = row.get("path")
            if not isinstance(path, str):
                errors.append("historical_receipt_document_path_mismatch")
                continue
            expected = EXPECTED_DOCUMENTS.get(path)
            if expected is None or path in seen:
                errors.append("historical_receipt_document_path_mismatch")
                continue
            seen.add(path)
            actual = (
                row.get("git_blob_sha1"),
                row.get("raw_sha256"),
                row.get("raw_bytes"),
                row.get("normalized_sha256"),
                row.get("normalized_bytes"),
            )
            if actual != expected:
                errors.append(f"historical_receipt_document_identity_mismatch:{path}")
            if row.get("normalization_id") != SOURCE["normalization_id"]:
                errors.append(f"historical_receipt_document_normalization_mismatch:{path}")
            if row.get("registry_exact_collision") is not False:
                errors.append(f"historical_receipt_registry_collision_changed:{path}")
            quality = row.get("quality")
            if not isinstance(quality, dict) or quality.get("decision") != "PASS":
                errors.append(f"historical_receipt_quality_changed:{path}")
            privacy = row.get("privacy")
            if not isinstance(privacy, dict) or (
                not _exact_int(privacy.get("email_shape_count"), 0)
                or privacy.get("matched_values_retained") is not False
                or not _zero_secret_counts(privacy.get("secret_finding_counts"))
            ):
                errors.append(f"historical_receipt_privacy_changed:{path}")
        if seen != set(EXPECTED_DOCUMENTS):
            errors.append("historical_receipt_document_set_mismatch")

    privacy = receipt.get("privacy")
    if not isinstance(privacy, dict) or (
        privacy.get("decision") != "PASS"
        or not _exact_int(privacy.get("email_shape_count"), 0)
        or privacy.get("matched_values_retained") is not False
        or not _zero_secret_counts(privacy.get("secret_finding_counts"))
    ):
        errors.append("historical_receipt_privacy_boundary_mismatch")

    dedup = receipt.get("dedup")
    if not isinstance(dedup, dict) or any(
        dedup.get(key) != expected
        for key, expected in {
            "bound_registry_exact_collision_absent": True,
            "family_lineage_collision_absent": True,
            "internal_exact_unique": True,
            "maximum_seven_word_shingle_jaccard": 0.000257,
            "near_duplicate_threshold": 0.85,
            "pairs_at_or_above_0_25": [],
        }.items()
    ):
        errors.append("historical_receipt_dedup_mismatch")

    gates = receipt.get("gates")
    if not isinstance(gates, dict) or not gates or any(value is not True for value in gates.values()):
        errors.append("historical_receipt_gates_not_all_true")

    boundary = receipt.get("claim_boundary")
    expected_boundary = {
        "source_family_qualified_for_training": True,
        "corpus_registry_mutated": False,
        "corpus_frozen": False,
        "representative_corpus_claimed": False,
        "evaluation_use_authorized": False,
        "downstream_d03_materialization_required": True,
        "downstream_incumbent_quality_privacy_dedup_required": True,
    }
    if boundary != expected_boundary:
        errors.append("historical_receipt_claim_boundary_mismatch")
    return errors


def validate_postgresql_source_authority(config: Any, receipt_bytes: bytes) -> list[str]:
    """Return fail-closed blockers for the current-main PostgreSQL source authority."""
    if not isinstance(config, dict):
        return ["config_root_must_be_object"]
    errors = _validate_receipt(receipt_bytes)
    expected_top = {
        "schema_version",
        "execution_profile",
        "project_authority",
        "bounded_source",
        "license",
        "historical_execution",
        "historical_dedup",
        "current_composition",
        "truth_boundary",
    }
    if set(config) != expected_top:
        errors.append("config_top_level_keys_mismatch")
    if config.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version_mismatch")
    if config.get("execution_profile") != "LOCAL_FREE":
        errors.append("execution_profile_must_be_local_free")
    if config.get("bounded_source") != SOURCE:
        errors.append("bounded_source_identity_mismatch")
    if config.get("license") != LICENSE:
        errors.append("license_identity_or_use_boundary_mismatch")

    expected_project = {
        "swarm_control_issue": 723,
        "convergence_issue": 2273,
        "source_pr": 448,
        "historical_prequal_issue": 1843,
    }
    if config.get("project_authority") != expected_project:
        errors.append("project_authority_mismatch")

    historical = config.get("historical_execution")
    expected_historical = {
        "source_head_sha": HISTORICAL_HEAD,
        "run_id": 33006023353,
        "job_id": 98299853774,
        "artifact_id": 9621106855,
        "artifact_zip_sha256": (
            "bf125d64ef0eb381e18256da45051e5034ceb65e569a631731d560bde6fb3cd0"
        ),
        "receipt_path": (
            "evidence/data/next100_040_postgresql_terminal_source_authority_v1.json"
        ),
        "receipt_git_blob_sha1": RECEIPT_GIT_BLOB_SHA1,
        "receipt_authority_identity_sha256": RECEIPT_AUTHORITY_ID,
    }
    if historical != expected_historical:
        errors.append("historical_execution_identity_mismatch")

    expected_dedup = {
        "historical_only": True,
        "bound_registry_exact_collision_absent": True,
        "family_lineage_collision_absent": True,
        "internal_exact_unique": True,
        "maximum_seven_word_shingle_jaccard": 0.000257,
        "near_duplicate_threshold": 0.85,
        "current_global_authority": False,
    }
    if config.get("historical_dedup") != expected_dedup:
        errors.append("historical_dedup_boundary_mismatch")

    composition = config.get("current_composition")
    if not isinstance(composition, dict):
        errors.append("current_composition_missing")
    else:
        expected_strings = {
            "global_dedup": "REQUIRED_NOT_EXECUTED_BY_THIS_AUTHORITY",
            "evaluation_firewall": "REQUIRED_AT_COMPOSITION",
            "current_retained_corpus_cleanliness": "NOT_PROVEN_BY_THIS_AUTHORITY",
            "source_authority_status": "VERIFIED_BOUNDED_SOURCE_AUTHORITY",
        }
        expected_credit_keys = {
            "canonical_capacity_credit_bytes",
            "canonical_family_credit",
            "canonical_files_credit",
        }
        if set(composition) != set(expected_strings) | expected_credit_keys:
            errors.append("current_composition_keys_mismatch")
        for key, expected in expected_strings.items():
            if composition.get(key) != expected:
                errors.append(f"current_composition_{key}_mismatch")
        for key in (
            "canonical_capacity_credit_bytes",
            "canonical_family_credit",
            "canonical_files_credit",
        ):
            if not _exact_int(composition.get(key), 0):
                errors.append(f"current_composition_{key}_must_be_exact_int_zero")

    truth = config.get("truth_boundary")
    expected_truth_keys = ZERO_INT_FIELDS | FALSE_BOOL_FIELDS
    if not isinstance(truth, dict):
        errors.append("truth_boundary_missing")
    elif set(truth) != expected_truth_keys:
        errors.append("truth_boundary_keys_mismatch")
    else:
        for key in ZERO_INT_FIELDS:
            if not _exact_int(truth.get(key), 0):
                errors.append(f"{key}_must_be_exact_int_zero")
        for key in FALSE_BOOL_FIELDS:
            if truth.get(key) is not False:
                errors.append(f"{key}_must_be_false")
    return errors


def validate_postgresql_source_authority_files(repo_root: str | Path = ".") -> list[str]:
    """Load repository-bound config/receipt and return blockers."""
    root = Path(repo_root)
    config = json.loads((root / CONFIG_PATH).read_text(encoding="utf-8"))
    historical = config.get("historical_execution", {})
    receipt_path = historical.get("receipt_path")
    if not isinstance(receipt_path, str):
        return ["historical_receipt_path_missing"]
    return validate_postgresql_source_authority(
        config,
        (root / receipt_path).read_bytes(),
    )
