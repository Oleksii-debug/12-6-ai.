"""Fail-closed validator for NEXT100-063 terminal source registry V5."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
V4_PATH = ROOT / "configs/data/next100_063_terminal_source_registry_v4.json"
V5_PATH = ROOT / "configs/data/next100_063_terminal_source_registry_v5.json"

EXPECTED_V4_IDENTITY = "9fc400a3144b46c481e45d043b0a3365eb2129c83bbacde6f9e7af8a41fadc58"
EXPECTED_V5_BLOB_SHA1 = "2dcc57cfba8ab6d600bc431a8713f7b8e305dcbf"
EXPECTED_ATTRS_HEAD = "cda0232d5574ef91eae0d7e0b7fa5efddcbe218b"
EXPECTED_ATTRS_RUN = 33006080831
EXPECTED_ATTRS_AUTHORITY = "151e593c3b67ae4c7686323983e6c45306a870b732573ee4820c0c017b65a7d4"
EXPECTED_ATTRS_ARTIFACT_ID = 9621650719
EXPECTED_ATTRS_ARTIFACT_DIGEST = "sha256:a8176b50a2254fcb50a6f80ca82b63459ba8e9cfddba904b16e5ac79f9c55ff2"
EXPECTED_ATTRS_CAPACITY = 170_435
EXPECTED_TOTAL = 2_215_615
EXPECTED_ENVELOPE = 2_217_976
EXPECTED_GAP = 17_784_385


class RegistryV5Error(ValueError):
    """Raised when the V5 terminal registry composition drifts."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RegistryV5Error(message)


def git_blob_sha1(raw: bytes) -> str:
    header = f"blob {len(raw)}\0".encode("ascii")
    return hashlib.sha1(header + raw).hexdigest()


def canonical_identity(data: dict[str, Any]) -> str:
    body = dict(data)
    body.pop("registry_identity_sha256", None)
    raw = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def validate(v4: dict[str, Any], v5: dict[str, Any], *, v5_blob_sha1: str) -> None:
    require(v5_blob_sha1 == EXPECTED_V5_BLOB_SHA1, "V5 config blob drift")
    require(v5.get("schema_version") == "12-6.next100-063-terminal-source-registry.v5", "V5 schema drift")
    require(v5.get("worker_id") == "NEXT100-063-CANONICAL-SOURCE-REGISTRY-CONVERGENCE-V5", "V5 worker drift")

    v4_claimed = v4.get("registry_identity_sha256")
    require(v4_claimed == EXPECTED_V4_IDENTITY, "V4 declared identity drift")
    require(canonical_identity(v4) == EXPECTED_V4_IDENTITY, "V4 content identity drift")

    base = v5.get("base_v4", {})
    require(base.get("path") == "configs/data/next100_063_terminal_source_registry_v4.json", "V4 path drift")
    require(base.get("registry_identity_sha256") == EXPECTED_V4_IDENTITY, "V4 binding drift")
    require(base.get("numeric_training_capacity_bytes") == 2_045_180, "V4 numeric capacity drift")
    require(base.get("source_normalized_envelope_bytes") == 2_047_541, "V4 envelope drift")
    require(base.get("uncredited_source_normalized_bytes") == 2_361, "V4 uncredited-byte drift")
    require(base.get("independent_family_count") == 14, "V4 family-count drift")
    require(
        base.get("by_stratum")
        == {
            "uk": {"family_count": 4, "numeric_training_capacity_bytes": 100_856},
            "en": {"family_count": 5, "numeric_training_capacity_bytes": 1_838_293},
            "code": {"family_count": 5, "numeric_training_capacity_bytes": 106_031},
        },
        "V4 stratum vector drift",
    )

    attrs = v5.get("terminal_addition", {})
    require(attrs.get("worker") == "NEXT100-053-CODE-ATTRS", "attrs worker drift")
    require(attrs.get("pr") == 474, "attrs PR drift")
    require(attrs.get("head") == EXPECTED_ATTRS_HEAD, "attrs head drift")
    require(attrs.get("dedicated_workflow_run") == EXPECTED_ATTRS_RUN, "attrs run drift")
    require(attrs.get("dedicated_workflow_conclusion") == "success", "attrs authority not terminal success")
    require(attrs.get("authority_identity") == EXPECTED_ATTRS_AUTHORITY, "attrs authority identity drift")
    require(attrs.get("artifact_id") == EXPECTED_ATTRS_ARTIFACT_ID, "attrs artifact id drift")
    require(attrs.get("artifact_digest") == EXPECTED_ATTRS_ARTIFACT_DIGEST, "attrs artifact digest drift")
    require(attrs.get("family") == "github:python-attrs/attrs", "attrs family drift")
    require(attrs.get("language") == "python" and attrs.get("modality") == "code", "attrs stratum drift")
    require(attrs.get("numeric_training_capacity_bytes") == EXPECTED_ATTRS_CAPACITY, "attrs capacity drift")
    require(attrs.get("source_normalized_bytes") == EXPECTED_ATTRS_CAPACITY, "attrs source-byte drift")
    require(attrs.get("source_file_count") == 4, "attrs file-count drift")
    require(attrs.get("training") == "ALLOWED", "attrs training permission drift")
    require(attrs.get("evaluation") == "NOT_SEPARATELY_ADMITTED", "attrs evaluation firewall drift")
    require(attrs.get("verdict") == "ADMIT", "attrs verdict drift")
    require(attrs.get("exact_duplicates") == [], "attrs exact-duplicate evidence drift")
    require(attrs.get("near_duplicate_pairs") == [], "attrs near-duplicate evidence drift")
    require(attrs.get("privacy_secrets") == "PASS", "attrs privacy/secrets gate drift")
    require(attrs.get("syntax_parse_compile") == "PASS", "attrs parse/compile gate drift")
    require(attrs.get("quality") == "PASS", "attrs quality gate drift")

    inv = v5.get("derived_pre_successor_global_dedup_inventory", {})
    require(2_045_180 + EXPECTED_ATTRS_CAPACITY == EXPECTED_TOTAL, "V5 total arithmetic invariant broken")
    require(2_047_541 + EXPECTED_ATTRS_CAPACITY == EXPECTED_ENVELOPE, "V5 envelope arithmetic invariant broken")
    require(inv.get("candidate_numeric_training_capacity_bytes") == EXPECTED_TOTAL, "V5 total drift")
    require(inv.get("candidate_source_normalized_envelope_bytes") == EXPECTED_ENVELOPE, "V5 envelope drift")
    require(inv.get("uncredited_source_normalized_bytes") == 2_361, "V5 uncredited-byte drift")
    require(inv.get("candidate_independent_family_count") == 15, "V5 family-count drift")
    require(
        inv.get("by_stratum")
        == {
            "uk": {"family_count": 4, "numeric_training_capacity_bytes": 100_856, "source_normalized_envelope_bytes": 100_856},
            "en": {"family_count": 5, "numeric_training_capacity_bytes": 1_838_293, "source_normalized_envelope_bytes": 1_840_654},
            "code": {"family_count": 6, "numeric_training_capacity_bytes": 276_466, "source_normalized_envelope_bytes": 276_466},
        },
        "V5 stratum vector drift",
    )
    require(inv.get("research_corpus_v1_acquisition_planning_target_bytes") == 20_000_000, "research target drift")
    require(inv.get("target_gap_numeric_training_capacity_bytes") == EXPECTED_GAP, "V5 gap drift")
    require(abs(float(inv.get("target_fraction_by_numeric_training_capacity")) - 0.11078075) < 1e-12, "V5 target fraction drift")

    policy = v5.get("composition_policy", {})
    require(policy.get("base_v4_identity_must_validate") is True, "V4 identity gate weakened")
    require(policy.get("only_exact_head_scoped_success_authorities_counted") is True, "terminality gate weakened")
    require(policy.get("one_independent_family_credit_per_canonical_lineage") is True, "family lineage gate weakened")
    require(policy.get("global_cross_source_dedup_required_before_corpus_identity") is True, "global dedup gate weakened")
    require(policy.get("evaluation_permission_never_inferred_from_training_permission") is True, "evaluation firewall weakened")

    gates = v5.get("downstream_gate_vector", {})
    require(gates.get("authorized_balanced_no_replay_loss_positions") == 0, "balanced training exposure must remain zero")
    require(gates.get("successor_global_cross_source_exact_near_dedup") == "REQUIRED_NEXT", "successor dedup gate weakened")
    require(gates.get("evaluation_decontamination") == "REQUIRED_AFTER_EXACT_CANDIDATE_INVENTORY", "decontamination gate weakened")
    require(gates.get("tokenizer_fit") == "BLOCKED", "tokenizer fit promoted")
    require(gates.get("long_training") == "BLOCKED", "long training promoted")
    require(gates.get("paid_compute") == "NOT_AUTHORIZED", "paid compute promoted")


def main() -> None:
    v4 = json.loads(V4_PATH.read_text(encoding="utf-8"))
    raw_v5 = V5_PATH.read_bytes()
    v5 = json.loads(raw_v5.decode("utf-8"))
    validate(v4, v5, v5_blob_sha1=git_blob_sha1(raw_v5))
    print(
        "NEXT100-063 V5 PASS "
        f"blob={EXPECTED_V5_BLOB_SHA1} numeric_capacity_bytes={EXPECTED_TOTAL} "
        f"code_numeric_bytes=276466 families=15 gap_bytes={EXPECTED_GAP}"
    )


if __name__ == "__main__":
    main()
