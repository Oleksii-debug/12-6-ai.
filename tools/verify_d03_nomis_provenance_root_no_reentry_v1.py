#!/usr/bin/env python3
"""Bind the prohibited PR462/Verba root to the physical Nomis-free successor.

This is an additive post-merge authority. It does not rerun corpus science and it
must not mutate the byte-exact SWARM-2065 physical producer. Instead it verifies
historical Git objects for the source-admission root, verifies the retained
SWARM-2065 release lineage, and exposes a fail-closed provenance predicate that
cannot be bypassed by renaming source aliases or changing payload bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PR462_HEAD = "d75edd497c7fb1054e86d892c9462f059c1f4aa9"
PR462_CONFIG = "configs/data/next100_027_ua_public_domain_lit_v1.json"
PR462_CONFIG_BLOB = "c09b8951aeaaa9d42da40ffcc180750fba4258c3"
PR462_AUTHORITY = "85f596e79b0ec6479d2ef815e2a6a9bdbfaa55993c797309c1ea4d93b1d9b0e7"
SOURCE_PR = 462

V4_HEAD = "991a0b6e939cddeff16c075922f7c407fa1e86cb"
V4_CONFIG = "configs/data/next100_063_terminal_source_registry_v4.json"
V4_CONFIG_BLOB = "60924a7cb76dc76bbff26a340184f54a2c374c83"
V4_IDENTITY = "9fc400a3144b46c481e45d043b0a3365eb2129c83bbacde6f9e7af8a41fadc58"

UPSTREAM_REPO = "dmytro-yemelianov/verbacorpus"
UPSTREAM_COMMIT = "34a2c10ac35e1febad6c270a88fc8b83790407da"
UPSTREAM_DATA_BLOB = "a8e31fd41bd3dbbde7d43ec3c04f56e5beb37d1b"
UPSTREAM_DATA_PATH = "app/public/data/landing.json"
SOURCE_FILTER = ["Nomis1864"]
SOURCE_FAMILY = "ua.verba.public-domain.nomis1864"
BLOCKED_NORMALIZED_SHA256 = (
    "1eb91dbd631898c6a2efe274b700a5be0deaca243c0a9d5d30994ddadcf43598"
)
BLOCKED_NORMALIZED_BYTES = 1659
BLOCKED_SOURCE_ID = "ua.verba.nomis1864.bounded24"

PHYSICAL_EXECUTION_HEAD = "3d7dd363f6b1701694c00c76c6353077f80d1492"
RELEASE_HEAD = "07754c5a1d61669061e608323ea35ddc093bb946"
PHYSICAL_RUN_ID = 34911721640
PHYSICAL_JOB_ID = 104200523132
PHYSICAL_ARTIFACT_ID = 10374891614
PHYSICAL_ARTIFACT_ZIP_SHA256 = (
    "663eec03d04252b6de574bf97ea0975703e0ba25779ca27340cb3acea941943a"
)
PHYSICAL_SOURCE_REPORT_SHA256 = (
    "db72eb1d4f86cd025741efd0c612c1f2e124ce24dfa133548c375d781331c93c"
)
PHYSICAL_SURVIVOR_AUTHORITY_SHA256 = (
    "e1c94f5eed4afa78a63d577fe33a67e28305c1e084c27cd1061061ad113f8ce5"
)
PHYSICAL_DATA526_EVIDENCE_IDENTITY_SHA256 = (
    "45ac421c9e4a5930330c1516af79e1b67d0535fc90b768ecfe9df54dd0a86c5c"
)

RELEASE_BLOBS = {
    "tools/run_d03_nomis_free_clean_successor_v1.py": (
        "bcc5c40c54f0a93f42acfd584f800048bbf49e7e"
    ),
    "tools/materialize_d03_nomis_free_data526_successor_v1.py": (
        "d183bb83df73ca4ca528d4360048c9ccc031cf33"
    ),
    "tools/run_d03_nomis_free_v7_data_only_v1.py": (
        "d1c478dd6266d46a0d564d77bb29f4e0d965013c"
    ),
    "tools/verify_d03_nomis_free_execution_authority_v1.py": (
        "48ba153da15735146beca833330807c0548d3515"
    ),
    "tools/verify_d03_nomis_free_release_authority_v1.py": (
        "f51ca785aa65c1a1644202de5a3018d238bb3431"
    ),
    "tests/test_d03_nomis_free_release_authority_v1.py": (
        "6fc856500c4b1094a13ae6d3eb925bdb124e05cb"
    ),
}

ROOT_KEYS = {
    "source_pr",
    "authority_identity_sha256",
    "source_head_sha",
    "upstream_repo",
    "upstream_commit",
    "upstream_data_object_git_blob_sha1",
    "upstream_data_object_path",
    "source_filter_exact",
}

TRUTH_BOUNDARY = {
    "current_retained_corpus_launch_authoritative": False,
    "tokenizer_fit_authorized": False,
    "authorized_optimized_target_exposure": 0,
    "optimizer_updates_executed_on_real_targets": 0,
    "training_executed": False,
    "learned_weights_created": False,
    "final_test_outcomes_read": False,
    "paid_compute_used": False,
    "foreign_pretrained_weights": False,
}


class ProvenanceAuthorityError(RuntimeError):
    """Base error for malformed or unprovable provenance authority."""


class ProhibitedProvenanceRoot(ProvenanceAuthorityError):
    """Raised when a candidate resolves to the prohibited PR462/Verba root."""


def req(condition: bool, message: str) -> None:
    if not condition:
        raise ProvenanceAuthorityError(message)


def canon(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def selfhash(value: Mapping[str, Any], key: str) -> str:
    body = dict(value)
    body.pop(key, None)
    return sha256(canon(body))


def git(
    root: Path,
    *args: str,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["GIT_NO_REPLACE_OBJECTS"] = "1"
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=check,
        capture_output=True,
        text=True,
        env=env,
    )


def gtext(root: Path, *args: str) -> str:
    return git(root, *args).stdout.strip()


def blob_sha(root: Path, ref: str, rel: str) -> str:
    try:
        return gtext(root, "rev-parse", f"{ref}:{rel}")
    except subprocess.CalledProcessError as exc:
        raise ProvenanceAuthorityError(
            f"missing bound historical path {ref}:{rel}"
        ) from exc


def json_at(root: Path, ref: str, rel: str) -> dict[str, Any]:
    try:
        value = json.loads(gtext(root, "show", f"{ref}:{rel}"))
    except (subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise ProvenanceAuthorityError(
            f"cannot load historical JSON {ref}:{rel}"
        ) from exc
    req(isinstance(value, dict), f"historical JSON root is not object: {rel}")
    return value


def prohibited_root() -> dict[str, Any]:
    return {
        "source_pr": SOURCE_PR,
        "authority_identity_sha256": PR462_AUTHORITY,
        "source_head_sha": PR462_HEAD,
        "upstream_repo": UPSTREAM_REPO,
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_data_object_git_blob_sha1": UPSTREAM_DATA_BLOB,
        "upstream_data_object_path": UPSTREAM_DATA_PATH,
        "source_filter_exact": list(SOURCE_FILTER),
    }


def validate_root_claim(value: Mapping[str, Any]) -> dict[str, Any]:
    req(set(value) == ROOT_KEYS, "provenance root key-set drift")
    req(type(value["source_pr"]) is int, "source_pr must be exact int")
    for key in ROOT_KEYS - {"source_pr", "source_filter_exact"}:
        req(type(value[key]) is str and value[key], f"invalid provenance field: {key}")
    source_filter = value["source_filter_exact"]
    req(
        isinstance(source_filter, list)
        and all(type(item) is str and item for item in source_filter),
        "source_filter_exact must be a nonempty string list",
    )
    return dict(value)


def _prohibited_anchor_matches(root: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "source_pr": root["source_pr"] == SOURCE_PR,
        "authority_identity": (
            root["authority_identity_sha256"] == PR462_AUTHORITY
        ),
        "source_head": root["source_head_sha"] == PR462_HEAD,
        "upstream_object": (
            root["upstream_repo"] == UPSTREAM_REPO
            and root["upstream_commit"] == UPSTREAM_COMMIT
            and root["upstream_data_object_git_blob_sha1"] == UPSTREAM_DATA_BLOB
            and root["upstream_data_object_path"] == UPSTREAM_DATA_PATH
        ),
        "source_filter_on_verba": (
            root["upstream_repo"] == UPSTREAM_REPO
            and root["source_filter_exact"] == SOURCE_FILTER
        ),
    }


def assert_root_admissible(value: Mapping[str, Any]) -> None:
    root = validate_root_claim(value)
    matches = _prohibited_anchor_matches(root)
    if any(matches.values()):
        labels = ",".join(sorted(key for key, matched in matches.items() if matched))
        raise ProhibitedProvenanceRoot(
            f"prohibited PR462/Verba provenance root: {labels}"
        )


def require_single_candidate_provenance(
    claims: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    req(len(claims) == 1, "exactly one source provenance root is required")
    root = validate_root_claim(claims[0])
    assert_root_admissible(root)
    return root


def candidate_is_admissible(candidate: Mapping[str, Any]) -> bool:
    claims = candidate.get("provenance_roots")
    req(
        isinstance(claims, list),
        "candidate provenance_roots must be an explicit list",
    )
    require_single_candidate_provenance(claims)
    return True


def _verify_pr462_config(config: Mapping[str, Any]) -> dict[str, Any]:
    req(
        config.get("authority_identity_sha256") == PR462_AUTHORITY,
        "PR462 authority identity drift",
    )
    req(
        selfhash(config, "authority_identity_sha256") == PR462_AUTHORITY,
        "PR462 authority self-hash drift",
    )
    digital = config.get("digital_edition")
    selection = config.get("selection")
    snapshot = config.get("snapshot")
    family = config.get("family")
    req(isinstance(digital, Mapping), "PR462 digital_edition missing")
    req(isinstance(selection, Mapping), "PR462 selection missing")
    req(isinstance(snapshot, Mapping), "PR462 snapshot missing")
    req(isinstance(family, Mapping), "PR462 family missing")
    transcription = digital.get("transcription_compilation")
    req(isinstance(transcription, Mapping), "PR462 transcription root missing")
    root = {
        "source_pr": SOURCE_PR,
        "authority_identity_sha256": PR462_AUTHORITY,
        "source_head_sha": PR462_HEAD,
        "upstream_repo": transcription.get("upstream_repo"),
        "upstream_commit": transcription.get("upstream_commit"),
        "upstream_data_object_git_blob_sha1": transcription.get(
            "upstream_data_object_git_blob_sha1"
        ),
        "upstream_data_object_path": transcription.get(
            "upstream_data_object_path"
        ),
        "source_filter_exact": selection.get("source_filter_exact"),
    }
    req(canon(validate_root_claim(root)) == canon(prohibited_root()), "root drift")
    req(family.get("source_family") == SOURCE_FAMILY, "PR462 family drift")
    req(
        snapshot.get("normalized_sha256") == BLOCKED_NORMALIZED_SHA256,
        "PR462 normalized SHA-256 drift",
    )
    req(
        snapshot.get("normalized_bytes") == BLOCKED_NORMALIZED_BYTES,
        "PR462 normalized byte count drift",
    )
    return root


def _verify_v4_registry(registry: Mapping[str, Any]) -> None:
    req(
        registry.get("registry_identity_sha256") == V4_IDENTITY,
        "V4 registry identity drift",
    )
    req(
        selfhash(registry, "registry_identity_sha256") == V4_IDENTITY,
        "V4 registry self-hash drift",
    )
    rows = registry.get("terminal_late_additions")
    req(isinstance(rows, list), "V4 terminal_late_additions missing")
    anchored = [
        row
        for row in rows
        if isinstance(row, Mapping)
        and (
            row.get("pr") == SOURCE_PR
            or row.get("authority_identity") == PR462_AUTHORITY
            or row.get("head") == PR462_HEAD
            or row.get("family") == SOURCE_FAMILY
        )
    ]
    req(len(anchored) == 1, "V4 PR462 root cardinality drift")
    row = anchored[0]
    expected = {
        "pr": SOURCE_PR,
        "authority_identity": PR462_AUTHORITY,
        "head": PR462_HEAD,
        "family": SOURCE_FAMILY,
        "numeric_training_capacity_bytes": BLOCKED_NORMALIZED_BYTES,
        "source_normalized_bytes": BLOCKED_NORMALIZED_BYTES,
        "dedicated_workflow_run": 32998503672,
        "dedicated_workflow_conclusion": "success",
        "worker": "NEXT100-027-DATA-UA-PUBLIC-DOMAIN-LIT",
        "verdict": "ADMIT",
    }
    for key, value in expected.items():
        req(
            type(row.get(key)) is type(value) and row.get(key) == value,
            f"V4 PR462 row drift: {key}",
        )


def _verify_release_lineage(root: Path) -> None:
    req(
        git(
            root,
            "merge-base",
            "--is-ancestor",
            PHYSICAL_EXECUTION_HEAD,
            RELEASE_HEAD,
            check=False,
        ).returncode
        == 0,
        "physical execution head is not ancestor of release head",
    )
    req(
        git(
            root,
            "merge-base",
            "--is-ancestor",
            RELEASE_HEAD,
            "HEAD",
            check=False,
        ).returncode
        == 0,
        "clean successor release is not ancestor of current Product head",
    )
    for rel, expected_blob in RELEASE_BLOBS.items():
        req(
            blob_sha(root, RELEASE_HEAD, rel) == expected_blob,
            f"clean successor release blob drift: {rel}",
        )


def verify_provenance_root_authority(root: Path) -> dict[str, Any]:
    root = root.resolve()
    req(
        gtext(root, "for-each-ref", "--format=%(refname)", "refs/replace") == "",
        "replace refs present",
    )
    req(
        blob_sha(root, PR462_HEAD, PR462_CONFIG) == PR462_CONFIG_BLOB,
        "PR462 config blob drift",
    )
    req(
        blob_sha(root, V4_HEAD, V4_CONFIG) == V4_CONFIG_BLOB,
        "V4 registry config blob drift",
    )
    source_root = _verify_pr462_config(json_at(root, PR462_HEAD, PR462_CONFIG))
    _verify_v4_registry(json_at(root, V4_HEAD, V4_CONFIG))
    _verify_release_lineage(root)

    root_identity = sha256(canon(source_root))
    return {
        "schema_version": "12-6.d03-nomis-provenance-root-no-reentry.v1",
        "prohibited_root_identity_sha256": root_identity,
        "source_authority_identity_sha256": PR462_AUTHORITY,
        "source_pr": SOURCE_PR,
        "source_head_sha": PR462_HEAD,
        "source_family": SOURCE_FAMILY,
        "blocked_source_id_bridge": BLOCKED_SOURCE_ID,
        "blocked_normalized_sha256": BLOCKED_NORMALIZED_SHA256,
        "blocked_normalized_bytes": BLOCKED_NORMALIZED_BYTES,
        "physical_execution_head_sha": PHYSICAL_EXECUTION_HEAD,
        "release_head_sha": RELEASE_HEAD,
        "physical_run_id": PHYSICAL_RUN_ID,
        "physical_job_id": PHYSICAL_JOB_ID,
        "physical_artifact_id": PHYSICAL_ARTIFACT_ID,
        "physical_artifact_zip_sha256": PHYSICAL_ARTIFACT_ZIP_SHA256,
        "physical_source_report_sha256": PHYSICAL_SOURCE_REPORT_SHA256,
        "physical_survivor_authority_sha256": PHYSICAL_SURVIVOR_AUTHORITY_SHA256,
        "physical_data526_evidence_identity_sha256": (
            PHYSICAL_DATA526_EVIDENCE_IDENTITY_SHA256
        ),
        "changed_aliases_cannot_bypass_root": True,
        "changed_payload_cannot_bypass_root": True,
        "physical_removed_object_bound_to_prohibited_root": True,
        "truth_boundary": dict(TRUTH_BOUNDARY),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    return parser.parse_args()


def main() -> int:
    result = verify_provenance_root_authority(_parse_args().repo_root)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
