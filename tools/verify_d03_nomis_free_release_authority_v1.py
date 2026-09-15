#!/usr/bin/env python3
"""Verify the workflow-free SWARM-2065 release lineage without rerunning science.

The physical clean-successor execution happened on PHYSICAL_EXECUTION_HEAD. A
terminal Product release may delete the temporary execution workflow, carry the
exact post-run Ruff-only regression repairs, and add this release verifier plus
its tests. The physical producers and evidence verifier must remain byte-exact.

This checker intentionally does not turn retained execution evidence into
corpus/tokenizer/training authority. It only proves that a later workflow-free
release still denotes the exact Product science that was physically executed.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any

PHYSICAL_EXECUTION_HEAD = "3d7dd363f6b1701694c00c76c6353077f80d1492"
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

TEMP_WORKFLOW = ".github/workflows/d03-nomis-free-clean-successor-v1.yml"
TEMP_WORKFLOW_BLOB = "bc648c9eb78425204df0a72abc1cd2a65cde2a9d"
RELEASE_VERIFIER = "tools/verify_d03_nomis_free_release_authority_v1.py"
RELEASE_TEST = "tests/test_d03_nomis_free_release_authority_v1.py"

PHYSICAL_SCIENCE_BLOBS = {
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
}

RELEASE_TEST_REPAIR_BLOBS = {
    "tests/test_d03_nomis_free_clean_successor_v1.py": (
        "0beb4d8e1c0bbae21448fe66caf81af0406b60a4"
    ),
    "tests/test_d03_nomis_free_data526_successor_v1.py": (
        "f122662c952aeeb2cad09e9adc485f430b39884e"
    ),
    "tests/test_d03_nomis_free_v7_data_only_v1.py": (
        "5fdfdf7fcebef6c27287603c3d557eb88a04cbf5"
    ),
}

EXPECTED_RELEASE_DELTA = {
    TEMP_WORKFLOW: "D",
    RELEASE_VERIFIER: "A",
    RELEASE_TEST: "A",
    **{rel: "M" for rel in RELEASE_TEST_REPAIR_BLOBS},
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


class ReleaseAuthorityError(RuntimeError):
    pass


def req(condition: bool, message: str) -> None:
    if not condition:
        raise ReleaseAuthorityError(message)


def git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
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


def _valid_sha(value: str) -> bool:
    return len(value) == 40 and all(c in "0123456789abcdef" for c in value)


def _blob_sha(root: Path, ref: str, rel: str) -> str:
    try:
        return gtext(root, "rev-parse", f"{ref}:{rel}")
    except subprocess.CalledProcessError as exc:
        raise ReleaseAuthorityError(f"missing bound path {ref}:{rel}") from exc


def _path_exists(root: Path, ref: str, rel: str) -> bool:
    return git(root, "cat-file", "-e", f"{ref}:{rel}", check=False).returncode == 0


def _release_delta(root: Path, release_head: str) -> dict[str, str]:
    raw = gtext(
        root,
        "diff",
        "--name-status",
        "--no-renames",
        f"{PHYSICAL_EXECUTION_HEAD}..{release_head}",
    )
    result: dict[str, str] = {}
    if not raw:
        return result
    for line in raw.splitlines():
        parts = line.split("\t")
        req(len(parts) == 2, f"unexpected release diff row: {line!r}")
        status, path = parts
        req(path not in result, f"duplicate release diff path: {path}")
        result[path] = status
    return result


def require_release_delta(actual: dict[str, str]) -> None:
    """Require the exact workflow-free release-only Product delta."""
    req(actual == EXPECTED_RELEASE_DELTA, f"release delta drift: {actual!r}")


def verify_release_checkout(root: Path, expected_release_head: str) -> dict[str, Any]:
    root = root.resolve()
    req(_valid_sha(expected_release_head), "invalid expected release head")
    req(
        gtext(root, "for-each-ref", "--format=%(refname)", "refs/replace") == "",
        "replace refs present",
    )
    req(
        git(
            root,
            "merge-base",
            "--is-ancestor",
            PHYSICAL_EXECUTION_HEAD,
            expected_release_head,
            check=False,
        ).returncode
        == 0,
        "physical execution head is not an ancestor of release head",
    )

    require_release_delta(_release_delta(root, expected_release_head))

    req(
        _blob_sha(root, PHYSICAL_EXECUTION_HEAD, TEMP_WORKFLOW) == TEMP_WORKFLOW_BLOB,
        "physical execution workflow blob drift",
    )
    req(
        not _path_exists(root, expected_release_head, TEMP_WORKFLOW),
        "temporary execution workflow still present on release head",
    )

    for rel, expected_blob in PHYSICAL_SCIENCE_BLOBS.items():
        req(
            _blob_sha(root, PHYSICAL_EXECUTION_HEAD, rel) == expected_blob,
            f"physical science blob drift: {rel}",
        )
        req(
            _blob_sha(root, expected_release_head, rel) == expected_blob,
            f"release science blob drift: {rel}",
        )

    for rel, expected_blob in RELEASE_TEST_REPAIR_BLOBS.items():
        req(
            _blob_sha(root, expected_release_head, rel) == expected_blob,
            f"release Ruff-only test blob drift: {rel}",
        )

    return {
        "schema_version": "12-6.d03-nomis-free-release-authority.v1",
        "physical_execution_head_sha": PHYSICAL_EXECUTION_HEAD,
        "release_head_sha": expected_release_head,
        "physical_run_id": PHYSICAL_RUN_ID,
        "physical_job_id": PHYSICAL_JOB_ID,
        "physical_artifact_id": PHYSICAL_ARTIFACT_ID,
        "physical_artifact_zip_sha256": PHYSICAL_ARTIFACT_ZIP_SHA256,
        "physical_source_report_sha256": PHYSICAL_SOURCE_REPORT_SHA256,
        "physical_survivor_authority_sha256": PHYSICAL_SURVIVOR_AUTHORITY_SHA256,
        "physical_data526_evidence_identity_sha256": (
            PHYSICAL_DATA526_EVIDENCE_IDENTITY_SHA256
        ),
        "physical_science_blob_count": len(PHYSICAL_SCIENCE_BLOBS),
        "release_test_repair_blob_count": len(RELEASE_TEST_REPAIR_BLOBS),
        "temporary_execution_workflow_removed": True,
        "physical_science_bytes_unchanged": True,
        "truth_boundary": dict(TRUTH_BOUNDARY),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--expected-release-head", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = verify_release_checkout(args.repo_root, args.expected_release_head)
    # Hash-only / identity-only output. No corpus payload is read or printed.
    for key in sorted(result):
        value = result[key]
        if key == "truth_boundary":
            continue
        print(f"{key}={value}")
    boundary_sha = hashlib.sha256(
        repr(sorted(result["truth_boundary"].items())).encode("utf-8")
    ).hexdigest()
    print(f"truth_boundary_sha256={boundary_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
