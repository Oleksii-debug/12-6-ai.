"""Versioned repository release for the qualification fabric."""

from __future__ import annotations

import hashlib
import re
from io import BytesIO
from pathlib import Path
from typing import Any
from zipfile import ZipFile

from twelve_six.ai_qa_control import _canonical_json_bytes
from twelve_six.evidence_policy import _strict_object
from twelve_six.integration_qualification import _checked_git, _require_ancestor, _tree

_SCHEMA = "12-6.plan8-qualification-fabric-release.v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMPONENT_PATHS = (
    "configs/control/ai_qa_policy_v1.json",
    "configs/control/plan8_evidence_requirements_v1.json",
    "configs/control/product_capabilities_v1.json",
    "configs/control/sil_scenario_v1.json",
    "configs/plan8/chaos/component-recovery-fixtures-v1.json",
    "configs/plan8/scenarios/v1/plan2-reserved-eval-firewall.json",
    "configs/plan8/scenarios/v1/plan5-agent-goal-observe-report.json",
    "configs/plan8/scenarios/v1/plan8-sil-synthetic-journey.json",
    "pyproject.toml",
    "src/twelve_six/ai_qa_control.py",
    "src/twelve_six/capability_map.py",
    "src/twelve_six/chaos_harness.py",
    "src/twelve_six/checkpoint/durability.py",
    "src/twelve_six/checkpoint/recovery_lock.py",
    "src/twelve_six/evidence_control_bridge.py",
    "src/twelve_six/evidence_lineage.py",
    "src/twelve_six/evidence_policy.py",
    "src/twelve_six/integration_qualification.py",
    "src/twelve_six/physical_qualification.py",
    "src/twelve_six/scenario_harness.py",
    "src/twelve_six/sil_qualification.py",
    "src/twelve_six/swarm_protocol_v2.py",
    "tools/plan8_evidence_bridge.py",
    "tools/plan8_physical_operator.py",
)
_HISTORICAL_HEAD = "d8f6846b4c1ae9cf5095f48c09ac0cd0e2310108"
_HISTORICAL_ACCEPTED = "94cb029f60aaaaffda9df288da6d3e7883cd0a98"
_HISTORICAL_TREE = "b026b85c568afc4575a43594cbf6bbea8d112377"
_HISTORICAL_ARCHIVE = "docs/evidence/plan8/section3-sil-11762209803.zip"
_HISTORICAL_ARCHIVE_SHA = "4d71ce518ee7607807f8e48ec43aecdf3fe751a7d3cac9f22ce6b46948538dab"
_HISTORICAL_ARCHIVE_BYTES = 6559
_HISTORICAL_MEMBERS = {
    "sil-environment.json": "f31e05858cd4fee82a11f20ff46bacc18b59c16a7be1600cc7ee872cc8e9ef9a",
    "sil-evidence.json": "6d52094d85413f496b6facdd52d482398a1439562cb4659d1150cc5233a8eaee",
    "sil.log": "4516608be779a786002aa6160307507da20286256120b30183cc9dd06fb0b879",
}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _component_identity(root: Path, source: str, path: str) -> dict[str, str]:
    listing = _checked_git(root, "ls-tree", "-z", source, "--", path)
    metadata, separator, filename = listing.partition(b"\t")
    if separator != b"\t" or filename != path.encode("ascii") + b"\0":
        raise ValueError(f"release component is missing or ambiguous: {path}")
    mode, kind, blob = metadata.decode("ascii").split()
    if mode not in {"100644", "100755"} or kind != "blob":
        raise ValueError(f"release component is not a regular Git file: {path}")
    content = _checked_git(root, "cat-file", "blob", blob)
    return {"path": path, "git_blob_sha": blob, "sha256": _sha(content)}


def _release(root: Path, source: str) -> dict[str, Any]:
    tree = _tree(root, source)
    _require_ancestor(root, _HISTORICAL_ACCEPTED, source)
    parents = _checked_git(root, "show", "-s", "--format=%P", _HISTORICAL_ACCEPTED).decode("ascii").split()
    if (
        _HISTORICAL_HEAD not in parents
        or _tree(root, _HISTORICAL_HEAD) != _HISTORICAL_TREE
        or _tree(root, _HISTORICAL_ACCEPTED) != _HISTORICAL_TREE
    ):
        raise ValueError("historical Linux SIL Git tree differs from its accepted provenance")
    return {
        "schema_version": _SCHEMA,
        "version": "1.0.0",
        "source": {"git_sha": source, "tree_sha": tree},
        "components": [_component_identity(root, source, path) for path in _COMPONENT_PATHS],
        "historical_linux_sil": {
            "candidate_sha": _HISTORICAL_HEAD,
            "accepted_sha": _HISTORICAL_ACCEPTED,
            "tree_sha": _HISTORICAL_TREE,
            "pull_request": 3098,
            "run_id": 38285068328,
            "job_id": 114933484308,
            "artifact_id": 11762209803,
            "archive_path": _HISTORICAL_ARCHIVE,
            "archive_sha256": _HISTORICAL_ARCHIVE_SHA,
            "archive_bytes": _HISTORICAL_ARCHIVE_BYTES,
            "member_sha256": _HISTORICAL_MEMBERS,
            "evidence_identity_sha256": "dbdea9c91b3ec3c3c8f215b472f7440e9c3eb27f6031b069abe63fdd3961e375",
            "vectors": 17,
            "journeys": 15,
            "environment": "ubuntu-24.04",
            "conclusion": "SUCCESS",
        },
        "qualification_campaigns": [
            {
                "scope": "LOCAL_SIL_CONTRACT_FIXTURE_AND_PHYSICAL_SIMULATION",
                "test": "tests/test_plan8_qualification_fabric_release.py::test_local_fabric_campaign",
            },
            {
                "scope": "Q0_COMPONENT_CHAOS_WITH_LOCAL_EFFECTS",
                "test": "tests/test_plan8_chaos_harness.py::ChaosHarnessTests"
                "::test_complete_versioned_q0_campaign_with_real_local_effects",
            },
        ],
    }


def build_qualification_fabric_release(repo_root: str | Path, source_sha: str) -> bytes:
    """Build canonical release bytes from exact Git objects chosen by the release operator."""
    return _canonical_json_bytes(_release(Path(repo_root), source_sha)) + b"\n"


def _verify_historical_archive(root: Path) -> None:
    with (root / _HISTORICAL_ARCHIVE).open("rb") as stream:
        raw = stream.read(_HISTORICAL_ARCHIVE_BYTES + 1)
    if len(raw) != _HISTORICAL_ARCHIVE_BYTES or _sha(raw) != _HISTORICAL_ARCHIVE_SHA:
        raise ValueError("historical Linux SIL archive differs from its trusted bytes")
    with ZipFile(BytesIO(raw)) as archive:
        if sorted(archive.namelist()) != sorted(_HISTORICAL_MEMBERS):
            raise ValueError("historical Linux SIL archive has unexpected members")
        for name, digest in _HISTORICAL_MEMBERS.items():
            if _sha(archive.read(name)) != digest:
                raise ValueError(f"historical Linux SIL archive member changed: {name}")


def verify_qualification_fabric_release(
    repo_root: str | Path,
    path: str | Path,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    """Read back a trusted release and the matching repository component objects."""
    if type(expected_sha256) is not str or not _SHA256.fullmatch(expected_sha256):
        raise ValueError("expected release SHA-256 must be exact")
    with Path(path).open("rb") as stream:
        raw = stream.read(65_537)
    if _sha(raw) != expected_sha256:
        raise ValueError("qualification fabric release differs from the trusted SHA-256")
    record = _strict_object(raw)
    source = record.get("source")
    if type(source) is not dict or type(source.get("git_sha")) is not str:
        raise ValueError("qualification fabric release has no exact source identity")
    root = Path(repo_root)
    source_sha = source["git_sha"]
    if raw != build_qualification_fabric_release(root, source_sha):
        raise ValueError("qualification fabric release differs from the exact Git component objects")
    head = _checked_git(root, "rev-parse", "HEAD").decode("ascii").strip()
    _require_ancestor(root, source_sha, head)
    for expected in record["components"]:
        if _component_identity(root, head, expected["path"]) != expected:
            raise ValueError(f"qualification fabric component changed: {expected['path']}")
    _verify_historical_archive(root)
    return {**record, "release_sha256": expected_sha256}
