"""Live GitHub authority checks for promotion evidence.

This layer composes the offline D10 release-attestation validator with live GitHub
workflow, artifact, audit-comment, and promotion-authority observations. It is
fail-closed: unavailable, mutable, stale, queued, rerun, or inconsistent evidence
cannot authorize a promotion transition.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .dependency_lock import (
    SUPPORTED_PROFILES,
    canonical_distribution_name,
    canonical_json_bytes,
    sha256_bytes,
)
from .manifest import AuditEvidence, CandidateStatus, ComponentDisposition, StageCandidateManifest
from .release_attestation import (
    ArtifactBinding,
    CandidateCIEvidence,
    ReleaseAttestation,
    ReleaseAttestationError,
    validate_release_attestation,
)

CANONICAL_REPOSITORY = "Oleksii-debug/12-6-ai."
GITHUB_WEB_ROOT = f"https://github.com/{CANONICAL_REPOSITORY}"
GITHUB_API_ROOT = f"https://api.github.com/repos/{CANONICAL_REPOSITORY}"
CANDIDATE_WORKFLOW_NAME = "CI"
DEPENDENCY_WORKFLOW_NAME = "Dependency Security Evidence"
DEPENDENCY_SBOM_SCHEMA = "12-6.dependency-sbom.v1"
DEPENDENCY_EVIDENCE_SCHEMA = "12-6.dependency-security-evidence.v1"
DEPENDENCY_CLEAN_STATUS = "EVIDENCE_COMPLETE_NO_REVIEW_FINDINGS"

_RUN_REF_RE = re.compile(
    rf"^{re.escape(GITHUB_WEB_ROOT)}/actions/runs/(?P<run_id>[1-9][0-9]*)$"
)
_ARTIFACT_REF_RE = re.compile(
    rf"^{re.escape(GITHUB_WEB_ROOT)}/actions/runs/(?P<run_id>[1-9][0-9]*)/"
    r"artifacts/(?P<artifact_id>[1-9][0-9]*)$"
)
_AUDIT_REF_RE = re.compile(
    rf"^{re.escape(GITHUB_WEB_ROOT)}/issues/(?P<issue>13|14)"
    r"#issuecomment-(?P<comment_id>[1-9][0-9]*)$"
)
_PROMOTION_REF_RE = re.compile(
    rf"^{re.escape(GITHUB_WEB_ROOT)}/issues/1"
    r"#issuecomment-(?P<comment_id>[1-9][0-9]*)$"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_VERDICT_RE = re.compile(
    r"(?im)^\s*verdict\s*:\s*"
    r"(PASS_WITH_NOTES|CHANGES_REQUIRED|BLOCKED|NOT_RUN|PASS)\b"
)

JsonGetter = Callable[[str], Mapping[str, Any]]


class LiveAuthorityError(ReleaseAttestationError):
    """Raised when a claimed durable reference disagrees with live GitHub authority."""


def _parse_timestamp(value: str, field_name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LiveAuthorityError(f"{field_name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise LiveAuthorityError(f"{field_name} must include a timezone offset")
    return parsed.astimezone(UTC)


def _require_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise LiveAuthorityError(f"{field_name} must be a positive integer")
    return value


def _require_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LiveAuthorityError(f"{field_name} must be an object")
    return value


def github_json_get(url: str) -> Mapping[str, Any]:
    """Read one GitHub REST JSON object without exposing credentials in diagnostics."""

    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "12-6-ai-live-promotion-authority/1",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            status = response.status
            payload = json.load(response)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
        raise LiveAuthorityError("live GitHub evidence lookup failed closed") from exc
    if status != 200:
        raise LiveAuthorityError(f"live GitHub evidence returned HTTP {status}")
    return _require_mapping(payload, "GitHub response")


def _repository_full_name(run: Mapping[str, Any]) -> str | None:
    for field_name in ("repository", "head_repository"):
        value = run.get(field_name)
        if isinstance(value, Mapping):
            full_name = value.get("full_name")
            if isinstance(full_name, str):
                return full_name
    return None


def _require_canonical_run_ref(evidence_ref: str, run_id: int) -> None:
    match = _RUN_REF_RE.fullmatch(evidence_ref)
    if match is None or int(match.group("run_id")) != run_id:
        raise LiveAuthorityError(
            "CI evidence_ref must be the exact canonical GitHub Actions run URL"
        )


def verify_workflow_run(
    *,
    run_id: int,
    expected_head_sha: str,
    evidence_ref: str,
    get_json: JsonGetter = github_json_get,
    expected_workflow_name: str = CANDIDATE_WORKFLOW_NAME,
    expected_completed_at: str | None = None,
) -> Mapping[str, Any]:
    """Verify one exact, first-attempt, completed-success GitHub Actions run."""

    _require_canonical_run_ref(evidence_ref, run_id)
    run = get_json(f"{GITHUB_API_ROOT}/actions/runs/{run_id}")
    if _require_int(run.get("id"), "workflow run id") != run_id:
        raise LiveAuthorityError("workflow run id differs from evidence")
    if run.get("status") != "completed":
        raise LiveAuthorityError("workflow run is not completed")
    if run.get("conclusion") != "success":
        raise LiveAuthorityError("workflow run did not conclude success")
    if run.get("head_sha") != expected_head_sha:
        raise LiveAuthorityError("workflow run head SHA is stale for claimed source")
    if run.get("name") != expected_workflow_name:
        raise LiveAuthorityError("workflow run is not the authoritative CI workflow")
    if run.get("html_url") != evidence_ref:
        raise LiveAuthorityError("workflow run canonical URL differs from evidence_ref")
    if run.get("run_attempt") != 1:
        raise LiveAuthorityError("workflow reruns are not bindable by evidence schema v1")
    if _repository_full_name(run) != CANONICAL_REPOSITORY:
        raise LiveAuthorityError("workflow run repository identity mismatch")
    if expected_completed_at is not None:
        observed = run.get("updated_at")
        if not isinstance(observed, str):
            raise LiveAuthorityError("workflow run completion timestamp is missing")
        if _parse_timestamp(observed, "workflow run updated_at") != _parse_timestamp(
            expected_completed_at,
            "claimed CI completed_at_utc",
        ):
            raise LiveAuthorityError("claimed CI completion timestamp differs from live run")
    return run


def _run_artifacts(run_id: int, get_json: JsonGetter) -> dict[int, Mapping[str, Any]]:
    artifacts: dict[int, Mapping[str, Any]] = {}
    page = 1
    total_count: int | None = None
    while page <= 10:
        payload = get_json(
            f"{GITHUB_API_ROOT}/actions/runs/{run_id}/artifacts?per_page=100&page={page}"
        )
        raw_items = payload.get("artifacts")
        if not isinstance(raw_items, list):
            raise LiveAuthorityError("workflow artifact listing is malformed")
        raw_total = payload.get("total_count")
        if isinstance(raw_total, bool) or not isinstance(raw_total, int) or raw_total < 0:
            raise LiveAuthorityError("workflow artifact total_count is malformed")
        total_count = raw_total
        for raw in raw_items:
            item = _require_mapping(raw, "workflow artifact")
            artifact_id = _require_int(item.get("id"), "workflow artifact id")
            if artifact_id in artifacts:
                raise LiveAuthorityError("workflow artifact listing contains duplicate IDs")
            artifacts[artifact_id] = item
        if len(artifacts) >= total_count:
            break
        if not raw_items:
            break
        page += 1
    if total_count is None or len(artifacts) != total_count:
        raise LiveAuthorityError("workflow artifact listing is incomplete")
    return artifacts


def _verify_artifact_record(
    *,
    record: Mapping[str, Any],
    artifact_id: int,
    run_id: int,
    expected_head_sha: str,
    expected_sha256: str | None = None,
) -> None:
    if _require_int(record.get("id"), "workflow artifact id") != artifact_id:
        raise LiveAuthorityError("workflow artifact ID mismatch")
    if record.get("expired") is not False:
        raise LiveAuthorityError("workflow artifact is expired or expiration state is unknown")
    workflow_run = record.get("workflow_run")
    if isinstance(workflow_run, Mapping):
        if workflow_run.get("id") != run_id:
            raise LiveAuthorityError("workflow artifact belongs to a different run")
        if workflow_run.get("head_sha") not in {None, expected_head_sha}:
            raise LiveAuthorityError("workflow artifact belongs to a stale source head")
    digest = record.get("digest")
    if expected_sha256 is not None and digest is not None:
        if digest != f"sha256:{expected_sha256}":
            raise LiveAuthorityError("workflow artifact digest differs from bound SHA-256")


def _verify_environment_artifacts(
    attestation: ReleaseAttestation,
    *,
    run_id: int,
    expected_head_sha: str,
    get_json: JsonGetter,
) -> dict[int, Mapping[str, Any]]:
    artifacts = _run_artifacts(run_id, get_json)
    for environment in attestation.environment_evidence:
        if environment.run_id != run_id or environment.source_sha != expected_head_sha:
            raise LiveAuthorityError("environment evidence is stale for live authority run")
        _require_canonical_run_ref(environment.evidence_ref, run_id)
        record = artifacts.get(environment.artifact_id)
        if record is None:
            raise LiveAuthorityError(
                f"environment artifact {environment.artifact_id} is missing from live run"
            )
        _verify_artifact_record(
            record=record,
            artifact_id=environment.artifact_id,
            run_id=run_id,
            expected_head_sha=expected_head_sha,
            expected_sha256=environment.archive_sha256,
        )
    return artifacts


def _parse_artifact_ref(evidence_ref: str) -> tuple[int, int]:
    match = _ARTIFACT_REF_RE.fullmatch(evidence_ref)
    if match is None:
        raise LiveAuthorityError(
            "gated artifact evidence_ref must identify one canonical GitHub Actions artifact"
        )
    return int(match.group("run_id")), int(match.group("artifact_id"))


def _verify_candidate_artifact_refs(
    attestation: ReleaseAttestation,
    *,
    run_id: int,
    expected_head_sha: str,
    artifacts: Mapping[int, Mapping[str, Any]],
) -> None:
    bound: tuple[ArtifactBinding, ...] = (
        *attestation.checkpoint_artifacts,
        *((attestation.release_artifact,) if attestation.release_artifact is not None else ()),
    )
    for item in bound:
        ref_run_id, artifact_id = _parse_artifact_ref(item.evidence_ref)
        if ref_run_id != run_id:
            raise LiveAuthorityError(
                f"artifact {item.kind!r} is not from exact candidate CI run"
            )
        record = artifacts.get(artifact_id)
        if record is None:
            raise LiveAuthorityError(
                f"artifact {item.kind!r} is missing from exact candidate run"
            )
        _verify_artifact_record(
            record=record,
            artifact_id=artifact_id,
            run_id=run_id,
            expected_head_sha=expected_head_sha,
            expected_sha256=None,
        )


def _verify_supply_chain_artifact_refs(
    attestation: ReleaseAttestation,
    *,
    expected_head_sha: str,
    get_json: JsonGetter,
) -> None:
    parsed = [
        (*_parse_artifact_ref(item.evidence_ref), item)
        for item in attestation.supply_chain_artifacts
    ]
    run_ids = {run_id for run_id, _, _ in parsed}
    if len(run_ids) != 1:
        raise LiveAuthorityError(
            "supply-chain artifacts must come from one exact dedicated workflow run"
        )
    supply_run_id = next(iter(run_ids))
    supply_run_ref = f"{GITHUB_WEB_ROOT}/actions/runs/{supply_run_id}"
    verify_workflow_run(
        run_id=supply_run_id,
        expected_head_sha=expected_head_sha,
        evidence_ref=supply_run_ref,
        get_json=get_json,
        expected_workflow_name=DEPENDENCY_WORKFLOW_NAME,
    )
    artifacts = _run_artifacts(supply_run_id, get_json)
    for _, artifact_id, item in parsed:
        record = artifacts.get(artifact_id)
        if record is None:
            raise LiveAuthorityError(
                f"supply-chain artifact {item.kind!r} is missing from dedicated run"
            )
        _verify_artifact_record(
            record=record,
            artifact_id=artifact_id,
            run_id=supply_run_id,
            expected_head_sha=expected_head_sha,
            expected_sha256=None,
        )


def _clean_markdown_for_authority(body: str) -> str:
    return body.replace("**", "").replace("`", "")


def verify_audit_evidence(
    audit: AuditEvidence,
    *,
    expected_issue: int,
    candidate_ci_completed_at: str,
    get_json: JsonGetter = github_json_get,
) -> Mapping[str, Any]:
    """Verify exact-candidate audit authority from the canonical audit issue comment."""

    match = _AUDIT_REF_RE.fullmatch(audit.evidence_ref)
    if match is None or int(match.group("issue")) != expected_issue:
        raise LiveAuthorityError("audit evidence_ref is not from the canonical auditor issue")
    comment_id = int(match.group("comment_id"))
    comment = get_json(f"{GITHUB_API_ROOT}/issues/comments/{comment_id}")
    if comment.get("html_url") != audit.evidence_ref:
        raise LiveAuthorityError("audit comment canonical URL differs from evidence_ref")
    body = comment.get("body")
    if not isinstance(body, str):
        raise LiveAuthorityError("audit comment body is missing")
    clean = _clean_markdown_for_authority(body)
    if audit.candidate_sha not in clean:
        raise LiveAuthorityError("audit comment does not bind the exact candidate SHA")
    verdicts = _VERDICT_RE.findall(clean)
    if verdicts != [audit.verdict.value]:
        raise LiveAuthorityError("audit comment verdict differs from structured audit evidence")
    if audit.cutoff_utc not in clean:
        raise LiveAuthorityError("audit comment does not bind the exact structured cutoff")
    created_at = comment.get("created_at")
    if not isinstance(created_at, str):
        raise LiveAuthorityError("audit comment created_at is missing")
    published = _parse_timestamp(created_at, "audit comment created_at")
    cutoff = _parse_timestamp(audit.cutoff_utc, "audit cutoff_utc")
    candidate_ci = _parse_timestamp(candidate_ci_completed_at, "candidate CI completed_at_utc")
    if published < cutoff or published < candidate_ci:
        raise LiveAuthorityError("audit comment predates its cutoff or exact candidate CI")
    return comment


def verify_promotion_authority(
    evidence_ref: str,
    *,
    candidate_sha: str,
    not_before: datetime,
    get_json: JsonGetter = github_json_get,
) -> Mapping[str, Any]:
    """Verify an explicit owner STABLE authorization on permanent control Issue #1."""

    match = _PROMOTION_REF_RE.fullmatch(evidence_ref)
    if match is None:
        raise LiveAuthorityError("STABLE promotion authority must be an Issue #1 comment")
    comment_id = int(match.group("comment_id"))
    comment = get_json(f"{GITHUB_API_ROOT}/issues/comments/{comment_id}")
    if comment.get("html_url") != evidence_ref:
        raise LiveAuthorityError("promotion authority canonical URL differs from evidence_ref")
    body = comment.get("body")
    if not isinstance(body, str):
        raise LiveAuthorityError("promotion authority comment body is missing")
    clean = _clean_markdown_for_authority(body)
    if candidate_sha not in clean:
        raise LiveAuthorityError("promotion authority does not bind exact candidate SHA")
    if re.search(r"(?im)^\s*promotion_authorized\s*:\s*stable\s*$", clean) is None:
        raise LiveAuthorityError("promotion authority lacks explicit PROMOTION_AUTHORIZED: STABLE")
    created_at = comment.get("created_at")
    if not isinstance(created_at, str):
        raise LiveAuthorityError("promotion authority created_at is missing")
    if _parse_timestamp(created_at, "promotion authority created_at") < not_before:
        raise LiveAuthorityError("promotion authority predates required CI/audit evidence")
    return comment


def _load_json_object(path: Path, field_name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LiveAuthorityError(f"{field_name} is not readable canonical JSON") from exc
    if not isinstance(value, dict):
        raise LiveAuthorityError(f"{field_name} must contain a JSON object")
    return value


def _self_hash(document: Mapping[str, Any], field_name: str) -> str:
    payload = dict(document)
    claimed = payload.pop(field_name, None)
    if not isinstance(claimed, str):
        raise LiveAuthorityError(f"{field_name} is missing")
    if sha256_bytes(canonical_json_bytes(payload)) != claimed:
        raise LiveAuthorityError(f"{field_name} mismatch")
    return claimed


def verify_dependency_security_evidence(
    attestation: ReleaseAttestation,
    *,
    artifact_root: str | Path,
) -> None:
    """Require exact-green PR #62 dependency-evidence semantics for a gated candidate."""

    if attestation.candidate_sha is None or attestation.dependency_lock is None:
        raise LiveAuthorityError(
            "candidate dependency-security validation lacks candidate identity"
        )
    by_kind = {item.kind: item for item in attestation.supply_chain_artifacts}
    sbom_binding = by_kind.get("sbom")
    report_binding = by_kind.get("dependency_report")
    if sbom_binding is None or report_binding is None:
        raise LiveAuthorityError("candidate dependency-security artifacts are incomplete")
    root = Path(artifact_root).resolve()
    sbom = _load_json_object((root / sbom_binding.path).resolve(), "dependency SBOM")
    report = _load_json_object((root / report_binding.path).resolve(), "dependency report")
    sbom_sha = _self_hash(sbom, "sbom_sha256")
    _self_hash(report, "evidence_sha256")
    if sbom.get("schema_version") != DEPENDENCY_SBOM_SCHEMA:
        raise LiveAuthorityError("unsupported dependency SBOM schema")
    if report.get("schema_version") != DEPENDENCY_EVIDENCE_SCHEMA:
        raise LiveAuthorityError("unsupported dependency security evidence schema")
    for label, document in (("SBOM", sbom), ("dependency report", report)):
        if document.get("repository_full_name") != CANONICAL_REPOSITORY:
            raise LiveAuthorityError(f"{label} repository identity mismatch")
        if document.get("source_sha") != attestation.candidate_sha:
            raise LiveAuthorityError(f"{label} source SHA is stale for candidate")
    expected_lock = {
        "semantic_sha256": attestation.dependency_lock.index_sha256,
        "file_sha256": attestation.dependency_lock.file_sha256,
    }
    if sbom.get("lock_index") != expected_lock or report.get("lock_index") != expected_lock:
        raise LiveAuthorityError("dependency evidence lock identity differs from attestation")
    if report.get("sbom_sha256") != sbom_sha:
        raise LiveAuthorityError("dependency report is not bound to exact candidate SBOM")
    profiles = sbom.get("profiles")
    if not isinstance(profiles, Mapping) or set(profiles) != set(SUPPORTED_PROFILES):
        raise LiveAuthorityError("dependency SBOM profile set is incomplete")
    expected_components: dict[str, dict[str, Any]] = {}
    for profile_id, raw_profile in profiles.items():
        profile = _require_mapping(raw_profile, f"SBOM profile {profile_id}")
        raw_components = profile.get("components")
        if not isinstance(raw_components, list) or not raw_components:
            raise LiveAuthorityError(f"SBOM profile {profile_id} has no components")
        if profile.get("component_count") != len(raw_components):
            raise LiveAuthorityError(f"SBOM profile {profile_id} component count mismatch")
        seen_profile: set[str] = set()
        for raw_component in raw_components:
            component = _require_mapping(raw_component, "SBOM component")
            name = component.get("name")
            version = component.get("version")
            if not isinstance(name, str) or not name.strip():
                raise LiveAuthorityError("SBOM component name is missing")
            if not isinstance(version, str) or not version.strip():
                raise LiveAuthorityError("SBOM component version is missing")
            canonical_name = canonical_distribution_name(name)
            key = f"{canonical_name}=={version}"
            if key in seen_profile:
                raise LiveAuthorityError("SBOM profile contains duplicate components")
            seen_profile.add(key)
            if component.get("purl") != f"pkg:pypi/{canonical_name}@{version}":
                raise LiveAuthorityError("SBOM component PURL mismatch")
            artifact_hashes = component.get("artifact_sha256")
            if (
                not isinstance(artifact_hashes, list)
                or not artifact_hashes
                or any(
                    not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None
                    for value in artifact_hashes
                )
            ):
                raise LiveAuthorityError("SBOM component artifact hashes are incomplete")
            current = expected_components.setdefault(
                key,
                {
                    "name": canonical_name,
                    "version": version,
                    "purl": component["purl"],
                    "profiles": set(),
                },
            )
            if current["version"] != version or current["purl"] != component["purl"]:
                raise LiveAuthorityError("SBOM component identity conflicts across profiles")
            current["profiles"].add(profile_id)

    sources = report.get("scan_sources")
    if not isinstance(sources, Mapping) or set(sources) != {"osv", "pypi"}:
        raise LiveAuthorityError("dependency scan source set is incomplete")
    for source in sources.values():
        if not isinstance(source, Mapping) or source.get("status") != "SUCCESS":
            raise LiveAuthorityError("dependency scan source is not successful")

    records = report.get("components")
    if not isinstance(records, list) or not records:
        raise LiveAuthorityError("dependency report component evidence is missing")
    actual_keys: set[str] = set()
    for raw_record in records:
        record = _require_mapping(raw_record, "dependency report component")
        key = record.get("key")
        if not isinstance(key, str) or key in actual_keys:
            raise LiveAuthorityError("dependency report component key is invalid or duplicated")
        expected = expected_components.get(key)
        if expected is None:
            raise LiveAuthorityError("dependency report contains component outside exact SBOM")
        actual_keys.add(key)
        for field_name in ("name", "version", "purl"):
            if record.get(field_name) != expected[field_name]:
                raise LiveAuthorityError(
                    f"dependency report component {field_name} differs from SBOM"
                )
        profiles_value = record.get("profiles")
        if not isinstance(profiles_value, list) or set(profiles_value) != expected["profiles"]:
            raise LiveAuthorityError("dependency report component profile membership mismatch")
        license_record = _require_mapping(record.get("license"), "dependency license evidence")
        if license_record.get("status") != "DECLARED":
            raise LiveAuthorityError("dependency license evidence remains unresolved")
        license_digest = license_record.get("metadata_sha256")
        if not isinstance(license_digest, str) or _SHA256_RE.fullmatch(license_digest) is None:
            raise LiveAuthorityError("dependency license metadata digest is invalid")
        advisory = _require_mapping(record.get("advisories"), "dependency advisory evidence")
        if advisory.get("status") != "QUERIED":
            raise LiveAuthorityError("dependency advisory source was not queried")
        advisory_digest = advisory.get("response_sha256")
        if not isinstance(advisory_digest, str) or _SHA256_RE.fullmatch(advisory_digest) is None:
            raise LiveAuthorityError("dependency advisory response digest is invalid")
        if advisory.get("vulnerabilities") != []:
            raise LiveAuthorityError("dependency vulnerability findings remain unresolved")
    if actual_keys != set(expected_components):
        raise LiveAuthorityError("dependency report does not cover the exact SBOM component set")

    if report.get("status") != DEPENDENCY_CLEAN_STATUS:
        raise LiveAuthorityError("dependency evidence still requires vulnerability/license review")
    if report.get("truth_boundary") != {
        "audit_verdict": False,
        "license_approval": False,
        "vulnerability_risk_acceptance": False,
        "promotion_authority": False,
    }:
        raise LiveAuthorityError("dependency evidence truth boundary was weakened")


def validate_live_promotion_authority(
    attestation: ReleaseAttestation,
    *,
    repo_root: str | Path = ".",
    artifact_root: str | Path | None = None,
    get_json: JsonGetter = github_json_get,
) -> StageCandidateManifest | None:
    """Compose offline attestation validation with live GitHub promotion authority."""

    stage_manifest = validate_release_attestation(
        attestation,
        repo_root=repo_root,
        artifact_root=artifact_root,
    )
    material_root = (
        Path(artifact_root).resolve()
        if artifact_root is not None
        else Path(repo_root).resolve()
    )

    if attestation.status is CandidateStatus.EXPERIMENTAL:
        runs: dict[tuple[int, str], Mapping[str, Any]] = {}
        for environment in attestation.environment_evidence:
            key = (environment.run_id, environment.source_sha)
            if key not in runs:
                runs[key] = verify_workflow_run(
                    run_id=environment.run_id,
                    expected_head_sha=environment.source_sha,
                    evidence_ref=environment.evidence_ref,
                    get_json=get_json,
                )
            _verify_environment_artifacts(
                attestation,
                run_id=environment.run_id,
                expected_head_sha=environment.source_sha,
                get_json=get_json,
            )
        return stage_manifest

    if (
        stage_manifest is None
        or attestation.candidate_sha is None
        or attestation.candidate_ci is None
    ):
        raise LiveAuthorityError("gated live authority validation lacks exact candidate evidence")
    candidate_ci: CandidateCIEvidence = attestation.candidate_ci
    verify_workflow_run(
        run_id=candidate_ci.run_id,
        expected_head_sha=attestation.candidate_sha,
        evidence_ref=candidate_ci.evidence_ref,
        get_json=get_json,
        expected_completed_at=candidate_ci.completed_at_utc,
    )

    for component in stage_manifest.components:
        if component.disposition is not ComponentDisposition.ACCEPTED:
            continue
        if component.ci_evidence is None:
            raise LiveAuthorityError(f"accepted component {component.lane} lacks CI evidence")
        verify_workflow_run(
            run_id=component.ci_evidence.run_id,
            expected_head_sha=component.source_sha,
            evidence_ref=component.ci_evidence.evidence_ref,
            get_json=get_json,
        )

    artifacts = _verify_environment_artifacts(
        attestation,
        run_id=candidate_ci.run_id,
        expected_head_sha=attestation.candidate_sha,
        get_json=get_json,
    )
    _verify_candidate_artifact_refs(
        attestation,
        run_id=candidate_ci.run_id,
        expected_head_sha=attestation.candidate_sha,
        artifacts=artifacts,
    )
    _verify_supply_chain_artifact_refs(
        attestation,
        expected_head_sha=attestation.candidate_sha,
        get_json=get_json,
    )
    verify_dependency_security_evidence(attestation, artifact_root=material_root)

    audits = ((13, stage_manifest.audit_a), (14, stage_manifest.audit_b))
    for issue_number, audit in audits:
        if audit is not None:
            verify_audit_evidence(
                audit,
                expected_issue=issue_number,
                candidate_ci_completed_at=candidate_ci.completed_at_utc,
                get_json=get_json,
            )

    if attestation.status is CandidateStatus.STABLE:
        if attestation.promotion_authority_ref is None:
            raise LiveAuthorityError("STABLE lacks external promotion authority")
        audit_cutoffs = [
            _parse_timestamp(audit.cutoff_utc, "audit cutoff_utc")
            for audit in (stage_manifest.audit_a, stage_manifest.audit_b)
            if audit is not None
        ]
        not_before = max(
            [
                _parse_timestamp(
                    candidate_ci.completed_at_utc,
                    "candidate CI completed_at_utc",
                ),
                *audit_cutoffs,
            ]
        )
        verify_promotion_authority(
            attestation.promotion_authority_ref,
            candidate_sha=attestation.candidate_sha,
            not_before=not_before,
            get_json=get_json,
        )
    return stage_manifest
