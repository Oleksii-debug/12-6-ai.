"""NEXT100-041 bounded CPython source-code admission authority."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import urllib.request
from pathlib import Path
from typing import Any

from twelve_six.data.code_normalization import decode_code_bytes
from twelve_six.data.external_sources import (
    PROJECT_RIGHTS_POLICY_REF,
    RIGHTS_APPROVED,
    USE_ALLOWED,
    EligibilityResolver,
    ExternalSourceSpec,
    RightsDecision,
    RightsEvidenceRef,
    SnapshotSpec,
    UsePermissions,
    build_external_source_registry,
    verify_local_snapshot,
)

SCHEMA = "12-6.next100-041-cpython-code-admission.v1"
AUTHORITY = "EXTERNAL_REAL_CODE_D03_ADMISSION_LOCAL_FREE"
POLICY_PATH = Path("configs/data/next100_041_cpython_code_rights_policy_v1.json")
REPORT_NAME = "next100-041-cpython-code-admission.json"
NORMALIZATION_POLICY = "STRICT_UTF8_IDENTITY_PRESERVE_V1"
NEAR_SHINGLE_SIZE = 5
NEAR_THRESHOLD = 0.85

_SECRET_PATTERNS = {
    "private_key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "github_token": re.compile(rb"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    "openai_key": re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "aws_access_key": re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    "jwt": re.compile(rb"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
}
_EMAIL_PATTERN = re.compile(rb"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_ALT_LICENSE_PATTERN = re.compile(
    rb"(?im)^\s*(?:#\s*)?(?:SPDX-License-Identifier\s*:|Licensed under|License:)"
)


class CpythonAdmissionError(RuntimeError):
    pass


def _cjson(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def _download(url: str, *, max_bytes: int) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "12-6-NEXT100-041-cpython-admission/1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise CpythonAdmissionError(f"bounded download exceeded {max_bytes} bytes: {url}")
    return data


def _require_head(repo: Path, source_sha: str) -> None:
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if actual != source_sha:
        raise CpythonAdmissionError(f"exact-head mismatch: {actual} != {source_sha}")


def _load_policy(repo: Path) -> dict[str, Any]:
    policy = json.loads((repo / POLICY_PATH).read_bytes())
    if policy.get("schema_version") != "12-6.next100-041-cpython-code-rights-policy.v1":
        raise CpythonAdmissionError("unsupported NEXT100-041 rights policy schema")
    if policy.get("worker_id") != "NEXT100-041-CODE-CPYTHON":
        raise CpythonAdmissionError("worker identity mismatch")
    if policy.get("policy_ref") != PROJECT_RIGHTS_POLICY_REF:
        raise CpythonAdmissionError("project rights policy mismatch")
    inventory = policy.get("inventory")
    limits = policy.get("limits", {})
    if not isinstance(inventory, list) or len(inventory) != limits.get("max_files"):
        raise CpythonAdmissionError("inventory must exactly fill the declared bounded file count")
    return policy


def _assert_code_only_path(path: str) -> None:
    posix = Path(path).as_posix()
    if not posix.startswith("Lib/") or not posix.endswith(".py"):
        raise CpythonAdmissionError(f"only first-party Lib/*.py code is admissible: {path}")
    lowered = posix.casefold()
    banned = ("lib/test/", "doc/", "/vendor/", "/vendored/", "/third_party/", "/generated/")
    if any(part in lowered for part in banned):
        raise CpythonAdmissionError(f"documentation/test/vendored/generated path excluded: {path}")


def _scan_privacy_and_secrets(raw: bytes, path: str) -> dict[str, Any]:
    secret_hits = {name: len(pattern.findall(raw)) for name, pattern in _SECRET_PATTERNS.items()}
    email_hits = len(_EMAIL_PATTERN.findall(raw))
    if any(secret_hits.values()):
        raise CpythonAdmissionError(f"secret-like material excluded: {path}: {secret_hits}")
    if email_hits:
        raise CpythonAdmissionError(f"personal email-like material excluded from bounded code object: {path}")
    return {"secret_pattern_hits": secret_hits, "email_like_hits": email_hits, "passed": True}


def _assert_no_file_local_alternate_license(raw: bytes, path: str) -> None:
    if _ALT_LICENSE_PATTERN.search(raw[:8192]):
        raise CpythonAdmissionError(f"file-local alternate-license marker requires separate review: {path}")


def _parse_python(text: str, path: str) -> dict[str, Any]:
    tree = ast.parse(text, filename=path, mode="exec")
    return {"passed": True, "ast_node_count": sum(1 for _ in ast.walk(tree))}


def _token_shingles(text: str) -> set[tuple[str, ...]]:
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\s]", text.casefold())
    if len(tokens) < NEAR_SHINGLE_SIZE:
        return set()
    return {tuple(tokens[i : i + NEAR_SHINGLE_SIZE]) for i in range(len(tokens) - NEAR_SHINGLE_SIZE + 1)}


def _near_jaccard(left: str, right: str) -> float:
    a = _token_shingles(left)
    b = _token_shingles(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _repository_identity(repo_url: str) -> dict[str, Any]:
    if repo_url != "https://github.com/python/cpython":
        raise CpythonAdmissionError("canonical CPython repository URL mismatch")
    payload = json.loads(_download("https://api.github.com/repos/python/cpython", max_bytes=100_000))
    if payload.get("html_url") != repo_url:
        raise CpythonAdmissionError("GitHub canonical repository identity mismatch")
    if payload.get("fork") is not False or payload.get("mirror_url") is not None:
        raise CpythonAdmissionError("fork/mirror CPython source is excluded")
    return {"repository": repo_url, "fork": False, "mirror": False}


def _family_identity(policy: dict[str, Any]) -> str:
    upstream = policy["upstream"]
    core = {
        "canonical_family_id": upstream["canonical_family_id"],
        "family_scope": upstream["family_scope"],
        "repository": upstream["repository"],
    }
    return _sha256(_cjson(core))


def _rights(
    policy: dict[str, Any],
    item: dict[str, Any],
    *,
    license_sha256: str,
    policy_sha256: str,
) -> RightsDecision:
    commit = policy["upstream"]["commit"]
    source_version = f"git:{commit}"
    captured_at = "2026-08-26T18:01:00Z"
    license_data = policy["license"]
    permissions = UsePermissions(
        acquisition=USE_ALLOWED,
        storage=USE_ALLOWED,
        analysis=USE_ALLOWED,
        model_training=USE_ALLOWED,
        redistribution=USE_ALLOWED,
    )
    return RightsDecision(
        status=RIGHTS_APPROVED,
        license_id=license_data["license_id"],
        terms_url=license_data["url"],
        allows_model_training=True,
        allows_derivatives=True,
        allows_redistribution=True,
        policy_ref=PROJECT_RIGHTS_POLICY_REF,
        reviewed_at=policy["reviewed_at"],
        reviewer_ref=policy["reviewer_ref"],
        uses=permissions,
        evidence_refs=(
            RightsEvidenceRef(
                evidence_id=f"{item['source_id']}.license",
                evidence_kind="license_text",
                uri=license_data["url"],
                sha256=license_sha256,
                captured_at=captured_at,
                source_id=item["source_id"],
                source_version=source_version,
            ),
            RightsEvidenceRef(
                evidence_id=f"{item['source_id']}.policy-decision",
                evidence_kind="policy_decision",
                uri=f"file:{POLICY_PATH.as_posix()}",
                sha256=policy_sha256,
                captured_at=captured_at,
                source_id=item["source_id"],
                source_version=source_version,
            ),
        ),
    )


def _materialize_one(
    repo: Path,
    output: Path,
    policy: dict[str, Any],
    item: dict[str, Any],
    *,
    license_sha256: str,
    policy_sha256: str,
) -> tuple[ExternalSourceSpec, dict[str, Any], str]:
    limits = policy["limits"]
    _assert_code_only_path(item["path"])
    raw = _download(item["raw_url"], max_bytes=limits["max_bytes_per_file"])
    if _git_blob_sha1(raw) != item["blob_sha1"]:
        raise CpythonAdmissionError(f"raw Git blob identity drift: {item['source_id']}")
    _assert_no_file_local_alternate_license(raw, item["path"])
    privacy = _scan_privacy_and_secrets(raw, item["path"])
    normalized, norm = decode_code_bytes(raw, language="python", path=item["path"])
    normalized_bytes = normalized.encode("utf-8")
    if normalized_bytes != raw or norm.policy != NORMALIZATION_POLICY:
        raise CpythonAdmissionError(f"identity-preserving normalization failed: {item['source_id']}")
    parse = _parse_python(normalized, item["path"])
    raw_sha256 = _sha256(raw)
    if raw_sha256 in set(policy["current_registry_binding"]["existing_raw_sha256"]):
        raise CpythonAdmissionError(f"current registry exact duplicate: {item['source_id']}")

    snapshot_rel = Path("data/external/snapshots/sha256") / raw_sha256 / "payload"
    snapshot_path = repo / snapshot_rel
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_bytes(raw)
    snapshot = SnapshotSpec(
        uri=f"file:{snapshot_rel.as_posix()}",
        sha256=raw_sha256,
        size_bytes=len(raw),
        retrieved_at="2026-08-26T18:01:00Z",
        upstream_version=f"git:{policy['upstream']['commit']}",
        retrieval_method="NEXT100-041 exact raw GitHub fetch + Git blob SHA-1 gate",
    )
    verify_local_snapshot(snapshot, snapshot_path)
    source = ExternalSourceSpec(
        source_id=item["source_id"],
        source_version=f"git:{policy['upstream']['commit']}",
        provider="Python Software Foundation / CPython contributors",
        source_url=item["raw_url"],
        source_kind="source_code",
        purpose="pretraining",
        synthetic=False,
        benchmark_material=False,
        held_out=False,
        snapshot=snapshot,
        rights=_rights(policy, item, license_sha256=license_sha256, policy_sha256=policy_sha256),
    )
    evidence = {
        "source_id": item["source_id"],
        "source_family": policy["upstream"]["canonical_family_id"],
        "commit": policy["upstream"]["commit"],
        "path": item["path"],
        "git_blob_sha1": item["blob_sha1"],
        "raw_sha256": raw_sha256,
        "raw_bytes": len(raw),
        "normalized_sha256": norm.normalized_sha256,
        "normalized_bytes": len(normalized_bytes),
        "normalization_policy": norm.policy,
        "parse_validity": parse,
        "privacy_secret_scan": privacy,
        "file_local_alternate_license_marker": False,
        "license_id": policy["license"]["license_id"],
        "license_blob_sha1": policy["license"]["blob_sha1"],
        "license_sha256": license_sha256,
        "training_purpose_decision": "ALLOWED",
        "redistribution_decision": "ALLOWED",
        "redistribution_conditions": policy["license"]["redistribution_conditions"],
        "evaluation_use": "NOT_ADMITTED",
        "reserved_for_evaluation": False,
        "source_manifest_sha256": source.source_manifest_sha256,
    }
    return source, evidence, normalized


def _deduplicate(policy: dict[str, Any], objects: list[dict[str, Any]], texts: list[tuple[str, str]]) -> dict[str, Any]:
    hashes = [item["raw_sha256"] for item in objects]
    exact_internal = sorted({digest for digest in hashes if hashes.count(digest) > 1})
    if exact_internal:
        raise CpythonAdmissionError(f"internal exact duplicates: {exact_internal}")

    near_internal: list[dict[str, Any]] = []
    for i, (left_id, left_text) in enumerate(texts):
        for right_id, right_text in texts[i + 1 :]:
            score = _near_jaccard(left_text, right_text)
            if score >= NEAR_THRESHOLD:
                near_internal.append({"left": left_id, "right": right_id, "jaccard": score})
    if near_internal:
        raise CpythonAdmissionError(f"internal near duplicates: {near_internal}")

    cross_registry: list[dict[str, Any]] = []
    for existing in policy["current_registry_binding"]["existing_code_objects"]:
        raw = _download(existing["raw_url"], max_bytes=250_000)
        if _sha256(raw) != existing["raw_sha256"]:
            raise CpythonAdmissionError(f"bound registry comparison object drift: {existing['source_id']}")
        existing_text = raw.decode("utf-8", errors="strict")
        for source_id, text in texts:
            score = _near_jaccard(text, existing_text)
            cross_registry.append({"candidate": source_id, "existing": existing["source_id"], "jaccard": score})
            if score >= NEAR_THRESHOLD:
                raise CpythonAdmissionError(
                    f"near duplicate with current code registry: {source_id} vs {existing['source_id']} ({score})"
                )
    return {
        "exact_internal_sha256": exact_internal,
        "near_internal": near_internal,
        "current_registry_exact_overlap": [],
        "current_registry_code_near_pairs": cross_registry,
        "near_duplicate_threshold": NEAR_THRESHOLD,
        "passed": True,
    }


def run(repo: Path, output: Path, source_sha: str) -> dict[str, Any]:
    _require_head(repo, source_sha)
    policy = _load_policy(repo)
    policy_bytes = (repo / POLICY_PATH).read_bytes()
    policy_sha256 = _sha256(policy_bytes)
    repository = _repository_identity(policy["upstream"]["repository"])

    license_data = policy["license"]
    license_bytes = _download(license_data["url"], max_bytes=100_000)
    if _git_blob_sha1(license_bytes) != license_data["blob_sha1"]:
        raise CpythonAdmissionError("PSF LICENSE Git blob identity drift")
    license_sha256 = _sha256(license_bytes)
    rights_dir = output / "rights-evidence"
    rights_dir.mkdir(parents=True, exist_ok=True)
    (rights_dir / "cpython-LICENSE.txt").write_bytes(license_bytes)

    sources: list[ExternalSourceSpec] = []
    objects: list[dict[str, Any]] = []
    texts: list[tuple[str, str]] = []
    total = 0
    for item in policy["inventory"]:
        source, evidence, text = _materialize_one(
            repo,
            output,
            policy,
            item,
            license_sha256=license_sha256,
            policy_sha256=policy_sha256,
        )
        total += evidence["raw_bytes"]
        sources.append(source)
        objects.append(evidence)
        texts.append((source.source_id, text))
    if total > policy["limits"]["max_total_raw_bytes"]:
        raise CpythonAdmissionError("bounded total raw byte limit exceeded")

    dedup = _deduplicate(policy, objects, texts)
    registry = build_external_source_registry(sources)
    eligibility = EligibilityResolver(registry).inventory()
    if eligibility["model_training_allowed"] != len(objects) or eligibility["model_training_blocked"] != 0:
        raise CpythonAdmissionError("D03 model-training eligibility did not admit every bounded CPython object")

    family_identity = _family_identity(policy)
    core = {
        "schema_version": SCHEMA,
        "authority": AUTHORITY,
        "worker_id": policy["worker_id"],
        "local_free_only": True,
        "source_head": source_sha,
        "upstream": {
            **repository,
            "commit": policy["upstream"]["commit"],
            "tree": policy["upstream"]["tree"],
            "canonical_family_id": policy["upstream"]["canonical_family_id"],
            "family_identity_sha256": family_identity,
            "independence_boundary": "One independent upstream family. Multiple files do not count as multiple families; any future CPython documentation snapshot must collapse to this same upstream family for independence accounting.",
        },
        "scope": {
            "modality": "code",
            "language": "python",
            "files": len(objects),
            "raw_bytes": total,
            "docs_included": False,
            "tests_included": False,
            "evaluation_objects_included": False,
        },
        "license_evidence": {
            "license_id": license_data["license_id"],
            "path": license_data["path"],
            "git_blob_sha1": license_data["blob_sha1"],
            "sha256": license_sha256,
            "redistribution_conditions": license_data["redistribution_conditions"],
            "policy_sha256": policy_sha256,
            "training_purpose_authority": policy["training_purpose_authority"],
        },
        "evaluation_boundary": policy["evaluation_boundary"],
        "registry_concurrency_binding": policy["current_registry_binding"],
        "objects": objects,
        "deduplication": dedup,
        "d03_registry": registry,
        "eligibility_inventory": eligibility,
        "terminal_verdict": "ADMIT",
        "truth_boundary": "Bounded external-real CPython code source admission for model training only; no evaluation permission, representativeness, benchmark cleanliness beyond the recorded reservation check, or whole-CPython-corpus admission is claimed.",
    }
    report = {**core, "authority_identity_sha256": _sha256(_cjson(core))}
    output.mkdir(parents=True, exist_ok=True)
    (output / REPORT_NAME).write_bytes(_cjson(report))
    return report


def validate(report: dict[str, Any], *, expected_source_sha: str | None = None) -> None:
    if report.get("schema_version") != SCHEMA or report.get("authority") != AUTHORITY:
        raise CpythonAdmissionError("authority schema/type mismatch")
    if expected_source_sha is not None and report.get("source_head") != expected_source_sha:
        raise CpythonAdmissionError("report source head mismatch")
    identity = report.get("authority_identity_sha256")
    core = {k: v for k, v in report.items() if k != "authority_identity_sha256"}
    if identity != _sha256(_cjson(core)):
        raise CpythonAdmissionError("authority identity mismatch")
    if report.get("terminal_verdict") != "ADMIT":
        raise CpythonAdmissionError("terminal verdict is not ADMIT")
    scope = report["scope"]
    if scope != {
        "modality": "code",
        "language": "python",
        "files": 3,
        "raw_bytes": scope["raw_bytes"],
        "docs_included": False,
        "tests_included": False,
        "evaluation_objects_included": False,
    }:
        raise CpythonAdmissionError("scope boundary mismatch")
    if len(report["objects"]) != 3:
        raise CpythonAdmissionError("bounded inventory count mismatch")
    if any(not item["parse_validity"]["passed"] for item in report["objects"]):
        raise CpythonAdmissionError("parse validity missing")
    if any(not item["privacy_secret_scan"]["passed"] for item in report["objects"]):
        raise CpythonAdmissionError("privacy/secret scan missing")
    if any(item["reserved_for_evaluation"] for item in report["objects"]):
        raise CpythonAdmissionError("evaluation-reserved object admitted")
    if any(item["evaluation_use"] != "NOT_ADMITTED" for item in report["objects"]):
        raise CpythonAdmissionError("evaluation use boundary drift")
    if not report["deduplication"]["passed"]:
        raise CpythonAdmissionError("deduplication did not pass")
    eligibility = report["eligibility_inventory"]
    if eligibility["model_training_allowed"] != 3 or eligibility["model_training_blocked"] != 0:
        raise CpythonAdmissionError("D03 eligibility mismatch")


def _main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--repo-root", default=".")
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument("--source-sha", required=True)
    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("report")
    validate_parser.add_argument("--expected-source-sha")
    args = parser.parse_args()
    if args.command == "run":
        report = run(Path(args.repo_root), Path(args.output_dir), args.source_sha)
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    else:
        report = json.loads(Path(args.report).read_bytes())
        validate(report, expected_source_sha=args.expected_source_sha)
        print(json.dumps({"status": "PASS", "authority_identity_sha256": report["authority_identity_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    _main()
