"""NEXT100-044 bounded FastAPI source-code admission authority."""

from __future__ import annotations

import argparse
import ast
import hashlib
import ipaddress
import json
import re
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from twelve_six.data.code_normalization import decode_code_bytes

SCHEMA = "12-6.next100-044-fastapi-code-admission.v1"
WORKER_ID = "NEXT100-044-CODE-FASTAPI"
POLICY_PATH = Path("configs/data/next100_044_fastapi_code_rights_policy_v1.json")
REPORT_NAME = "next100-044-fastapi-code-admission.json"
NORMALIZATION_POLICY = "STRICT_UTF8_IDENTITY_PRESERVE_V1"
NEAR_SHINGLE_SIZE = 5
NEAR_THRESHOLD = 0.85

_SECRET_PATTERNS = {
    "private_key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "github_token": re.compile(rb"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    "openai_key": re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "aws_access_key": re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    "jwt": re.compile(rb"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
    "credential_assignment": re.compile(
        rb"(?i)\b(?:api[_-]?key|password|passwd|secret|access[_-]?token)\b\s*[:=]\s*[\"'][^\"'\r\n]{12,}[\"']"
    ),
}
_EMAIL_PATTERN = re.compile(rb"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_URL_PATTERN = re.compile(rb"https?://[A-Za-z0-9._:\-\[\]]+(?:/[^\s\"']*)?")
_ALT_LICENSE_PATTERN = re.compile(
    rb"(?im)^\s*(?:#\s*)?(?:SPDX-License-Identifier\s*:|Licensed under|License:)"
)


class FastapiAdmissionError(RuntimeError):
    pass


def _cjson(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def _download(url: str, *, max_bytes: int) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "12-6-NEXT100-044-fastapi-admission/1"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise FastapiAdmissionError(f"bounded download exceeded {max_bytes} bytes: {url}")
    return data


def _require_head(repo: Path, source_sha: str) -> None:
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if actual != source_sha:
        raise FastapiAdmissionError(f"exact-head mismatch: {actual} != {source_sha}")


def _load_policy(repo: Path) -> dict[str, Any]:
    policy = json.loads((repo / POLICY_PATH).read_text(encoding="utf-8"))
    if policy.get("schema_version") != "12-6.next100-044-fastapi-code-rights-policy.v1":
        raise FastapiAdmissionError("unsupported NEXT100-044 rights policy schema")
    if policy.get("worker_id") != WORKER_ID:
        raise FastapiAdmissionError("worker identity mismatch")
    inventory = policy.get("inventory")
    limits = policy.get("limits", {})
    if not isinstance(inventory, list) or len(inventory) != limits.get("max_files"):
        raise FastapiAdmissionError("inventory must exactly fill declared bounded file count")
    if policy["upstream"]["canonical_family_id"] in policy["current_registry_binding"]["existing_families"]:
        raise FastapiAdmissionError("FastAPI family already present in bound registry")
    if policy["evaluation_boundary"]["code_record_count"] != 0:
        raise FastapiAdmissionError("bound code evaluation set is non-empty; exact overlap proof required")
    return policy


def _assert_code_only_path(path: str) -> None:
    allowed = {
        "fastapi/encoders.py",
        "fastapi/exceptions.py",
        "fastapi/datastructures.py",
    }
    if path not in allowed:
        raise FastapiAdmissionError(f"path outside bounded FastAPI implementation inventory: {path}")
    lowered = path.casefold()
    banned = ("tests/", "docs/", "docs_src/", "fixtures/", "vendor/", "vendored/", "third_party/", "generated/")
    if any(part in lowered for part in banned) or not lowered.endswith(".py"):
        raise FastapiAdmissionError(f"test/doc/vendored/generated/non-Python path excluded: {path}")


def _is_private_host(host: str) -> bool:
    host = host.strip("[]").casefold()
    if host in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved


def _private_endpoint_hits(raw: bytes) -> list[str]:
    hits: list[str] = []
    for match in _URL_PATTERN.finditer(raw):
        url = match.group(0).decode("utf-8", errors="ignore")
        host = urllib.parse.urlsplit(url).hostname
        if host and _is_private_host(host):
            hits.append(url)
    return sorted(set(hits))


def _scan_privacy_and_secrets(raw: bytes, path: str) -> dict[str, Any]:
    secret_hits = {name: len(pattern.findall(raw)) for name, pattern in _SECRET_PATTERNS.items()}
    email_hits = len(_EMAIL_PATTERN.findall(raw))
    private_endpoints = _private_endpoint_hits(raw)
    if any(secret_hits.values()):
        raise FastapiAdmissionError(f"secret-like material excluded: {path}: {secret_hits}")
    if email_hits:
        raise FastapiAdmissionError(f"email-like personal data excluded: {path}: {email_hits}")
    if private_endpoints:
        raise FastapiAdmissionError(f"private endpoint material excluded: {path}: {private_endpoints}")
    return {
        "secret_pattern_hits": secret_hits,
        "email_like_hits": email_hits,
        "private_endpoint_hits": private_endpoints,
        "passed": True,
    }


def _assert_no_file_local_alternate_license(raw: bytes, path: str) -> None:
    if _ALT_LICENSE_PATTERN.search(raw[:8192]):
        raise FastapiAdmissionError(f"file-local alternate-license marker requires separate review: {path}")


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
    if repo_url != "https://github.com/fastapi/fastapi":
        raise FastapiAdmissionError("canonical FastAPI repository URL mismatch")
    payload = json.loads(_download("https://api.github.com/repos/fastapi/fastapi", max_bytes=100_000))
    if payload.get("html_url") != repo_url:
        raise FastapiAdmissionError("GitHub canonical repository identity mismatch")
    if payload.get("fork") is not False or payload.get("mirror_url") is not None:
        raise FastapiAdmissionError("fork/mirror FastAPI source is excluded")
    return {"repository": repo_url, "fork": False, "mirror": False}


def _family_identity(policy: dict[str, Any]) -> str:
    upstream = policy["upstream"]
    core = {
        "canonical_family_id": upstream["canonical_family_id"],
        "family_scope": upstream["family_scope"],
        "repository": upstream["repository"],
    }
    return _sha256(_cjson(core))


def _verify_license(policy: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    license_data = policy["license"]
    raw = _download(license_data["url"], max_bytes=20_000)
    if len(raw) != license_data["expected_bytes"]:
        raise FastapiAdmissionError("license byte-size drift")
    if _git_blob_sha1(raw) != license_data["blob_sha1"]:
        raise FastapiAdmissionError("license Git blob identity drift")
    text = raw.decode("utf-8", errors="strict")
    required = (
        "MIT License",
        "Permission is hereby granted, free of charge",
        "The above copyright notice and this permission notice",
    )
    if not all(marker in text for marker in required):
        raise FastapiAdmissionError("MIT license grant/notice markers not found")
    return raw, {
        "license_id": license_data["license_id"],
        "path": license_data["path"],
        "git_blob_sha1": license_data["blob_sha1"],
        "sha256": _sha256(raw),
        "bytes": len(raw),
        "training_decision": "ALLOWED",
        "redistribution_decision": "ALLOWED_WITH_NOTICE",
        "redistribution_conditions": license_data["redistribution_conditions"],
    }


def _materialize_one(
    repo: Path,
    policy: dict[str, Any],
    item: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    limits = policy["limits"]
    _assert_code_only_path(item["path"])
    raw = _download(item["raw_url"], max_bytes=limits["max_bytes_per_file"])
    if len(raw) != item["expected_bytes"]:
        raise FastapiAdmissionError(f"raw byte-size drift: {item['source_id']}")
    if _git_blob_sha1(raw) != item["blob_sha1"]:
        raise FastapiAdmissionError(f"raw Git blob identity drift: {item['source_id']}")
    _assert_no_file_local_alternate_license(raw, item["path"])
    privacy = _scan_privacy_and_secrets(raw, item["path"])
    normalized, norm = decode_code_bytes(raw, language="python", path=item["path"])
    normalized_bytes = normalized.encode("utf-8")
    if normalized_bytes != raw or norm.policy != NORMALIZATION_POLICY:
        raise FastapiAdmissionError(f"identity-preserving normalization failed: {item['source_id']}")
    parse = _parse_python(normalized, item["path"])
    raw_sha256 = _sha256(raw)
    if raw_sha256 in set(policy["current_registry_binding"]["existing_raw_sha256"]):
        raise FastapiAdmissionError(f"current registry exact duplicate: {item['source_id']}")

    snapshot_rel = Path("data/external/snapshots/sha256") / raw_sha256 / "payload"
    snapshot_path = repo / snapshot_rel
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_bytes(raw)
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
        "training_purpose_decision": "ALLOWED",
        "redistribution_decision": "ALLOWED_WITH_NOTICE",
        "evaluation_use": "NOT_ADMITTED",
        "benchmark_material": False,
        "held_out": False,
        "reserved_for_evaluation": False,
        "snapshot_uri": f"file:{snapshot_rel.as_posix()}",
    }
    return evidence, normalized


def _deduplicate(policy: dict[str, Any], objects: list[dict[str, Any]], texts: list[tuple[str, str]]) -> dict[str, Any]:
    hashes = [item["raw_sha256"] for item in objects]
    exact_internal = sorted({digest for digest in hashes if hashes.count(digest) > 1})
    if exact_internal:
        raise FastapiAdmissionError(f"internal exact duplicates: {exact_internal}")

    near_internal: list[dict[str, Any]] = []
    for i, (left_id, left_text) in enumerate(texts):
        for right_id, right_text in texts[i + 1 :]:
            score = _near_jaccard(left_text, right_text)
            pair = {"left": left_id, "right": right_id, "jaccard": round(score, 9)}
            near_internal.append(pair)
            if score >= NEAR_THRESHOLD:
                raise FastapiAdmissionError(f"internal near duplicate: {pair}")

    cross_registry: list[dict[str, Any]] = []
    for existing in policy["current_registry_binding"]["existing_code_objects"]:
        raw = _download(existing["raw_url"], max_bytes=250_000)
        if _sha256(raw) != existing["raw_sha256"]:
            raise FastapiAdmissionError(f"bound registry comparison object drift: {existing['source_id']}")
        if _git_blob_sha1(raw) != existing["git_blob_sha1"]:
            raise FastapiAdmissionError(f"bound registry Git blob drift: {existing['source_id']}")
        existing_text = raw.decode("utf-8", errors="strict")
        for source_id, text in texts:
            score = _near_jaccard(text, existing_text)
            pair = {
                "candidate": source_id,
                "existing": existing["source_id"],
                "existing_family": existing["family"],
                "jaccard": round(score, 9),
            }
            cross_registry.append(pair)
            if score >= NEAR_THRESHOLD:
                raise FastapiAdmissionError(f"near duplicate with current code registry: {pair}")

    return {
        "exact_internal_sha256": exact_internal,
        "near_internal_pairs": near_internal,
        "current_registry_exact_overlap": [],
        "current_registry_code_near_pairs": cross_registry,
        "near_duplicate_threshold": NEAR_THRESHOLD,
        "passed": True,
    }


def _authority_identity(report_without_identity: dict[str, Any]) -> str:
    return _sha256(_cjson(report_without_identity))


def run(repo: Path, output: Path, source_sha: str) -> dict[str, Any]:
    _require_head(repo, source_sha)
    policy = _load_policy(repo)
    policy_sha256 = _sha256((repo / POLICY_PATH).read_bytes())
    repository = _repository_identity(policy["upstream"]["repository"])
    license_bytes, license_evidence = _verify_license(policy)

    license_rel = Path("data/external/licenses/sha256") / license_evidence["sha256"] / "LICENSE"
    license_path = repo / license_rel
    license_path.parent.mkdir(parents=True, exist_ok=True)
    license_path.write_bytes(license_bytes)
    license_evidence["snapshot_uri"] = f"file:{license_rel.as_posix()}"

    objects: list[dict[str, Any]] = []
    texts: list[tuple[str, str]] = []
    total = 0
    for item in policy["inventory"]:
        evidence, text = _materialize_one(repo, policy, item)
        objects.append(evidence)
        texts.append((item["source_id"], text))
        total += evidence["raw_bytes"]
    if total > policy["limits"]["max_total_raw_bytes"]:
        raise FastapiAdmissionError("total raw byte budget exceeded")

    dedup = _deduplicate(policy, objects, texts)
    report: dict[str, Any] = {
        "schema_version": SCHEMA,
        "worker_id": WORKER_ID,
        "terminal_verdict": "ADMIT",
        "execution_profile": "LOCAL_FREE",
        "source_head_sha": source_sha,
        "policy_sha256": policy_sha256,
        "upstream": {
            **repository,
            "commit": policy["upstream"]["commit"],
            "tree": policy["upstream"]["tree"],
            "canonical_family_id": policy["upstream"]["canonical_family_id"],
            "family_identity_sha256": _family_identity(policy),
            "independent_family_credit": 1,
        },
        "license": license_evidence,
        "scope": {
            "files": len(objects),
            "raw_bytes": total,
            "docs_included": False,
            "tests_or_fixtures_included": False,
            "vendored_or_generated_included": False,
            "evaluation_objects_included": False,
        },
        "objects": objects,
        "deduplication": dedup,
        "rights": {
            "training": policy["training_purpose_authority"]["decision"],
            "redistribution": policy["training_purpose_authority"]["redistribution"],
            "basis": policy["training_purpose_authority"]["basis"],
        },
        "evaluation_boundary": {
            **policy["evaluation_boundary"],
            "candidate_exact_overlap_count": 0,
            "candidate_reserved_count": 0,
            "passed": True,
        },
        "registry_binding": {
            "authority": policy["current_registry_binding"]["authority"],
            "producer_head": policy["current_registry_binding"]["producer_head"],
            "registry_identity_sha256": policy["current_registry_binding"]["registry_identity_sha256"],
            "fastapi_family_present_before_admission": False,
        },
        "promotion_boundary": (
            "This is terminal source-specific admission evidence only. It does not mutate DATA-300's frozen source inventory "
            "or grant evaluation use; a successor corpus/source registry must explicitly consume this authority."
        ),
    }
    report["authority_identity_sha256"] = _authority_identity(report)
    output.mkdir(parents=True, exist_ok=True)
    path = output / REPORT_NAME
    path.write_bytes(_cjson(report))
    return report


def validate(path: Path, expected_source_sha: str) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    identity = report.pop("authority_identity_sha256", None)
    if not isinstance(identity, str) or identity != _authority_identity(report):
        raise FastapiAdmissionError("authority identity mismatch")
    report["authority_identity_sha256"] = identity
    if report.get("schema_version") != SCHEMA or report.get("worker_id") != WORKER_ID:
        raise FastapiAdmissionError("authority schema/worker mismatch")
    if report.get("terminal_verdict") != "ADMIT":
        raise FastapiAdmissionError("authority is not ADMIT")
    if report.get("source_head_sha") != expected_source_sha:
        raise FastapiAdmissionError("validated authority does not bind exact worker head")
    if report["scope"]["files"] != 3 or report["scope"]["raw_bytes"] > 30000:
        raise FastapiAdmissionError("bounded scope mismatch")
    if not report["deduplication"]["passed"] or not report["evaluation_boundary"]["passed"]:
        raise FastapiAdmissionError("dedup/evaluation boundary did not pass")
    for obj in report["objects"]:
        if obj["raw_sha256"] != obj["normalized_sha256"]:
            raise FastapiAdmissionError("identity normalization mismatch")
        if not obj["parse_validity"]["passed"] or not obj["privacy_secret_scan"]["passed"]:
            raise FastapiAdmissionError("parse/privacy evidence not passing")
        if obj["reserved_for_evaluation"] or obj["evaluation_use"] != "NOT_ADMITTED":
            raise FastapiAdmissionError("evaluation boundary violation")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--repo-root", type=Path, default=Path("."))
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--source-sha", required=True)
    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("report", type=Path)
    validate_parser.add_argument("--expected-source-sha", required=True)
    args = parser.parse_args()

    if args.command == "run":
        report = run(args.repo_root.resolve(), args.output_dir, args.source_sha)
    else:
        report = validate(args.report, args.expected_source_sha)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
