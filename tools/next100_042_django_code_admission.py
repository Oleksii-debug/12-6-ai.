#!/usr/bin/env python3
"""NEXT100-042 bounded Django source-code admission verifier.

Stdlib-only and LOCAL_FREE. The verifier reacquires exact upstream objects,
checks immutable Git identities, rights evidence, parse/privacy/secret gates,
and deduplicates the proposed Django family against the incumbent DATA-227
code objects. It never grants or reserves evaluation use.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

SCHEMA = "12-6.next100-042-django-code-authority.v1"
AUTHORITY = "NEXT100_042_DJANGO_CODE_SOURCE_ADMISSION"
USER_AGENT = "12-6-NEXT100-042-django-code-admission/1"
SHINGLE_SIZE = 5

SECRET_PATTERNS = {
    "private_key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "github_token": re.compile(rb"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    "openai_like_key": re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "aws_access_key": re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    "jwt_like": re.compile(rb"\beyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
}
EMAIL_PATTERN = re.compile(rb"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\s]")


class AdmissionError(RuntimeError):
    pass


def cjson(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def fetch_bytes(url: str, *, max_bytes: int = 300_000) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=30) as response:
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise AdmissionError(f"bounded fetch exceeded {max_bytes} bytes: {url}")
    return data


def fetch_json(url: str, *, max_bytes: int = 300_000) -> dict[str, Any]:
    value = json.loads(fetch_bytes(url, max_bytes=max_bytes))
    if not isinstance(value, dict):
        raise AdmissionError(f"expected JSON object: {url}")
    return value


def raw_url(repository_url: str, commit: str, path: str) -> str:
    prefix = "https://github.com/"
    if not repository_url.startswith(prefix):
        raise AdmissionError(f"noncanonical GitHub repository: {repository_url}")
    slug = repository_url.removeprefix(prefix).strip("/")
    if slug.count("/") != 1:
        raise AdmissionError(f"invalid GitHub repository slug: {slug}")
    return f"https://raw.githubusercontent.com/{slug}/{commit}/{path}"


def assert_exact_head(repo: Path, expected: str | None) -> str:
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if expected and actual != expected:
        raise AdmissionError(f"exact-head mismatch: {actual} != {expected}")
    return actual


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_bytes())
    if config.get("schema_version") != "12-6.next100-042-django-code-source.v1":
        raise AdmissionError("unsupported NEXT100-042 config schema")
    if config.get("authority_id") != "NEXT100-042-CODE-DJANGO":
        raise AdmissionError("authority id mismatch")
    if config.get("execution_profile") != "LOCAL_FREE":
        raise AdmissionError("NEXT100-042 must remain LOCAL_FREE")
    rights = config.get("rights_decision", {})
    if rights.get("model_training") != "ALLOWED" or rights.get("redistribution") != "ALLOWED":
        raise AdmissionError("training and redistribution must be explicitly ALLOWED")
    if rights.get("evaluation_use") != "NOT_SEPARATELY_AUTHORIZED":
        raise AdmissionError("evaluation use must remain not separately authorized")
    if rights.get("evaluation_reserved") is not False:
        raise AdmissionError("NEXT100-042 must not reserve training objects for evaluation")
    selected = config.get("selected_files")
    if not isinstance(selected, list) or len(selected) != 3:
        raise AdmissionError("NEXT100-042 exact scope is three selected implementation files")
    if len({item["path"] for item in selected}) != len(selected):
        raise AdmissionError("duplicate selected path")
    banned_fragments = ("/tests/", "/test/", "/docs/", "/generated/", "/vendor/", "/vendored/")
    for item in selected:
        path_text = "/" + item["path"].lower()
        if not item["path"].endswith(".py") or any(fragment in path_text for fragment in banned_fragments):
            raise AdmissionError(f"non-substantive or excluded path: {item['path']}")
        if item["path"] in {"django/__init__.py", "django/__main__.py"}:
            raise AdmissionError(f"version/bootstrap metadata excluded: {item['path']}")
    return config


def verify_upstream_identity(config: dict[str, Any]) -> dict[str, Any]:
    upstream = config["upstream"]
    repository_url = upstream["repository_url"]
    slug = repository_url.removeprefix("https://github.com/")
    repo = fetch_json(f"https://api.github.com/repos/{slug}", max_bytes=100_000)
    if repo.get("html_url") != repository_url:
        raise AdmissionError("upstream repository canonical identity mismatch")
    if repo.get("fork") is not False or repo.get("mirror_url") is not None:
        raise AdmissionError("fork/mirror cannot establish an independent source family")

    ref = fetch_json(f"https://api.github.com/repos/{slug}/git/ref/tags/{upstream['tag']}", max_bytes=100_000)
    ref_object = ref.get("object", {})
    if ref_object.get("type") != "tag" or ref_object.get("sha") != upstream["tag_object_sha1"]:
        raise AdmissionError("annotated tag object drift")
    tag = fetch_json(f"https://api.github.com/repos/{slug}/git/tags/{upstream['tag_object_sha1']}", max_bytes=150_000)
    if tag.get("tag") != upstream["tag"]:
        raise AdmissionError("tag name mismatch")
    if tag.get("object", {}).get("type") != "commit" or tag.get("object", {}).get("sha") != upstream["commit"]:
        raise AdmissionError("tag no longer resolves to pinned commit")
    verification = tag.get("verification", {})
    if verification.get("verified") is not True:
        raise AdmissionError("annotated upstream release tag is not GitHub-verified")

    commit = fetch_json(f"https://api.github.com/repos/{slug}/git/commits/{upstream['commit']}", max_bytes=150_000)
    if commit.get("tree", {}).get("sha") != upstream["tree_sha1"]:
        raise AdmissionError("pinned upstream tree drift")
    return {
        "repository_url": repository_url,
        "fork": repo.get("fork"),
        "mirror_url": repo.get("mirror_url"),
        "tag": upstream["tag"],
        "tag_object_sha1": upstream["tag_object_sha1"],
        "tag_signature_verified": True,
        "commit": upstream["commit"],
        "tree_sha1": upstream["tree_sha1"],
    }


def scan_sensitive(data: bytes, path: str) -> dict[str, Any]:
    secret_hits = [name for name, pattern in SECRET_PATTERNS.items() if pattern.search(data)]
    email_hits = sorted({match.decode("ascii", "replace") for match in EMAIL_PATTERN.findall(data)})
    if secret_hits:
        raise AdmissionError(f"secret-like material detected in {path}: {secret_hits}")
    if email_hits:
        raise AdmissionError(f"email/PII-like literal detected in {path}: {email_hits}")
    return {"secret_hits": [], "privacy_email_hits": [], "passed": True}


def token_shingles(text: str) -> set[tuple[str, ...]]:
    tokens = TOKEN_PATTERN.findall(text.casefold())
    if len(tokens) < SHINGLE_SIZE:
        return set()
    return {tuple(tokens[i : i + SHINGLE_SIZE]) for i in range(len(tokens) - SHINGLE_SIZE + 1)}


def jaccard(left: str, right: str) -> float:
    a = token_shingles(left)
    b = token_shingles(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def verify_license_objects(config: dict[str, Any]) -> list[dict[str, Any]]:
    upstream = config["upstream"]
    out: list[dict[str, Any]] = []
    for item in config["licenses"]:
        url = raw_url(upstream["repository_url"], upstream["commit"], item["path"])
        data = fetch_bytes(url)
        if len(data) != item["size_bytes"]:
            raise AdmissionError(f"license size drift: {item['path']}")
        actual_blob = git_blob_sha1(data)
        if actual_blob != item["blob_sha1"]:
            raise AdmissionError(f"license Git blob drift: {item['path']}")
        text = data.decode("utf-8", "strict")
        if item["path"] == "LICENSE":
            required = (
                "Redistribution and use in source and binary forms, with or without modification",
                "Neither the name of Django nor the names of its contributors may be used",
            )
            if not all(fragment in text for fragment in required):
                raise AdmissionError("Django BSD terms evidence missing required grant/condition text")
        if item["path"] == "LICENSE.python":
            required = (
                "Django includes code from the Python standard library",
                "grants Licensee a nonexclusive, royalty-free, world-wide license",
            )
            if not all(fragment in text for fragment in required):
                raise AdmissionError("Python-derived-code compliance evidence missing required text")
        out.append({
            "path": item["path"],
            "spdx": item["spdx"],
            "blob_sha1": actual_blob,
            "raw_sha256": sha256(data),
            "size_bytes": len(data),
            "url": url,
        })
    return out


def verify_code_object(config: dict[str, Any], item: dict[str, Any]) -> tuple[dict[str, Any], str]:
    upstream = config["upstream"]
    url = raw_url(upstream["repository_url"], upstream["commit"], item["path"])
    raw = fetch_bytes(url)
    if len(raw) != item["size_bytes"]:
        raise AdmissionError(f"raw size drift: {item['source_id']}")
    actual_blob = git_blob_sha1(raw)
    if actual_blob != item["blob_sha1"]:
        raise AdmissionError(f"raw Git blob identity drift: {item['source_id']}")
    if b"\x00" in raw:
        raise AdmissionError(f"NUL byte rejected: {item['path']}")
    text = raw.decode("utf-8", "strict")
    if text.encode("utf-8") != raw:
        raise AdmissionError(f"strict UTF-8 identity normalization failed: {item['path']}")
    try:
        tree = ast.parse(text, filename=item["path"], mode="exec")
    except SyntaxError as exc:
        raise AdmissionError(f"Python parse failure in {item['path']}: {exc}") from exc
    sensitive = scan_sensitive(raw, item["path"])
    node_count = sum(1 for _ in ast.walk(tree))
    if node_count < 25:
        raise AdmissionError(f"implementation file is not substantive enough: {item['path']}")
    evidence = {
        "source_id": item["source_id"],
        "source_family": upstream["source_family"],
        "repository_url": upstream["repository_url"],
        "commit": upstream["commit"],
        "path": item["path"],
        "substantive_role": item["substantive_role"],
        "blob_sha1": actual_blob,
        "raw_sha256": sha256(raw),
        "normalization_sha256": sha256(text.encode("utf-8")),
        "size_bytes": len(raw),
        "python_ast_parse": "PASS",
        "ast_node_count": node_count,
        "sensitive_scan": sensitive,
        "training_purpose_decision": "ALLOWED",
        "redistribution_decision": "ALLOWED",
        "evaluation_use_decision": "NOT_SEPARATELY_AUTHORIZED",
        "evaluation_reserved": False,
        "raw_url": url,
    }
    return evidence, text


def acquire_incumbents(config: dict[str, Any]) -> list[tuple[dict[str, Any], str]]:
    acquired: list[tuple[dict[str, Any], str]] = []
    anchor = config["incumbent_code_registry_anchor"]
    if anchor.get("authority") != "DATA-227-REAL-CODE-SOURCE-ADMISSION-V2":
        raise AdmissionError("unexpected incumbent code registry authority")
    for item in anchor["objects"]:
        url = raw_url(item["repository_url"], item["commit"], item["path"])
        raw = fetch_bytes(url)
        if len(raw) != item["size_bytes"] or git_blob_sha1(raw) != item["blob_sha1"]:
            raise AdmissionError(f"incumbent identity drift: {item['source_id']}")
        text = raw.decode("utf-8", "strict")
        acquired.append(({
            "source_id": item["source_id"],
            "repository_url": item["repository_url"],
            "commit": item["commit"],
            "path": item["path"],
            "blob_sha1": item["blob_sha1"],
            "raw_sha256": sha256(raw),
        }, text))
    return acquired


def deduplicate(config: dict[str, Any], proposed: list[tuple[dict[str, Any], str]], incumbents: list[tuple[dict[str, Any], str]]) -> dict[str, Any]:
    threshold = float(config["quality_gates"]["dedup"]["threshold"])
    all_items = proposed + incumbents
    exact_pairs: list[dict[str, str]] = []
    near_pairs: list[dict[str, Any]] = []
    for i, (left, left_text) in enumerate(all_items):
        for right, right_text in all_items[i + 1 :]:
            if left["raw_sha256"] == right["raw_sha256"]:
                exact_pairs.append({"left": left["source_id"], "right": right["source_id"], "sha256": left["raw_sha256"]})
            score = jaccard(left_text, right_text)
            if score >= threshold:
                near_pairs.append({"left": left["source_id"], "right": right["source_id"], "jaccard": round(score, 8)})
    if exact_pairs or near_pairs:
        raise AdmissionError(f"dedup gate failed: exact={exact_pairs}, near={near_pairs}")
    return {
        "exact_method": "raw SHA-256 equality",
        "near_method": f"{SHINGLE_SIZE}-token shingle Jaccard",
        "near_threshold": threshold,
        "exact_duplicate_pairs": [],
        "near_duplicate_pairs": [],
        "compared_object_count": len(all_items),
        "incumbent_object_count": len(incumbents),
        "proposed_object_count": len(proposed),
    }


def build_report(repo: Path, config_path: Path, expected_head: str | None) -> dict[str, Any]:
    head = assert_exact_head(repo, expected_head)
    config = load_config(config_path)
    config_sha = sha256(config_path.read_bytes())
    upstream_evidence = verify_upstream_identity(config)
    license_evidence = verify_license_objects(config)

    proposed: list[tuple[dict[str, Any], str]] = []
    for item in config["selected_files"]:
        proposed.append(verify_code_object(config, item))
    incumbents = acquire_incumbents(config)
    dedup = deduplicate(config, proposed, incumbents)

    incumbent_families = set(config["incumbent_code_registry_anchor"]["families"])
    proposed_family = config["upstream"]["source_family"]
    if proposed_family in incumbent_families:
        raise AdmissionError("Django family already present in incumbent family set")
    if config["upstream"]["repository_url"] in {item[0]["repository_url"] for item in incumbents}:
        raise AdmissionError("Django repository identity collides with incumbent repository")

    objects = [item[0] for item in proposed]
    report = {
        "schema_version": SCHEMA,
        "authority": AUTHORITY,
        "worker_id": "NEXT100-042-CODE-DJANGO",
        "decision": "ADMIT",
        "terminal": True,
        "execution_profile": "LOCAL_FREE",
        "project_source_head": head,
        "config_path": config_path.as_posix(),
        "config_sha256": config_sha,
        "upstream": upstream_evidence,
        "source_family": {
            "id": proposed_family,
            "family_credit": 1,
            "independent_of_incumbents": True,
            "incumbent_families_checked": sorted(incumbent_families),
            "rule": "multiple files from one canonical upstream repository count as one family; forks/mirrors do not create new family credit",
        },
        "selected_object_count": len(objects),
        "selected_raw_bytes": sum(item["size_bytes"] for item in objects),
        "objects": objects,
        "license_evidence": license_evidence,
        "rights": config["rights_decision"],
        "quality": {
            "all_strict_utf8_identity": all(item["raw_sha256"] == item["normalization_sha256"] for item in objects),
            "all_python_ast_parse": all(item["python_ast_parse"] == "PASS" for item in objects),
            "all_privacy_secret_scans_clear": all(item["sensitive_scan"]["passed"] for item in objects),
            "substantive_implementation_only": True,
            "excluded_generated_version_metadata": True,
        },
        "deduplication": dedup,
        "evaluation_boundary": {
            "evaluation_use": "NOT_SEPARATELY_AUTHORIZED",
            "evaluation_reserved": False,
            "statement": "No object in this authority is reserved for evaluation. Evaluation use requires a separate explicit authority.",
        },
        "scope_exclusions": config["scope_exclusions"],
        "late_registry_check_required_before_final": True,
    }
    if report["selected_object_count"] != 3 or report["selected_raw_bytes"] != 54156:
        raise AdmissionError("bounded snapshot cardinality/byte total drift")
    return report


def validate_report(report: dict[str, Any]) -> None:
    if report.get("schema_version") != SCHEMA or report.get("authority") != AUTHORITY:
        raise AdmissionError("report schema/authority mismatch")
    if report.get("decision") != "ADMIT" or report.get("terminal") is not True:
        raise AdmissionError("report is not terminal ADMIT")
    if report.get("execution_profile") != "LOCAL_FREE":
        raise AdmissionError("report violates LOCAL_FREE")
    if report.get("selected_object_count") != 3 or report.get("selected_raw_bytes") != 54156:
        raise AdmissionError("snapshot scope mismatch")
    if report["source_family"]["id"] != "github:django/django" or report["source_family"]["family_credit"] != 1:
        raise AdmissionError("source family identity mismatch")
    if report["source_family"]["independent_of_incumbents"] is not True:
        raise AdmissionError("independent family check did not pass")
    if not report["quality"]["all_strict_utf8_identity"] or not report["quality"]["all_python_ast_parse"]:
        raise AdmissionError("encoding/parse quality gate failed")
    if not report["quality"]["all_privacy_secret_scans_clear"]:
        raise AdmissionError("privacy/secret gate failed")
    if report["deduplication"]["exact_duplicate_pairs"] or report["deduplication"]["near_duplicate_pairs"]:
        raise AdmissionError("dedup report contains collisions")
    if report["rights"]["model_training"] != "ALLOWED" or report["rights"]["redistribution"] != "ALLOWED":
        raise AdmissionError("rights decision mismatch")
    if report["evaluation_boundary"]["evaluation_use"] != "NOT_SEPARATELY_AUTHORIZED":
        raise AdmissionError("evaluation boundary widened")
    if report["evaluation_boundary"]["evaluation_reserved"] is not False:
        raise AdmissionError("evaluation reservation was created")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("configs/data/next100_042_django_code_source_v1.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-head")
    args = parser.parse_args(argv)

    repo = args.repo_root.resolve()
    config_path = (repo / args.config).resolve() if not args.config.is_absolute() else args.config
    output = (repo / args.output).resolve() if not args.output.is_absolute() else args.output
    try:
        report = build_report(repo, config_path, args.expected_head)
        validate_report(report)
    except Exception as exc:
        print(json.dumps({"authority": AUTHORITY, "decision": "REJECT", "terminal": True, "reason": str(exc)}, sort_keys=True))
        return 2
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(cjson(report))
    print(cjson(report).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
