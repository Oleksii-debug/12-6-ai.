#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import keyword
import re
import token
import tokenize
import urllib.request
from pathlib import Path
from typing import Any

SECRET_PATTERNS = {
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    "slack_token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    "openai_key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "credential_url": re.compile(r"https?://[^/\s:@]+:[^/\s@]+@"),
    "literal_credential": re.compile(
        r"(?i)\b(?:password|passwd|secret|api[_-]?key|access[_-]?token)\b\s*=\s*['\"][^'\"\n]{8,}['\"]"
    ),
}
PRIVACY_PATTERNS = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "ipv4": re.compile(r"(?<![0-9])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9])"),
    "private_endpoint": re.compile(
        r"(?i)https?://(?:localhost|127\.0\.0\.1|10\.|192\.168\.|172\.(?:1[6-9]|2[0-9]|3[01])\.)"
    ),
}


def request_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "NEXT100-052-source-verifier"})
    with urllib.request.urlopen(req, timeout=30) as response:
        return response.read()


def api_json(url: str) -> Any:
    return json.loads(request_bytes(url).decode("utf-8"))


def fetch_raw(repository: str, commit: str, path: str) -> bytes:
    return request_bytes(f"https://raw.githubusercontent.com/{repository}/{commit}/{path}")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def normalize_text(data: bytes) -> bytes:
    text = data.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.split("\n")).encode("utf-8")


def python_skeleton(data: bytes) -> list[str]:
    text = data.decode("utf-8")
    out: list[str] = []
    ignored = {
        token.ENCODING,
        token.ENDMARKER,
        token.INDENT,
        token.DEDENT,
        token.NEWLINE,
        tokenize.NL,
        token.COMMENT,
    }
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type in ignored:
            continue
        if tok.type == token.NAME:
            out.append(tok.string if keyword.iskeyword(tok.string) else "ID")
        elif tok.type == token.STRING:
            out.append("STR")
        elif tok.type == token.NUMBER:
            out.append("NUM")
        else:
            out.append(tok.string)
    return out


def shingles(items: list[str], width: int = 5) -> set[tuple[str, ...]]:
    if len(items) < width:
        return {tuple(items)} if items else set()
    return {tuple(items[i : i + width]) for i in range(len(items) - width + 1)}


def similarity(a: bytes, b: bytes) -> dict[str, float | bool]:
    sa = shingles(python_skeleton(a))
    sb = shingles(python_skeleton(b))
    intersection = len(sa & sb)
    union = len(sa | sb)
    minimum = min(len(sa), len(sb))
    return {
        "raw_exact": a == b,
        "normalized_exact": normalize_text(a) == normalize_text(b),
        "skeleton_jaccard": intersection / union if union else 1.0,
        "skeleton_containment": intersection / minimum if minimum else 1.0,
    }


def scan_text(data: bytes) -> dict[str, list[str]]:
    text = data.decode("utf-8")
    return {
        "secret_hits": [name for name, pattern in SECRET_PATTERNS.items() if pattern.search(text)],
        "privacy_hits": [name for name, pattern in PRIVACY_PATTERNS.items() if pattern.search(text)],
    }


def tree_blobs(repository: str, commit: str) -> dict[str, str]:
    commit_obj = api_json(f"https://api.github.com/repos/{repository}/commits/{commit}")
    tree_sha = commit_obj["commit"]["tree"]["sha"]
    tree = api_json(f"https://api.github.com/repos/{repository}/git/trees/{tree_sha}?recursive=1")
    return {
        item["path"]: item["sha"]
        for item in tree["tree"]
        if item.get("type") == "blob"
    }


def rejected(sim: dict[str, float | bool], policy: dict[str, Any]) -> bool:
    return bool(
        sim["raw_exact"]
        or sim["normalized_exact"]
        or float(sim["skeleton_jaccard"])
        >= float(policy["python_skeleton_5gram_jaccard_reject_at_or_above"])
        or float(sim["skeleton_containment"])
        >= float(policy["python_skeleton_5gram_containment_reject_at_or_above"])
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", default="configs/data/next100_052_typer_code_source_v1.json"
    )
    parser.add_argument(
        "--output", default="evidence/next100_052/typer_source_admission.json"
    )
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    upstream = manifest["upstream"]
    repository = upstream["repository"]
    commit = upstream["commit"]
    policy = manifest["dedup_policy"]
    failures: list[str] = []

    source_bytes: dict[str, bytes] = {}
    file_reports: list[dict[str, Any]] = []
    for item in manifest["selected_files"]:
        data = fetch_raw(repository, commit, item["path"])
        source_bytes[item["path"]] = data
        report: dict[str, Any] = {
            "path": item["path"],
            "bytes": len(data),
            "raw_sha256": sha256(data),
            "git_blob_sha1": git_blob_sha1(data),
        }
        try:
            ast.parse(data.decode("utf-8"), filename=item["path"])
            report["parse_valid"] = True
        except (SyntaxError, UnicodeDecodeError) as exc:
            report["parse_valid"] = False
            report["parse_error"] = str(exc)
            failures.append(f"parse:{item['path']}")
        report.update(scan_text(data))
        if report["secret_hits"]:
            failures.append(f"secrets:{item['path']}")
        if report["privacy_hits"]:
            failures.append(f"privacy:{item['path']}")
        for field in ("bytes", "raw_sha256", "git_blob_sha1"):
            if report[field] != item[field]:
                failures.append(f"identity:{item['path']}:{field}")
        file_reports.append(report)

    lic = manifest["license"]
    license_bytes = fetch_raw(repository, commit, lic["path"])
    license_report = {
        "path": lic["path"],
        "bytes": len(license_bytes),
        "raw_sha256": sha256(license_bytes),
        "git_blob_sha1": git_blob_sha1(license_bytes),
        "spdx": lic["spdx"],
        "training_decision": lic["training_decision"],
        "redistribution_decision": lic["redistribution_decision"],
    }
    for field in ("bytes", "raw_sha256", "git_blob_sha1"):
        if license_report[field] != lic[field]:
            failures.append(f"license_identity:{field}")
    if lic["training_decision"] != "ALLOW":
        failures.append("training_rights")
    if lic["redistribution_decision"] != "ALLOW_WITH_NOTICE":
        failures.append("redistribution_rights")

    selected_blob_ids = {item["git_blob_sha1"] for item in manifest["selected_files"]}
    reserve_overlaps = [
        reserved
        for reserved in manifest["known_evaluation_reservations"]
        if reserved["git_blob_sha1"] in selected_blob_ids
    ]
    if reserve_overlaps:
        failures.append("evaluation_reservation_overlap")

    registry_pairs: list[dict[str, Any]] = []
    for selected_path, selected_data in source_bytes.items():
        for other in manifest["terminal_registry_code_objects"]:
            other_data = fetch_raw(other["repository"], other["commit"], other["path"])
            sim = similarity(selected_data, other_data)
            pair = {
                "selected": selected_path,
                "other_family": other["family"],
                "other_path": other["path"],
                **sim,
            }
            registry_pairs.append(pair)
            if rejected(sim, policy):
                failures.append(
                    f"registry_dedup:{selected_path}:{other['family']}:{other['path']}"
                )

    lineage_reports: list[dict[str, Any]] = []
    for comparator in manifest["lineage_comparators"]:
        blobs = tree_blobs(comparator["repository"], comparator["commit"])
        exact_hits = [
            path for path, blob_sha in blobs.items() if blob_sha in selected_blob_ids
        ]
        near_pairs: list[dict[str, Any]] = []
        for selected_path, selected_data in source_bytes.items():
            for other_path in comparator["near_paths"]:
                if other_path not in blobs:
                    failures.append(
                        f"lineage_missing_path:{comparator['family']}:{other_path}"
                    )
                    continue
                other_data = fetch_raw(
                    comparator["repository"], comparator["commit"], other_path
                )
                sim = similarity(selected_data, other_data)
                pair = {"selected": selected_path, "other_path": other_path, **sim}
                near_pairs.append(pair)
                if rejected(sim, policy):
                    failures.append(
                        f"lineage_near_duplicate:{selected_path}:{comparator['family']}:{other_path}"
                    )
        if exact_hits:
            failures.append(f"lineage_exact_blob:{comparator['family']}")
        lineage_reports.append(
            {
                "family": comparator["family"],
                "repository": comparator["repository"],
                "release": comparator["release"],
                "commit": comparator["commit"],
                "exact_blob_hits_anywhere_in_tree": exact_hits,
                "near_pairs": near_pairs,
            }
        )

    report = {
        "schema": "next100.external_code_source_admission.evidence.v1",
        "worker_id": manifest["worker_id"],
        "verdict": "ADMIT" if not failures else "REJECT",
        "upstream": upstream,
        "family": manifest["family"],
        "license": license_report,
        "files": file_reports,
        "registry_dedup_pairs": registry_pairs,
        "lineage_reports": lineage_reports,
        "evaluation_reservation_overlaps": reserve_overlaps,
        "selection_exclusions": manifest["selection_exclusions"],
        "failures": failures,
        "training_executed": False,
        "local_free_only": True,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
