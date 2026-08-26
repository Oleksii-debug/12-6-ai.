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
    "private_endpoint": re.compile(r"(?i)https?://(?:localhost|127\.0\.0\.1|10\.|192\.168\.|172\.(?:1[6-9]|2[0-9]|3[01])\.)"),
}


def fetch(repository: str, commit: str, path: str) -> bytes:
    url = f"https://raw.githubusercontent.com/{repository}/{commit}/{path}"
    req = urllib.request.Request(url, headers={"User-Agent": "NEXT100-045-source-verifier"})
    with urllib.request.urlopen(req, timeout=30) as response:
        data = response.read()
    return data


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def normalize_text(data: bytes) -> bytes:
    text = data.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return ("\n".join(line.rstrip() for line in text.split("\n"))).encode("utf-8")


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="configs/data/next100_045_starlette_code_source_v1.json")
    parser.add_argument("--output", default="evidence/next100_045/starlette_source_admission.json")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    upstream = manifest["upstream"]
    repository = upstream["repository"]
    commit = upstream["commit"]

    failures: list[str] = []
    source_bytes: dict[str, bytes] = {}
    file_reports: list[dict[str, Any]] = []

    for item in manifest["selected_files"]:
        data = fetch(repository, commit, item["path"])
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
    license_bytes = fetch(repository, commit, lic["path"])
    license_report = {
        "path": lic["path"],
        "bytes": len(license_bytes),
        "raw_sha256": sha256(license_bytes),
        "git_blob_sha1": git_blob_sha1(license_bytes),
        "spdx": lic["spdx"],
        "training_decision": lic["training_decision"],
        "redistribution_decision": lic["redistribution_decision"],
    }
    if license_report["raw_sha256"] != lic["raw_sha256"] or license_report["git_blob_sha1"] != lic["git_blob_sha1"]:
        failures.append("license_identity")
    if lic["training_decision"] != "ALLOW" or lic["redistribution_decision"] != "ALLOW_WITH_NOTICE":
        failures.append("rights_decision")

    reserve_overlaps: list[dict[str, Any]] = []
    for selected in manifest["selected_files"]:
        for reserved in manifest["known_evaluation_reservations"]:
            if selected["git_blob_sha1"] == reserved["git_blob_sha1"]:
                reserve_overlaps.append({"selected": selected["path"], "reserved": reserved})
            if repository == reserved["repository"] and commit == reserved["commit"] and selected["path"] == reserved["path"]:
                reserve_overlaps.append({"selected": selected["path"], "reserved": reserved})
    if reserve_overlaps:
        failures.append("evaluation_reservation_overlap")

    registry_bytes: dict[str, bytes] = {}
    registry_identity: list[dict[str, Any]] = []
    for item in manifest["terminal_registry_code_objects"]:
        data = fetch(item["repository"], item["commit"], item["path"])
        registry_bytes[item["family"]] = data
        identity = {
            "family": item["family"],
            "path": item["path"],
            "raw_sha256": sha256(data),
            "git_blob_sha1": git_blob_sha1(data),
        }
        registry_identity.append(identity)
        if identity["raw_sha256"] != item["raw_sha256"] or identity["git_blob_sha1"] != item["git_blob_sha1"]:
            failures.append(f"registry_identity:{item['family']}")

    policy = manifest["dedup_policy"]
    dedup_pairs: list[dict[str, Any]] = []
    selected_items = list(source_bytes.items())
    for index, (left_path, left_bytes) in enumerate(selected_items):
        for right_path, right_bytes in selected_items[index + 1 :]:
            metrics = similarity(left_bytes, right_bytes)
            dedup_pairs.append({"left": left_path, "right": right_path, **metrics})
        for family, other_bytes in registry_bytes.items():
            metrics = similarity(left_bytes, other_bytes)
            dedup_pairs.append({"left": left_path, "right": family, **metrics})

    for pair in dedup_pairs:
        if pair["raw_exact"] or pair["normalized_exact"]:
            failures.append(f"dedup_exact:{pair['left']}:{pair['right']}")
        if pair["skeleton_jaccard"] >= policy["python_skeleton_5gram_jaccard_reject_at_or_above"]:
            failures.append(f"dedup_jaccard:{pair['left']}:{pair['right']}")
        if pair["skeleton_containment"] >= policy["python_skeleton_5gram_containment_reject_at_or_above"]:
            failures.append(f"dedup_containment:{pair['left']}:{pair['right']}")

    family = manifest["family"]
    if family["stable_repository_id"] != upstream["repository_id"]:
        failures.append("family_repository_identity")
    if any(obj["family"] == family["id"] for obj in manifest["terminal_registry_code_objects"]):
        failures.append("family_already_registered")

    result = {
        "schema": "next100.external_code_source_admission.report.v1",
        "worker_id": manifest["worker_id"],
        "verdict": "ADMIT" if not failures else "REJECT",
        "training_executed": False,
        "local_free_only": manifest["local_free_only"],
        "upstream": upstream,
        "family": family,
        "license": license_report,
        "files": file_reports,
        "evaluation_reservation_overlaps": reserve_overlaps,
        "registry_identity": registry_identity,
        "dedup_pairs": dedup_pairs,
        "failures": sorted(set(failures)),
        "admission_scope": manifest["admission_scope"],
        "evaluation_use": manifest["evaluation_use"],
    }
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    result["report_sha256_without_self"] = sha256(canonical)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if result["verdict"] == "ADMIT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
