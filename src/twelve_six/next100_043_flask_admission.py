"""NEXT100-043 bounded external-real Flask code admission authority."""

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

SCHEMA = "12-6.next100-043-flask-code-admission-report.v1"
MANIFEST_PATH = Path("configs/data/next100_043_flask_code_rights_v1.json")
REPORT_NAME = "next100-043-flask-code-admission.json"
BANNED_PATH_PARTS = frozenset(
    {
        "vendor",
        "vendored",
        "vendors",
        "third_party",
        "third-party",
        "node_modules",
        "dist",
        "build",
        "generated",
        "gen",
        "deps",
        "dependencies",
        ".venv",
        "site-packages",
    }
)
SECRET_PATTERNS = {
    "private_key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "github_token": re.compile(rb"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    "openai_key": re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "aws_access_key": re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    "google_api_key": re.compile(rb"\bAIza[0-9A-Za-z_-]{30,}\b"),
    "slack_token": re.compile(rb"\bxox[baprs]-[0-9A-Za-z-]{10,}\b"),
}
SSN_PATTERN = re.compile(rb"\b\d{3}-\d{2}-\d{4}\b")
PAN_PATTERN = re.compile(rb"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\s]")


class AdmissionError(RuntimeError):
    pass


def _cjson(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def _download(url: str, *, max_bytes: int = 250_000) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": "12-6-NEXT100-043-Flask-admission/1"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise AdmissionError(f"bounded download exceeded {max_bytes} bytes: {url}")
    return data


def _download_json(url: str, *, max_bytes: int = 250_000) -> dict[str, Any]:
    return json.loads(_download(url, max_bytes=max_bytes))


def _require_head(repo: Path, source_sha: str) -> None:
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    if actual != source_sha:
        raise AdmissionError(f"exact-head mismatch: {actual} != {source_sha}")


def load_manifest(repo: Path) -> dict[str, Any]:
    value = json.loads((repo / MANIFEST_PATH).read_text(encoding="utf-8"))
    validate_manifest(value)
    return value


def validate_manifest(value: dict[str, Any]) -> None:
    if value.get("schema_version") != "12-6.next100-043-flask-code-admission.v1":
        raise AdmissionError("unsupported manifest schema")
    if value.get("worker_id") != "NEXT100-043-CODE-FLASK":
        raise AdmissionError("worker identity mismatch")
    upstream = value["upstream"]
    if upstream["repository"] != "https://github.com/pallets/flask":
        raise AdmissionError("unexpected upstream repository")
    if upstream["source_family"] != "github:pallets/flask":
        raise AdmissionError("unexpected source family")
    selected = value["selected_files"]
    if not isinstance(selected, list) or len(selected) != 8:
        raise AdmissionError("bounded Flask allowlist must contain exactly eight files")
    paths = [item["path"] for item in selected]
    if len(paths) != len(set(paths)):
        raise AdmissionError("duplicate selected path")
    if sum(item["size_bytes"] for item in selected) != value["candidate_bytes"]:
        raise AdmissionError("candidate byte total mismatch")
    if value["license_review"]["uses"]["model_training"] != "ALLOWED":
        raise AdmissionError("training use is not explicitly allowed")
    if value["license_review"]["uses"]["redistribution"] != "ALLOWED":
        raise AdmissionError("redistribution is not explicitly allowed")
    if value["evaluation_boundary"]["candidate_role"] != "TRAINING_ONLY":
        raise AdmissionError("Flask admission must remain training-only")
    if value["evaluation_boundary"]["evaluation_use_authorized"] is not False:
        raise AdmissionError("evaluation use must remain unauthorized")


def _assert_path_allowed(path: str) -> None:
    p = Path(path)
    parts = [part.casefold() for part in p.parts]
    if p.is_absolute() or ".." in p.parts:
        raise AdmissionError(f"unsafe source path: {path}")
    if any(part in BANNED_PATH_PARTS for part in parts):
        raise AdmissionError(f"vendored/generated/build path excluded: {path}")
    if not path.startswith("src/flask/") or not path.endswith(".py"):
        raise AdmissionError(f"not first-party Flask Python implementation: {path}")


def _scan_secrets(data: bytes) -> list[str]:
    return [name for name, pattern in SECRET_PATTERNS.items() if pattern.search(data)]


def _luhn_valid(number: str) -> bool:
    digits = [int(ch) for ch in number]
    total = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def _scan_privacy(data: bytes) -> list[str]:
    findings: list[str] = []
    if SSN_PATTERN.search(data):
        findings.append("us_ssn_like")
    for match in PAN_PATTERN.finditer(data):
        digits = re.sub(rb"\D", b"", match.group()).decode("ascii")
        if 13 <= len(digits) <= 19 and _luhn_valid(digits):
            findings.append("payment_pan_like")
            break
    return findings


def _token_shingles(text: str, size: int) -> set[tuple[str, ...]]:
    tokens = TOKEN_RE.findall(text.casefold())
    if len(tokens) < size:
        return set()
    return {tuple(tokens[i : i + size]) for i in range(len(tokens) - size + 1)}


def _near_jaccard(left: str, right: str, size: int = 5) -> float:
    a = _token_shingles(left, size)
    b = _token_shingles(right, size)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _verify_upstream_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    upstream = manifest["upstream"]
    repo_meta = _download_json("https://api.github.com/repos/pallets/flask")
    if repo_meta.get("html_url") != upstream["repository"]:
        raise AdmissionError("canonical Flask repository identity drift")
    if repo_meta.get("fork") is not False or repo_meta.get("mirror_url") is not None:
        raise AdmissionError("Flask upstream is a fork or mirror")

    ref = _download_json(
        f"https://api.github.com/repos/pallets/flask/git/ref/tags/{upstream['release_tag']}"
    )
    if ref["object"]["sha"] != upstream["annotated_tag_sha1"]:
        raise AdmissionError("Flask release tag object drift")

    tag = _download_json(
        f"https://api.github.com/repos/pallets/flask/git/tags/{upstream['annotated_tag_sha1']}"
    )
    if tag["object"]["type"] != "commit" or tag["object"]["sha"] != upstream["commit"]:
        raise AdmissionError("Flask annotated tag target drift")
    if tag.get("verification", {}).get("verified") is not True:
        raise AdmissionError("Flask annotated tag is not GitHub-verified")

    commit = _download_json(
        f"https://api.github.com/repos/pallets/flask/git/commits/{upstream['commit']}"
    )
    if commit["tree"]["sha"] != upstream["tree_sha1"]:
        raise AdmissionError("Flask release commit tree drift")
    if commit.get("verification", {}).get("verified") is not True:
        raise AdmissionError("Flask release commit is not GitHub-verified")
    return {
        "repository": upstream["repository"],
        "fork": False,
        "mirror_url": None,
        "tag_verification": "PASS",
        "commit_verification": "PASS",
    }


def _verify_license(
    manifest: dict[str, Any], output: Path
) -> tuple[dict[str, Any], bytes]:
    review = manifest["license_review"]
    data = _download(review["license_url"], max_bytes=10_000)
    if len(data) != review["license_size_bytes"]:
        raise AdmissionError("Flask license size drift")
    if _git_blob_sha1(data) != review["license_blob_sha1"]:
        raise AdmissionError("Flask license Git blob drift")
    text = data.decode("utf-8")
    required = (
        "Redistribution and use in source and binary forms",
        "with or without modification",
        "Neither the name of the copyright holder",
        'THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS\n"AS IS"',
    )
    if not all(fragment in text for fragment in required):
        raise AdmissionError("reviewed BSD-3-Clause grant text drift")
    rights_dir = output / "rights-evidence"
    rights_dir.mkdir(parents=True, exist_ok=True)
    (rights_dir / "LICENSE.txt").write_bytes(data)
    return {
        "license_id": review["license_id"],
        "license_blob_sha1": review["license_blob_sha1"],
        "license_sha256": _sha256(data),
        "training_purpose_decision": "ALLOWED",
        "redistribution_decision": "ALLOWED",
        "redistribution_conditions": review["redistribution_conditions"],
    }, data


def _verify_candidate(
    manifest: dict[str, Any], item: dict[str, Any], output: Path
) -> tuple[dict[str, Any], str]:
    path = item["path"]
    _assert_path_allowed(path)
    commit = manifest["upstream"]["commit"]
    url = f"https://raw.githubusercontent.com/pallets/flask/{commit}/{path}"
    raw = _download(url)
    if len(raw) != item["size_bytes"]:
        raise AdmissionError(f"{path}: size drift")
    if _git_blob_sha1(raw) != item["blob_sha1"]:
        raise AdmissionError(f"{path}: Git blob identity drift")
    secret_findings = _scan_secrets(raw)
    if secret_findings:
        raise AdmissionError(f"{path}: secret-like material: {secret_findings}")
    privacy_findings = _scan_privacy(raw)
    if privacy_findings:
        raise AdmissionError(f"{path}: high-risk privacy material: {privacy_findings}")
    if b"\x00" in raw:
        raise AdmissionError(f"{path}: NUL byte")
    text = raw.decode("utf-8")
    if text.encode("utf-8") != raw:
        raise AdmissionError(f"{path}: UTF-8 identity normalization drift")
    ast.parse(text, filename=path)

    snapshot_path = output / "snapshots" / path
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_bytes(raw)
    evidence = {
        "path": path,
        "size_bytes": len(raw),
        "blob_sha1": item["blob_sha1"],
        "raw_sha256": _sha256(raw),
        "normalization_sha256": _sha256(raw),
        "normalization_policy": "STRICT_UTF8_IDENTITY_PRESERVE_V1",
        "parse": "PASS",
        "secret_scan": "PASS",
        "privacy_scan": "PASS",
        "raw_url": url,
    }
    return evidence, text


def _verify_baseline_objects(
    manifest: dict[str, Any],
) -> list[tuple[dict[str, Any], str]]:
    verified: list[tuple[dict[str, Any], str]] = []
    for item in manifest["terminal_code_registry_baseline"]["objects"]:
        raw = _download(item["raw_url"])
        if len(raw) != item["size_bytes"]:
            raise AdmissionError(f"baseline {item['source_id']}: size drift")
        if _git_blob_sha1(raw) != item["blob_sha1"]:
            raise AdmissionError(f"baseline {item['source_id']}: Git blob drift")
        if _sha256(raw) != item["raw_sha256"]:
            raise AdmissionError(f"baseline {item['source_id']}: SHA-256 drift")
        verified.append((item, raw.decode("utf-8")))
    return verified


def _dedup(
    candidates: list[tuple[dict[str, Any], str]],
    baseline: list[tuple[dict[str, Any], str]],
    threshold: float,
) -> dict[str, Any]:
    exact: list[dict[str, str]] = []
    near: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    for evidence, _ in candidates:
        digest = evidence["raw_sha256"]
        if digest in seen:
            exact.append({"left": seen[digest], "right": evidence["path"]})
        seen[digest] = evidence["path"]
    for item, _ in baseline:
        digest = item["raw_sha256"]
        if digest in seen:
            exact.append({"left": seen[digest], "right": item["source_id"]})
        seen[digest] = item["source_id"]

    for i, (left, left_text) in enumerate(candidates):
        for right, right_text in candidates[i + 1 :]:
            score = _near_jaccard(left_text, right_text)
            if score >= threshold:
                near.append(
                    {
                        "left": left["path"],
                        "right": right["path"],
                        "jaccard": score,
                        "scope": "within_flask",
                    }
                )
        for baseline_item, baseline_text in baseline:
            score = _near_jaccard(left_text, baseline_text)
            if score >= threshold:
                near.append(
                    {
                        "left": left["path"],
                        "right": baseline_item["source_id"],
                        "jaccard": score,
                        "scope": "terminal_baseline",
                    }
                )
    if exact or near:
        raise AdmissionError(
            f"dedup rejected bounded Flask snapshot: exact={exact}, near={near}"
        )
    return {
        "exact_duplicate_sha256": exact,
        "near_pairs_at_or_above_threshold": near,
        "near_threshold": threshold,
        "token_shingle_size": 5,
        "compared_against_terminal_data227": True,
        "result": "PASS",
    }


def _self_hash(report: dict[str, Any]) -> str:
    value = dict(report)
    value.pop("report_sha256", None)
    return _sha256(_cjson(value))


def run(repo: Path, output: Path, source_sha: str) -> dict[str, Any]:
    _require_head(repo, source_sha)
    manifest = load_manifest(repo)
    output.mkdir(parents=True, exist_ok=True)
    upstream_identity = _verify_upstream_identity(manifest)
    license_evidence, _ = _verify_license(manifest, output)

    candidates: list[tuple[dict[str, Any], str]] = []
    for item in manifest["selected_files"]:
        candidates.append(_verify_candidate(manifest, item, output))
    baseline = _verify_baseline_objects(manifest)
    dedup = _dedup(
        candidates,
        baseline,
        float(manifest["verification"]["near_dedup"]["jaccard_threshold"]),
    )
    bytes_admitted = sum(item[0]["size_bytes"] for item in candidates)
    if bytes_admitted != manifest["candidate_bytes"]:
        raise AdmissionError("admitted byte count mismatch")

    candidate_families = {manifest["upstream"]["source_family"]}
    baseline_families = set(
        manifest["terminal_code_registry_baseline"]["source_families"]
    )
    if candidate_families & baseline_families:
        raise AdmissionError("Flask family aliases an incumbent family")

    objects = [item[0] for item in candidates]
    snapshot_manifest_sha256 = _sha256(
        _cjson(
            [
                {
                    "path": item["path"],
                    "blob_sha1": item["blob_sha1"],
                    "raw_sha256": item["raw_sha256"],
                    "size_bytes": item["size_bytes"],
                }
                for item in objects
            ]
        )
    )
    report: dict[str, Any] = {
        "schema_version": SCHEMA,
        "authority": "EXTERNAL_REAL_CODE_D03_ADMISSION_LOCAL_FREE",
        "worker_id": manifest["worker_id"],
        "source_sha": source_sha,
        "verdict": "ADMIT_TRAINING_ONLY",
        "training_authorized": True,
        "redistribution_authorized": True,
        "evaluation_authorized": False,
        "upstream": manifest["upstream"],
        "upstream_identity": upstream_identity,
        "license": license_evidence,
        "source_family": manifest["upstream"]["source_family"],
        "source_family_count_delta": 1,
        "selected_object_count": len(objects),
        "selected_objects": objects,
        "admitted_raw_bytes": bytes_admitted,
        "conservative_unique_bytes_delta": bytes_admitted,
        "snapshot_manifest_sha256": snapshot_manifest_sha256,
        "normalization": "STRICT_UTF8_IDENTITY_PRESERVE_V1",
        "parse_checks": {"required": len(objects), "passed": len(objects)},
        "secret_scan": {"required": len(objects), "passed": len(objects)},
        "privacy_scan": {
            "required": len(objects),
            "passed": len(objects),
            "scope": manifest["verification"]["privacy_scan"],
        },
        "deduplication": dedup,
        "terminal_code_registry_baseline": manifest[
            "terminal_code_registry_baseline"
        ],
        "registry_after_if_no_newer_conflict": {
            "source_families": sorted(baseline_families | candidate_families),
            "source_family_count": len(baseline_families | candidate_families),
            "admitted_raw_bytes": manifest["terminal_code_registry_baseline"][
                "admitted_raw_bytes"
            ]
            + bytes_admitted,
        },
        "evaluation_boundary": manifest["evaluation_boundary"],
        "excluded_capacity_policy": manifest["selection_policy"],
        "vendored_or_generated_counted_bytes": 0,
        "model_training_executed": False,
        "optimizer_steps": 0,
        "local_free_only": True,
    }
    report["report_sha256"] = _self_hash(report)
    report_path = output / REPORT_NAME
    report_path.write_bytes(_cjson(report))
    return report


def validate_report(repo: Path, path: Path, expected_source_sha: str) -> None:
    manifest = load_manifest(repo)
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema_version") != SCHEMA:
        raise AdmissionError("report schema mismatch")
    if report.get("source_sha") != expected_source_sha:
        raise AdmissionError("report source head mismatch")
    if report.get("verdict") != "ADMIT_TRAINING_ONLY":
        raise AdmissionError("report verdict is not terminal training admission")
    if report.get("report_sha256") != _self_hash(report):
        raise AdmissionError("report self-hash mismatch")
    if report.get("admitted_raw_bytes") != manifest["candidate_bytes"]:
        raise AdmissionError("report byte count mismatch")
    if report.get("selected_object_count") != len(manifest["selected_files"]):
        raise AdmissionError("report object count mismatch")
    if report.get("source_family_count_delta") != 1:
        raise AdmissionError("Flask must add exactly one independent family")
    if report.get("vendored_or_generated_counted_bytes") != 0:
        raise AdmissionError("vendored/generated capacity was counted")
    if report.get("evaluation_authorized") is not False:
        raise AdmissionError("evaluation boundary violated")
    if report.get("model_training_executed") is not False:
        raise AdmissionError("admission task must not train a model")
    if report.get("local_free_only") is not True:
        raise AdmissionError("LOCAL_FREE boundary violated")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--repo-root", type=Path, default=Path("."))
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--source-sha", required=True)
    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("report", type=Path)
    validate_parser.add_argument("--repo-root", type=Path, default=Path("."))
    validate_parser.add_argument("--expected-source-sha", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "run":
        report = run(args.repo_root, args.output_dir, args.source_sha)
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        validate_report(args.repo_root, args.report, args.expected_source_sha)
        print("NEXT100-043 Flask admission report: PASS")


if __name__ == "__main__":
    main()
