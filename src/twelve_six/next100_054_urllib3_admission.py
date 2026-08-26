"""NEXT100-054 bounded external-real urllib3 code admission authority."""

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

SCHEMA = "12-6.next100-054-urllib3-code-admission-report.v1"
MANIFEST_PATH = Path("configs/data/next100_054_urllib3_code_rights_v1.json")
REPORT_NAME = "next100-054-urllib3-code-admission.json"
BANNED_PATH_PARTS = frozenset(
    {
        "vendor", "vendored", "vendors", "third_party", "third-party",
        "node_modules", "dist", "build", "generated", "gen", "deps",
        "dependencies", ".venv", "site-packages",
    }
)
EXCLUDED_URLLIB3_SUBTREES = ("src/urllib3/contrib/", "src/urllib3/http2/")
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
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def _download(url: str, *, max_bytes: int = 350_000) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": "12-6-NEXT100-054-urllib3-admission/1"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise AdmissionError(f"bounded download exceeded {max_bytes} bytes: {url}")
    return data


def _download_json(url: str, *, max_bytes: int = 350_000) -> dict[str, Any]:
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
    if value.get("schema_version") != "12-6.next100-054-urllib3-code-admission.v1":
        raise AdmissionError("unsupported manifest schema")
    if value.get("worker_id") != "NEXT100-054-CODE-URLLIB3":
        raise AdmissionError("worker identity mismatch")
    upstream = value["upstream"]
    if upstream["repository"] != "https://github.com/urllib3/urllib3":
        raise AdmissionError("unexpected upstream repository")
    if upstream["source_family"] != "github:urllib3/urllib3":
        raise AdmissionError("unexpected source family")
    selected = value["selected_files"]
    if not isinstance(selected, list) or len(selected) != 8:
        raise AdmissionError("bounded urllib3 allowlist must contain exactly eight files")
    paths = [item["path"] for item in selected]
    if len(paths) != len(set(paths)):
        raise AdmissionError("duplicate selected path")
    if sum(item["size_bytes"] for item in selected) != value["candidate_bytes"]:
        raise AdmissionError("candidate byte total mismatch")
    if value["candidate_bytes"] != 228836:
        raise AdmissionError("unexpected candidate byte total")
    uses = value["license_review"]["uses"]
    if uses["model_training"] != "ALLOWED" or uses["redistribution"] != "ALLOWED":
        raise AdmissionError("training and redistribution must be explicitly allowed")
    boundary = value["evaluation_boundary"]
    if boundary["candidate_role"] != "TRAINING_ONLY":
        raise AdmissionError("urllib3 admission must remain training-only")
    if boundary["evaluation_use_authorized"] is not False:
        raise AdmissionError("evaluation use must remain unauthorized")
    if boundary["selection_records_at_review"] != 0 or boundary["final_test_records_at_review"] != 0:
        raise AdmissionError("reviewed code evaluation reservation is non-empty")


def _assert_path_allowed(path: str) -> None:
    p = Path(path)
    parts = [part.casefold() for part in p.parts]
    if p.is_absolute() or ".." in p.parts:
        raise AdmissionError(f"unsafe source path: {path}")
    if any(part in BANNED_PATH_PARTS for part in parts):
        raise AdmissionError(f"vendored/generated/build path excluded: {path}")
    if any(path.startswith(prefix) for prefix in EXCLUDED_URLLIB3_SUBTREES):
        raise AdmissionError(f"explicitly excluded urllib3 subtree: {path}")
    if not path.startswith("src/urllib3/") or not path.endswith(".py"):
        raise AdmissionError(f"not first-party urllib3 Python implementation: {path}")


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
    repo_meta = _download_json("https://api.github.com/repos/urllib3/urllib3")
    if repo_meta.get("html_url") != upstream["repository"]:
        raise AdmissionError("canonical urllib3 repository identity drift")
    if repo_meta.get("fork") is not False or repo_meta.get("mirror_url") is not None:
        raise AdmissionError("urllib3 upstream is a fork or mirror")

    ref = _download_json(
        f"https://api.github.com/repos/urllib3/urllib3/git/ref/tags/{upstream['release_tag']}"
    )
    if ref["object"]["type"] != "tag" or ref["object"]["sha"] != upstream["annotated_tag_sha1"]:
        raise AdmissionError("urllib3 release tag object drift")

    tag = _download_json(
        f"https://api.github.com/repos/urllib3/urllib3/git/tags/{upstream['annotated_tag_sha1']}"
    )
    if tag["object"]["type"] != "commit" or tag["object"]["sha"] != upstream["commit"]:
        raise AdmissionError("urllib3 annotated tag target drift")
    if tag.get("verification", {}).get("verified") is not True:
        raise AdmissionError("urllib3 annotated tag is not GitHub-verified")

    commit = _download_json(
        f"https://api.github.com/repos/urllib3/urllib3/git/commits/{upstream['commit']}"
    )
    if commit["tree"]["sha"] != upstream["tree_sha1"]:
        raise AdmissionError("urllib3 release commit tree drift")
    if commit.get("verification", {}).get("verified") is not True:
        raise AdmissionError("urllib3 release commit is not GitHub-verified")

    return {
        "repository": upstream["repository"],
        "fork": False,
        "mirror_url": None,
        "tag_verification": "PASS",
        "commit_verification": "PASS",
    }


def _verify_license(manifest: dict[str, Any], output: Path) -> dict[str, Any]:
    review = manifest["license_review"]
    data = _download(review["license_url"], max_bytes=10_000)
    if len(data) != review["license_size_bytes"]:
        raise AdmissionError("urllib3 license size drift")
    if _git_blob_sha1(data) != review["license_blob_sha1"]:
        raise AdmissionError("urllib3 license Git blob drift")
    text = data.decode("utf-8")
    required = (
        "MIT License",
        "Permission is hereby granted, free of charge",
        "to deal in the Software without restriction",
        "publish, distribute, sublicense, and/or sell",
        "The above copyright notice and this permission notice shall be included",
        'THE SOFTWARE IS PROVIDED "AS IS"',
    )
    if not all(fragment in text for fragment in required):
        raise AdmissionError("reviewed MIT grant text drift")
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
    }


def _verify_candidate(
    manifest: dict[str, Any], item: dict[str, Any], output: Path
) -> tuple[dict[str, Any], str]:
    path = item["path"]
    _assert_path_allowed(path)
    commit = manifest["upstream"]["commit"]
    url = f"https://raw.githubusercontent.com/urllib3/urllib3/{commit}/{path}"
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


def _verify_requests_lineage(manifest: dict[str, Any]) -> dict[str, Any]:
    audit = manifest["requests_lineage_audit"]
    commit = audit["requests_commit"]
    base = f"https://raw.githubusercontent.com/psf/requests/{commit}/"
    checks: dict[str, str] = {}
    for key in ("current_dependency", "current_compat_alias", "historical_lineage"):
        item = audit[key]
        raw = _download(base + item["path"])
        if _git_blob_sha1(raw) != item["blob_sha1"]:
            raise AdmissionError(f"Requests lineage evidence drift: {item['path']}")
        text = raw.decode("utf-8")
        if item["required_text"] not in text:
            raise AdmissionError(f"Requests lineage evidence missing text: {item['path']}")
        checks[key] = "PASS"
    return {
        "requests_commit": commit,
        "current_external_dependency": checks["current_dependency"],
        "current_compat_alias_to_external_urllib3": checks["current_compat_alias"],
        "historical_vendoring_lineage": checks["historical_lineage"],
    }


def _verify_evaluation_boundary(manifest: dict[str, Any]) -> dict[str, Any]:
    boundary = manifest["evaluation_boundary"]
    url = (
        "https://raw.githubusercontent.com/Oleksii-debug/12-6-ai./"
        f"{boundary['head_at_review']}/{boundary['evidence_path']}"
    )
    raw = _download(url, max_bytes=20_000)
    if _git_blob_sha1(raw) != boundary["evidence_blob_sha1"]:
        raise AdmissionError("EVAL-322 evidence Git blob drift")
    evidence = json.loads(raw)
    unsigned = dict(evidence)
    expected_identity = unsigned.pop("evidence_identity_sha256")
    calculated_identity = _sha256(_cjson(unsigned))
    if expected_identity != boundary["evidence_identity_sha256"]:
        raise AdmissionError("EVAL-322 configured evidence identity mismatch")
    if calculated_identity != expected_identity:
        raise AdmissionError("EVAL-322 canonical evidence identity mismatch")
    gate = evidence["code_gate"]
    if gate["verdict"] != "BLOCKED":
        raise AdmissionError("EVAL-322 code gate is no longer blocked at reviewed head")
    if gate["selection_records"] != 0 or gate["final_test_records"] != 0:
        raise AdmissionError("reviewed EVAL-322 code reservation is non-empty")
    if evidence["selection_validation"]["component_identities"]["code_records"] != 0:
        raise AdmissionError("reviewed selection-validation contains code records")
    if evidence["final_test"]["code_included"] is not False:
        raise AdmissionError("reviewed final-test includes code")
    return {
        "authority": boundary["authority"],
        "head": boundary["head_at_review"],
        "evidence_identity_sha256": expected_identity,
        "code_gate": "BLOCKED_NO_PRISTINE_CODE_OBJECTS",
        "selection_records": 0,
        "final_test_records": 0,
        "selected_blob_intersection_count": 0,
        "status": "PASS_NO_CODE_EVALUATION_RESERVATION_AT_REVIEWED_AUTHORITY",
    }


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
                        "scope": "within_urllib3",
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
            f"dedup rejected bounded urllib3 snapshot: exact={exact}, near={near}"
        )
    return {
        "exact_duplicate_sha256": exact,
        "near_pairs_at_or_above_threshold": near,
        "near_threshold": threshold,
        "token_shingle_size": 5,
        "candidate_object_count": len(candidates),
        "baseline_object_count": len(baseline),
        "status": "PASS",
    }


def _family_decision(
    lineage: dict[str, Any], dedup: dict[str, Any], baseline: list[tuple[dict[str, Any], str]]
) -> dict[str, Any]:
    requests_present = any(item["source_family"] == "github:psf/requests" for item, _ in baseline)
    if not requests_present:
        raise AdmissionError("terminal Requests baseline is missing")
    if (
        lineage["current_external_dependency"] != "PASS"
        or lineage["current_compat_alias_to_external_urllib3"] != "PASS"
        or lineage["historical_vendoring_lineage"] != "PASS"
    ):
        raise AdmissionError("Requests lineage audit incomplete")
    if dedup["status"] != "PASS":
        raise AdmissionError("family decision requires clean object dedup")
    return {
        "relationship": "RELATED_LINEAGE",
        "historical_requests_vendoring": True,
        "current_requests_uses_external_urllib3_dependency": True,
        "current_requests_compatibility_aliases_external_urllib3": True,
        "current_bounded_object_overlap": "NONE_AT_EXACT_OR_JACCARD_GE_0.85",
        "decision": "SEPARATE_CURRENT_BOUNDED_FAMILIES_WITH_LINEAGE_CAVEAT",
        "urllib3_family": "github:urllib3/urllib3",
        "requests_family": "github:psf/requests",
        "future_rule": "Any Requests object copied or vendored from urllib3 must be attributed to urllib3 lineage and collapsed or excluded rather than counted as independent capacity.",
    }


def run(repo: Path, output: Path, source_sha: str) -> dict[str, Any]:
    _require_head(repo, source_sha)
    manifest = load_manifest(repo)
    output.mkdir(parents=True, exist_ok=True)

    upstream = _verify_upstream_identity(manifest)
    rights = _verify_license(manifest, output)
    evaluation = _verify_evaluation_boundary(manifest)
    lineage = _verify_requests_lineage(manifest)

    candidates = [
        _verify_candidate(manifest, item, output) for item in manifest["selected_files"]
    ]
    baseline = _verify_baseline_objects(manifest)
    threshold = manifest["verification"]["near_dedup"]["jaccard_threshold"]
    dedup = _dedup(candidates, baseline, threshold)
    family = _family_decision(lineage, dedup, baseline)

    objects = [evidence for evidence, _ in candidates]
    snapshot_manifest = {
        "source_family": manifest["upstream"]["source_family"],
        "commit": manifest["upstream"]["commit"],
        "objects": [
            {
                "path": item["path"],
                "blob_sha1": item["blob_sha1"],
                "raw_sha256": item["raw_sha256"],
                "size_bytes": item["size_bytes"],
            }
            for item in objects
        ],
        "total_bytes": sum(item["size_bytes"] for item in objects),
    }
    snapshot_manifest["identity_sha256"] = _sha256(_cjson(snapshot_manifest))

    report: dict[str, Any] = {
        "schema_version": SCHEMA,
        "worker_id": "NEXT100-054-CODE-URLLIB3",
        "authority": "EXTERNAL_REAL_CODE_SOURCE_ADMISSION_LOCAL_FREE",
        "terminal": True,
        "verdict": "ADMIT_TRAINING_ONLY",
        "source_head": source_sha,
        "upstream": manifest["upstream"],
        "upstream_verification": upstream,
        "rights": rights,
        "selection": {
            "object_count": len(objects),
            "raw_bytes": sum(item["size_bytes"] for item in objects),
            "source_family_count": 1,
            "source_family": manifest["upstream"]["source_family"],
        },
        "objects": objects,
        "snapshot_manifest": snapshot_manifest,
        "privacy_secret_parse": {
            "strict_utf8_identity_all": True,
            "python_ast_parse_all": True,
            "secret_scan_all": "PASS",
            "privacy_scan_all": "PASS",
        },
        "deduplication": dedup,
        "requests_lineage": lineage,
        "source_family_decision": family,
        "evaluation_boundary": evaluation,
        "terminal_code_registry_baseline": {
            "authority": manifest["terminal_code_registry_baseline"]["authority"],
            "head": manifest["terminal_code_registry_baseline"]["head"],
            "source_families": manifest["terminal_code_registry_baseline"]["source_families"],
            "object_count": len(baseline),
        },
        "execution_class": "LOCAL_FREE",
        "training_performed": False,
        "paid_compute_used": False,
        "terminal_conditions": {
            "exact_revision_bound": True,
            "license_bound": True,
            "blob_inventory_bound": True,
            "training_rights_allowed": True,
            "redistribution_allowed": True,
            "privacy_secret_parse_pass": True,
            "dedup_pass": True,
            "requests_lineage_decided": True,
            "evaluation_collision_count": 0,
        },
    }
    report["manifest_identity_sha256"] = _sha256(_cjson(manifest))
    report["report_identity_sha256"] = _sha256(_cjson(report))
    (output / REPORT_NAME).write_bytes(_cjson(report))
    return report


def validate_report(path: Path, expected_source_sha: str) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema_version") != SCHEMA:
        raise AdmissionError("unsupported report schema")
    if report.get("source_head") != expected_source_sha:
        raise AdmissionError("report source head mismatch")
    if report.get("verdict") != "ADMIT_TRAINING_ONLY" or report.get("terminal") is not True:
        raise AdmissionError("report is not terminal ADMIT_TRAINING_ONLY")
    identity = report.pop("report_identity_sha256", None)
    calculated = _sha256(_cjson(report))
    if identity != calculated:
        raise AdmissionError("report identity mismatch")
    if report["selection"]["object_count"] != 8 or report["selection"]["raw_bytes"] != 228836:
        raise AdmissionError("report selection inventory mismatch")
    if report["deduplication"]["status"] != "PASS":
        raise AdmissionError("dedup did not pass")
    if report["evaluation_boundary"]["selected_blob_intersection_count"] != 0:
        raise AdmissionError("evaluation collision detected")
    if report["source_family_decision"]["decision"] != "SEPARATE_CURRENT_BOUNDED_FAMILIES_WITH_LINEAGE_CAVEAT":
        raise AdmissionError("Requests lineage decision mismatch")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--repo-root", type=Path, default=Path("."))
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--source-sha", required=True)
    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("report", type=Path)
    validate_parser.add_argument("--expected-source-sha", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "run":
        report = run(args.repo_root.resolve(), args.output_dir, args.source_sha)
        print(json.dumps({
            "verdict": report["verdict"],
            "report_identity_sha256": report["report_identity_sha256"],
            "snapshot_identity_sha256": report["snapshot_manifest"]["identity_sha256"],
        }, sort_keys=True))
    else:
        report = validate_report(args.report, args.expected_source_sha)
        print(json.dumps({
            "verdict": report["verdict"],
            "report_identity_sha256": _sha256(_cjson(report)),
        }, sort_keys=True))


if __name__ == "__main__":
    main()
