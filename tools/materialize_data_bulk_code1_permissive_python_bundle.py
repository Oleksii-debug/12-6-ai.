#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

DEFAULT_CONFIG = Path("configs/data/data_bulk_code1_permissive_python_bundle_v1.json")
DEFAULT_OUTPUT = Path("evidence/data_bulk_code1/permissive_python_bundle_v1.json")
SCHEMA = "12-6.data-bulk-code1-permissive-python-bundle.v1"
REPORT_SCHEMA = "12-6.data-bulk-code1-permissive-python-bundle-report.v1"
EXPECTED_CONTRACT_IDENTITY = "7fd2228208f928859ebe68e947a72c977cda6952035a654d12923ce3a19a7dd6"

CREDENTIAL_PATTERNS = {
    "private_key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "aws_access_key": re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    "github_classic_token": re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    "github_fine_grained_token": re.compile(rb"\bgithub_pat_[A-Za-z0-9_]{50,}\b"),
    "openai_style_secret": re.compile(rb"\bsk-[A-Za-z0-9]{32,}\b"),
}


class MaterializationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MaterializationError(message)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _contract_identity(config: dict[str, Any]) -> str:
    payload = dict(config)
    payload.pop("contract_identity_sha256", None)
    return _sha256(_canonical_bytes(payload))


def _run(cmd: list[str], *, cwd: Path | None = None) -> str:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    if proc.returncode != 0:
        raise MaterializationError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr.strip()}"
        )
    return proc.stdout.strip()


def load_contract(path: Path) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MaterializationError(f"cannot read contract: {exc}") from exc

    _require(config.get("schema_version") == SCHEMA, "schema drift")
    _require(config.get("worker_id") == "DATA-BULK-CODE-1-PERMISSIVE-PYTHON-BUNDLE", "worker drift")
    _require(config.get("issue") == 635, "issue binding drift")
    _require(config.get("execution_profile") == "LOCAL_FREE", "execution profile weakened")
    _require(config.get("contract_identity_sha256") == EXPECTED_CONTRACT_IDENTITY, "contract identity declaration drift")
    _require(_contract_identity(config) == EXPECTED_CONTRACT_IDENTITY, "contract content identity mismatch")

    parent = config.get("parent", {})
    _require(parent.get("pr") == 594, "parent PR drift")
    _require(parent.get("head_sha") == "a546a1cb1434bee7db09d8fc8eff6040491de8ff", "parent head drift")
    _require(parent.get("package_id") == "code-permissive-python-implementation-pool", "parent package drift")

    rights = config.get("rights_boundary", {})
    _require(rights.get("evaluation") == "NOT_SEPARATELY_ADMITTED", "evaluation firewall weakened")
    _require(rights.get("final_test") == "PROHIBITED", "final-test firewall weakened")
    _require(rights.get("automatic_canonical_capacity_credit") is False, "automatic capacity promotion forbidden")

    policy = config.get("selection_policy", {})
    _require(policy.get("extensions") == [".py"], "extension policy drift")
    _require(policy.get("strict_utf8") is True, "UTF-8 gate weakened")
    _require(policy.get("python_ast_parse") is True, "AST gate weakened")
    _require(policy.get("max_file_bytes") == 524288, "max file bound drift")
    _require(policy.get("reject_symlinks") is True, "symlink gate weakened")
    _require(policy.get("secret_scan") == "BOUND_CREDENTIAL_PATTERNS_V1", "secret scan drift")

    expected_sources = {
        "pallets/flask": ("d318b683471101618febed18996405ad26462110", "src/flask", "BSD-3-Clause", "LICENSE.txt", "9d227a0cc43c3268d15722b763bd94ad298645a1"),
        "pallets/click": ("68e7ea7228ca144c52e4d1d282cc09da59f7771f", "src/click", "BSD-3-Clause", "LICENSE.txt", "d12a849186982399c537c5b9a8fd77bf2edd5eab"),
        "pallets/jinja": ("5ef70112a1ff19c05324ff889dd30405b1002044", "src/jinja2", "BSD-3-Clause", "LICENSE.txt", "c37cae49ec77ad6ebb25568c1605f1fee5313cfb"),
        "pallets/werkzeug": ("0005c79e09bae5f4cc2bd8ccd468d7dafe24a455", "src/werkzeug", "BSD-3-Clause", "LICENSE.txt", "c37cae49ec77ad6ebb25568c1605f1fee5313cfb"),
        "agronholm/anyio": ("ae250440c90020b030ba4e83cccc37e9a84512c5", "src/anyio", "MIT", "LICENSE", "104eebf5a3002fccdaceef3a4cb936173c1c2035"),
        "pytest-dev/pytest": ("28549a5f6b82bc916bb2ec5cb9fbfffe9b79fc66", "src/_pytest", "MIT", "LICENSE", "c3f1657fce94589bd1ec7cead810639047f3d359"),
    }
    sources = config.get("sources")
    _require(isinstance(sources, list) and len(sources) == len(expected_sources), "source cardinality drift")
    families: set[str] = set()
    for source in sources:
        repo = source.get("repository")
        _require(repo in expected_sources, f"unexpected source: {repo}")
        expected = expected_sources[repo]
        observed = (
            source.get("commit"), source.get("source_root"), source.get("license_id"),
            source.get("license_path"), source.get("license_git_blob_sha1"),
        )
        _require(observed == expected, f"source authority drift: {repo}")
        family = source.get("family_id")
        _require(family == f"github:{repo}", f"family identity drift: {repo}")
        _require(family not in families, f"duplicate family: {family}")
        families.add(family)

    truth = config.get("truth_boundary", {})
    _require(truth.get("corpus_identity") is None and truth.get("shard_identity") is None, "corpus/shard identity fabricated")
    _require(truth.get("authorized_training_exposure") == 0, "training exposure must remain zero")
    _require(truth.get("tokenizer_fit_authorized") is False, "tokenizer fit must remain blocked")
    _require(truth.get("long_training_authorized") is False, "long training must remain blocked")
    _require(truth.get("paid_compute_authorized") is False, "paid compute must remain blocked")
    return config


def _verify_license(repo_dir: Path, source: dict[str, Any]) -> dict[str, Any]:
    path = repo_dir / source["license_path"]
    _require(path.is_file() and not path.is_symlink(), f"license missing or symlinked: {source['repository']}")
    blob = _run(["git", "hash-object", source["license_path"]], cwd=repo_dir)
    _require(blob == source["license_git_blob_sha1"], f"license blob mismatch: {source['repository']}")
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="strict")
    if source["license_id"] == "BSD-3-Clause":
        _require("Redistribution and use in source and binary forms" in text, "BSD license text missing grant")
        _require("Neither the name of the copyright holder" in text, "BSD third condition missing")
    elif source["license_id"] == "MIT":
        _require("The MIT License" in text and "Permission is hereby granted, free of charge" in text, "MIT license text mismatch")
    else:
        raise MaterializationError(f"unhandled license: {source['license_id']}")
    return {
        "license_id": source["license_id"],
        "license_path": source["license_path"],
        "license_git_blob_sha1": blob,
        "license_sha256": _sha256(raw),
    }


def _checkout_source(source: dict[str, Any], destination: Path) -> Path:
    repo_dir = destination / source["repository"].replace("/", "__")
    repo_dir.mkdir(parents=True, exist_ok=False)
    _run(["git", "init", "-q"], cwd=repo_dir)
    _run(["git", "remote", "add", "origin", f"https://github.com/{source['repository']}.git"], cwd=repo_dir)
    _run(["git", "-c", "protocol.version=2", "fetch", "--depth=1", "--no-tags", "origin", source["commit"]], cwd=repo_dir)
    _run(["git", "checkout", "--detach", "--quiet", "FETCH_HEAD"], cwd=repo_dir)
    head = _run(["git", "rev-parse", "HEAD"], cwd=repo_dir)
    _require(head == source["commit"], f"exact commit mismatch: {source['repository']}")
    return repo_dir


def _credential_hits(raw: bytes) -> list[str]:
    return [name for name, pattern in CREDENTIAL_PATTERNS.items() if pattern.search(raw)]


def _eligible_file_record(repo_dir: Path, root: Path, path: Path, policy: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    rel = path.relative_to(repo_dir).as_posix()
    root_rel = path.relative_to(root)
    excluded_parts = set(policy["exclude_directory_components"])
    if any(part.lower() in excluded_parts for part in root_rel.parts[:-1]):
        return None, {"path": rel, "reason": "excluded_directory_component"}
    if path.is_symlink():
        if policy["reject_symlinks"]:
            return None, {"path": rel, "reason": "symlink"}
    if path.suffix not in policy["extensions"]:
        return None, {"path": rel, "reason": "extension"}
    raw = path.read_bytes()
    if len(raw) > int(policy["max_file_bytes"]):
        return None, {"path": rel, "reason": "oversized", "utf8_bytes": len(raw)}
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return None, {"path": rel, "reason": "non_utf8"}
    hits = _credential_hits(raw)
    if hits:
        return None, {"path": rel, "reason": "credential_pattern", "patterns": hits}
    try:
        ast.parse(text, filename=rel)
    except SyntaxError as exc:
        return None, {"path": rel, "reason": "ast_parse_failure", "line": exc.lineno}
    return {
        "path": rel,
        "sha256": _sha256(raw),
        "utf8_bytes": len(raw),
    }, None


def materialize(config: dict[str, Any], workspace: Path) -> dict[str, Any]:
    policy = config["selection_policy"]
    source_reports: list[dict[str, Any]] = []
    all_hashes: set[str] = set()
    total_files = 0
    total_bytes = 0

    for source in config["sources"]:
        repo_dir = _checkout_source(source, workspace)
        license_record = _verify_license(repo_dir, source)
        root = repo_dir / source["source_root"]
        _require(root.is_dir() and not root.is_symlink(), f"source root missing or symlinked: {source['repository']}")

        files: list[dict[str, Any]] = []
        exclusions: list[dict[str, Any]] = []
        for path in sorted(root.rglob("*.py"), key=lambda p: p.as_posix()):
            record, exclusion = _eligible_file_record(repo_dir, root, path, policy)
            if record is not None:
                _require(record["sha256"] not in all_hashes, f"exact duplicate file hash across bundle: {record['path']}")
                all_hashes.add(record["sha256"])
                files.append(record)
            elif exclusion is not None:
                exclusions.append(exclusion)

        _require(files, f"no eligible implementation files: {source['repository']}")
        family_bytes = sum(row["utf8_bytes"] for row in files)
        family_identity = _sha256(_canonical_bytes({
            "repository": source["repository"],
            "family_id": source["family_id"],
            "commit": source["commit"],
            "license": license_record,
            "files": files,
        }))
        source_reports.append({
            "repository": source["repository"],
            "family_id": source["family_id"],
            "commit": source["commit"],
            "source_root": source["source_root"],
            "license": license_record,
            "eligible_file_count": len(files),
            "eligible_utf8_bytes": family_bytes,
            "excluded_file_count": len(exclusions),
            "exclusions": exclusions,
            "files": files,
            "family_materialization_identity_sha256": family_identity,
        })
        total_files += len(files)
        total_bytes += family_bytes

    report_core = {
        "schema_version": REPORT_SCHEMA,
        "worker_id": config["worker_id"],
        "contract_identity_sha256": config["contract_identity_sha256"],
        "execution_profile": "LOCAL_FREE",
        "source_family_count": len(source_reports),
        "eligible_file_count": total_files,
        "eligible_utf8_bytes": total_bytes,
        "sources": source_reports,
        "purpose_firewall": {
            "model_training_source_use": config["rights_boundary"]["model_training_source_use"],
            "evaluation": "NOT_SEPARATELY_ADMITTED",
            "final_test": "PROHIBITED",
            "automatic_canonical_capacity_credit": False,
        },
        "downstream_required": config["downstream_required"],
        "claim_boundary": {
            "corpus_identity": None,
            "shard_identity": None,
            "post_dedup_capacity_claimed": False,
            "authorized_training_exposure": 0,
            "tokenizer_fit_authorized": False,
            "long_training_authorized": False,
            "paid_compute_authorized": False,
        },
    }
    report = dict(report_core)
    report["report_identity_sha256"] = _sha256(_canonical_bytes(report_core))
    return report


def validate_report(config: dict[str, Any], report: dict[str, Any]) -> None:
    _require(report.get("schema_version") == REPORT_SCHEMA, "report schema drift")
    _require(report.get("contract_identity_sha256") == EXPECTED_CONTRACT_IDENTITY, "report contract mismatch")
    _require(report.get("source_family_count") == 6, "family count mismatch")
    _require(report.get("eligible_file_count", 0) > 0, "empty file ledger")
    _require(report.get("eligible_utf8_bytes", 0) > 0, "empty byte ledger")
    sources = report.get("sources", [])
    _require(len(sources) == 6, "source report cardinality mismatch")
    _require(len({row["family_id"] for row in sources}) == 6, "family duplication")
    hashes: set[str] = set()
    bytes_sum = 0
    file_sum = 0
    for source in sources:
        _require(source["eligible_file_count"] == len(source["files"]), "family file count mismatch")
        _require(source["eligible_utf8_bytes"] == sum(row["utf8_bytes"] for row in source["files"]), "family byte count mismatch")
        for row in source["files"]:
            _require(re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is not None, "invalid file hash")
            _require(row["sha256"] not in hashes, "duplicate file hash")
            hashes.add(row["sha256"])
            _require(row["utf8_bytes"] > 0, "nonpositive file bytes")
        file_sum += source["eligible_file_count"]
        bytes_sum += source["eligible_utf8_bytes"]
    _require(file_sum == report["eligible_file_count"], "aggregate file count mismatch")
    _require(bytes_sum == report["eligible_utf8_bytes"], "aggregate byte count mismatch")
    firewall = report["purpose_firewall"]
    _require(firewall["evaluation"] == "NOT_SEPARATELY_ADMITTED", "evaluation leakage")
    _require(firewall["final_test"] == "PROHIBITED", "final-test leakage")
    _require(firewall["automatic_canonical_capacity_credit"] is False, "automatic capacity promotion")
    claim = report["claim_boundary"]
    _require(claim["corpus_identity"] is None and claim["shard_identity"] is None, "fabricated corpus/shard identity")
    _require(claim["authorized_training_exposure"] == 0, "training exposure promoted")
    _require(claim["tokenizer_fit_authorized"] is False and claim["long_training_authorized"] is False, "training gate promoted")
    _require(claim["paid_compute_authorized"] is False, "paid compute promoted")
    core = dict(report)
    identity = core.pop("report_identity_sha256", None)
    _require(identity == _sha256(_canonical_bytes(core)), "report identity mismatch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--check-contract-only", action="store_true")
    parser.add_argument("--validate-report", type=Path)
    args = parser.parse_args()

    config = load_contract(args.config)
    if args.validate_report is not None:
        report = json.loads(args.validate_report.read_text(encoding="utf-8"))
        validate_report(config, report)
        print(f"PASS report={report['report_identity_sha256']} bytes={report['eligible_utf8_bytes']}")
        return 0
    if args.check_contract_only:
        print(f"PASS contract={EXPECTED_CONTRACT_IDENTITY}")
        return 0

    if args.workspace is None:
        with tempfile.TemporaryDirectory(prefix="data-bulk-code1-") as tmp:
            report = materialize(config, Path(tmp))
    else:
        args.workspace.mkdir(parents=True, exist_ok=True)
        _require(not any(args.workspace.iterdir()), "workspace must be empty")
        report = materialize(config, args.workspace)
    validate_report(config, report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical_bytes(report) + b"\n")
    print(f"PASS report={report['report_identity_sha256']} families={report['source_family_count']} files={report['eligible_file_count']} bytes={report['eligible_utf8_bytes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
