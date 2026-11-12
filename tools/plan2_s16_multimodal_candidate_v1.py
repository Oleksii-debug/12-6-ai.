"""Replay the S16 multilingual and code candidate, without granting corpus admission."""
from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from tools import materialize_data_bulk_code1_permissive_python_bundle as code
from tools import next100_106_balance_gate as historical_balance
from tools import plan2_s16_independent_eval_binding_v1 as base
from tools.plan2_physical_materialization_v1 import _atomic_write, _read_destination
from twelve_six import d03_languk_postexecution_rights as languk_rights

OUTPUT = "multimodal-candidate.json"
CODE_AUTHORITY = "evidence/data_bulk_code1/permissive_python_bundle_v1_terminal.json"
CODE_AUTHORITY_BLOB = "b2b9b361dbf0e6c883dca4d7f006812c69cdd4e0"
CODE_REPORT_ID = "80f3a8f20dfb82825e6c89ac1f76f2f41233296b8a426a1cf466b0fcc0892985"
BALANCE_POLICY = "configs/data/next100_106_balance_gate_policy_v1.json"
BALANCE_POLICY_ID = "9a9242f47981c25e754fc95e2650050da4e4195aa1ef3a78f2c293f9e25d7ff7"


def _languk(root: Path, candidate: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    languk_rights.validate_languk_postexecution_rights(root, languk_rights.CONFIG_PATH)
    spec = base.post_admission.LANGUK
    base.need(candidate.is_file() and not candidate.is_symlink(), "invalid LangUK candidate file")
    with candidate.open("rb") as handle:
        candidate_raw = handle.read(4_000_001)
    base.need(0 < len(candidate_raw) <= 4_000_000, "LangUK candidate exceeds its byte limit")
    inventory, payloads, authority = base.post_admission._prepare_source(
        spec, authority_raw=base.books.read_checked(root, languk_rights.CONFIG_PATH),
        candidate_raw=candidate_raw,
    )
    rows, rejections = base._privacy_screen(
        inventory, payloads, "languk-supreme-court", spec.family, spec.modality,
    )
    base.need(bool(rows), "no LangUK candidate survived canonical privacy screening")
    return rows, {
        **authority, "accepted_records": len(rows), "privacy_rejections": rejections,
    }


def _git_blob(repo: Path, commit: str, name: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), "cat-file", "blob", f"{commit}:{name}"],
        check=True, capture_output=True, timeout=30,
    ).stdout


def _code(
    root: Path, report_path: Path, checkouts: Path,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    authority_raw = base.books.read_checked(root, CODE_AUTHORITY)
    base.need(base.books.git_blob(authority_raw) == CODE_AUTHORITY_BLOB,
              "code terminal authority changed")
    authority = json.loads(authority_raw, object_pairs_hook=base._pairs)
    report = json.loads(
        base.books.read_checked(report_path.parent.resolve(strict=True), report_path.name),
        object_pairs_hook=base._pairs,
    )
    config = code.load_contract(root / code.DEFAULT_CONFIG)
    code.validate_report(config, report)
    base.need(report["report_identity_sha256"] == authority["artifact"]["retained_report_identity_sha256"]
              == CODE_REPORT_ID, "code report is not the independently verified terminal result")
    expected = {row["family_id"]: row for row in authority["families"]}
    checkouts = checkouts.resolve(strict=True)
    rows: list[dict[str, str]] = []
    rejections: Counter[str] = Counter()
    for source in report["sources"]:
        family = source["family_id"]
        base.need(all(source[field] == expected[family][field] for field in (
            "family_materialization_identity_sha256", "eligible_file_count", "eligible_utf8_bytes",
        )), "code family provenance changed")
        repo = checkouts / source["repository"].replace("/", "__")
        base.need(repo.is_dir() and not repo.is_symlink()
                  and repo.resolve().parent == checkouts, "invalid source checkout")
        license_record = source["license"]
        license_bytes = _git_blob(repo, source["commit"], license_record["license_path"])
        base.need(base.books.git_blob(license_bytes) == license_record["license_git_blob_sha1"]
                  and base.books.sha(license_bytes) == license_record["license_sha256"],
                  "pinned code license changed")
        for item in source["files"]:
            raw = _git_blob(repo, source["commit"], item["path"])
            base.need(len(raw) == item["utf8_bytes"] and base.books.sha(raw) == item["sha256"],
                      "code file is not the exact authorized candidate")
            scan = base.privacy.hash_safe_scan(raw)
            if scan.action != "ALLOW":
                rejections[scan.action] += 1
                continue
            rows.append({
                "record_id": f"{family}:{item['path']}", "source_id": family,
                "source_family": family, "lineage_family": family, "modality": "code",
                "text": raw.decode("utf-8", "strict"),
            })
    base.need(bool(rows), "no code candidate survived canonical privacy screening")
    return rows, {
        "terminal_source_report_sha256": CODE_REPORT_ID,
        "verified_files": report["eligible_file_count"], "accepted_records": len(rows),
        "privacy_rejections": dict(sorted(rejections.items())),
    }


def _balance_preview(root: Path, projection: dict[str, Any]) -> dict[str, Any]:
    policy = historical_balance.load_json(root / BALANCE_POLICY)
    historical_balance.validate_policy(policy)
    base.need(policy["policy_identity_sha256"] == BALANCE_POLICY_ID, "historical balance policy changed")
    cfg = policy["policy"]
    families = [
        {"family_id": family, "stratum": "ua" if modality == "uk" else modality,
         "unique_bytes": size}
        for family, modalities in projection["surviving_utf8_bytes_by_family_and_modality"].items()
        for modality, size in modalities.items()
    ]
    target = cfg["target_total_source_bytes"]
    maximum = historical_balance.maximum_feasible_total(families, cfg)
    return {
        "scope": "HISTORICAL_20M_POLICY_BYTE_UPPER_BOUND_ONLY",
        "policy_sha256": policy["policy_identity_sha256"],
        "target_utf8_bytes": target,
        "maximum_feasible_utf8_bytes_before_whole_document_selection": maximum,
        "maximum_reaches_historical_target": maximum == target,
        "is_plan2_production_balance_authority": False,
    }


def inspect(
    root: Path, selection: tuple[Path, Path, Path, Path], franko: Path, arxiv: Path,
    languk: Path, code_report: Path, code_checkouts: Path,
) -> dict[str, Any]:
    receipt, _, _ = replay(root, selection, franko, arxiv, languk, code_report, code_checkouts)
    return receipt


def replay(
    root: Path, selection: tuple[Path, Path, Path, Path], franko: Path, arxiv: Path,
    languk: Path, code_report: Path, code_checkouts: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, str]]]:
    """Replay source evidence and keep the verified training survivors in memory."""
    root = root.resolve(strict=True)
    prior = base.inspect(root)
    sources = base.physical.inspect(root)
    final_rows, _, final_receipt = base.eval233.resolve_eval233_final_test(root)
    base.need(final_receipt["source_final_set_identity_sha256"]
              == prior["eval233_final_set_identity_sha256"], "final-test custody changed")
    python_rows, python_receipt = base._load_cpython_rows(root)
    franko_rows, franko_receipt = base._load_franko_rows(root, franko)
    arxiv_rows, arxiv_receipt = base._load_arxiv_rows(root, arxiv)
    languk_rows, languk_receipt = _languk(root, languk)
    code_rows, code_receipt = _code(root, code_report, code_checkouts)
    combined = python_rows + franko_rows + arxiv_rows + languk_rows + code_rows
    scan, survivors = base._scan_selection_with_survivors(
        root, sources, selection, extra_rows=combined, final_rows=final_rows,
    )
    core = {
        "schema": "12-6.plan2-s16-multimodal-candidate.v1",
        "base_authority_sha256": prior["manifest_sha256"],
        "final_documents_compared": len(final_rows),
        "source_evidence": {
            "cpython": python_receipt, "franko": franko_receipt, "arxiv": arxiv_receipt,
            "languk": languk_receipt, "code": code_receipt,
        },
        "canonical_privacy_policy": base.privacy.POLICY_VERSION,
        "decontamination": scan,
        "historical_balance_preview": _balance_preview(root, scan["candidate_dedup_projection"]),
        "native_s3_s9_corpus_admitted": False,
        "production_balance_approved": False,
        "tokenizer_fit_authorized": False,
        "training_authorized": False,
        "production_release_authorized": False,
        "terminal_done": False,
    }
    receipt = {**core, "manifest_sha256": base.books.sha(base.books.canonical(core))}
    return receipt, prior, survivors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--selection-reconstructed", type=Path, nargs=4, required=True)
    for argument in ("franko", "arxiv", "languk", "code-report", "code-checkouts"):
        parser.add_argument(f"--{argument}", type=Path, required=True)
    args = parser.parse_args()
    try:
        destination = args.out_dir
        base.need(not any(path.is_symlink() for path in (destination, *destination.parents)),
                  "unsafe multimodal evidence destination")
        evidence = inspect(
            args.root, tuple(args.selection_reconstructed), args.franko, args.arxiv, args.languk,
            args.code_report, args.code_checkouts,
        )
        raw = base.books.canonical(evidence)
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / OUTPUT
        if not target.exists() and not target.is_symlink():
            _atomic_write(destination, target, raw)
        base.need(_read_destination(target) == raw, "multimodal candidate receipt changed")
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(json.dumps({"decision": "DENIED", "error_type": type(exc).__name__}, sort_keys=True))
        return 2
    print(json.dumps({"decision": "CANDIDATE_ONLY", "manifest_sha256": evidence["manifest_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
