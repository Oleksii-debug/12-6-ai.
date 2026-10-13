"""Bind independent real EVAL-303 selection and EVAL-233 final custody for S16.

This does not admit the S15 training-source candidates to canonical S3--S9.
It reuses separate source-specific evaluation approvals and does not expose
heldout payload, final-test outcomes, model scores or optimizer targets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from tools import plan2_public_domain_books_v1 as books
from tools import plan2_s15_real_eval233_decontamination_v1 as final_audit
from tools import plan2_s15_three_family_physical_v1 as physical
from tools import plan2_s16_source_rights_privacy_v1 as rights
from tools.plan2_physical_materialization_v1 import _atomic_write, _read_destination
from twelve_six.data import eval233_final_test_resolver_v1 as eval233

SCHEMA = "12-6.plan2-s16-independent-real-eval-binding.v1"
OUTPUT = "independent-real-eval-binding.json"
SELECTION_CONFIG = "configs/evaluation/eval303_selection_validation_composite_v1.json"
SELECTION_MEMBERS = "data/evaluation/eval303/selection-validation/composite-membership.jsonl"
SELECTION_GIT_BLOB = "c2e5b27bac541e55be2c807ea65b0e2ed77b7019"
SELECTION_MEMBER_GIT_BLOB = "04cf8b3fc1857f0d7b51e581aab8e4dc68f7f47d"
SELECTION_IDENTITY = "7b97a9ab04469236dc5bc17fc80155cb43430b01c443bb6209fac090557258fd"
SELECTION_MEMBERS_SHA256 = "e4bb39dd7aa6a20c7ed34e093f563b5f4896ac16828151c6b375a83cd8a068c6"
ROLES = ("train", "validation", "test")
LANGS = ("uk", "en")


class IndependentEvalDenied(ValueError):
    """Physical selection/final membership or rights reservation drift."""


def need(ok: bool, reason: str) -> None:
    if not ok:
        raise IndependentEvalDenied(reason)


def _pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for k, v in values:
        need(k not in result, "duplicate heldout membership JSON key")
        result[k] = v
    return result


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "strict")).hexdigest()


def inspect_selection_rows(raw: bytes, config: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate *metadata only*: selected text cannot reenter training."""
    need(isinstance(raw, bytes) and bool(raw) and raw.endswith(b"\n"),
         "invalid selection membership bytes")
    need(config["schema_version"] == "12-6.eval303-selection-validation-composite.v1"
         and config["selection_identity_sha256"] == SELECTION_IDENTITY
         and config["purpose"] == "selection-validation"
         and config["rights_and_reservation"][
             "all_included_records_selection_validation_authorized"] is True
         and config["rights_and_reservation"][
             "all_included_records_training_prohibited"] is True
         and config["rights_and_reservation"][
             "all_included_records_tokenizer_fit_prohibited"] is True
         and config["rights_and_reservation"][
             "all_included_records_final_test_prohibited"] is True,
         "selection scope/rights/config drift")
    entries = raw.splitlines(keepends=True)
    need(len(entries) == config["composite_membership"]["documents"] == 10,
         "selection count drift")
    rows: list[dict[str, Any]] = []
    ids: set[str] = set()
    contents: set[str] = set()
    families: Counter[str] = Counter()
    langcounts: Counter[str] = Counter()
    for line in entries:
        need(line.endswith(b"\n"), "selection member lacks line terminator")
        try:
            row = json.loads(line.decode("utf-8", "strict"), object_pairs_hook=_pairs)
        except (UnicodeError, ValueError) as exc:
            raise IndependentEvalDenied("malformed selection row") from exc
        need(type(row) is dict, "invalid membership row")
        rid = row.get("record_id")
        fam = row.get("source_family")
        sid = row.get("source_id")
        lang = row.get("language")
        digest = row.get("content_sha256")
        length = row.get("utf8_bytes")
        need(all(type(v) is str and bool(v) for v in (rid, sid, fam))
             and type(lang) is str and lang in LANGS
             and type(digest) is str and len(digest) == 64
             and all(ch in "0123456789abcdef" for ch in digest)
             and type(length) is int and length > 0,
             "selected record source/language/hash invalid")
        need(row.get("purpose") == "selection-validation"
             and row.get("selection_stratum") in ("ua", "en")
             and row["selection_stratum"] == ("ua" if lang == "uk" else "en")
             and row.get("selection_eligible") is True
             and row.get("future_training_prohibited") is True
             and row.get("training_eligible") is False
             and row.get("tokenizer_fit_eligible") is False
             and row.get("final_test_eligible") is False
             and type(row.get("rights_authority_ref")) is str
             and bool(row["rights_authority_ref"])
             and type(row.get("reservation_authority_ref")) is str
             and bool(row["reservation_authority_ref"]),
             "selected record rights/reservation grant changed")
        need(rid not in ids and digest not in contents,
             "duplicate selection record or content")
        ids.add(rid)
        contents.add(digest)
        families[fam] += 1
        langcounts[lang] += 1
        rows.append(row)
    need(dict(families) == config["source_families"]["records_by_family"]
         and len(families) == config["source_families"]["independent_count"] == 4
         and langcounts == {"uk": 8, "en": 2},
         "selection language/family composition drift")
    return rows


def inspect(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    try:
        permissions = rights.inspect(root)
        sources = physical.inspect(root)
        finale = final_audit.inspect(root)
        selection_config_raw = books.read_checked(root, SELECTION_CONFIG)
        selected_raw = books.read_checked(root, SELECTION_MEMBERS)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise IndependentEvalDenied("real physical source or evaluation lineage denied") from exc
    need(books.git_blob(selection_config_raw) == SELECTION_GIT_BLOB,
         "frozen EVAL-303 configuration Git blob drift")
    need(books.git_blob(selected_raw) == SELECTION_MEMBER_GIT_BLOB
         and books.sha(selected_raw) == SELECTION_MEMBERS_SHA256,
         "frozen EVAL-303 membership bytes drift")
    try:
        config = json.loads(selection_config_raw.decode("utf-8", "strict"),
                            object_pairs_hook=_pairs)
    except (UnicodeError, ValueError) as exc:
        raise IndependentEvalDenied("invalid EVAL-303 configuration") from exc
    need(type(config) is dict
         and config["composite_membership"]["sha256"] == SELECTION_MEMBERS_SHA256
         and config["final_test_firewall"]["data232_final_test_identity_sha256"]
         == eval233.DATA232_FINAL_TEST_ID,
         "selection/final firewalls disconnected")
    selected = inspect_selection_rows(selected_raw, config)
    need(permissions["physical_manifest_sha256"] == sources["manifest_sha256"]
         and permissions["source_level_rights_verified"] is True
         and permissions["physical_privacy_executed"] is True
         and permissions["s3_s9_corpus_admitted"] is False
         and permissions["training_corpus_authorized"] is False,
         "unverified original training candidate rights/privacy")
    need(finale["physical_three_family_sha256"] == sources["manifest_sha256"]
         and finale["selection_identity_sha256"] == SELECTION_IDENTITY
         and finale["real_final_test_custody_verified"] is True
         and finale["real_final_test_outcomes_read"] is False
         and finale["real_final_decontamination_clean"] is True
         and finale["physical_s3_s9_admitted"] is False,
         "real DATA232 final reserve/decontamination not cleared")
    try:
        final_rows, _authority, final_receipt = eval233.resolve_eval233_final_test(root)
    except (OSError, ValueError, RuntimeError) as exc:
        raise IndependentEvalDenied("real final set cannot be independently resolved") from exc
    need(final_receipt["source_final_set_identity_sha256"] ==
         finale["real_eval233_final_membership_identity_sha256"]
         and final_receipt["final_test_outcomes_read"] is False
         and final_receipt["documents"] == 16,
         "real final-test source identity drift")
    train_families = {row["source_family"] for row in sources["source_members"]}
    val_families = {row["source_family"] for row in selected}
    test_families = {row["source_family"] for row in final_rows}
    need(len(train_families) == 3 and len(val_families) == 4
         and len(test_families) == 2
         and train_families.isdisjoint(val_families)
         and train_families.isdisjoint(test_families)
         and val_families.isdisjoint(test_families),
         "train, validation and final source-family overlap")
    train_ids = {row["source_id"] for row in sources["source_members"]}
    val_ids = {row["source_id"] for row in selected}
    test_ids = {row["source_id"] for row in final_rows}
    need(train_ids.isdisjoint(val_ids)
         and train_ids.isdisjoint(test_ids)
         and val_ids.isdisjoint(test_ids),
         "train, validation and final source identity overlap")
    train_hashes = {row["normalized_sha256"] for row in sources["source_members"]}
    val_hashes = {row["content_sha256"] for row in selected}
    test_hashes = {_sha_text(row["text"]) for row in final_rows}
    need(not train_hashes & val_hashes
         and not train_hashes & test_hashes
         and not val_hashes & test_hashes,
         "train, selection or final exact content leakage")
    by_role: dict[str, Counter[str]] = {
        "train": Counter(x["modality"] for x in sources["source_members"]),
        "validation": Counter(x["language"] for x in selected),
        "test": Counter("uk" if x["modality"] == "ua" else x["modality"]
                        for x in final_rows),
    }
    need(all(by_role[role][lang] > 0 for role in ROLES for lang in LANGS)
         and by_role["train"] == {"uk": 12, "en": 3}
         and by_role["validation"] == {"uk": 8, "en": 2}
         and by_role["test"] == {"uk": 8, "en": 8},
         "real independent UA/EN role coverage changed")
    byte_counts: dict[str, Counter[str]] = {
        "train": Counter(), "validation": Counter(), "test": Counter(),
    }
    for row in sources["source_members"]:
        byte_counts["train"][row["modality"]] += row["normalized_bytes"]
    for row in selected:
        byte_counts["validation"][row["language"]] += row["utf8_bytes"]
    for row in final_rows:
        lang = "uk" if row["modality"] == "ua" else "en"
        byte_counts["test"][lang] += len(row["text"].encode("utf-8"))
    core = {
        "schema_version": SCHEMA,
        "decision": "INDEPENDENT_REAL_UA_EN_EVAL_REUSED_NO_PRODUCTION_CORPUS",
        "physical_source_rights_sha256": permissions["manifest_sha256"],
        "physical_training_candidate_sha256": sources["manifest_sha256"],
        "eval303_selection_identity_sha256": SELECTION_IDENTITY,
        "eval303_selection_membership_sha256": SELECTION_MEMBERS_SHA256,
        "eval233_final_set_identity_sha256":
            final_receipt["source_final_set_identity_sha256"],
        "eval233_data232_report_sha256": finale["data232_report_sha256"],
        "role_language_records": {
            role: {lang: by_role[role][lang] for lang in LANGS}
            for role in ROLES
        },
        "role_language_utf8_bytes": {
            role: {lang: byte_counts[role][lang] for lang in LANGS}
            for role in ROLES
        },
        "role_family_counts": {
            "train": len(train_families), "validation": len(val_families),
            "test": len(test_families),
        },
        "all_three_roles_physical_ua_en": True,
        "all_source_families_disjoint_across_roles": True,
        "role_exact_content_overlap_count": 0,
        "evaluation_use_rights_separately_admitted": True,
        "heldout_fitted_to_tokenizer": False,
        "real_final_test_outcomes_read": False,
        "current_s15_train_sources_are_corpus_admitted": False,
        "production_balance_approved": False,
        "train_selection_cross_source_near_decontamination_verified": False,
        "production_s3_s9_dataset_authorized": False,
        "training_authorized": False,
        "plan2_evaluation_release_authorized": False,
        "tokenizer_fit_authorized": False,
        "production_release_authorized": False,
        "plan9_binding_authorized": False,
        "terminal_done": False,
    }
    return {**core, "manifest_sha256": books.sha(books.canonical(core))}


def stage(root: Path, destination: Path) -> dict[str, Any]:
    need(not any(p.is_symlink() for p in (destination, *destination.parents)),
         "unsafe independent evaluation destination")
    report = inspect(root)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / OUTPUT
    raw = books.canonical(report)
    if target.exists() or target.is_symlink():
        need(target.is_file() and not target.is_symlink()
             and _read_destination(target) == raw,
             "immutable independent-evaluation readback changed")
    else:
        _atomic_write(destination, target, raw)
    need(_read_destination(target) == raw, "readback mismatch")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = stage(args.root, args.out_dir)
    except (OSError, ValueError) as exc:
        print(json.dumps({"decision": "DENIED", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({
        "decision": report["decision"],
        "manifest_sha256": report["manifest_sha256"],
        "role_language_records": report["role_language_records"],
        "terminal_done": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
