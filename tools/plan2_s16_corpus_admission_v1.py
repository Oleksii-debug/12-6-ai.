"""Build the initial physical Plan 2 training corpus from verified sources."""
from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tools import materialize_data_bulk_code1_permissive_python_bundle as code
from tools import next100_106_balance_gate as historical_balance
from tools import plan2_s16_independent_eval_binding_v1 as base
from tools import plan2_s16_multimodal_candidate_v1 as candidate
from tools.plan2_physical_materialization_v1 import _atomic_write, _read_destination
from tools.plan2_s16_independent_eval_binding_v1 import need

SCHEMA = "12-6.plan2-s16-physical-corpus.v1"
DATA = "train.jsonl"
MANIFEST = "manifest.json"
STRATA = {"uk": "ua", "en": "en", "code": "code"}
TOLERANCE_PERCENTAGE_POINTS = 2
ALLOCATION_TOTAL_FAMILY_PERCENT = 24
ALLOCATION_WITHIN_STRATUM_FAMILY_PERCENT = 59
BASE_D03_CONFIG = "configs/data/plan2_s15_d03_physical_source_candidates_v1.json"
BOOK_CONFIG = "configs/data/plan2_public_domain_books_v1.json"
FRANKO_CONFIG = "configs/data/d03_franko1901_exact_materialization_v1.json"


def _source_terms(root: Path) -> dict[str, dict[str, Any]]:
    """Describe the source terms already verified by the physical replay."""
    books = historical_balance.load_json(root / BOOK_CONFIG)
    d03 = historical_balance.load_json(root / BASE_D03_CONFIG)
    franko = historical_balance.load_json(root / FRANKO_CONFIG)
    terms = {
        base.books.GUTENBERG_FAMILY: {
            "license": "public-domain-originals",
            "evidence": BOOK_CONFIG,
            "attribution": [
                {"author": item["author"], "title": item["title"], "source": item["upstream_source_url"]}
                for item in books["sources"]
            ],
        },
        "python.cpython.documentation": {
            "license": "PSF-2.0", "evidence": base.CPYTHON_AUTHORITY,
            "redistribution": "Preserve PSF license and copyright notice, and describe modifications.",
        },
        "ua.verba.public-domain.franko1901": {
            "license": "CC0-1.0", "evidence": FRANKO_CONFIG,
            "attribution": franko["rights_boundary"]["required_attribution"],
        },
        "common-pile/arxiv_abstracts": {
            "license": "CC0-1.0", "evidence": base.ARXIV_AUTHORITY,
            "scope": "descriptive metadata and abstracts",
        },
        "ua.languk.supreme-court-decisions": {
            "license": "MIT", "evidence": candidate.languk_rights.CONFIG_PATH,
            "attribution": "Andrii Ruda, lang-uk/court-decisions-uk, pseudonymized judicial decisions",
            "redistribution": "Preserve upstream copyright and permission notice.",
        },
    }
    notice_directories = {
        "php.manual.documentation": "php",
        "rust-book.documentation.uk-translation": "rust",
    }
    for source in d03["source_families"]:
        prefix = f"/licenses/{notice_directories[source['id']]}/"
        notices = [item for item in d03["source_license_files"] if prefix in item["path"]]
        need(bool(notices) and all(
            base.books.git_blob(base.books.read_checked(root, item["path"])) == item["git_blob_sha1"]
            for item in notices
        ), "source notice differs from the independently verified grant")
        terms[source["id"]] = {
            "license": source["source_rights"],
            "evidence": BASE_D03_CONFIG,
            "attribution": source["source_repository"],
            "notices": notices,
        }
    for source in code.load_contract(root / code.DEFAULT_CONFIG)["sources"]:
        terms[source["family_id"]] = {
            "license": source["license_id"], "evidence": candidate.CODE_AUTHORITY,
            "repository": source["repository"], "commit": source["commit"],
            "notice_path": source["license_path"], "notice_git_blob_sha1": source["license_git_blob_sha1"],
        }
    return terms


def _select(
    survivors: list[dict[str, str]], root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Choose whole records with deterministic per-family budgets."""
    policy = historical_balance.load_json(root / candidate.BALANCE_POLICY)
    historical_balance.validate_policy(policy)
    need(policy["policy_identity_sha256"] == candidate.BALANCE_POLICY_ID,
         "mixture ratios and caps changed")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    ids: set[str] = set()
    content: set[str] = set()
    for row in survivors:
        record_id = base._sha_text(row["record_id"])
        digest = base._sha_text(row["text"])
        size = len(row["text"].encode("utf-8"))
        need(size > 0 and record_id not in ids and digest not in content,
             "duplicate or empty post-dedup record")
        ids.add(record_id)
        content.add(digest)
        family, stratum = row["source_family"], STRATA[row["modality"]]
        grouped[family, stratum].append({
            "record_id_sha256": record_id,
            "source_id": row["source_id"],
            "source_family": family,
            "stratum": stratum,
            "utf8_bytes": size,
            "content_sha256": digest,
            "text": row["text"],
        })
    families = [
        {"family_id": family, "stratum": stratum,
         "unique_bytes": sum(row["utf8_bytes"] for row in rows)}
        for (family, stratum), rows in sorted(grouped.items())
    ]
    need(len({item["family_id"] for item in families}) == len(families),
         "a source family must have a single stratum")
    # Reserve headroom so whole-record rounding does not overconcentrate
    # the final mixture when the actual denominator falls below its budget.
    quantum = 2_000
    cfg = {
        **policy["policy"],
        "max_family_fraction_total": {"numerator": ALLOCATION_TOTAL_FAMILY_PERCENT, "denominator": 100},
        "max_family_fraction_own_stratum": {
            "numerator": ALLOCATION_WITHIN_STRATUM_FAMILY_PERCENT, "denominator": 100,
        },
        "budget_quantum_bytes": quantum,
        "target_total_source_bytes": sum(item["unique_bytes"] for item in families) // quantum * quantum,
    }
    budget = historical_balance.maximum_feasible_total(families, cfg)
    need(budget > 0, "independent source capacities cannot support the mixture")
    allocation = historical_balance.deterministic_allocation(budget, families, cfg)
    chosen: list[dict[str, Any]] = []
    for item in allocation:
        remaining = item["allocated_bytes"]
        rows = grouped[item["family_id"], item["stratum"]]
        for row in sorted(rows, key=lambda value: (-value["utf8_bytes"], value["record_id_sha256"])):
            size = row["utf8_bytes"]
            if size <= remaining:
                chosen.append(row)
                remaining -= size
    contributions: Counter[str] = Counter()
    family_bytes: Counter[tuple[str, str]] = Counter()
    domain_bytes: Counter[str] = Counter()
    source_bytes: Counter[str] = Counter()
    for row in chosen:
        stratum, family, size = row["stratum"], row["source_family"], row["utf8_bytes"]
        contributions[stratum] += size
        family_bytes[family, stratum] += size
        source_bytes[row["source_id"]] += size
        domain = "software" if stratum == "code" else "general_text"
        if row["source_id"] == "arxiv_abstracts":
            domain = "science_abstracts"
        domain_bytes[domain] += size
    total = sum(contributions.values())
    for stratum in historical_balance.STRATA:
        ratio = cfg["strata"][stratum]
        num, den = ratio["target_numerator"], ratio["target_denominator"]
        need(total > 0 and abs(contributions[stratum] * den - total * num) * 100
             <= total * den * TOLERANCE_PERCENTAGE_POINTS, "physical stratum balance is out of bounds")
        need(sum(1 for _, key in family_bytes if key == stratum)
             >= cfg["minimum_independent_families_per_stratum"], "insufficient selected families")
    for (_, stratum), size in family_bytes.items():
        need(size * 4 <= total and size * 5 <= contributions[stratum] * 3,
             "physical source family exceeds its cap")
    need(domain_bytes["science_abstracts"] > 0, "science coverage missing from selected corpus")
    return sorted(chosen, key=lambda row: row["record_id_sha256"]), {
        "ratios_and_caps_sha256": candidate.BALANCE_POLICY_ID,
        "allocation_total_family_cap_percent": ALLOCATION_TOTAL_FAMILY_PERCENT,
        "allocation_within_stratum_family_cap_percent": ALLOCATION_WITHIN_STRATUM_FAMILY_PERCENT,
        "maximum_whole_record_deviation_percentage_points": TOLERANCE_PERCENTAGE_POINTS,
        "selected_utf8_bytes": total,
        "available_utf8_bytes": sum(item["unique_bytes"] for item in families),
        "selected_bytes_by_stratum": dict(contributions),
        "selected_bytes_by_family": {family: size for (family, _), size in family_bytes.items()},
        "selected_bytes_by_domain": dict(domain_bytes),
        "selected_bytes_by_source": dict(source_bytes),
        "excluded_record_counts": {"WHOLE_RECORD_MIXTURE_BUDGET": len(survivors) - len(chosen)},
        "token_count_measurement_stage": "S17",
    }


def build(
    root: Path, selection: tuple[Path, Path, Path, Path], franko: Path, arxiv: Path,
    languk: Path, code_report: Path, code_checkouts: Path,
) -> tuple[dict[str, Any], bytes]:
    replay, holdouts, survivors = candidate.replay(
        root, selection, franko, arxiv, languk, code_report, code_checkouts,
    )
    scan = replay["decontamination"]
    need(holdouts["all_source_families_disjoint_across_roles"]
         and holdouts["evaluation_use_rights_separately_admitted"]
         and not holdouts["real_final_test_outcomes_read"]
         and scan["clean"], "reserved evaluation custody or decontamination failed")
    role_counts = holdouts["role_language_records"]
    need(all(role_counts[role].get(lang, 0) > 0 for role in ("validation", "test") for lang in ("uk", "en")),
         "reserved evaluation is missing a Ukrainian or English cohort")
    rows, mixture = _select(survivors, root)
    terms = _source_terms(root)
    used_families = {row["source_family"] for row in rows}
    need(used_families <= terms.keys(), "selected family has no independently verified source terms")
    source_records = Counter(row["source_id"] for row in rows)
    train_strata = Counter("uk" if row["stratum"] == "ua" else row["stratum"] for row in rows)
    payload = b"".join(base.books.canonical(row) for row in rows)
    core = {
        "schema": SCHEMA,
        "source_replay_sha256": replay["manifest_sha256"],
        "s3_s4_physical_and_normalized_source_evidence_sha256": holdouts["physical_training_candidate_sha256"],
        "s3_source_family_evidence": replay["source_evidence"],
        "selected_source_terms": {family: terms[family] for family in sorted(used_families)},
        "s5_privacy_policy": replay["canonical_privacy_policy"],
        "s6_s7_deduplication": scan["candidate_dedup_projection"],
        "s8_data232_report_sha256": scan["data232_report_sha256"],
        "s8_selection_sha256": holdouts["eval303_selection_identity_sha256"],
        "s8_final_test_sha256": holdouts["eval233_final_set_identity_sha256"],
        "s9_mixture": mixture,
        "role_counts": {
            "train": dict(train_strata),
            "validation": role_counts["validation"],
            "test": role_counts["test"],
        },
        "training_records_by_source": dict(source_records),
        "base_source_rights_sha256": holdouts["physical_source_rights_sha256"],
        "coverage": {
            "sk": "UNKNOWN", "cs": "UNKNOWN", "math": "UNKNOWN",
            "science_abstracts": "ENGLISH_SOURCE_ONLY",
            "code": "PYTHON_SOURCE",
            "science_evaluation": "UNKNOWN", "math_evaluation": "UNKNOWN",
            "code_evaluation": "UNKNOWN",
        },
        "payload_sha256": base.books.sha(payload),
        "payload_bytes": len(payload),
        "records": len(rows),
        "physical_s3_s9_admission_passed": True,
        "heldout_payloads_exported": False,
        "production_release_authorized": False,
        "model_training_executed": False,
    }
    return {**core, "manifest_sha256": base.books.sha(base.books.canonical(core))}, payload


def _stage(destination: Path, manifest: dict[str, Any], payload: bytes) -> None:
    need(not any(path.is_symlink() for path in (destination, *destination.parents)),
         "unsafe corpus destination")
    files = ((destination / DATA, payload), (destination / MANIFEST, base.books.canonical(manifest)))
    for path, raw in files:
        if path.exists() or path.is_symlink():
            need(_read_destination(path) == raw, "existing corpus differs from the verified rebuild")
    destination.mkdir(parents=True, exist_ok=True)
    for path, raw in files:
        if not path.exists():
            _atomic_write(destination, path, raw)
        need(_read_destination(path) == raw, "corpus readback failed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--selection-reconstructed", type=Path, nargs=4, required=True)
    for arg in ("franko", "arxiv", "languk", "code-report", "code-checkouts"):
        parser.add_argument(f"--{arg}", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest, payload = build(
            args.root, tuple(args.selection_reconstructed), args.franko, args.arxiv, args.languk,
            args.code_report, args.code_checkouts,
        )
        _stage(args.out_dir, manifest, payload)
    except base.IndependentEvalDenied as exc:
        print(json.dumps({"decision": "DENIED", "reason": str(exc)}))
        return 2
    except (KeyError, OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(json.dumps({"decision": "DENIED", "error_type": type(exc).__name__}))
        return 2
    print(json.dumps({"decision": "PHYSICAL_CORPUS_VERIFIED", "manifest_sha256": manifest["manifest_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
