"""S15 real 3-family S9 mixture mechanics, not a production S8/S9 receipt.

Reuses the canonical S9 deterministic quota composer only after real source
G06/DATA232 gates. Its S8-shaped input is explicitly synthetic: it must NEVER
be treated as an admitted multi-family S8 decontamination authority.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools import plan2_corpus_mixture_v1 as mixture
from tools import plan2_public_domain_books_v1 as books
from tools import plan2_s15_three_family_physical_v1 as combined
from tools import plan2_s15_three_family_split_probe_v1 as physical_split
from tools.plan2_physical_materialization_v1 import _atomic_write, _read_destination

SCHEMA = "12-6.plan2-s15-three-family-real-mixture-probe.v1"
OUTPUT = "three-family-mixture-probe.json"


class MixProbeDenied(ValueError):
    """Invalid upstream physical source or unauthorized corpus assertion."""


def need(value: bool, reason: str) -> None:
    if not value:
        raise MixProbeDenied(reason)


def inspect(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    physical = combined.inspect(root)
    s10 = physical_split.inspect(root)
    need(physical["source_family_count"] == 3
         and physical["document_identity_count"] == 15
         and physical["g06_rejected_record_ids"] == []
         and physical["data232_excluded_record_count"] == 0
         and physical["data232_quarantined_source_family_count"] == 0
         and physical["training_corpus_authorized"] is False
         and physical["physical_s3_s9_multi_family_admitted"] is False
         and s10["upstream_three_family_sha256"] == physical["manifest_sha256"]
         and s10["canonical_source_family_count"] == 3
         and s10["physical_s9_admitted"] is False,
         "missing physical and non-authorizing source privacy/rights controls")
    metadata = physical["source_members"]
    rows = physical_split.physical_rows(root, physical)
    by_record = {x["record_id"]: x for x in metadata}
    by_source: dict[str, dict[str, Any]] = {}
    enriched = []
    for row in rows:
        m = by_record[row["record_id"]]
        need(row["source_id"] == m["source_id"]
             and books.sha(row["text"].encode("utf-8")) == m["normalized_sha256"],
             "source changed after G06 and DATA232 audit")
        source = by_source.setdefault(m["source_id"], {
            "source_id": m["source_id"],
            "source_family": m["source_family"],
            "language": m["modality"],
            "domain": "literature" if m["modality"] == "en"
                      else "technical_documentation",
            "modality": "text",
            "sampling_weight": 1,
        })
        need(source["source_family"] == m["source_family"]
             and source["language"] == m["modality"],
             "one source credited across inconsistent families or languages")
        enriched.append({
            "record_id": m["record_id"],
            "source_id": m["source_id"],
            "modality": m["modality"],
            "text": row["text"],
        })
    need(len(enriched) == 15 and len(by_source) == 5
         and {x["source_family"] for x in by_source.values()} ==
             set(physical["families"]), "source mixture identity incomplete")
    policy_raw = books.read_checked(root, mixture.POLICY_PATH)
    need(books.git_blob(policy_raw) == mixture.POLICY_GIT_BLOB,
         "incumbent S9 source policy was replaced")
    upstream_policy = json.loads(policy_raw)
    incumbent = mixture._parse_policy(books.canonical(upstream_policy))
    need(incumbent["purpose"] == "LOCAL_FREE_CANDIDATE_MIXTURE"
         and incumbent["max_records_total"] == 50,
         "unqualified S9 deterministic policy source")
    policy = {
        **incumbent,
        "revision": "plan2-s15-physical-3-family-mechanics-v1",
        "sources": [by_source[sid] for sid in sorted(by_source)],
    }
    receipt = {
        "manifest_sha256": books.sha(books.canonical({
            "upstream_physical_source_sha256": physical["manifest_sha256"],
            "upstream_g06_input_sha256": physical["g06_input_sha256"],
            "upstream_data232_report_sha256": physical["data232_report_sha256"],
            "synthetic_s8_mechanics_only": True,
        })),
        "decontaminated_record_ids": sorted(by_record),
        "excluded_record_ids": [],
    }
    mixed = mixture.compose_mixture(receipt, enriched, policy)
    need(mixed["selected_record_count"] == 15
         and set(mixed["selected_record_ids"]) == set(by_record)
         and mixed["training_corpus_authorized"] is False
         and mixed["tokenizer_fit_authorized"] is False
         and mixed["actual_tokenizer_token_count"] is None,
         "S9 mixture lost source members or impersonates release tokens")
    core = {
        "schema_version": SCHEMA,
        "decision": "THREE_REAL_FAMILIES_S9_MIXTURE_MECHANICS_NOT_ADMISSION",
        "source_cohort_manifest_sha256": physical["manifest_sha256"],
        "source_s10_split_manifest_sha256": s10["s10_split_manifest_sha256"],
        "synthetic_s8_wrapper_sha256": receipt["manifest_sha256"],
        "s9_mechanics_mixture_sha256": mixed["dataset_candidate_sha256"],
        "s9_mixture_policy_sha256": mixed["policy_sha256"],
        "source_family_count": 3,
        "physical_source_count": 5,
        "selected_physical_document_count": mixed["selected_record_count"],
        "source_contributions": mixed["contributions"],
        "source_excluded_counts": mixed["excluded_reason_counts"],
        "token_count_is_proxy_only": True,
        "real_s8_authority_issued": False,
        "physical_s3_s9_admitted": False,
        "tokenizer_fit_authorized": False,
        "training_corpus_authorized": False,
        "production_release_authorized": False,
        "terminal_done": False,
    }
    return {**core, "manifest_sha256": books.sha(books.canonical(core))}


def stage(root: Path, destination: Path) -> dict[str, Any]:
    need(not any(p.is_symlink() for p in (destination, *destination.parents)),
         "symlink mixture destination")
    report = inspect(root)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / OUTPUT
    raw = books.canonical(report)
    if target.exists() or target.is_symlink():
        need(target.is_file() and not target.is_symlink()
             and _read_destination(target) == raw,
             "immutable three-family mixture receipt changed")
    else:
        _atomic_write(destination, target, raw)
    need(_read_destination(target) == raw, "physical mixture receipt readback drift")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    report = stage(args.root, args.out_dir)
    print(json.dumps({
        "decision": report["decision"],
        "manifest_sha256": report["manifest_sha256"],
        "terminal_done": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
