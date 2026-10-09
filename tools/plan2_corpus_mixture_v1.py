"""Plan 2 S9: deterministic, policy-pinned corpus mixture over S8 survivors.

LOCAL_FREE candidate-only; no tokenizer fit, training admission, or final-test
access. The token proxy is NOT a model-tokenizer count.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tools import plan2_reserved_eval_firewall_v1 as firewall
from tools.plan2_physical_materialization_v1 import (
    _atomic_write,
    _json,
    _read_destination,
    _read_source,
)

SCHEMA = "12-6.plan2-corpus-mixture-candidate.v1"
POLICY_SCHEMA = "12-6.plan2-corpus-mixture-policy.v1"
POLICY_PATH = "configs/data/plan2_corpus_mixture_policy_v1.json"
POLICY_GIT_BLOB = "5edb3c7c49669b8b627f83261ca9446ff3e0c47d"
BUCKETS = ("source", "family", "language", "domain", "modality")


class Plan2MixtureError(ValueError):
    """Mixture input, versioned policy, or immutable output not trusted."""


def _need(ok: bool, message: str) -> None:
    if not ok:
        raise Plan2MixtureError(message)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: Any) -> bytes:
    return (json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ) + "\n").encode("utf-8")


def _git_blob(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def _positive_int(value: Any) -> bool:
    return type(value) is int and 0 < value <= 1_000_000


def _parse_policy(raw: bytes) -> dict[str, Any]:
    try:
        policy = _json(raw)
    except (ValueError, UnicodeError, TypeError) as exc:
        raise Plan2MixtureError("mixture policy malformed") from exc
    _need(set(policy) == {
        "schema_version", "revision", "purpose", "seed", "max_records_total",
        "caps", "sources",
    }, "mixture policy keys drift")
    _need(
        policy["schema_version"] == POLICY_SCHEMA
        and policy["purpose"] == "LOCAL_FREE_CANDIDATE_MIXTURE"
        and type(policy["revision"]) is str
        and bool(policy["revision"].strip())
        and type(policy["seed"]) is str
        and re.fullmatch(r"[0-9a-f]{64}", policy["seed"]) is not None
        and _positive_int(policy["max_records_total"]),
        "mixture policy schema/seed/limits invalid",
    )
    caps = policy["caps"]
    _need(type(caps) is dict and set(caps) == set(BUCKETS)
          and all(_positive_int(x) for x in caps.values()),
          "mixture caps invalid")
    sources = policy["sources"]
    _need(type(sources) is list and bool(sources),
          "mixture source policy missing")
    seen = set()
    for row in sources:
        _need(type(row) is dict and set(row) == {
            "source_id", "source_family", "language", "domain", "modality",
            "sampling_weight",
        }, "mixture source policy shape invalid")
        _need(all(type(row[k]) is str and
                  re.fullmatch(r"[a-z][a-z0-9._:-]{1,127}", row[k])
                  for k in ("source_id", "source_family", "language",
                            "domain", "modality")),
              "mixture source dimension invalid")
        _need(_positive_int(row["sampling_weight"]),
              "mixture sampling weight invalid")
        _need(row["source_id"] not in seen, "duplicate mixture source policy")
        seen.add(row["source_id"])
    return policy


def _token_proxy(text: str) -> int:
    # Deliberately labeled proxy, never impersonates a frozen tokenizer.
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def compose_mixture(
    s8_receipt: Mapping[str, Any],
    s7_rows: Sequence[Mapping[str, str]],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Pure deterministic selector; called only after upstream S8 verification."""
    selected_pool = set(s8_receipt["decontaminated_record_ids"])
    excluded = set(s8_receipt["excluded_record_ids"])
    by_source = {x["source_id"]: x for x in policy["sources"]}
    records: list[tuple[int, str, dict[str, str], int]] = []
    seen: set[str] = set()
    audit: list[dict[str, Any]] = []
    for row in s7_rows:
        rid = row["record_id"]
        _need(type(rid) is str and rid not in seen, "duplicate mixture record")
        seen.add(rid)
        _need(rid in selected_pool or rid in excluded,
              "mixture row absent from verified S8 boundary")
        if rid in excluded:
            audit.append({"record_id": rid, "reason": "S8_DECONTAMINATED"})
            continue
        tags = by_source.get(row["source_id"])
        if tags is None:
            audit.append({"record_id": rid, "reason": "UNMAPPED_SOURCE"})
            continue
        _need(tags["language"] == row["modality"],
              "mixture claimed language inconsistent with verified S7/S8")
        _need(tags["modality"] == "text",
              "mixture claimed modality inconsistent with verified S7/S8")
        units = _token_proxy(row["text"])
        _need(units > 0, "empty token proxy record")
        rank = int(_sha((policy["seed"] + ":" + rid).encode()), 16)
        rank //= tags["sampling_weight"]
        records.append((rank, rid, tags, units))
    _need(seen == selected_pool | excluded, "mixture missing S8 records")
    _need(not selected_pool & excluded, "S8 selected/excluded overlap")
    counts: dict[str, Counter[str]] = {k: Counter() for k in BUCKETS}
    selected: list[str] = []
    totals: dict[str, dict[str, Any]] = {}
    for _, rid, tags, units in sorted(records, key=lambda r: (r[0], r[1])):
        dims = {
            "source": tags["source_id"],
            "family": tags["source_family"],
            "language": tags["language"],
            "domain": tags["domain"],
            "modality": tags["modality"],
        }
        reason = None
        if len(selected) >= policy["max_records_total"]:
            reason = "TOTAL_CAP"
        else:
            for key in BUCKETS:
                if counts[key][dims[key]] >= policy["caps"][key]:
                    reason = key.upper() + "_CAP"
                    break
        if reason:
            audit.append({"record_id": rid, "reason": reason})
            continue
        selected.append(rid)
        for key, name in dims.items():
            counts[key][name] += 1
            bucket = totals.setdefault(key, {}).setdefault(name, {
                "records": 0, "token_proxy_units": 0,
            })
            bucket["records"] += 1
            bucket["token_proxy_units"] += units
    _need(bool(selected), "mixture has no admissible survivors")
    core = {
        "schema_version": SCHEMA,
        "policy_schema_version": POLICY_SCHEMA,
        "policy_revision": policy["revision"],
        "policy_sha256": _sha(_canonical(policy)),
        "upstream_s8_manifest_sha256": s8_receipt["manifest_sha256"],
        "input_s8_survivor_count": len(selected_pool),
        "selected_record_count": len(selected),
        "selected_record_ids": sorted(selected),
        "excluded": sorted(audit, key=lambda r: (r["record_id"], r["reason"])),
        "excluded_reason_counts": dict(sorted(Counter(
            row["reason"] for row in audit
        ).items())),
        "contributions": totals,
        "token_measure": "DETERMINISTIC_UNICODE_TOKEN_PROXY_NOT_MODEL_TOKENS",
        "actual_tokenizer_token_count": None,
        "training_corpus_authorized": False,
        "tokenizer_fit_authorized": False,
        "real_final_test_material_accessed": False,
        "paid_compute_used": False,
    }
    return {**core, "dataset_candidate_sha256": _sha(_canonical(core))}


def inspect_mixture(
    cohorts: Sequence[tuple[Mapping[str, Any], bytes, Mapping[str, Any]]],
    reserved: bytes,
    policy_raw: bytes,
) -> dict[str, Any]:
    _need(_git_blob(policy_raw) == POLICY_GIT_BLOB,
          "unapproved mixture policy Git blob")
    policy = _parse_policy(policy_raw)
    try:
        s8 = firewall.inspect_firewall(cohorts, reserved)
        _s7, rows = firewall._retained_rows(cohorts)
        result = compose_mixture(s8, rows, policy)
    except (KeyError, TypeError, ValueError) as exc:
        raise Plan2MixtureError("verified S8 mixture input invalid") from exc
    return result


def stage_mixture(root: Path, destination: Path) -> dict[str, Any]:
    _need(not any(x.is_symlink() for x in (destination, *destination.parents)),
          "symlink mixture destination")
    try:
        policy = _read_source(root, POLICY_PATH)
        reserved = _read_source(root, firewall.RESERVE_PATH)
        source_seed = _json(_read_source(
            root, "configs/data/plan2_source_inventory_v1.json"))
        seed_rows = source_seed.get("sources")
        _need(type(seed_rows) is list and bool(seed_rows),
              "S1 source registry unavailable for family binding")
        source_families = {
            row["source_id"]: row["source_family"] for row in seed_rows
        }
        _need(len(source_families) == len(seed_rows) and all(
            row["source_family"] == source_families.get(row["source_id"])
            for row in _parse_policy(policy)["sources"]
        ), "mixture source family disagrees with canonical S1 registry")
        s8 = firewall.stage_firewall(root, destination / "firewall")
        prefix = destination / "firewall" / "near" / "exact" / "privacy"
        normalized = _json(_read_destination(
            prefix / "normalization" / "normalization-manifest.json"))
        payload = _read_destination(
            prefix / "normalization" / "cohort" / "normalized.utf8")
        privacy_receipt = _json(_read_destination(
            prefix / "privacy-manifest.json"))
        result = inspect_mixture(
            ((normalized, payload, privacy_receipt),), reserved, policy)
        _need(result["input_s8_survivor_count"] ==
              s8["decontaminated_record_count"],
              "physical S8 cohort does not match mixture")
    except (OSError, KeyError, ValueError, TypeError) as exc:
        raise Plan2MixtureError("mixture staging denied") from exc
    core = {k: v for k, v in result.items() if k != "dataset_candidate_sha256"}
    core["physical_s8_manifest_sha256"] = s8["manifest_sha256"]
    manifest = {**core, "dataset_candidate_sha256": _sha(_canonical(core))}
    target = destination / "corpus-mixture-manifest.json"
    blob = _canonical(manifest)
    if target.exists() or target.is_symlink():
        _need(_read_destination(target) == blob,
              "immutable mixture manifest drift")
    else:
        _atomic_write(destination, target, blob)
    _need(_read_destination(target) == blob, "mixture readback drift")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan-2 S9 LOCAL_FREE mixture")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    result = stage_mixture(args.root, args.out_dir)
    print(json.dumps({
        "status": "PASS_MIXTURE_CANDIDATE_ONLY",
        "dataset_candidate_sha256": result["dataset_candidate_sha256"],
        "selected_record_count": result["selected_record_count"],
        "training_corpus_authorized": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
