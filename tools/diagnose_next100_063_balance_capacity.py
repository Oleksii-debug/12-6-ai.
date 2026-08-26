#!/usr/bin/env python3
"""Diagnose family-capped 45/35/20 capacity of NEXT100-063 V2.

This tool is deliberately diagnostic only. It never promotes source bytes to
optimized loss positions and never authorizes training. It binds the exact
NEXT100-063 V2 terminal-source-registry identity and the frozen DATA-295 balance
policy, then computes the largest exact-mixture source-byte budget that can be
formed without replay while respecting family concentration caps.
"""
from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

REGISTRY_PATH = Path("configs/data/next100_063_terminal_source_registry_v2.json")
EXPECTED_REGISTRY_IDENTITY = "934933896a4b3b01dd58cd18d13bcc36245913f83412c6b3f697c64dd03e4d4d"
TARGET_BYTES = 20_000_000

# Frozen DATA-295 policy: 45% Ukrainian / 35% English / 20% code.
MIXTURE = {
    "uk": Fraction(9, 20),
    "en": Fraction(7, 20),
    "code": Fraction(1, 5),
}
GLOBAL_FAMILY_CAP = Fraction(1, 4)
WITHIN_STRATUM_FAMILY_CAP = Fraction(3, 5)

# Exact DATA-287 family capacities underlying the aggregate base vector in
# NEXT100-063. Keeping these explicit lets the diagnostic enforce concentration
# rather than incorrectly treating each stratum aggregate as one fungible pool.
BASE_FAMILY_BYTES = {
    "uk": {
        "ua.rada.open-data.laws-texts": 88_565,
    },
    "en": {
        "en.standardebooks.manual": 84_793,
    },
    "code": {
        "github:encode/httpx": 8_161,
        "github:psf/requests": 1_542,
    },
}


class CapacityDiagnosticError(ValueError):
    """Raised when the bound source registry no longer matches this diagnostic."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CapacityDiagnosticError(message)


def stratum_for(row: dict[str, Any]) -> str:
    if row["modality"] == "code":
        return "code"
    language = row.get("language")
    require(language in {"uk", "en"}, f"unsupported text language: {language!r}")
    return language


def load_family_capacities(registry: dict[str, Any]) -> dict[str, dict[str, int]]:
    require(
        registry["registry_identity_sha256"] == EXPECTED_REGISTRY_IDENTITY,
        "NEXT100-063 V2 registry identity drifted; refresh this diagnostic first",
    )

    families = {stratum: dict(values) for stratum, values in BASE_FAMILY_BYTES.items()}

    base = registry["base_registry"]["by_stratum"]
    for stratum in ("uk", "en", "code"):
        require(
            sum(families[stratum].values()) == base[stratum]["normalized_bytes"],
            f"DATA-287 {stratum} aggregate no longer matches bound family capacities",
        )

    for row in registry["terminal_additions"]:
        require(
            row.get("dedicated_workflow_conclusion") == "success",
            f"source lacks successful dedicated exact-head workflow: PR {row['pr']}",
        )
        require(row["training"].startswith("ALLOWED"), f"non-training source counted: PR {row['pr']}")
        require(row["verdict"].startswith("ADMIT"), f"non-terminal source counted: PR {row['pr']}")
        stratum = stratum_for(row)
        family = row["family"]
        require(family not in families[stratum], f"duplicate family credit: {family}")
        normalized_bytes = int(row["normalized_bytes"])
        require(normalized_bytes > 0, f"non-positive family capacity: {family}")
        families[stratum][family] = normalized_bytes

    inventory = registry["pre_global_dedup_inventory"]["by_stratum"]
    for stratum in ("uk", "en", "code"):
        require(
            sum(families[stratum].values()) == inventory[stratum]["normalized_bytes"],
            f"{stratum} byte arithmetic mismatch",
        )
        require(
            len(families[stratum]) == inventory[stratum]["family_count"],
            f"{stratum} family-count mismatch",
        )

    return families


def family_cap(total_bytes: int, stratum: str) -> Fraction:
    stratum_bytes = MIXTURE[stratum] * total_bytes
    return min(GLOBAL_FAMILY_CAP * total_bytes, WITHIN_STRATUM_FAMILY_CAP * stratum_bytes)


def exact_mixture(total_bytes: int) -> bool:
    return all((share * total_bytes).denominator == 1 for share in MIXTURE.values())


def feasible(total_bytes: int, families: dict[str, dict[str, int]]) -> bool:
    if not exact_mixture(total_bytes):
        return False
    for stratum, share in MIXTURE.items():
        required = share * total_bytes
        cap = family_cap(total_bytes, stratum)
        available = sum(min(Fraction(value), cap) for value in families[stratum].values())
        if available < required:
            return False
    return True


def max_exact_mixture_bytes(families: dict[str, dict[str, int]], target: int = TARGET_BYTES) -> int:
    # All frozen shares have denominator 20, so exact mixture totals are multiples
    # of 20 bytes. For every stratum, capped-available/T is non-increasing as T
    # grows, so feasibility is monotone and binary search is exact here.
    unit = 20
    lo = 0
    hi = target // unit
    while lo < hi:
        mid = (lo + hi + 1) // 2
        total = mid * unit
        if feasible(total, families):
            lo = mid
        else:
            hi = mid - 1
    return lo * unit


def build_report(registry: dict[str, Any]) -> dict[str, Any]:
    families = load_family_capacities(registry)
    feasible_total = max_exact_mixture_bytes(families)

    strata: dict[str, Any] = {}
    for stratum, share in MIXTURE.items():
        required = share * feasible_total
        cap = family_cap(feasible_total, stratum)
        capped = {
            family: min(Fraction(value), cap)
            for family, value in sorted(families[stratum].items())
        }
        raw_total = sum(families[stratum].values())
        target_stratum = int(share * TARGET_BYTES)
        strata[stratum] = {
            "family_count": len(families[stratum]),
            "raw_pre_global_dedup_bytes": raw_total,
            "20m_policy_target_bytes": target_stratum,
            "20m_raw_capacity_gap_bytes": max(target_stratum - raw_total, 0),
            "feasible_exact_mixture_required_bytes": int(required),
            "per_family_cap_at_feasible_total": float(cap),
            "capped_available_bytes_at_feasible_total": float(sum(capped.values())),
            "family_capacity_bytes": families[stratum],
        }

    limiting = []
    next_total = feasible_total + 20
    for stratum, share in MIXTURE.items():
        required = share * next_total
        cap = family_cap(next_total, stratum)
        available = sum(min(Fraction(v), cap) for v in families[stratum].values())
        if available < required:
            limiting.append(stratum)

    return {
        "schema_version": "12-6.next100-063-balance-capacity-diagnostic.v2",
        "source_registry_identity_sha256": EXPECTED_REGISTRY_IDENTITY,
        "policy": {
            "mixture": {key: float(value) for key, value in MIXTURE.items()},
            "max_global_family_share": float(GLOBAL_FAMILY_CAP),
            "max_within_stratum_family_share": float(WITHIN_STRATUM_FAMILY_CAP),
            "replay_allowed": False,
        },
        "raw_pre_global_dedup_bytes": registry["pre_global_dedup_inventory"]["candidate_normalized_bytes"],
        "diagnostic_exact_mixture_family_capped_source_bytes": feasible_total,
        "next_20_byte_increment_limiting_strata": limiting,
        "strata": strata,
        "truth_boundary": {
            "diagnostic_only": True,
            "global_dedup_completed": False,
            "evaluation_decontamination_completed": False,
            "post_pack_unique_loss_positions": 0,
            "training_authorized": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    report = build_report(registry)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(
            "NEXT100_063_FAMILY_CAPPED_EXACT_MIXTURE_BYTES="
            + str(report["diagnostic_exact_mixture_family_capped_source_bytes"])
        )
        print("NEXT100_063_LIMITING_STRATA=" + ",".join(report["next_20_byte_increment_limiting_strata"]))
        for stratum in ("uk", "en", "code"):
            row = report["strata"][stratum]
            print(f"NEXT100_063_{stratum.upper()}_20M_RAW_GAP={row['20m_raw_capacity_gap_bytes']}")
        print("NEXT100_063_TRAINING_AUTHORIZED=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
