"""Project V5 terminal attrs authority through the frozen 45/35/20 balance policy."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
V4_REGISTRY = ROOT / "configs/data/next100_063_terminal_source_registry_v4.json"
V5_REGISTRY = ROOT / "configs/data/next100_063_terminal_source_registry_v5.json"
BASE_TOOL = ROOT / "tools/diagnose_next100_063_balance_capacity.py"
V5_VALIDATOR = ROOT / "tools/validate_next100_063_terminal_source_registry_v5.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_module("next100_063_balance_v4", BASE_TOOL)
v5_validator = _load_module("next100_063_registry_v5", V5_VALIDATOR)


def build_report(v4: dict[str, Any], v5: dict[str, Any], *, v5_blob_sha1: str) -> dict[str, Any]:
    v5_validator.validate(v4, v5, v5_blob_sha1=v5_blob_sha1)
    families = base.load_family_capacities(v4)
    attrs = v5["terminal_addition"]
    family = str(attrs["family"])
    if family in families["code"]:
        raise v5_validator.RegistryV5Error("attrs family already present in V4 balance inventory")
    families["code"][family] = int(attrs["numeric_training_capacity_bytes"])

    feasible_total = base.max_exact_mixture_bytes(families)
    target = int(v5["derived_pre_successor_global_dedup_inventory"]["research_corpus_v1_acquisition_planning_target_bytes"])
    gaps: dict[str, int] = {}
    raw: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    for stratum, share in base.MIXTURE.items():
        raw[stratum] = sum(families[stratum].values())
        family_counts[stratum] = len(families[stratum])
        target_stratum = int(share * target)
        gaps[stratum] = max(target_stratum - raw[stratum], 0)

    next_total = feasible_total + 20
    limiting: list[str] = []
    for stratum, share in base.MIXTURE.items():
        required = share * next_total
        cap = base.family_cap(next_total, stratum)
        available = sum(min(base.Fraction(value), cap) for value in families[stratum].values())
        if available < required:
            limiting.append(stratum)

    return {
        "schema_version": "12-6.next100-063-balance-capacity-diagnostic.v5",
        "source_registry": {
            "base_v4_identity_sha256": v5_validator.EXPECTED_V4_IDENTITY,
            "v5_config_blob_sha1": v5_validator.EXPECTED_V5_BLOB_SHA1,
            "terminal_attrs_authority_identity": v5_validator.EXPECTED_ATTRS_AUTHORITY,
        },
        "policy": {
            "mixture": {key: float(value) for key, value in base.MIXTURE.items()},
            "max_global_family_share": float(base.GLOBAL_FAMILY_CAP),
            "max_within_stratum_family_share": float(base.WITHIN_STRATUM_FAMILY_CAP),
            "replay_allowed": False,
        },
        "raw_pre_successor_global_dedup_numeric_training_capacity_bytes": sum(raw.values()),
        "raw_capacity_by_stratum": raw,
        "family_count_by_stratum": family_counts,
        "20m_raw_capacity_gap_by_stratum": gaps,
        "diagnostic_exact_mixture_family_capped_source_bytes": feasible_total,
        "next_20_byte_increment_limiting_strata": limiting,
        "truth_boundary": {
            "diagnostic_only": True,
            "successor_global_dedup_completed": False,
            "evaluation_decontamination_completed": False,
            "post_pack_unique_loss_positions": 0,
            "training_authorized": False,
            "paid_compute_authorized": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    v4 = json.loads(V4_REGISTRY.read_text(encoding="utf-8"))
    raw_v5 = V5_REGISTRY.read_bytes()
    v5 = json.loads(raw_v5.decode("utf-8"))
    report = build_report(v4, v5, v5_blob_sha1=v5_validator.git_blob_sha1(raw_v5))
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(
            "NEXT100_063_V5_FAMILY_CAPPED_EXACT_MIXTURE_BYTES="
            + str(report["diagnostic_exact_mixture_family_capped_source_bytes"])
        )
        print("NEXT100_063_V5_LIMITING_STRATA=" + ",".join(report["next_20_byte_increment_limiting_strata"]))
        for stratum in ("uk", "en", "code"):
            print(
                f"NEXT100_063_V5_{stratum.upper()}_20M_RAW_GAP="
                f"{report['20m_raw_capacity_gap_by_stratum'][stratum]}"
            )
        print("NEXT100_063_V5_TRAINING_AUTHORIZED=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
