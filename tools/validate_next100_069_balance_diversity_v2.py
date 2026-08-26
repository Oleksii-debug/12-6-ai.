#!/usr/bin/env python3
"""Validate NEXT100-069 balance/diversity authority without network or model results."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUTH_PATH = ROOT / "configs/data/next100_069_balance_diversity_v2.json"
DEDUP_PATH = ROOT / "configs/data/next100_065_cross_source_dedup_v3.json"

EXPECTED_POLICY = {
    "target_total_source_bytes": 20_000_000,
    "minimum_independent_families_per_stratum": 2,
    "max_family_fraction_total": "1/4",
    "max_family_fraction_own_stratum": "3/5",
}
EXPECTED_TARGETS = {"ua": 9_000_000, "en": 7_000_000, "code": 4_000_000}
EXPECTED_LOSS_PARTIAL = {"total": 173_355, "ua_rada": 88_564, "en_standardebooks": 84_791}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_sha_without_identity(data: dict) -> str:
    body = dict(data)
    body.pop("authority_identity_sha256", None)
    payload = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def hill_q2(values: list[int]) -> Fraction:
    total = sum(values)
    if total <= 0:
        raise AssertionError("Hill q2 undefined for empty capacity")
    return Fraction(total * total, sum(value * value for value in values))


def validate(authority: dict, dedup: dict) -> None:
    assert authority["schema_version"] == "12-6.next100-069-balance-diversity-v2.v1"
    assert authority["worker_id"] == "NEXT100-069-BALANCE-DIVERSITY-V2"
    assert authority["execution_class"] == "LOCAL_FREE"
    assert authority["model_training_executed"] is False
    assert authority["model_result_guided_tuning"] is False
    assert authority["policy"]["replay_or_duplication_to_meet_quota"] is False
    assert authority["policy"]["mixture_retuning_from_model_bpb"] is False
    for key, value in EXPECTED_POLICY.items():
        assert authority["policy"][key] == value
    assert {k: v["target_bytes"] for k, v in authority["policy"]["strata"].items()} == EXPECTED_TARGETS

    assert dedup["schema_version"] == "12-6.next100-065-cross-source-dedup.v3"
    assert dedup["worker_id"] == "NEXT100-065-CROSSSOURCE-DEDUP-V3"
    assert dedup["local_free_only"] is True
    assert dedup["model_training_executed"] is False
    assert authority["source_vector"]["head_sha"] == "90065ffc97a5133e76cacdee6991eb171c4ea2ba"
    assert authority["source_vector"]["config_blob_sha"] == "c1e05f09490e25f6fed765dfb70d900717528f4d"
    assert authority["source_vector"]["terminal_refresh_cutoff_utc"] == dedup["terminal_refresh_cutoff_utc"]

    by_stratum = defaultdict(int)
    by_family = defaultdict(int)
    family_stratum: dict[str, str] = {}
    modality_map = {"uk": "ua", "en": "en", "code": "code"}
    for source in dedup["sources"]:
        stratum = modality_map[source["modality"]]
        family = source["source_family"]
        value = int(source["declared_capacity_bytes"])
        by_stratum[stratum] += value
        by_family[family] += value
        previous = family_stratum.setdefault(family, stratum)
        assert previous == stratum, f"family crosses strata: {family}"

    expected_by_stratum = authority["available_unique_source_bytes"]["by_stratum"]
    assert dict(by_stratum) == expected_by_stratum
    assert sum(by_stratum.values()) == authority["available_unique_source_bytes"]["total"]

    authority_families = {entry["family_id"]: entry for entry in authority["families"]}
    assert set(authority_families) == set(by_family)
    for family, value in by_family.items():
        assert authority_families[family]["bytes"] == value
        assert authority_families[family]["stratum"] == family_stratum[family]

    family_counts = {
        stratum: sum(1 for family in by_family if family_stratum[family] == stratum)
        for stratum in ("ua", "en", "code")
    }
    assert family_counts == {k: authority["family_count"][k] for k in ("ua", "en", "code")}
    assert sum(family_counts.values()) == authority["family_count"]["global"]

    for stratum in ("ua", "en", "code"):
        values = [by_family[f] for f in by_family if family_stratum[f] == stratum]
        observed = hill_q2(values)
        expected = Fraction(authority["effective_family_count_hill_q2"][stratum]["exact_fraction"])
        assert observed == expected
    observed_global = hill_q2(list(by_family.values()))
    assert observed_global == Fraction(authority["effective_family_count_hill_q2"]["global"]["exact_fraction"])

    total = authority["available_unique_source_bytes"]["total"]
    global_violations = sorted(
        family for family, value in by_family.items() if Fraction(value, total) > Fraction(1, 4)
    )
    own_violations = sorted(
        family
        for family, value in by_family.items()
        if Fraction(value, by_stratum[family_stratum[family]]) > Fraction(3, 5)
    )
    pool_gate = authority["family_gate"]["whole_available_pool_if_consumed_without_subsampling"]
    assert global_violations == sorted(pool_gate["global_25pct_violations"])
    assert own_violations == sorted(pool_gate["own_stratum_60pct_violations"])

    min_families = authority["policy"]["minimum_independent_families_per_stratum"]
    assert family_counts["ua"] >= min_families
    assert family_counts["en"] < min_families
    assert family_counts["code"] >= min_families
    assert authority["family_gate"]["fixed_mixture_nonreplay_feasible_source_bytes"] == 0
    assert authority["family_gate"]["fixed_mixture_status"] == "FAIL_EN_HAS_ONLY_ONE_TERMINAL_DEDUP_CERTIFIED_FAMILY"

    gaps = {key: EXPECTED_TARGETS[key] - by_stratum[key] for key in EXPECTED_TARGETS}
    assert gaps == authority["acquisition_gaps_to_20m_source_bytes"]["by_stratum"]
    assert sum(gaps.values()) == authority["acquisition_gaps_to_20m_source_bytes"]["total"]
    assert authority["acquisition_gaps_to_20m_source_bytes"]["target_family_byte_caps"] == {
        "ua": 5_000_000,
        "en": 4_200_000,
        "code": 2_400_000,
    }

    loss = authority["unique_loss_positions"]
    assert loss["full_current_vector_exact_total"] is None
    assert loss["certified_partial"] == EXPECTED_LOSS_PARTIAL
    assert loss["source_bytes_covered_by_legacy_loss_ledger"] == 173_358
    assert loss["source_bytes_not_yet_loss_ledgered"] == total - 173_358
    assert loss["no_source_bytes_relabelled_as_loss_positions"] is True

    assert authority["verdict"] == "FAIL_RETAIN_45_35_20_POLICY_ACQUIRE_MORE_DATA_NO_BPB_RETUNING"
    assert canonical_sha_without_identity(authority) == authority["authority_identity_sha256"]


def main() -> None:
    authority = load_json(AUTH_PATH)
    dedup = load_json(DEDUP_PATH)
    validate(authority, dedup)
    print("NEXT100-069 balance/diversity V2 authority: PASS")
    print(f"verdict={authority['verdict']}")
    print(f"available_unique_source_bytes={authority['available_unique_source_bytes']['total']}")
    print(f"certified_partial_unique_loss_positions={authority['unique_loss_positions']['certified_partial']['total']}")


if __name__ == "__main__":
    main()
