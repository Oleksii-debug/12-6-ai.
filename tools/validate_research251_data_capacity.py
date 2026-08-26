#!/usr/bin/env python3
"""Validate RESEARCH-251 machine-readable data-capacity evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "research251" / "data_capacity_scale_gate_20260826.json"


def canonical_without_identity(payload: dict) -> bytes:
    clone = dict(payload)
    clone.pop("evidence_identity_sha256", None)
    return json.dumps(
        clone,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def main() -> None:
    payload = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    assert payload["worker_id"] == "RESEARCH-251-DATA-CAPACITY-SCALE-GATE"
    assert payload["local_free_only"] is True
    assert payload["training_executed"] is False
    accounting = payload["unique_loss_token_accounting"]
    assert accounting["padded_tokens_count_as_data"] is False
    assert accounting["strongest_execution_verified_no_replay_unique_loss_tokens"] == 2_000_060
    assert accounting["terminal_source_token_ceiling"] == 20_000_775
    assert payload["current_data"]["external_real_training_corpus_authority"] is False
    assert payload["current_data"]["latest_external_real_registry_code_source_count"] == 0

    scales = {item["scale"]: item for item in payload["scales"]}
    assert set(scales) == {"10M", "100M", "400M", "1B"}
    assert scales["10M"]["minimum_meaningful_unique_token_range"] == [5_000_000, 20_000_000]
    assert scales["100M"]["minimum_meaningful_unique_token_range"] == [50_000_000, 200_000_000]
    assert scales["400M"]["minimum_meaningful_unique_token_range"] == [250_000_000, 1_000_000_000]
    assert scales["1B"]["minimum_meaningful_unique_token_range"] == [625_000_000, 2_500_000_000]
    assert all(item["data_limited"] is True for item in scales.values())

    source_ceiling = accounting["terminal_source_token_ceiling"]
    verified_floor = accounting["strongest_execution_verified_no_replay_unique_loss_tokens"]
    for item in scales.values():
        params = item["parameter_count"]
        assert abs(item["current_source_tokens_per_parameter"] - source_ceiling / params) < 1e-12
        assert abs(item["current_verified_loss_tokens_per_parameter"] - verified_floor / params) < 1e-12
        cap = item["maximum_recommended_exposure_before_uncontrolled_recycling"]
        assert cap["execution_verified_without_replay"] == verified_floor
        assert cap["hard_source_token_ceiling"] == source_ceiling

    expected = hashlib.sha256(canonical_without_identity(payload)).hexdigest()
    assert payload["evidence_identity_sha256"] == expected
    print("RESEARCH-251 data-capacity gate: PASS")
    print(expected)


if __name__ == "__main__":
    main()
