#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "research313" / "20m_data_capacity_gate_v1.json"

EXPECTED = {
    "schema_version": "12-6.research313-20m-data-capacity-gate.v1",
    "worker_id": "RESEARCH-313-20M-DATA-CAPACITY-GATE",
    "execution_profile": "LOCAL_FREE",
    "verdict": "BLOCKED_NO_TERMINAL_FINAL_CORPUS_LEDGER",
    "data300_head_sha": "8ea7f830e50a23754d189dd4134f4afad76a7ee9",
    "data300_contract_identity_sha256": "07d7beaaff4616e839450de6af3d407855c832bf75a24a959d1a12de5d9364e5",
    "data301_head_sha": "8820ba1b255f6bb95c7db0531fd846078a1aae01",
    "data301_evidence_identity_sha256": "939065abeefff8aed924415589608ff3fc721fe4b0a57fc200146a4b6a137e81",
    "data294_ledger_identity_sha256": "9a1cd57c52459bdc6e4bb2d46047a47713e10d9a5be7b0a4b86f041ba6f62bd0",
    "text_unique_positions": 173355,
    "max_safe_positions_now": 0,
}


def canonical_identity(payload: dict) -> str:
    unsigned = dict(payload)
    unsigned.pop("evidence_identity_sha256", None)
    encoded = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate() -> None:
    data = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    assert data["schema_version"] == EXPECTED["schema_version"]
    assert data["worker_id"] == EXPECTED["worker_id"]
    assert data["execution_profile"] == EXPECTED["execution_profile"]
    assert data["verdict"] == EXPECTED["verdict"]

    cutoff = data["authority_cutoff"]
    assert cutoff["data300_head_sha"] == EXPECTED["data300_head_sha"]
    assert (
        cutoff["data300_contract_identity_sha256"]
        == EXPECTED["data300_contract_identity_sha256"]
    )
    assert cutoff["data301_head_sha"] == EXPECTED["data301_head_sha"]
    assert (
        cutoff["data301_evidence_identity_sha256"]
        == EXPECTED["data301_evidence_identity_sha256"]
    )
    assert cutoff["data301_status"] == "TERMINAL_BLOCKED"
    assert (
        cutoff["data294_ledger_identity_sha256"]
        == EXPECTED["data294_ledger_identity_sha256"]
    )

    actual = data["actual_unique_loss_evidence"]
    assert (
        actual["unique_nonignored_causal_loss_positions_one_pass"]
        == EXPECTED["text_unique_positions"]
    )
    assert actual["padding_positions_counted"] == 0
    assert actual["replay_permitted"] is False
    assert actual["full_five_source_ledger_exists"] is False
    assert sum(actual["by_language"].values()) == EXPECTED["text_unique_positions"]

    inventory = data["candidate_inventory"]
    assert inventory["families_by_stratum"] == {"uk": 1, "en": 1, "code": 2}
    assert inventory["family_gate_minimum_per_stratum"] == 2
    assert inventory["family_constrained_no_replay_budget_source_bytes"] == 0

    safe = data["maximum_safe_preregistered_exposure"]
    assert safe["unit"] == "unique_nonignored_causal_loss_positions"
    assert safe["positions_now"] == EXPECTED["max_safe_positions_now"]
    assert safe["before_any_replay"] is True

    learned = data["budget_classes"]["meaningful_learned_campaign"]
    assert learned["project_native_decision_envelope_unique_positions"] == {
        "lower": 10000000,
        "upper": 40000000,
        "derivation": (
            "RESEARCH-251 local 0.5-2 unique-targets/parameter decision band "
            "applied only as a project planning envelope, not as an optimum"
        ),
    }
    assert learned["external_real_final_candidate_authorized_positions_now"] == 0

    assert data["evidence_identity_sha256"] == canonical_identity(data)


if __name__ == "__main__":
    validate()
    print("RESEARCH-313 evidence: VALID")
