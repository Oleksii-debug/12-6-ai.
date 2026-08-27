from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from twelve_six import html_extraction_bakeoff as bakeoff


CONTRACT_PATH = Path("configs/research/html_extraction_bakeoff_v1.json")


def _contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _rehash_contract(contract: dict) -> None:
    contract["contract_sha256"] = bakeoff.canonical_sha256(
        bakeoff._contract_hash_payload(contract)
    )


def _gold_adapter(contract: dict, suffix: str = ""):
    def extract(html: str) -> str:
        for fixture in contract["fixtures"]:
            anchor = fixture["required_anchors"][0]
            if anchor in html:
                return fixture["gold_text"] + suffix
        raise AssertionError("fixture marker not found")

    return extract


def test_frozen_contract_is_valid() -> None:
    contract = bakeoff.load_contract(CONTRACT_PATH)
    assert contract["base_sha"] == "5020afd671a3885c1b738c8b4eafe7525f630546"
    assert {f["language"] for f in contract["fixtures"]} == {"en", "uk"}
    assert {f["payload_kind"] for f in contract["fixtures"]} == {"html", "warc_response"}


def test_fixture_tamper_fails_even_if_contract_hash_is_recomputed() -> None:
    contract = _contract()
    contract["fixtures"][0]["payload"] += " tamper"
    _rehash_contract(contract)
    with pytest.raises(bakeoff.ContractError, match="fixture hash mismatch"):
        bakeoff.validate_contract(contract)


def test_contract_hash_drift_fails_closed() -> None:
    contract = _contract()
    contract["selection_rule"]["minimum_macro_anchor_recall"] = 0.5
    with pytest.raises(bakeoff.ContractError, match="contract hash mismatch"):
        bakeoff.validate_contract(contract)


def test_malformed_warc_envelope_fails_closed() -> None:
    contract = _contract()
    fixture = copy.deepcopy(
        next(f for f in contract["fixtures"] if f["payload_kind"] == "warc_response")
    )
    fixture["payload"] = fixture["payload"].replace("WARC-Type: response", "WARC-Type: request")
    fixture["payload_sha256"] = (
        __import__("hashlib").sha256(fixture["payload"].encode()).hexdigest()
    )
    with pytest.raises(bakeoff.ContractError, match="not a response"):
        bakeoff.extract_html_payload(fixture)


def test_runtime_version_drift_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    contract = _contract()
    spec = contract["extractors"]["trafilatura"]
    monkeypatch.setattr(bakeoff.importlib.metadata, "version", lambda _: "0.0.0")
    with pytest.raises(bakeoff.RuntimeIdentityError, match="version mismatch"):
        bakeoff.resolve_runtime_extractor("trafilatura", spec)


def test_missing_runtime_is_a_retest_not_a_winner(monkeypatch: pytest.MonkeyPatch) -> None:
    contract = bakeoff.load_contract(CONTRACT_PATH)

    def missing(name: str, spec: dict):
        raise bakeoff.RuntimeIdentityError(f"missing runtime distribution: {spec['distribution']}")

    monkeypatch.setattr(bakeoff, "resolve_runtime_extractor", missing)
    report = bakeoff.run_bakeoff(contract)
    assert report["terminal_state"] == "RETEST_RUNTIME_REQUIRED"
    assert set(report["runtime_errors"]) == {"trafilatura", "resiliparse"}
    bakeoff.validate_report(report, contract)


def test_nondeterministic_extractor_forces_retest() -> None:
    contract = bakeoff.load_contract(CONTRACT_PATH)
    counter = {"n": 0}
    good = _gold_adapter(contract)

    def unstable(html: str) -> str:
        counter["n"] += 1
        return good(html) + ("" if counter["n"] % 2 else " changed")

    report = bakeoff.run_bakeoff(
        contract,
        adapters={"trafilatura": unstable, "resiliparse": good},
    )
    assert report["terminal_state"] == "RETEST_NONDETERMINISTIC"
    assert report["results"]["trafilatura"]["all_deterministic"] is False


def test_preregistered_rule_can_select_candidate_without_adoption() -> None:
    contract = bakeoff.load_contract(CONTRACT_PATH)
    gold = _gold_adapter(contract)
    noisier = _gold_adapter(contract, " extra noise tokens repeated extra noise tokens repeated")
    report = bakeoff.run_bakeoff(
        contract,
        adapters={"trafilatura": gold, "resiliparse": noisier},
    )
    assert report["terminal_state"] == "CANDIDATE_TRAFILATURA"
    assert report["production_extractor_replacement_authorized"] is False
    assert report["training_authorized_bytes"] == 0
    assert report["corpus_capacity_credited"] == 0


def test_evidence_hash_ignores_non_authoritative_timing() -> None:
    contract = bakeoff.load_contract(CONTRACT_PATH)
    gold = _gold_adapter(contract)
    adapters = {"trafilatura": gold, "resiliparse": gold}
    first = bakeoff.run_bakeoff(contract, adapters=adapters)
    second = bakeoff.run_bakeoff(contract, adapters=adapters)
    assert first["deterministic_evidence_sha256"] == second["deterministic_evidence_sha256"]
    assert first["terminal_state"] == "NO_CLEAR_WINNER"


def test_truth_boundary_rejects_adopted_and_training_credit() -> None:
    contract = bakeoff.load_contract(CONTRACT_PATH)
    gold = _gold_adapter(contract)
    report = bakeoff.run_bakeoff(
        contract,
        adapters={"trafilatura": gold, "resiliparse": gold},
    )

    adopted = copy.deepcopy(report)
    adopted["terminal_state"] = "ADOPTED"
    adopted["deterministic_evidence_sha256"] = bakeoff.canonical_sha256(
        bakeoff.deterministic_report_projection(adopted)
    )
    with pytest.raises(bakeoff.ContractError, match="forbidden or unknown terminal state"):
        bakeoff.validate_report(adopted, contract)

    credited = copy.deepcopy(report)
    credited["training_authorized_bytes"] = 1
    credited["deterministic_evidence_sha256"] = bakeoff.canonical_sha256(
        bakeoff.deterministic_report_projection(credited)
    )
    with pytest.raises(bakeoff.ContractError, match="cannot authorize training bytes"):
        bakeoff.validate_report(credited, contract)
