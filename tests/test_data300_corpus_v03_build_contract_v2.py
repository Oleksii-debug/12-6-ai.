#!/usr/bin/env python3
"""Focused regression tests for DATA-300 contract v2."""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "validate_data300_corpus_v03_build_contract_v2.py"
CONTRACT_PATH = (
    ROOT / "configs" / "data" / "data300_corpus_v03_frozen_build_contract_v2.json"
)

spec = importlib.util.spec_from_file_location("data300v2", MODULE_PATH)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def load_contract():
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def rehash(contract):
    contract["contract_identity_sha256"] = mod.contract_identity(contract)
    return contract


def expect_fail(contract, needle):
    try:
        mod.validate_contract(contract)
    except mod.ContractError as exc:
        assert needle in str(exc), (needle, str(exc))
    else:
        raise AssertionError(f"expected failure containing {needle!r}")


def test_contract_passes():
    mod.validate_contract(load_contract())


def test_self_hash_tamper_fails():
    contract = load_contract()
    contract["repository"] = "wrong/repo"
    expect_fail(contract, "repository identity drift")


def test_rehashed_corpus_promotion_fails():
    contract = load_contract()
    contract["corpus_state"] = "CORPUS_FROZEN"
    rehash(contract)
    expect_fail(contract, "must not declare the corpus frozen")


def test_source_mutation_fails_even_rehashed():
    contract = load_contract()
    contract["exact_training_candidate_inventory"]["sources"].pop()
    contract["exact_training_candidate_inventory"]["source_count"] = 4
    rehash(contract)
    expect_fail(contract, "exact source inventory/order drift")


def test_red_successor_cannot_be_promoted():
    contract = load_contract()
    contract["terminal_component_lock"]["quality_policy"][
        "successor_data296_terminal"
    ] = True
    rehash(contract)
    expect_fail(contract, "red DATA-296 must not be promoted")


def test_repetition_switch_fails():
    contract = load_contract()
    contract["artificial_repetition"]["sampling_with_replacement"] = True
    rehash(contract)
    expect_fail(contract, "artificial-repetition")


def test_clean_build_count_fails():
    contract = load_contract()
    contract["build_determinism"]["clean_build_count"] = 1
    rehash(contract)
    expect_fail(contract, "exactly two clean builds")


def test_clean_tree_compare():
    with tempfile.TemporaryDirectory() as temp:
        base = Path(temp)
        a = base / "a"
        b = base / "b"
        a.mkdir()
        b.mkdir()
        (a / "x").write_bytes(b"same")
        (b / "x").write_bytes(b"same")
        mod.compare_clean_builds(a, b)
        (b / "x").write_bytes(b"different")
        try:
            mod.compare_clean_builds(a, b)
        except mod.ContractError as exc:
            assert "clean builds differ" in str(exc)
        else:
            raise AssertionError("different clean trees were accepted")


def main():
    tests = [
        test_contract_passes,
        test_self_hash_tamper_fails,
        test_rehashed_corpus_promotion_fails,
        test_source_mutation_fails_even_rehashed,
        test_red_successor_cannot_be_promoted,
        test_repetition_switch_fails,
        test_clean_build_count_fails,
        test_clean_tree_compare,
    ]
    for fn in tests:
        fn()
    print(f"DATA-300 v2 tests PASS: {len(tests)}")


if __name__ == "__main__":
    main()
