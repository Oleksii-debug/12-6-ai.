from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "run_data296_quality_external_audit.py"
SPEC = importlib.util.spec_from_file_location("data296_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _plan() -> dict:
    return json.loads(
        (ROOT / "configs" / "data" / "data296_quality_filter_external_audit_v1.json").read_text(
            encoding="utf-8"
        )
    )


def test_preregistration_and_incumbent_blob_identities_are_locked() -> None:
    plan = _plan()
    MODULE.verify_preregistration(plan, ROOT)


def test_all_policy_identities_match_preregistration() -> None:
    plan = _plan()
    observed = {
        raw["role"]: MODULE.policy_from_plan(raw).manifest()["policy_sha256"]
        for raw in plan["policies"]
    }
    expected = {raw["role"]: raw["expected_policy_sha256"] for raw in plan["policies"]}
    assert observed == expected
    assert set(observed) == {
        "INCUMBENT",
        "PERMISSIVE_PREREGISTERED",
        "STRICT_PREREGISTERED",
    }


def test_alternatives_change_only_preregistered_narrow_thresholds() -> None:
    plan = _plan()
    policies = {raw["role"]: raw for raw in plan["policies"]}
    incumbent = policies["INCUMBENT"]
    for role in ("PERMISSIVE_PREREGISTERED", "STRICT_PREREGISTERED"):
        alternative = policies[role]
        for mode in ("uk", "en"):
            changed = {
                key
                for key, value in alternative[mode].items()
                if value != incumbent[mode][key]
            }
            assert changed == {"max_symbol_ratio", "min_distinct_token_ratio"}
        changed_code = {
            key
            for key, value in alternative["code"].items()
            if value != incumbent["code"][key]
        }
        assert changed_code == {"max_symbol_ratio", "min_code_structure_score"}


def test_line_pack_partition_is_exact_and_nonoverlapping() -> None:
    text = "alpha\n" + ("beta gamma\n" * 20) + "omega"
    packs = MODULE.line_packs(text, 37)
    assert "".join(packs) == text
    assert len(packs) > 1
    assert sum(len(pack.encode("utf-8")) for pack in packs) == len(text.encode("utf-8"))


def test_diagnostic_categories_are_interpretable() -> None:
    uk = "Україна і її мова та дані. Це український текст."
    rst = ".. note:: Documentation\n\n:class: example\n\n``literal``\n"
    code = "def f(x):\n    return x + 1\n"
    assert MODULE.is_ukrainian_dominant(uk)
    assert MODULE.is_rst_syntax_bearing(rst)
    assert MODULE.is_parse_valid_python(code)
    assert not MODULE.is_parse_valid_python("def broken(:\n")
