from __future__ import annotations

import ast
import json
from itertools import pairwise
from pathlib import Path

from twelve_six.data.quality_granularity import (
    FROZEN_POLICY_SHA256,
    apply_frozen_granularity,
    compare_three_granularities,
    fixed_diagnostic_packs,
    frozen_granularity_policy,
    natural_quality_windows,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs" / "data" / "next100_067_quality_granularity_v1.json"


def _alpha_suffix(index: int, alphabet: str) -> str:
    base = len(alphabet)
    value = index
    chars: list[str] = []
    while True:
        chars.append(alphabet[value % base])
        value = value // base - 1
        if value < 0:
            return "".join(reversed(chars))


def _uk_legal_fixture() -> str:
    repeated = "стаття стаття стаття.\n" * 500
    alphabet = "абвгґдеєжзиіїйклмнопрстуфхцчшщьюя"
    valid = " ".join(
        "норма" + _alpha_suffix(index, alphabet) for index in range(3000)
    )
    return repeated + valid + ".\n"


def _en_documentation_fixture() -> str:
    repeated = ".. note:: documentation documentation documentation\n" * 350
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    valid = " ".join(
        "section" + _alpha_suffix(index, alphabet) for index in range(3000)
    )
    return repeated + valid + ".\n"


def test_frozen_manifest_matches_preregistration_and_keeps_thresholds() -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    policy = frozen_granularity_policy()
    assert policy.manifest()["policy_sha256"] == FROZEN_POLICY_SHA256
    assert policy.manifest() == plan["frozen_policy"]
    assert plan["authority_boundary"]["thresholds_changed_by_next100_067"] is False
    assert plan["model_results_may_be_read"] is False
    assert plan["final_test_outcomes_may_be_read"] is False
    assert plan["threshold_tuning_from_outcomes"] == "FORBIDDEN"


def test_natural_window_partition_is_exact_nonoverlapping_and_bounded() -> None:
    text = ("alpha beta gamma delta\n" * 3000) + "tail"
    policy = frozen_granularity_policy()
    windows = natural_quality_windows(text, policy=policy)
    assert len(windows) > 1
    assert "".join(window.text for window in windows) == text
    assert windows[0].start_char == 0
    assert windows[-1].end_char == len(text)
    assert all(left.end_char == right.start_char for left, right in pairwise(windows))
    assert all(window.chars <= policy.natural_window_max_chars for window in windows)


def test_ua_legal_text_localizes_rejection_instead_of_evicting_valid_remainder() -> None:
    text = _uk_legal_fixture()
    comparison = compare_three_granularities(
        [{"id": "ua-legal-large", "mode": "uk", "text": text}]
    )
    frozen = comparison["frozen_policy"]["items"][0]
    assert comparison["whole_source"]["accepted"] is False
    assert frozen["authoritative_unit"] == "BOUNDED_NATURAL_LANGUAGE_WINDOW"
    assert frozen["status"] == "RETAIN_PARTIAL"
    assert frozen["retained_utf8_bytes"] > 0
    assert frozen["rejected_utf8_bytes"] > 0
    assert frozen["family_eviction_authority"] is False
    assert comparison["predeclared_metrics"]["family_soft_quality_eviction_count"] == 0


def test_english_documentation_localizes_rejection_and_retains_valid_sections() -> None:
    text = _en_documentation_fixture()
    comparison = compare_three_granularities(
        [{"id": "en-docs-large", "mode": "en", "text": text}]
    )
    frozen = comparison["frozen_policy"]["items"][0]
    assert comparison["whole_source"]["accepted"] is False
    assert frozen["authoritative_unit"] == "BOUNDED_NATURAL_LANGUAGE_WINDOW"
    assert frozen["status"] == "RETAIN_PARTIAL"
    assert frozen["retained_utf8_bytes"] > 0
    assert frozen["rejected_utf8_bytes"] > 0
    assert frozen["family_eviction_authority"] is False


def test_real_code_uses_source_native_file_not_arbitrary_pack_for_rejection() -> None:
    path = ROOT / "src" / "twelve_six" / "data" / "document_quality.py"
    code = path.read_text(encoding="utf-8")
    ast.parse(code)
    result = apply_frozen_granularity("document_quality.py", code, "code")
    assert result["authoritative_unit"] == "DOCUMENT"
    assert result["windows"] == []
    assert result["family_eviction_authority"] is False

    packs = fixed_diagnostic_packs(code, 1024)
    assert len(packs) > 1
    parse_failures = 0
    for pack in packs:
        try:
            ast.parse(pack)
        except SyntaxError:
            parse_failures += 1
    assert parse_failures > 0


def test_code_pack_comparison_is_explicitly_non_authoritative() -> None:
    path = ROOT / "src" / "twelve_six" / "data" / "document_quality.py"
    code = path.read_text(encoding="utf-8")
    comparison = compare_three_granularities(
        [{"id": "real-project-code", "mode": "code", "text": code}],
        diagnostic_pack_chars=1024,
    )
    assert comparison["bounded_packs"]["authoritative"] is False
    assert comparison["frozen_policy"]["items"][0]["authoritative_unit"] == "DOCUMENT"
    assert comparison["predeclared_metrics"]["code_authoritative_parse_preservation"] == "SOURCE_NATIVE_UNIT_ONLY"
    assert comparison["model_results_read"] is False
    assert comparison["final_test_outcomes_read"] is False


def main() -> int:
    checks = (
        test_frozen_manifest_matches_preregistration_and_keeps_thresholds,
        test_natural_window_partition_is_exact_nonoverlapping_and_bounded,
        test_ua_legal_text_localizes_rejection_instead_of_evicting_valid_remainder,
        test_english_documentation_localizes_rejection_and_retains_valid_sections,
        test_real_code_uses_source_native_file_not_arbitrary_pack_for_rejection,
        test_code_pack_comparison_is_explicitly_non_authoritative,
    )
    for check in checks:
        check()
        print(f"NEXT100_067_PASS={check.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())