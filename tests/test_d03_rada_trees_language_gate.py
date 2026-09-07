from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "derive_d03_rada_trees_language_gate.py"
EVIDENCE = (
    ROOT
    / "evidence"
    / "d03-rada-trees"
    / "secondary-plaintext-language-gate-v1.json"
)

spec = importlib.util.spec_from_file_location("rada_language_gate", MODULE_PATH)
assert spec is not None and spec.loader is not None
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def policy() -> dict[str, float | int]:
    return {
        "minimum_cyrillic_letter_fraction": 0.90,
        "minimum_ukrainian_specific_letter_count": 1,
        "minimum_letters": 100,
        "maximum_tab_fraction": 0.10,
    }


def test_language_metrics_accept_clear_ukrainian() -> None:
    accepted, reasons = mod.assess_metrics(
        {
            "letters": 1000,
            "cyrillic_letter_fraction": 0.99,
            "ukrainian_specific_letter_count": 40,
            "tab_fraction": 0.0,
        },
        policy(),
    )
    assert accepted is True
    assert reasons == ()


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("cyrillic_letter_fraction", 0.89, "cyrillic_fraction"),
        ("ukrainian_specific_letter_count", 0, "ukrainian_specific_letters"),
        ("letters", 99, "letter_count"),
        ("tab_fraction", 0.11, "tab_fraction"),
    ],
)
def test_language_metrics_fail_closed_at_each_boundary(
    field: str, value: float, reason: str
) -> None:
    metrics = {
        "letters": 1000,
        "cyrillic_letter_fraction": 0.99,
        "ukrainian_specific_letter_count": 40,
        "tab_fraction": 0.0,
    }
    metrics[field] = value
    accepted, reasons = mod.assess_metrics(metrics, policy())
    assert accepted is False
    assert reason in reasons


def test_missing_metric_fails_closed() -> None:
    with pytest.raises(mod.LanguageGateError, match="missing/invalid language metric"):
        mod.assess_metrics(
            {
                "letters": 1000,
                "cyrillic_letter_fraction": 0.99,
                "ukrainian_specific_letter_count": 40,
            },
            policy(),
        )


def test_terminal_report_is_hash_valid_and_zero_credit() -> None:
    report = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    claimed = report["report_sha256"]
    core = dict(report)
    core.pop("report_sha256")
    assert mod.canonical_sha256(core) == claimed
    assert claimed == "adb89b227d2b0631ca7b35ba59ff744d2efa74ec9c9f6f2f8334738213074291"
    assert report["language_result"]["language_pass_members"] == 4384
    assert report["language_result"]["language_pass_bytes"] == 877_899_128
    assert report["language_result"]["language_reject_members"] == 0
    assert report["decision"]["raw_text_emitted"] is False
    assert report["decision"]["quality_complete"] is False
    assert report["decision"]["privacy_complete"] is False
    assert report["claim_boundary"]["training_authorized_bytes"] == 0
    assert report["claim_boundary"]["unique_causal_loss_positions_authorized"] == 0
    assert report["claim_boundary"]["model_training_executed"] is False
