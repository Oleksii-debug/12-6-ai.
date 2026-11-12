"""The S16 multimodal replay cannot reuse changed code authority or grant release."""
from pathlib import Path

import pytest

from tools import plan2_s16_multimodal_candidate_v1 as pilot
from tools.plan2_s16_independent_eval_binding_v1 import IndependentEvalDenied

ROOT = Path(__file__).resolve().parents[1]


def test_code_report_needs_independently_verified_terminal_identity(tmp_path, monkeypatch):
    report = tmp_path / "report.json"
    report.write_bytes(b'{"report_identity_sha256":"changed"}')
    monkeypatch.setattr(pilot.code, "validate_report", lambda *_: None)

    def refuse_git_read(*_):
        pytest.fail("code bytes read before checking independent terminal evidence")

    monkeypatch.setattr(pilot, "_git_blob", refuse_git_read)
    with pytest.raises(IndependentEvalDenied, match="independently verified terminal result"):
        pilot._code(ROOT, report, tmp_path)


def test_balance_preview_has_zero_feasibility_without_independent_code_families():
    summary = pilot._balance_preview(ROOT, {
        "surviving_utf8_bytes_by_family_and_modality": {
            "uk1": {"uk": 20_000_000}, "uk2": {"uk": 20_000_000},
            "en1": {"en": 20_000_000}, "en2": {"en": 20_000_000},
            "code1": {"code": 20_000_000},
        },
    })
    assert summary["maximum_feasible_utf8_bytes_before_whole_document_selection"] == 0
    assert not summary["is_plan2_production_balance_authority"]
