from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest

from twelve_six.data.wikisource_pd_contract import (
    WikisourceIntakeError,
    validate_control_contract,
)


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools/materialize_d03_wikisource_lesia1892.py"
CANONICAL_CONTRACT = ROOT / "configs/data/d03_wikisource_lesia1892_current_main_v1.json"


def _load_tool() -> dict[str, object]:
    return runpy.run_path(str(TOOL_PATH))


def test_strict_control_loader_accepts_canonical_contract() -> None:
    loader = _load_tool()["load_strict_json_object"]
    assert callable(loader)
    contract = loader(CANONICAL_CONTRACT.read_text(encoding="utf-8"))
    validate_control_contract(contract)


@pytest.mark.parametrize(
    "raw",
    [
        '{"execution_class":"PAID","execution_class":"LOCAL_FREE"}',
        '{"truth_boundary":{"training_authorized_bytes":1,"training_authorized_bytes":0}}',
        '{"\\u0065xecution_class":"PAID","execution_class":"LOCAL_FREE"}',
    ],
)
def test_strict_control_loader_rejects_duplicate_members(raw: str) -> None:
    loader = _load_tool()["load_strict_json_object"]
    assert callable(loader)
    with pytest.raises(WikisourceIntakeError, match="duplicate JSON object member"):
        loader(raw)


@pytest.mark.parametrize(
    "raw",
    [
        '{"value":NaN}',
        '{"value":Infinity}',
        '{"value":-Infinity}',
        '{"value":1e400}',
        '{"value":-1e400}',
    ],
)
def test_strict_control_loader_rejects_non_finite_numbers(raw: str) -> None:
    loader = _load_tool()["load_strict_json_object"]
    assert callable(loader)
    with pytest.raises(WikisourceIntakeError, match="non-finite JSON"):
        loader(raw)


@pytest.mark.parametrize("raw", ["[]", "null", '"text"', "0"])
def test_strict_control_loader_requires_object_root(raw: str) -> None:
    loader = _load_tool()["load_strict_json_object"]
    assert callable(loader)
    with pytest.raises(WikisourceIntakeError, match="must be a JSON object"):
        loader(raw)


def test_cli_rejects_ambiguous_contract_before_materialization_or_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_tool()
    main = module["main"]
    assert callable(main)

    def fail_if_materialized(*args: object, **kwargs: object) -> None:
        pytest.fail("materialize_live must not run for an ambiguous control contract")

    main.__globals__["materialize_live"] = fail_if_materialized

    contract_path = tmp_path / "ambiguous.json"
    candidate_path = tmp_path / "candidate.jsonl"
    report_path = tmp_path / "report.json"
    contract_path.write_text(
        '{"execution_class":"PAID","execution_class":"LOCAL_FREE"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(TOOL_PATH),
            "--contract",
            str(contract_path),
            "--candidate-out",
            str(candidate_path),
            "--report-out",
            str(report_path),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 2
    assert not candidate_path.exists()
    assert not report_path.exists()
