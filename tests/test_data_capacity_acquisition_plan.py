from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.validate_data_capacity_acquisition_plan import validate

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/data/data_capacity_acquisition_plan_v1.json"


def _write_mutation(tmp_path: Path, mutate) -> Path:
    payload = json.loads(PLAN.read_text(encoding="utf-8"))
    mutate(payload)
    path = tmp_path / "mutated.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def test_plan_validates() -> None:
    result = validate(PLAN)
    assert result["status"] == "PASS"
    assert result["conservative_training_bytes_total"] == 186_199
    assert result["conservative_20m_source_byte_gap"] == 19_813_801
    assert result["long_training_authorized"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p["truth_boundary"].__setitem__("long_training_authorized", True),
        lambda p: p["observed_training_source_capacity"].__setitem__(
            "authorized_unique_loss_positions", 1
        ),
        lambda p: p["authority_vector"]["data301_terminal_build"].__setitem__(
            "corpus_identity", "fabricated"
        ),
        lambda p: p["mixture_policy"].__setitem__("replay_allowed", True),
        lambda p: p["primary_20m_acquisition_gap"]["conservative_gap_by_stratum"].__setitem__(
            "uk", 0
        ),
    ],
)
def test_mutations_fail_closed(tmp_path: Path, mutation) -> None:
    path = _write_mutation(tmp_path, mutation)
    with pytest.raises(SystemExit):
        validate(path)
