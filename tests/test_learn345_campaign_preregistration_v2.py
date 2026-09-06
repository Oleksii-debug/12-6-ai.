from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from tools.validate_learn345_campaign_preregistration_v2 import (
    canonical_without_identity,
    validate,
)

EVIDENCE = Path("evidence/learn345/20m_campaign_preregistration_v2.json")


def _load() -> dict[str, object]:
    return json.loads(EVIDENCE.read_text(encoding="utf-8"))


def _write(tmp_path: Path, data: dict[str, object]) -> Path:
    data = copy.deepcopy(data)
    data["evidence_identity_sha256"] = hashlib.sha256(
        canonical_without_identity(data)
    ).hexdigest()
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return path


def test_current_v2_preregistration_validates() -> None:
    validate(EVIDENCE)


def test_rejects_fabricated_positive_exposure(tmp_path: Path) -> None:
    data = _load()
    data["observed_authorities"]["d04_packed_exposure"]["real_postpack_unique_loss_positions"] = 1
    with pytest.raises(SystemExit, match="current real post-pack exposure must remain zero"):
        validate(_write(tmp_path, data))


def test_rejects_optimizer_mechanics_as_lr_selection_authority(tmp_path: Path) -> None:
    data = _load()
    data["recipe_contract"]["learning_rate_selection_authority"] = "TRAIN-344B"
    with pytest.raises(SystemExit, match="synthetic mechanics may not self-select"):
        validate(_write(tmp_path, data))


def test_requires_next_exposure_resume_binding(tmp_path: Path) -> None:
    data = _load()
    data["checkpoint_and_resume"]["mandatory_state"].remove("next_exposure_identity")
    with pytest.raises(SystemExit, match="resume must bind exact next exposure"):
        validate(_write(tmp_path, data))


def test_final_test_cannot_select_checkpoint(tmp_path: Path) -> None:
    data = _load()
    data["evaluation"]["final_test_may_influence_selection"] = True
    with pytest.raises(SystemExit, match="final test may not influence selection"):
        validate(_write(tmp_path, data))


def test_replay_cannot_manufacture_capacity(tmp_path: Path) -> None:
    data = _load()
    data["campaign"]["replay_allowed"] = True
    with pytest.raises(SystemExit, match="capacity firewall weakened"):
        validate(_write(tmp_path, data))
