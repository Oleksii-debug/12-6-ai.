from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools.validate_learn345_campaign_preregistration_v2 import (
    canonical_without_identity,
    validate,
)

EVIDENCE = Path("evidence/learn345/20m_campaign_preregistration_v2.json")


def _load() -> dict[str, object]:
    return json.loads(EVIDENCE.read_text(encoding="utf-8"))


def _write(directory: Path, data: dict[str, object]) -> Path:
    data = copy.deepcopy(data)
    data["evidence_identity_sha256"] = hashlib.sha256(
        canonical_without_identity(data)
    ).hexdigest()
    path = directory / "evidence.json"
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return path


class Learn345V2Tests(unittest.TestCase):
    def _assert_rejected(self, data: dict[str, object], pattern: str) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = _write(Path(temp_dir), data)
            with self.assertRaisesRegex(SystemExit, pattern):
                validate(path)

    def test_current_v2_preregistration_validates(self) -> None:
        validate(EVIDENCE)

    def test_rejects_fabricated_positive_exposure(self) -> None:
        data = _load()
        data["observed_authorities"]["d04_packed_exposure"][
            "real_postpack_unique_loss_positions"
        ] = 1
        self._assert_rejected(data, "current real post-pack exposure must remain zero")

    def test_rejects_optimizer_mechanics_as_lr_selection_authority(self) -> None:
        data = _load()
        data["recipe_contract"]["learning_rate_selection_authority"] = "TRAIN-344B"
        self._assert_rejected(data, "synthetic mechanics may not self-select")

    def test_requires_next_exposure_resume_binding(self) -> None:
        data = _load()
        data["checkpoint_and_resume"]["mandatory_state"].remove("next_exposure_identity")
        self._assert_rejected(data, "resume must bind exact next exposure")

    def test_final_test_cannot_select_checkpoint(self) -> None:
        data = _load()
        data["evaluation"]["final_test_may_influence_selection"] = True
        self._assert_rejected(data, "final test may not influence selection")

    def test_replay_cannot_manufacture_capacity(self) -> None:
        data = _load()
        data["campaign"]["replay_allowed"] = True
        self._assert_rejected(data, "capacity firewall weakened")


if __name__ == "__main__":
    unittest.main()
