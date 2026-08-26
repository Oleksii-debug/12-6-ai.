from __future__ import annotations

import json
from pathlib import Path

import pytest

from test_postbase353_redteam import (
    ProjectOwnedToyBackend,
    _authority_path,
    _base_root,
    _dataset,
    _plan,
)
from twelve_six.post_base.contract import snapshot_directory
from twelve_six.post_base.sft_recovery import terminalize_sft_mechanics
from twelve_six.post_base.sft_runner import SFT_CHECKPOINT_NAMESPACE, SFTMechanicsExample


def test_generation_bytes_plus_manifest_rehash_forgery_fails_closed(tmp_path: Path) -> None:
    base_root = _base_root(tmp_path)
    dataset = _dataset()
    plan = _plan(dataset=dataset, base_root=base_root, max_steps=1)
    experiment_root = tmp_path / "coordinated-generation-forgery"

    class CoordinatedGenerationForger(ProjectOwnedToyBackend):
        def train_step(
            self,
            state: object,
            example: SFTMechanicsExample,
            *,
            step: int,
            seed: int,
        ) -> dict[str, float]:
            generation_root = (
                experiment_root / SFT_CHECKPOINT_NAMESPACE / "generation_000000"
            )
            backend_root = generation_root / "backend"
            (backend_root / "weights.json").write_text(
                '{"weight": 555.0}', encoding="utf-8"
            )
            manifest_path = generation_root / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["backend_snapshot_sha256"] = snapshot_directory(
                backend_root
            ).identity_sha256
            manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
            return super().train_step(state, example, step=step, seed=seed)

    with pytest.raises(RuntimeError, match="publication digest drift"):
        terminalize_sft_mechanics(
            plan=plan,
            dataset=dataset,
            canonical_base_root=base_root,
            experiment_root=experiment_root,
            backend=CoordinatedGenerationForger(),
        )
    assert not _authority_path(experiment_root).exists()
