from __future__ import annotations

import json
from pathlib import Path

from twelve_six.data.external_sources import build_eligibility_inventory


def _json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_committed_eligibility_inventory_matches_canonical_registry() -> None:
    registry = _json("data/external/external_sources.json")
    report = _json("reports/data24_training_eligibility_inventory.json")
    assert report["registered_inventory"] == build_eligibility_inventory(registry)
    assert report["source_registry_identity_sha256"] == registry["registry_identity_sha256"]


def test_multilingual_recipe_binds_same_registry_and_blocks_unregistered_fixture() -> None:
    registry = _json("data/external/external_sources.json")
    recipe = _json("configs/data/multilingual_uk_en_code_v1.experimental.json")
    admission = recipe["source_admission"]
    assert admission["registry_identity_sha256"] == registry["registry_identity_sha256"]
    assert admission["eligibility_resolver_required"] is True
    assert admission["project_authored_synthetic_requires_registry_entry_and_evidence"] is True
    assert recipe["local_mechanics_corpus"]["canonical_training_eligibility"] == "BLOCKED_UNREGISTERED"
