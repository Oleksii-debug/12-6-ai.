from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/run_next100_070_two_clean_build.py"
SPEC = importlib.util.spec_from_file_location("next100_070", SCRIPT)
assert SPEC and SPEC.loader
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def _fixture_authorities(tmp: Path):
    config = MOD.load_json(ROOT / "configs/data/next100_070_two_clean_build_v1.json")
    contract = MOD.load_json(ROOT / config["base_contract"]["path"])
    registry_sources = []
    rights_rows = []
    families = set()
    for frozen in contract["exact_training_candidate_inventory"]["sources"]:
        source_id = MOD.normalized_source_id(frozen["source_id"])
        family = frozen["family"]
        families.add(family)
        normalized_sha = frozen.get("normalized_sha256", frozen["raw_sha256"])
        modality = frozen["modality"]
        language = "python" if modality == "code" else frozen["language"]
        registry_sources.append(
            {
                "source_id": source_id,
                "independent_source_family": {"family_id": family},
                "language": language,
                "modality": modality,
                "snapshot": {
                    "normalized_bytes": frozen["normalized_bytes"],
                    "normalized_sha256": normalized_sha,
                    "raw_sha256": frozen["raw_sha256"],
                    "normalization_policy": "TEST_IDENTITY_POLICY",
                },
                "rights": {"model_training": {"status": "ALLOWED"}},
            }
        )
        rights_rows.append(
            {
                "source_id": source_id,
                "source_family": family,
                "license_id": frozen.get("license", "TEST-RIGHTS"),
                "rights": {
                    "model_training": "ALLOWED",
                    "redistribution": "ALLOWED_TEST_FIXTURE",
                    "evaluation": "NOT_SEPARATELY_ADMITTED",
                },
            }
        )
    registry = {
        "registry_identity_sha256": "test-registry-identity",
        "local_free_only": True,
        "source_count": len(registry_sources),
        "independent_source_family_count": len(families),
        "byte_report": {"unique_normalized_bytes": sum(row["snapshot"]["normalized_bytes"] for row in registry_sources)},
        "sources": registry_sources,
    }
    rights = {"local_free_only": True, "admitted": rights_rows}
    registry_path = tmp / "registry.json"
    rights_path = tmp / "rights.json"
    registry_path.write_bytes(MOD.canonical_bytes(registry))
    rights_path.write_bytes(MOD.canonical_bytes(rights))
    config["late_bound_authorities"]["registry"]["git_blob_sha1"] = MOD.git_blob_sha1_bytes(registry_path.read_bytes())
    config["late_bound_authorities"]["registry"]["registry_identity_sha256"] = registry["registry_identity_sha256"]
    config["late_bound_authorities"]["rights"]["git_blob_sha1"] = MOD.git_blob_sha1_bytes(rights_path.read_bytes())
    return config, registry_path, rights_path


def _tree(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_two_clean_roots_are_byte_identical() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        config, registry_path, rights_path = _fixture_authorities(tmp)
        contract, registry, rights, records = MOD.validate_authorities(config, registry_path, rights_path)
        root_a = tmp / "clean-a"
        root_b = tmp / "clean-b"
        result_a = MOD.build_outputs(config, contract, registry, rights, records, root_a)
        result_b = MOD.build_outputs(config, contract, registry, rights, records, root_b)
        assert result_a["tree_listing_sha256"] == result_b["tree_listing_sha256"]
        assert _tree(root_a) == _tree(root_b)
        assert set(_tree(root_a)) == set(config["determinism_contract"]["required_surfaces"])
        assert json.loads((root_a / "gate_report.json").read_text(encoding="utf-8"))["status"] == "BLOCKED_DETERMINISTIC_PREFLIGHT_COMPLETE"
        assert json.loads((root_a / "shards/manifest.json").read_text(encoding="utf-8"))["payload_file_count"] == 0


def test_dirty_output_root_fails_closed() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        config, registry_path, rights_path = _fixture_authorities(tmp)
        contract, registry, rights, records = MOD.validate_authorities(config, registry_path, rights_path)
        dirty = tmp / "dirty"
        dirty.mkdir()
        (dirty / "residue").write_text("shared cache residue", encoding="utf-8")
        with pytest.raises(MOD.ValidationError, match="clean and empty"):
            MOD.build_outputs(config, contract, registry, rights, records, dirty)
