from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.scientific_scope_gate import owned_tests, semantic_sources


ROOT = Path(__file__).resolve().parents[1]


def test_ownership_rules_are_additive_for_training_surface() -> None:
    ownership = json.loads((ROOT / "configs/ci/scientific_scope_ownership.v1.json").read_text(encoding="utf-8"))
    tests, hits = owned_tests(["src/twelve_six/training/trainer.py"], ownership)
    assert "first-party-minimum" in hits["src/twelve_six/training/trainer.py"]
    assert "training-engine" in hits["src/twelve_six/training/trainer.py"]
    assert "tests/test_s0_convergence_integration.py" in tests
    assert "tests/test_training_engine.py" in tests


def test_ownership_config_references_existing_tests() -> None:
    ownership = json.loads((ROOT / "configs/ci/scientific_scope_ownership.v1.json").read_text(encoding="utf-8"))
    for rule in ownership["rules"]:
        for test in rule["tests"]:
            assert (ROOT / test).is_file(), (rule["id"], test)


def test_unowned_first_party_surface_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="unowned first-party scientific surface"):
        owned_tests(["src/twelve_six/new_unowned_module.py"], {"rules": []})


def test_semantic_sources_follows_internal_import_closure(tmp_path: Path) -> None:
    (tmp_path / "src/twelve_six/training").mkdir(parents=True)
    (tmp_path / "tools").mkdir()
    (tmp_path / "src/twelve_six/__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "src/twelve_six/model.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "src/twelve_six/training/__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "src/twelve_six/training/probe.py").write_text("from .. import model\n", encoding="utf-8")
    (tmp_path / "tools/experiment.py").write_text("import twelve_six.training.probe\n", encoding="utf-8")

    result = semantic_sources(tmp_path, ["tools/experiment.py"])
    assert "tools/experiment.py" in result
    assert "src/twelve_six/training/probe.py" in result
    assert "src/twelve_six/model.py" in result
