from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tools import run_d03_expanded_global_dedup_v9 as runner


def _historical_tree(root: Path) -> Path:
    source = root / "src"
    package = source / "twelve_six"
    data = package / "data"
    data.mkdir(parents=True)
    (package / "__init__.py").write_text(
        'raise RuntimeError("historical package initializer must never execute")\n',
        encoding="utf-8",
    )
    (data / "__init__.py").write_text("", encoding="utf-8")
    (data / "cross_source_capacity_audit_v7.py").write_text(
        'MARKER = "historical-v7"\n',
        encoding="utf-8",
    )
    (data / "cross_source_capacity_audit_v3.py").write_text(
        'MARKER = "historical-v3"\n',
        encoding="utf-8",
    )
    return source


def _project_snapshot() -> dict[str, ModuleType]:
    return {
        name: module
        for name, module in sys.modules.items()
        if runner._is_twelve_six_module(name)
    }


def _assert_snapshot_restored(before: dict[str, ModuleType]) -> None:
    after = _project_snapshot()
    assert set(after) == set(before)
    for name, module in before.items():
        assert after[name] is module


def test_historical_namespace_bypasses_preloaded_current_package_cache(
    tmp_path: Path,
) -> None:
    v7_root = tmp_path / "v7"
    source = _historical_tree(v7_root)
    before = _project_snapshot()
    before_path = list(sys.path)
    before_bytecode = sys.dont_write_bytecode

    with runner._isolated_historical_v7_imports(v7_root):
        historical = importlib.import_module(
            "twelve_six.data.cross_source_capacity_audit_v7"
        )
        assert historical.MARKER == "historical-v7"
        assert source.resolve() in Path(historical.__file__).resolve().parents
        package = sys.modules["twelve_six"]
        assert getattr(package, "__file__", None) is None
        assert list(package.__path__) == [str(source.resolve() / "twelve_six")]
        for name, module in before.items():
            assert sys.modules.get(name) is not module

    _assert_snapshot_restored(before)
    assert sys.path == before_path
    assert sys.dont_write_bytecode is before_bytecode
    assert not list(v7_root.rglob("__pycache__"))


def test_historical_namespace_restores_current_modules_after_failure(
    tmp_path: Path,
) -> None:
    v7_root = tmp_path / "v7"
    _historical_tree(v7_root)
    before = _project_snapshot()
    before_path = list(sys.path)

    with pytest.raises(RuntimeError, match="synthetic historical failure"):
        with runner._isolated_historical_v7_imports(v7_root):
            importlib.import_module("twelve_six.data.cross_source_capacity_audit_v7")
            raise RuntimeError("synthetic historical failure")

    _assert_snapshot_restored(before)
    assert sys.path == before_path


def test_historical_namespace_rejects_project_module_outside_v7_root(
    tmp_path: Path,
) -> None:
    v7_root = tmp_path / "v7"
    _historical_tree(v7_root)
    outside = tmp_path / "outside.py"
    outside.write_text("VALUE = 1\n", encoding="utf-8")
    before = _project_snapshot()

    with pytest.raises(
        runner.ExpandedDedupError,
        match="historical project module escaped V7 source",
    ):
        with runner._isolated_historical_v7_imports(v7_root):
            importlib.import_module("twelve_six.data.cross_source_capacity_audit_v7")
            escaped = ModuleType("twelve_six.escaped")
            escaped.__file__ = str(outside)
            sys.modules[escaped.__name__] = escaped

    _assert_snapshot_restored(before)


def test_reconstruct_captures_v7_in_isolation_then_restores_current_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v7_root = tmp_path / "v7"
    _historical_tree(v7_root)
    before = _project_snapshot()
    validations: list[Path] = []
    bulk_observed_restored: list[bool] = []

    def validate(root: Path) -> Path:
        resolved = root.resolve()
        validations.append(resolved)
        return resolved

    def capture(
        root: Path,
        _config: dict[str, Any],
    ) -> tuple[object, dict[str, Any], dict[str, Any], dict[str, bytes]]:
        historical = importlib.import_module(
            "twelve_six.data.cross_source_capacity_audit_v7"
        )
        assert historical.MARKER == "historical-v7"
        assert root == v7_root.resolve()
        return (
            object(),
            {},
            {"sources": [{"source_id": "v7"}]},
            {"v7": b"v7"},
        )

    def materialize_bulk(
        _root: Path,
        _workspace: Path,
        _config: dict[str, Any],
    ) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, bytes]]:
        current = _project_snapshot()
        bulk_observed_restored.append(
            set(current) == set(before)
            and all(current[name] is module for name, module in before.items())
        )
        return {}, [{"source_id": "bulk"}], {"bulk": b"bulk"}

    monkeypatch.setattr(runner, "validate_v7_checkout", validate)
    monkeypatch.setattr(runner.v8, "_capture_terminal_v7", capture)
    monkeypatch.setattr(runner.v8, "_materialize_bulk", materialize_bulk)

    inventory, payloads = runner.reconstruct_v8_source_inputs(
        v7_root=v7_root,
        bulk_workspace=tmp_path / "bulk",
        v8_config={},
    )

    assert validations == [v7_root.resolve(), v7_root.resolve()]
    assert bulk_observed_restored == [True]
    assert inventory["sources"] == [{"source_id": "v7"}, {"source_id": "bulk"}]
    assert payloads == {"v7": b"v7", "bulk": b"bulk"}
    _assert_snapshot_restored(before)


def test_historical_namespace_rejects_symlink_source_root(tmp_path: Path) -> None:
    actual = tmp_path / "actual"
    _historical_tree(actual)
    linked = tmp_path / "linked"
    try:
        linked.symlink_to(actual, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable")

    with pytest.raises(
        runner.ExpandedDedupError,
        match="historical V7 source root must not be a symlink",
    ):
        with runner._isolated_historical_v7_imports(linked):
            pytest.fail("symlinked historical root was accepted")
