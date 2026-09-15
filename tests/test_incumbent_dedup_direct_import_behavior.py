from __future__ import annotations

import collections
import importlib.util
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType

from twelve_six.data import incumbent_dedup_indexed_execution as indexed


def _load_source_module(tmp_path: Path, name: str, source: str) -> ModuleType:
    path = tmp_path / f"{name}.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_frozen_direct_import_inventory_covers_pinned_stdlib_behavior() -> None:
    actual = {
        (label, module_name, imported_name, bound_name)
        for label, module_name, imported_name, bound_name, _state
        in indexed._FROZEN_DIRECT_IMPORTED_BEHAVIOR
    }
    assert actual == {
        ("DATA232", "collections", "defaultdict", "defaultdict"),
        ("DATA232", "typing", "Mapping", "Mapping"),
        ("DATA232", "typing", "Sequence", "Sequence"),
        ("V1", "collections", "defaultdict", "defaultdict"),
        ("V1", "collections.abc", "Mapping", "Mapping"),
        ("V1", "collections.abc", "Sequence", "Sequence"),
        ("V1", "pathlib", "Path", "Path"),
        ("V1", "urllib.request", "Request", "Request"),
        ("V1", "urllib.request", "urlopen", "urlopen"),
        ("V3", "collections", "Counter", "Counter"),
        ("V3", "collections", "defaultdict", "defaultdict"),
        ("V3", "collections.abc", "Mapping", "Mapping"),
        ("V3", "collections.abc", "Sequence", "Sequence"),
        ("V3", "pathlib", "Path", "Path"),
    }


def test_attestation_rejects_counter_class_member_replacement(tmp_path: Path) -> None:
    module = _load_source_module(
        tmp_path,
        "direct_counter_member_replacement",
        (
            "from collections import Counter\n"
            "def behavior(values):\n"
            "    return Counter(values)\n"
        ),
    )
    indexed._attest_executable_module(module, "V3")

    original = Counter.update
    caught: str | None = None

    def replacement(self: Counter, *args: object, **kwargs: object) -> None:
        del self, args, kwargs

    try:
        Counter.update = replacement
        try:
            indexed._attest_executable_module(module, "V3")
        except indexed.IndexedExecutionError as exc:
            caught = str(exc)
    finally:
        Counter.update = original

    assert caught == "V3 direct imported behavior drift: collections.Counter"


def test_loader_dependency_attestation_rejects_counter_count_helper_rebinding() -> None:
    counter_state = next(
        state
        for label, module_name, imported_name, bound_name, state
        in indexed._FROZEN_DIRECT_IMPORTED_BEHAVIOR
        if (label, module_name, imported_name, bound_name)
        == ("V3", "collections", "Counter", "Counter")
    )
    assert indexed._direct_behavior_state_matches(Counter, counter_state)
    assert Counter(["a", "a", "b"]) == Counter({"a": 2, "b": 1})

    original = collections._count_elements
    caught_helper: str | None = None
    caught_runtime: str | None = None

    class MustNotReadV1:
        @property
        def v1(self) -> object:
            raise AssertionError("v1 accessed before loader dependency attestation")

    def replacement(mapping: object, iterable: object) -> None:
        del mapping, iterable

    try:
        collections._count_elements = replacement
        assert indexed._direct_behavior_state_matches(Counter, counter_state)
        assert Counter(["a", "a", "b"]) == Counter()
        try:
            indexed._attest_loader_frozen_runtime_dependencies()
        except indexed.IndexedExecutionError as exc:
            caught_helper = str(exc)
        try:
            indexed.attest_incumbent_runtime(MustNotReadV1())
        except indexed.IndexedExecutionError as exc:
            caught_runtime = str(exc)
    finally:
        collections._count_elements = original

    assert caught_helper == "collections._count_elements runtime drift"
    assert caught_runtime == "collections._count_elements runtime drift"
    indexed._attest_loader_frozen_runtime_dependencies()


def test_attestation_rejects_unfrozen_direct_behavior_import(tmp_path: Path) -> None:
    module = _load_source_module(
        tmp_path,
        "direct_userdict_unfrozen",
        (
            "from collections import UserDict\n"
            "def behavior(value):\n"
            "    return UserDict(value)\n"
        ),
    )

    caught: str | None = None
    try:
        indexed._attest_executable_module(module, "V3")
    except indexed.IndexedExecutionError as exc:
        caught = str(exc)

    assert caught == (
        "V3 direct imported behavior not frozen: collections.UserDict as UserDict"
    )


def test_counter_restored_after_adversarial_regression() -> None:
    counter = Counter(["a", "a", "b"])
    assert counter == Counter({"a": 2, "b": 1})