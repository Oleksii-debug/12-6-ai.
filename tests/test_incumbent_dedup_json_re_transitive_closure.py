from __future__ import annotations

import functools
import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

_ISOLATED_INDEXED_TEST_NODE = "TWELVE_SIX_ISOLATED_INDEXED_TEST_NODE"


def _isolated_indexed_test(test):
    @functools.wraps(test)
    def wrapper(*args, **kwargs):
        current = os.environ.get("PYTEST_CURRENT_TEST", "").rsplit(" (", 1)[0]
        if os.environ.get(_ISOLATED_INDEXED_TEST_NODE) == current and current:
            return test(*args, **kwargs)
        if not current:
            return test(*args, **kwargs)

        env = os.environ.copy()
        env[_ISOLATED_INDEXED_TEST_NODE] = current
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", current],
            cwd=Path(__file__).resolve().parent.parent,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=180,
        )
        assert completed.returncode == 0, (
            f"isolated indexed test failed: {current}\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )

    return wrapper


class _LazyIndexed:
    _module: ModuleType | None = None

    def _load_module(self) -> ModuleType:
        module = object.__getattribute__(self, "_module")
        if module is None:
            module = importlib.import_module(
                "twelve_six.data.incumbent_dedup_indexed_execution"
            )
            object.__setattr__(self, "_module", module)
        return module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._load_module(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_module":
            object.__setattr__(self, name, value)
            return
        setattr(self._load_module(), name, value)

    def __delattr__(self, name: str) -> None:
        if name == "_module":
            object.__delattr__(self, name)
            return
        delattr(self._load_module(), name)


indexed = _LazyIndexed()


@_isolated_indexed_test
def test_core_executor_bytes_are_preserved_exactly() -> None:
    payload = indexed._core.__file__
    assert isinstance(payload, str)
    with open(payload, "rb") as handle:
        assert indexed._git_blob_sha1(handle.read()) == "af7be7909501ea9d76604ebed084cec32fbd9456"


@_isolated_indexed_test
def test_loader_rejects_json_default_encoder_rebinding(monkeypatch: pytest.MonkeyPatch) -> None:
    with monkeypatch.context() as patch:
        patch.setattr(json, "_default_encoder", json.JSONEncoder())
        with pytest.raises(
            indexed.IndexedExecutionError,
            match=r"transitive behavior drift: json\._default_encoder",
        ):
            indexed._attest_loader_frozen_runtime_dependencies()

    indexed._attest_loader_frozen_runtime_dependencies()


@_isolated_indexed_test
def test_loader_rejects_json_default_encoder_in_place_state_drift() -> None:
    encoder = json._default_encoder
    original = encoder.ensure_ascii
    try:
        encoder.ensure_ascii = not original
        with pytest.raises(
            indexed.IndexedExecutionError,
            match=r"transitive behavior drift: json\._default_encoder",
        ):
            indexed._attest_loader_frozen_runtime_dependencies()
    finally:
        encoder.ensure_ascii = original

    indexed._attest_loader_frozen_runtime_dependencies()


@_isolated_indexed_test
def test_loader_rejects_re_compiler_module_rebinding(monkeypatch: pytest.MonkeyPatch) -> None:
    cache = re._cache
    original_cache = dict(cache)
    caught: indexed.IndexedExecutionError | None = None
    try:
        with monkeypatch.context() as patch:
            patch.setattr(re, "_compiler", ModuleType("synthetic_re_compiler"))
            try:
                indexed._attest_loader_frozen_runtime_dependencies()
            except indexed.IndexedExecutionError as exc:
                caught = exc
    finally:
        cache.clear()
        cache.update(original_cache)

    assert caught is not None
    assert str(caught) == "transitive behavior drift: re._compiler"
    indexed._attest_loader_frozen_runtime_dependencies()


@pytest.mark.parametrize("member_name", ("compile", "isstring"))
@_isolated_indexed_test
def test_loader_rejects_re_compiler_behavior_rebinding(
    monkeypatch: pytest.MonkeyPatch,
    member_name: str,
) -> None:
    def replacement(*args: object, **kwargs: object) -> object:
        del args, kwargs
        return object()

    cache = re._cache
    original_cache = dict(cache)
    caught: indexed.IndexedExecutionError | None = None
    try:
        with monkeypatch.context() as patch:
            patch.setattr(re._compiler, member_name, replacement)
            try:
                indexed._attest_loader_frozen_runtime_dependencies()
            except indexed.IndexedExecutionError as exc:
                caught = exc
    finally:
        cache.clear()
        cache.update(original_cache)

    assert caught is not None
    assert str(caught) == f"transitive behavior drift: re._compiler.{member_name}"
    indexed._attest_loader_frozen_runtime_dependencies()


@_isolated_indexed_test
def test_loader_rejects_re_cache_rebinding(monkeypatch: pytest.MonkeyPatch) -> None:
    cache = re._cache
    original_cache = dict(cache)
    caught: indexed.IndexedExecutionError | None = None
    try:
        with monkeypatch.context() as patch:
            patch.setattr(re, "_cache", {})
            try:
                indexed._attest_loader_frozen_runtime_dependencies()
            except indexed.IndexedExecutionError as exc:
                caught = exc
    finally:
        cache.clear()
        cache.update(original_cache)

    assert caught is not None
    assert str(caught) == "transitive behavior drift: re._cache"
    indexed._attest_loader_frozen_runtime_dependencies()


@_isolated_indexed_test
def test_loader_rejects_re_maxcache_non_exact_int(monkeypatch: pytest.MonkeyPatch) -> None:
    caught: indexed.IndexedExecutionError | None = None
    with monkeypatch.context() as patch:
        patch.setattr(re, "_MAXCACHE", float(re._MAXCACHE))
        try:
            indexed._attest_loader_frozen_runtime_dependencies()
        except indexed.IndexedExecutionError as exc:
            caught = exc

    assert caught is not None
    assert str(caught) == "transitive behavior drift: re._MAXCACHE"
    indexed._attest_loader_frozen_runtime_dependencies()


@_isolated_indexed_test
def test_loader_binds_runtime_specific_re_cache2_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    caught: indexed.IndexedExecutionError | None = None
    if indexed._FROZEN_RE_HAS_CACHE2:
        original_cache2 = re._cache2
        with monkeypatch.context() as patch:
            patch.setattr(re, "_cache2", {})
            try:
                indexed._attest_loader_frozen_runtime_dependencies()
            except indexed.IndexedExecutionError as exc:
                caught = exc
        expected = "transitive behavior drift: re._cache2"
        assert re._cache2 is original_cache2
    else:
        with monkeypatch.context() as patch:
            patch.setattr(re, "_cache2", {}, raising=False)
            try:
                indexed._attest_loader_frozen_runtime_dependencies()
            except indexed.IndexedExecutionError as exc:
                caught = exc
        expected = "transitive behavior drift: re._cache2 presence"
        assert not hasattr(re, "_cache2")

    assert caught is not None
    assert str(caught) == expected
    indexed._attest_loader_frozen_runtime_dependencies()


@_isolated_indexed_test
def test_loader_binds_runtime_specific_re_maxcache2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not indexed._FROZEN_RE_HAS_MAXCACHE2:
        pytest.skip("runtime has no re._MAXCACHE2")

    caught: indexed.IndexedExecutionError | None = None
    with monkeypatch.context() as patch:
        patch.setattr(re, "_MAXCACHE2", float(re._MAXCACHE2))
        try:
            indexed._attest_loader_frozen_runtime_dependencies()
        except indexed.IndexedExecutionError as exc:
            caught = exc

    assert caught is not None
    assert str(caught) == "transitive behavior drift: re._MAXCACHE2"
    indexed._attest_loader_frozen_runtime_dependencies()


@_isolated_indexed_test
def test_runtime_specific_re_cache2_is_neutralized_when_present() -> None:
    if not indexed._FROZEN_RE_HAS_CACHE2:
        pytest.skip("runtime has no re._cache2")

    cache2 = re._cache2
    original = dict(cache2)
    key = (str, "__cache2_probe__", 0)
    poison = re._compiler.compile(".*", 0)
    try:
        cache2[key] = poison
        indexed._attest_loader_frozen_runtime_dependencies()
        indexed._neutralize_verified_re_cache()
        assert re._cache2 is cache2
        assert key not in cache2
        assert cache2 == {}
    finally:
        cache2.clear()
        cache2.update(original)

    indexed._attest_loader_frozen_runtime_dependencies()


@_isolated_indexed_test
def test_verified_regex_cache_is_neutralized_not_trusted() -> None:
    class Poison:
        def sub(self, repl: object, string: object, count: int = 0) -> str:
            del repl, string, count
            return "POISONED"

    cache = re._cache
    original = dict(cache)
    key = (str, "needle", 0)
    try:
        cache[key] = Poison()
        indexed._attest_loader_frozen_runtime_dependencies()
        assert re.sub("needle", "safe", "needle") == "POISONED"

        indexed._neutralize_verified_re_cache()
        assert re._cache is cache
        assert key not in cache
        assert re.sub("needle", "safe", "needle") == "safe"
    finally:
        cache.clear()
        cache.update(original)

    indexed._attest_loader_frozen_runtime_dependencies()


@_isolated_indexed_test
def test_lazy_proxy_monkeypatch_mutates_real_authority_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = indexed._load_module()
    original = module._CORE_RUNTIME_ATTEST

    def replacement(_v3: object) -> None:
        return None

    with monkeypatch.context() as patch:
        patch.setattr(indexed, "_CORE_RUNTIME_ATTEST", replacement)
        assert module._CORE_RUNTIME_ATTEST is replacement
        assert indexed._CORE_RUNTIME_ATTEST is replacement

    assert module._CORE_RUNTIME_ATTEST is original
    assert indexed._CORE_RUNTIME_ATTEST is original


@_isolated_indexed_test
def test_runtime_attestation_neutralizes_cache_before_and_after_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Poison:
        def sub(self, repl: object, string: object, count: int = 0) -> str:
            del repl, string, count
            return "POISONED"

    cache = re._cache
    original = dict(cache)
    key = (str, "two-phase-probe", 0)
    cache2 = getattr(re, "_cache2", None)
    original_cache2 = None if cache2 is None else dict(cache2)
    key2 = (str, "two-phase-cache2-probe", 0)
    poison2 = None if cache2 is None else re._compiler.compile(".*", 0)
    events: list[str] = []

    def fake_core_attest(v3: object) -> None:
        del v3
        assert key not in cache
        if cache2 is not None:
            assert key2 not in cache2
        events.append("core")
        cache[key] = Poison()
        if cache2 is not None:
            cache2[key2] = poison2

    try:
        cache[key] = Poison()
        if cache2 is not None:
            cache2[key2] = poison2
        with monkeypatch.context() as patch:
            patch.setattr(indexed, "_CORE_RUNTIME_ATTEST", fake_core_attest)
            indexed.attest_incumbent_runtime(object())
        assert events == ["core"]
        assert key not in cache
        if cache2 is not None:
            assert key2 not in cache2
    finally:
        cache.clear()
        cache.update(original)
        if cache2 is not None and original_cache2 is not None:
            cache2.clear()
            cache2.update(original_cache2)

    indexed._attest_loader_frozen_runtime_dependencies()


@_isolated_indexed_test
def test_byte_preserved_core_resolves_hardened_hooks() -> None:
    assert indexed._core._attest_loader_frozen_runtime_dependencies is indexed._attest_loader_frozen_runtime_dependencies
    assert indexed._core.attest_incumbent_runtime is indexed.attest_incumbent_runtime
