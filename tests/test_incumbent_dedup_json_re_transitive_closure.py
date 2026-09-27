from __future__ import annotations

import json
import re
from types import ModuleType

import pytest

from twelve_six.data import incumbent_dedup_indexed_execution as indexed


def test_core_executor_bytes_are_preserved_exactly() -> None:
    payload = indexed._core.__file__
    assert isinstance(payload, str)
    with open(payload, "rb") as handle:
        assert indexed._git_blob_sha1(handle.read()) == "af7be7909501ea9d76604ebed084cec32fbd9456"


def test_loader_rejects_json_default_encoder_rebinding(monkeypatch: pytest.MonkeyPatch) -> None:
    with monkeypatch.context() as patch:
        patch.setattr(json, "_default_encoder", json.JSONEncoder())
        with pytest.raises(
            indexed.IndexedExecutionError,
            match=r"transitive behavior drift: json\._default_encoder",
        ):
            indexed._attest_loader_frozen_runtime_dependencies()

    indexed._attest_loader_frozen_runtime_dependencies()


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


def test_loader_rejects_re_compiler_module_rebinding(monkeypatch: pytest.MonkeyPatch) -> None:
    with monkeypatch.context() as patch:
        patch.setattr(re, "_compiler", ModuleType("synthetic_re_compiler"))
        with pytest.raises(
            indexed.IndexedExecutionError,
            match=r"transitive behavior drift: re\._compiler$",
        ):
            indexed._attest_loader_frozen_runtime_dependencies()

    indexed._attest_loader_frozen_runtime_dependencies()


@pytest.mark.parametrize("member_name", ("compile", "isstring"))
def test_loader_rejects_re_compiler_behavior_rebinding(
    monkeypatch: pytest.MonkeyPatch,
    member_name: str,
) -> None:
    def replacement(*args: object, **kwargs: object) -> object:
        del args, kwargs
        return object()

    with monkeypatch.context() as patch:
        patch.setattr(re._compiler, member_name, replacement)
        with pytest.raises(
            indexed.IndexedExecutionError,
            match=rf"transitive behavior drift: re\._compiler\.{member_name}$",
        ):
            indexed._attest_loader_frozen_runtime_dependencies()

    indexed._attest_loader_frozen_runtime_dependencies()


def test_loader_rejects_re_cache_rebinding(monkeypatch: pytest.MonkeyPatch) -> None:
    with monkeypatch.context() as patch:
        patch.setattr(re, "_cache", {})
        with pytest.raises(
            indexed.IndexedExecutionError,
            match=r"transitive behavior drift: re\._cache",
        ):
            indexed._attest_loader_frozen_runtime_dependencies()

    indexed._attest_loader_frozen_runtime_dependencies()


def test_loader_rejects_re_maxcache_non_exact_int(monkeypatch: pytest.MonkeyPatch) -> None:
    with monkeypatch.context() as patch:
        patch.setattr(re, "_MAXCACHE", float(re._MAXCACHE))
        with pytest.raises(
            indexed.IndexedExecutionError,
            match=r"transitive behavior drift: re\._MAXCACHE",
        ):
            indexed._attest_loader_frozen_runtime_dependencies()

    indexed._attest_loader_frozen_runtime_dependencies()


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
    events: list[str] = []

    def fake_core_attest(v3: object) -> None:
        del v3
        assert key not in cache
        events.append("core")
        cache[key] = Poison()

    try:
        cache[key] = Poison()
        with monkeypatch.context() as patch:
            patch.setattr(indexed, "_CORE_RUNTIME_ATTEST", fake_core_attest)
            indexed.attest_incumbent_runtime(object())
        assert events == ["core"]
        assert key not in cache
    finally:
        cache.clear()
        cache.update(original)

    indexed._attest_loader_frozen_runtime_dependencies()


def test_byte_preserved_core_resolves_hardened_hooks() -> None:
    assert indexed._core._attest_loader_frozen_runtime_dependencies is indexed._attest_loader_frozen_runtime_dependencies
    assert indexed._core.attest_incumbent_runtime is indexed.attest_incumbent_runtime
