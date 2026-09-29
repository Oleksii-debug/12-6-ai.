from __future__ import annotations

import ast
import functools
import html
import importlib.util
import inspect
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


def _load_source_module(tmp_path: Path, name: str, source: str) -> ModuleType:
    path = tmp_path / f"{name}.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("label", "source", "global_name"),
    (
        (
            "DATA232",
            (
                "import re\nimport unicodedata\n"
                "def normalize(value):\n"
                "    return re.sub(r'\\\\s+', ' ', unicodedata.normalize('NFKC', value))\n"
            ),
            "unicodedata",
        ),
        (
            "DATA232",
            (
                "import re\nimport unicodedata\n"
                "def normalize(value):\n"
                "    return re.sub(r'\\\\s+', ' ', unicodedata.normalize('NFKC', value))\n"
            ),
            "re",
        ),
        (
            "V1",
            (
                "import hashlib\n"
                "def digest(value):\n"
                "    return hashlib.sha256(value).hexdigest()\n"
            ),
            "hashlib",
        ),
        (
            "V3",
            (
                "import html\n"
                "def normalize(value):\n"
                "    return html.unescape(value)\n"
            ),
            "html",
        ),
    ),
)
@_isolated_indexed_test
def test_attestation_rejects_imported_behavior_global_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    source: str,
    global_name: str,
) -> None:
    module = _load_source_module(tmp_path, f"synthetic_{label.lower()}_{global_name}", source)
    indexed._attest_executable_module(module, label)

    monkeypatch.setattr(module, global_name, object())
    with pytest.raises(
        indexed.IndexedExecutionError,
        match=rf"{label} referenced global drift: {global_name}",
    ):
        indexed._attest_executable_module(module, label)


@pytest.mark.parametrize(
    ("label", "source", "global_name", "member_name"),
    (
        (
            "DATA232",
            (
                "import re\nimport unicodedata\n"
                "def normalize(value):\n"
                "    return re.sub(r'\\s+', ' ', unicodedata.normalize('NFKC', value))\n"
            ),
            "unicodedata",
            "normalize",
        ),
        (
            "DATA232",
            (
                "import re\nimport unicodedata\n"
                "def normalize(value):\n"
                "    return re.sub(r'\\s+', ' ', unicodedata.normalize('NFKC', value))\n"
            ),
            "re",
            "sub",
        ),
        (
            "V1",
            (
                "import hashlib\n"
                "def digest(value):\n"
                "    return hashlib.sha256(value).hexdigest()\n"
            ),
            "hashlib",
            "sha256",
        ),
        (
            "V3",
            (
                "import html\n"
                "def normalize(value):\n"
                "    return html.unescape(value)\n"
            ),
            "html",
            "unescape",
        ),
    ),
)
@_isolated_indexed_test
def test_attestation_rejects_imported_behavior_member_in_place_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    source: str,
    global_name: str,
    member_name: str,
) -> None:
    module = _load_source_module(
        tmp_path,
        f"synthetic_{label.lower()}_{global_name}_{member_name}",
        source,
    )
    indexed._attest_executable_module(module, label)

    imported = getattr(module, global_name)

    def replacement(*args: object, **kwargs: object) -> None:
        del args, kwargs

    with monkeypatch.context() as patch:
        patch.setattr(imported, member_name, replacement)
        with pytest.raises(
            indexed.IndexedExecutionError,
            match=rf"{label} imported behavior drift: {global_name}\.{member_name}",
        ):
            indexed._attest_executable_module(module, label)


@pytest.mark.parametrize(
    ("label", "module", "member_name"),
    (
        ("ast.parse", ast, "parse"),
        ("inspect.isclass", inspect, "isclass"),
        ("inspect.isfunction", inspect, "isfunction"),
    ),
)
@_isolated_indexed_test
def test_loader_bootstrap_rejects_python_function_code_drift(
    label: str,
    module: ModuleType,
    member_name: str,
) -> None:
    function = getattr(module, member_name)
    original_code = function.__code__
    caught: str | None = None

    def replacement(*args: object, **kwargs: object) -> None:
        del args, kwargs

    try:
        function.__code__ = replacement.__code__
        try:
            indexed._attest_loader_frozen_runtime_dependencies()
        except indexed.IndexedExecutionError as exc:
            caught = str(exc)
    finally:
        function.__code__ = original_code

    assert caught == f"{label} runtime drift"
    indexed._attest_loader_frozen_runtime_dependencies()


@_isolated_indexed_test
def test_imported_python_member_code_drift_is_rejected_without_rebinding(
    tmp_path: Path,
) -> None:
    module = _load_source_module(
        tmp_path,
        "synthetic_v1_json_dumps_in_place",
        "import json\ndef encode(value):\n    return json.dumps(value)\n",
    )
    indexed._attest_executable_module(module, "V1")
    original_code = json.dumps.__code__
    caught: str | None = None

    def replacement(*args: object, **kwargs: object) -> str:
        del args, kwargs
        return "drifted"

    try:
        json.dumps.__code__ = replacement.__code__
        assert module.json.dumps is json.dumps
        try:
            indexed._attest_executable_module(module, "V1")
        except indexed.IndexedExecutionError as exc:
            caught = str(exc)
    finally:
        json.dumps.__code__ = original_code

    assert caught == "V1 imported behavior drift: json.dumps"
    indexed._attest_executable_module(module, "V1")


@_isolated_indexed_test
def test_loader_rejects_re_compile_transitive_rebinding() -> None:
    original = re._compile
    caught: str | None = None

    def replacement(*args: object, **kwargs: object) -> object:
        del args, kwargs
        return object()

    try:
        re._compile = replacement
        try:
            indexed._attest_loader_frozen_runtime_dependencies()
        except indexed.IndexedExecutionError as exc:
            caught = str(exc)
    finally:
        re._compile = original

    assert caught == "transitive behavior drift: re._compile"
    indexed._attest_loader_frozen_runtime_dependencies()


@_isolated_indexed_test
def test_loader_rejects_json_encoder_transitive_rebinding() -> None:
    original = json.JSONEncoder
    caught: str | None = None

    class ReplacementEncoder:
        pass

    try:
        json.JSONEncoder = ReplacementEncoder
        try:
            indexed._attest_loader_frozen_runtime_dependencies()
        except indexed.IndexedExecutionError as exc:
            caught = str(exc)
    finally:
        json.JSONEncoder = original

    assert caught == "transitive behavior drift: json.JSONEncoder"
    indexed._attest_loader_frozen_runtime_dependencies()


@_isolated_indexed_test
def test_verifier_attests_before_reference_callable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import verify_incumbent_dedup_indexed_equivalence as verifier

    events: list[str] = []

    class DummyV3:
        def audit_payloads(self, inventory: object, payloads: object) -> dict[str, object]:
            events.append("reference")
            return {}

    v3 = DummyV3()
    monkeypatch.setattr(verifier.importlib, "import_module", lambda _: v3)

    def reject_runtime(module: object) -> None:
        assert module is v3
        events.append("attest")
        raise indexed.IndexedExecutionError("synthetic runtime drift")

    monkeypatch.setattr(verifier, "attest_incumbent_runtime", reject_runtime)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_incumbent_dedup_indexed_equivalence.py",
            "--v3-module",
            "synthetic.v3",
            "--inventory",
            "not-read-before-attestation.json",
            "--payload-map",
            "not-read-before-attestation-payload-map.json",
        ],
    )

    with pytest.raises(indexed.IndexedExecutionError, match="synthetic runtime drift"):
        verifier.main()
    assert events == ["attest"]

@_isolated_indexed_test
def test_loader_rejects_html_replace_charref_transitive_rebinding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen_unescape = html.unescape

    def replacement(_match: object) -> str:
        return "drifted"

    with monkeypatch.context() as patch:
        patch.setattr(html, "_replace_charref", replacement)
        assert html.unescape is frozen_unescape
        assert html.unescape("&amp;") == "drifted"
        with pytest.raises(
            indexed.IndexedExecutionError,
            match=r"transitive behavior drift: html\._replace_charref",
        ):
            indexed._attest_loader_frozen_runtime_dependencies()

    indexed._attest_loader_frozen_runtime_dependencies()


@_isolated_indexed_test
def test_loader_rejects_html_charref_transitive_rebinding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with monkeypatch.context() as patch:
        patch.setattr(html, "_charref", re.compile(r"never-match"))
        with pytest.raises(
            indexed.IndexedExecutionError,
            match=r"transitive behavior drift: html\._charref",
        ):
            indexed._attest_loader_frozen_runtime_dependencies()

    indexed._attest_loader_frozen_runtime_dependencies()


@_isolated_indexed_test
def test_loader_rejects_html_invalid_charrefs_in_place_mutation() -> None:
    original = dict(html._invalid_charrefs)
    try:
        html._invalid_charrefs[-1] = "drifted"
        with pytest.raises(
            indexed.IndexedExecutionError,
            match=r"transitive behavior drift: html\._invalid_charrefs",
        ):
            indexed._attest_loader_frozen_runtime_dependencies()
    finally:
        html._invalid_charrefs.clear()
        html._invalid_charrefs.update(original)

    indexed._attest_loader_frozen_runtime_dependencies()


@_isolated_indexed_test
def test_loader_rejects_html_invalid_codepoints_in_place_mutation() -> None:
    original = set(html._invalid_codepoints)
    try:
        html._invalid_codepoints.add(-1)
        with pytest.raises(
            indexed.IndexedExecutionError,
            match=r"transitive behavior drift: html\._invalid_codepoints",
        ):
            indexed._attest_loader_frozen_runtime_dependencies()
    finally:
        html._invalid_codepoints.clear()
        html._invalid_codepoints.update(original)

    indexed._attest_loader_frozen_runtime_dependencies()


@_isolated_indexed_test
def test_loader_rejects_html5_in_place_mutation() -> None:
    original = dict(html._html5)
    try:
        html._html5["__swarm_transitive_probe__"] = "drifted"
        with pytest.raises(
            indexed.IndexedExecutionError,
            match=r"transitive behavior drift: html\._html5",
        ):
            indexed._attest_loader_frozen_runtime_dependencies()
    finally:
        html._html5.clear()
        html._html5.update(original)

    indexed._attest_loader_frozen_runtime_dependencies()
