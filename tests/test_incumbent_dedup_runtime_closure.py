from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest

from twelve_six.data.incumbent_dedup_indexed_execution import (
    IndexedExecutionError,
    _attest_executable_module,
    _attest_loader_frozen_runtime_dependencies,
)


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
def test_attestation_rejects_imported_behavior_global_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    source: str,
    global_name: str,
) -> None:
    module = _load_source_module(tmp_path, f"synthetic_{label.lower()}_{global_name}", source)
    _attest_executable_module(module, label)

    monkeypatch.setattr(module, global_name, object())
    with pytest.raises(
        IndexedExecutionError,
        match=rf"{label} referenced global drift: {global_name}",
    ):
        _attest_executable_module(module, label)


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
    _attest_executable_module(module, label)

    imported = getattr(module, global_name)

    def replacement(*args: object, **kwargs: object) -> None:
        del args, kwargs

    with monkeypatch.context() as patch:
        patch.setattr(imported, member_name, replacement)
        with pytest.raises(
            IndexedExecutionError,
            match=rf"{label} imported behavior drift: {global_name}\.{member_name}",
        ):
            _attest_executable_module(module, label)


@pytest.mark.parametrize(
    ("label", "module", "member_name"),
    (
        ("ast.parse", ast, "parse"),
        ("inspect.isclass", inspect, "isclass"),
        ("inspect.isfunction", inspect, "isfunction"),
    ),
)
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
            _attest_loader_frozen_runtime_dependencies()
        except IndexedExecutionError as exc:
            caught = str(exc)
    finally:
        function.__code__ = original_code

    assert caught == f"{label} runtime drift"
    _attest_loader_frozen_runtime_dependencies()


def test_imported_python_member_code_drift_is_rejected_without_rebinding(
    tmp_path: Path,
) -> None:
    module = _load_source_module(
        tmp_path,
        "synthetic_v1_json_dumps_in_place",
        "import json\ndef encode(value):\n    return json.dumps(value)\n",
    )
    _attest_executable_module(module, "V1")
    original_code = json.dumps.__code__
    caught: str | None = None

    def replacement(*args: object, **kwargs: object) -> str:
        del args, kwargs
        return "drifted"

    try:
        json.dumps.__code__ = replacement.__code__
        assert module.json.dumps is json.dumps
        try:
            _attest_executable_module(module, "V1")
        except IndexedExecutionError as exc:
            caught = str(exc)
    finally:
        json.dumps.__code__ = original_code

    assert caught == "V1 imported behavior drift: json.dumps"
    _attest_executable_module(module, "V1")


def test_loader_rejects_re_compile_transitive_rebinding() -> None:
    original = re._compile
    caught: str | None = None

    def replacement(*args: object, **kwargs: object) -> object:
        del args, kwargs
        return object()

    try:
        re._compile = replacement
        try:
            _attest_loader_frozen_runtime_dependencies()
        except IndexedExecutionError as exc:
            caught = str(exc)
    finally:
        re._compile = original

    assert caught == "transitive behavior drift: re._compile"
    _attest_loader_frozen_runtime_dependencies()


def test_loader_rejects_json_encoder_transitive_rebinding() -> None:
    original = json.JSONEncoder
    caught: str | None = None

    class ReplacementEncoder:
        pass

    try:
        json.JSONEncoder = ReplacementEncoder
        try:
            _attest_loader_frozen_runtime_dependencies()
        except IndexedExecutionError as exc:
            caught = str(exc)
    finally:
        json.JSONEncoder = original

    assert caught == "transitive behavior drift: json.JSONEncoder"
    _attest_loader_frozen_runtime_dependencies()


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
        raise IndexedExecutionError("synthetic runtime drift")

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

    with pytest.raises(IndexedExecutionError, match="synthetic runtime drift"):
        verifier.main()
    assert events == ["attest"]