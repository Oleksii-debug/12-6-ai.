from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from twelve_six.data.incumbent_dedup_indexed_execution import (
    IndexedExecutionError,
    _attest_executable_module,
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
            "import re\nimport unicodedata\n"
            "def normalize(value):\n"
            "    return re.sub(r'\\\\s+', ' ', unicodedata.normalize('NFKC', value))\n",
            "unicodedata",
        ),
        (
            "DATA232",
            "import re\nimport unicodedata\n"
            "def normalize(value):\n"
            "    return re.sub(r'\\\\s+', ' ', unicodedata.normalize('NFKC', value))\n",
            "re",
        ),
        (
            "V1",
            "import hashlib\n"
            "def digest(value):\n"
            "    return hashlib.sha256(value).hexdigest()\n",
            "hashlib",
        ),
        (
            "V3",
            "import html\n"
            "def normalize(value):\n"
            "    return html.unescape(value)\n",
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
