from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

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


EXPECTED_FROZEN_MEMBERS = {
    ("DATA232", "hashlib", "sha256"),
    ("DATA232", "json", "dumps"),
    ("DATA232", "re", "fullmatch"),
    ("DATA232", "re", "sub"),
    ("DATA232", "unicodedata", "normalize"),
    ("V1", "hashlib", "sha1"),
    ("V1", "hashlib", "sha256"),
    ("V1", "json", "dumps"),
    ("V1", "re", "fullmatch"),
    ("V3", "html", "unescape"),
    ("V3", "re", "S"),
    ("V3", "re", "escape"),
    ("V3", "re", "fullmatch"),
    ("V3", "re", "match"),
    ("V3", "re", "search"),
    ("V3", "re", "sub"),
    ("V3", "unicodedata", "normalize"),
}


def test_frozen_member_inventory_matches_exact_pinned_closure() -> None:
    actual = {
        (label, global_name, member_name)
        for label, global_name, member_name, _expected
        in indexed._FROZEN_IMPORTED_BEHAVIOR_MEMBERS
    }
    assert actual == EXPECTED_FROZEN_MEMBERS


@pytest.mark.parametrize(
    ("label", "source", "global_name", "member_name"),
    (
        (
            "DATA232",
            "import hashlib\ndef behavior(value):\n    return hashlib.sha256(value).hexdigest()\n",
            "hashlib",
            "sha256",
        ),
        (
            "DATA232",
            "import json\ndef behavior(value):\n    return json.dumps(value, sort_keys=True)\n",
            "json",
            "dumps",
        ),
        (
            "DATA232",
            "import re\ndef behavior(value):\n    return re.fullmatch(r'x+', value)\n",
            "re",
            "fullmatch",
        ),
        (
            "DATA232",
            "import re\ndef behavior(value):\n    return re.sub(r'x+', 'x', value)\n",
            "re",
            "sub",
        ),
        (
            "DATA232",
            (
                "import unicodedata\n"
                "def behavior(value):\n"
                "    return unicodedata.normalize('NFKC', value)\n"
            ),
            "unicodedata",
            "normalize",
        ),
        (
            "V1",
            "import hashlib\ndef behavior(value):\n    return hashlib.sha1(value).hexdigest()\n",
            "hashlib",
            "sha1",
        ),
        (
            "V1",
            "import hashlib\ndef behavior(value):\n    return hashlib.sha256(value).hexdigest()\n",
            "hashlib",
            "sha256",
        ),
        (
            "V1",
            "import json\ndef behavior(value):\n    return json.dumps(value, sort_keys=True)\n",
            "json",
            "dumps",
        ),
        (
            "V1",
            "import re\ndef behavior(value):\n    return re.fullmatch(r'x+', value)\n",
            "re",
            "fullmatch",
        ),
        (
            "V3",
            "import html\ndef behavior(value):\n    return html.unescape(value)\n",
            "html",
            "unescape",
        ),
        (
            "V3",
            "import re\ndef behavior(value):\n    return re.fullmatch(r'x+', value)\n",
            "re",
            "fullmatch",
        ),
        (
            "V3",
            "import re\ndef behavior(value):\n    return re.match(r'x+', value)\n",
            "re",
            "match",
        ),
        (
            "V3",
            "import re\ndef behavior(value):\n    return re.search(r'x+', value)\n",
            "re",
            "search",
        ),
        (
            "V3",
            "import re\ndef behavior(value):\n    return re.escape(value)\n",
            "re",
            "escape",
        ),
        (
            "V3",
            "import re\ndef behavior(value):\n    return re.sub(r'x+', 'x', value)\n",
            "re",
            "sub",
        ),
        (
            "V3",
            (
                "import re\n"
                "def behavior(value):\n"
                "    return re.sub(r'x+', 'x', value, flags=re.S)\n"
            ),
            "re",
            "S",
        ),
        (
            "V3",
            (
                "import unicodedata\n"
                "def behavior(value):\n"
                "    return unicodedata.normalize('NFKC', value)\n"
            ),
            "unicodedata",
            "normalize",
        ),
    ),
)
def test_attestation_rejects_every_exact_imported_member_in_place_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    source: str,
    global_name: str,
    member_name: str,
) -> None:
    module = _load_source_module(
        tmp_path,
        f"frozen_{label.lower()}_{global_name}_{member_name}",
        source,
    )
    indexed._attest_executable_module(module, label)
    imported = getattr(module, global_name)

    with monkeypatch.context() as patch:
        patch.setattr(imported, member_name, object())
        with pytest.raises(indexed.IndexedExecutionError) as exc_info:
            indexed._attest_executable_module(module, label)

    assert str(exc_info.value) == f"{label} imported behavior drift: {global_name}.{member_name}"


def test_attestation_fails_closed_on_unfrozen_direct_member_reference(tmp_path: Path) -> None:
    module = _load_source_module(
        tmp_path,
        "unfrozen_v1_re_split",
        "import re\ndef behavior(value):\n    return re.split(r'x+', value)\n",
    )
    with pytest.raises(
        indexed.IndexedExecutionError,
        match=r"V1 imported behavior member not frozen: re\.split",
    ):
        indexed._attest_executable_module(module, "V1")


def test_git_blob_identity_uses_loader_frozen_sha1(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"authority-bytes"
    expected = indexed._git_blob_sha1(payload)

    def explode(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("live hashlib.sha1 must not be used")

    with monkeypatch.context() as patch:
        patch.setattr(indexed.hashlib, "sha1", explode)
        assert indexed._git_blob_sha1(payload) == expected


def test_code_attestation_uses_loader_frozen_marshal_dumps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_source_module(
        tmp_path,
        "frozen_v1_marshal_digest",
        "def behavior(value):\n    return value\n",
    )
    indexed._attest_executable_module(module, "V1")
    module.behavior.__code__ = (lambda value: value + "!").__code__

    with monkeypatch.context() as patch:
        patch.setattr(indexed.marshal, "dumps", lambda _code: b"forged-equal-digest")
        with pytest.raises(
            indexed.IndexedExecutionError,
            match=r"V1 callable code drift: behavior",
        ):
            indexed._attest_executable_module(module, "V1")
