from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_artifact_manifest():
    path = ROOT / "packaging" / "windows" / "artifact_manifest.py"
    spec = importlib.util.spec_from_file_location("windows_artifact_manifest_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ARTIFACT_MANIFEST = _load_artifact_manifest()


def _write_wheel(root: Path, members: list[str]) -> Path:
    wheel = root / "twelve_six_ai-0.0.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for member in members:
            archive.writestr(member, b"fixture")
    return wheel


@pytest.mark.parametrize(
    "member",
    [
        "twelve_six/checkpoint/__init__.py",
        "twelve_six/checkpoint/core.py",
        "twelve_six/checkpoint/state_tree.py",
        "twelve_six/model/checkpoint_policy.json",
    ],
)
def test_checkpoint_namespace_source_files_are_not_payload(member: str) -> None:
    assert not ARTIFACT_MANIFEST._is_model_checkpoint_payload(member)


@pytest.mark.parametrize(
    "member",
    [
        "weights.safetensors",
        "nested/model.pt",
        "nested/model.PTH",
        "checkpoint/run.ckpt",
        "exports/model.onnx",
        "models/weights.gguf",
        "models/weights.ggml",
    ],
)
def test_serialized_model_payload_formats_are_rejected(member: str) -> None:
    assert ARTIFACT_MANIFEST._is_model_checkpoint_payload(member)


def test_application_manifest_allows_checkpoint_python_namespace(tmp_path: Path) -> None:
    _write_wheel(
        tmp_path,
        [
            "twelve_six/checkpoint/__init__.py",
            "twelve_six/checkpoint/core.py",
            "twelve_six/checkpoint/hf_export.py",
        ],
    )

    ARTIFACT_MANIFEST._application(tmp_path, "a" * 40)

    document = json.loads((tmp_path / "app-manifest.json").read_text(encoding="utf-8"))
    assert document["contains_checkpoint"] is False
    assert document["source_sha"] == "a" * 40


def test_application_manifest_rejects_real_checkpoint_payload(tmp_path: Path) -> None:
    _write_wheel(
        tmp_path,
        [
            "twelve_six/checkpoint/core.py",
            "twelve_six/assets/weights.safetensors",
        ],
    )

    with pytest.raises(RuntimeError, match="application wheel contains model/checkpoint bytes"):
        ARTIFACT_MANIFEST._application(tmp_path, "b" * 40)

    assert not (tmp_path / "app-manifest.json").exists()


def test_runtime_manifest_rejects_non_safetensors_checkpoint_payload(tmp_path: Path) -> None:
    lock = tmp_path / "12-6-lock"
    lock.mkdir()
    (lock / "profile.json").write_text(
        json.dumps(
            {
                "profile_id": "windows-x86_64",
                "python": {"version": "3.11.9"},
                "manifest_sha256": "c" * 64,
            }
        ),
        encoding="utf-8",
    )
    payload = tmp_path / "wheelhouse" / "model.ckpt"
    payload.parent.mkdir()
    payload.write_bytes(b"not-real-model-data")

    with pytest.raises(RuntimeError, match="runtime artifact must not contain model/checkpoint bytes"):
        ARTIFACT_MANIFEST._runtime(tmp_path)

    assert not (tmp_path / "runtime-manifest.json").exists()
