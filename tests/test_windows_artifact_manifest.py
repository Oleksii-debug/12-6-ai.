from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ARTIFACT_MANIFEST = _load_module(
    "windows_artifact_manifest_test",
    "packaging/windows/artifact_manifest.py",
)
WINDOWS_LAUNCHER = _load_module(
    "windows_launcher_test",
    "packaging/windows/launcher.py",
)


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


@pytest.mark.parametrize(
    "payload",
    [
        '{"profile_id":"a","profile_id":"b"}',
        '{"outer":{"manifest_sha256":"a","manifest_sha256":"b"}}',
        '{"value":NaN}',
        '{"value":Infinity}',
        '{"value":-Infinity}',
        '{"value":1e400}',
        '{"outer":[{"value":-1e400}]}',
    ],
)
def test_read_json_rejects_ambiguous_or_nonfinite_authority(
    tmp_path: Path, payload: str
) -> None:
    path = tmp_path / "authority.json"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError):
        ARTIFACT_MANIFEST._read_json(path)


def test_read_json_preserves_valid_nested_json(tmp_path: Path) -> None:
    path = tmp_path / "authority.json"
    path.write_text(
        '{"profile_id":"windows-x86_64","nested":{"finite":1.25},"items":[0,true,null]}',
        encoding="utf-8",
    )

    assert ARTIFACT_MANIFEST._read_json(path) == {
        "profile_id": "windows-x86_64",
        "nested": {"finite": 1.25},
        "items": [0, True, None],
    }


def test_windows_workflow_checks_actual_stderr_variable() -> None:
    workflow = (
        ROOT / ".github" / "workflows" / "d08-windows-product-packaging.yml"
    ).read_text(encoding="utf-8")
    assert "$stdertText" not in workflow
    assert "if ($stderrText -match 'Український stdin без GUI')" in workflow


def test_application_manifest_rejects_checkpoint_sidecar_outside_wheel(
    tmp_path: Path,
) -> None:
    _write_wheel(tmp_path, ["twelve_six/__init__.py"])
    sidecar = tmp_path / "assets" / "weights.safetensors"
    sidecar.parent.mkdir()
    sidecar.write_bytes(b"serialized-model-fixture")

    with pytest.raises(
        RuntimeError,
        match="application artifact contains model/checkpoint sidecar bytes",
    ):
        ARTIFACT_MANIFEST._application(tmp_path, "c" * 40)

    assert not (tmp_path / "app-manifest.json").exists()


def test_runtime_manifest_rejects_nested_application_wheel(tmp_path: Path) -> None:
    lock = tmp_path / "12-6-lock"
    lock.mkdir()
    (lock / "profile.json").write_text(
        json.dumps(
            {
                "profile_id": "windows-x86_64",
                "python": {"version": "3.11.9"},
                "manifest_sha256": "d" * 64,
            }
        ),
        encoding="utf-8",
    )
    nested = tmp_path / "nested" / "wheel-cache"
    nested.mkdir(parents=True)
    (nested / "twelve_six_ai-0.0.0-py3-none-any.whl").write_bytes(b"fixture")

    with pytest.raises(
        RuntimeError,
        match="runtime artifact must not contain the application wheel",
    ):
        ARTIFACT_MANIFEST._runtime(tmp_path)

    assert not (tmp_path / "runtime-manifest.json").exists()


def test_runtime_report_normalizes_missing_runtime_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = tmp_path / "12-6-lock"
    lock.mkdir()
    profile = json.loads(
        (ROOT / "requirements" / "locks" / "windows-x86_64" / "profile.json").read_text(
            encoding="utf-8"
        )
    )
    (lock / "profile.json").write_text(
        json.dumps(profile),
        encoding="utf-8",
    )
    monkeypatch.setattr(WINDOWS_LAUNCHER, "_lock_dir", lambda: lock)

    _, errors = WINDOWS_LAUNCHER._runtime_report()

    assert "cannot read installed D08 runtime lock" in errors


def test_windows_workflow_triggers_manifest_regression_suite() -> None:
    workflow = (
        ROOT / ".github" / "workflows" / "d08-windows-product-packaging.yml"
    ).read_text(encoding="utf-8")
    assert '- "tests/test_windows_artifact_manifest.py"' in workflow


def test_windows_workflow_prefixes_artifact_digests_for_evidence() -> None:
    workflow = (
        ROOT / ".github" / "workflows" / "d08-windows-product-packaging.yml"
    ).read_text(encoding="utf-8")
    assert (
        "--app-artifact-digest "
        "'sha256:${{ needs.source-and-app.outputs.app_artifact_digest }}'"
        in workflow
    )
    assert (
        "--runtime-artifact-digest "
        "'sha256:${{ needs.windows-runtime-bundle.outputs.runtime_artifact_digest }}'"
        in workflow
    )


def test_windows_workflow_rejects_failed_generation_stdout() -> None:
    workflow = (
        ROOT / ".github" / "workflows" / "d08-windows-product-packaging.yml"
    ).read_text(encoding="utf-8")
    assert "prompt leaked into stdout" in workflow
    assert "failed generation unexpectedly wrote stdout" in workflow


def test_windows_locked_profile_forces_utf8_for_unicode_path() -> None:
    workflow = (
        ROOT / ".github" / "workflows" / "d08-windows-product-packaging.yml"
    ).read_text(encoding="utf-8")
    assert "$env:PYTHONUTF8 = '1'" in workflow
    assert "$env:PYTHONIOENCODING = 'utf-8'" in workflow


@pytest.mark.parametrize(
    "raw",
    [
        '{"profile_id":"windows-x86_64","profile_id":"other"}',
        '{"nested":{"version":"a","version":"b"}}',
        '{"value":NaN}',
        '{"value":Infinity}',
        '{"value":-Infinity}',
        '{"value":1e400}',
        '{"nested":[{"value":-1e400}]}',
    ],
)
def test_launcher_profile_rejects_ambiguous_or_nonfinite_json(
    tmp_path: Path,
    raw: str,
) -> None:
    lock = tmp_path / "12-6-lock"
    lock.mkdir()
    (lock / "profile.json").write_text(raw, encoding="utf-8")

    with pytest.raises(RuntimeError, match="cannot read installed D08 Windows lock profile"):
        WINDOWS_LAUNCHER._load_profile(lock)


def test_launcher_pins_committed_windows_profile_authority() -> None:
    profile = json.loads(
        (ROOT / "requirements" / "locks" / "windows-x86_64" / "profile.json").read_text(
            encoding="utf-8"
        )
    )
    assert profile["manifest_sha256"] == WINDOWS_LAUNCHER.PROFILE_MANIFEST_SHA256


def test_launcher_rejects_self_consistent_noncanonical_profile(tmp_path: Path) -> None:
    lock = tmp_path / "12-6-lock"
    lock.mkdir()
    profile = {
        "profile_id": "windows-x86_64",
        "python": {"version": "3.11.9"},
        "locks": {"runtime": {"sha256": "f" * 64}},
        "extra_policy": "attacker-controlled",
    }
    profile["manifest_sha256"] = WINDOWS_LAUNCHER.hashlib.sha256(
        WINDOWS_LAUNCHER._canonical_json_bytes(profile)
    ).hexdigest()
    (lock / "profile.json").write_text(json.dumps(profile), encoding="utf-8")

    with pytest.raises(RuntimeError, match="not the canonical profile"):
        WINDOWS_LAUNCHER._load_profile(lock)


def _self_hashed_manifest(payload: dict[str, object]) -> dict[str, object]:
    unsigned = dict(payload)
    unsigned["manifest_sha256"] = ARTIFACT_MANIFEST.hashlib.sha256(
        ARTIFACT_MANIFEST._canonical(payload)
    ).hexdigest()
    return unsigned


def test_read_self_hashed_manifest_rejects_tamper(tmp_path: Path) -> None:
    path = tmp_path / "app-manifest.json"
    payload = _self_hashed_manifest(
        {
            "schema_version": "12-6.windows-application-artifact.v1",
            "source_sha": "a" * 40,
            "contains_runtime_wheels": False,
            "contains_checkpoint": False,
            "files": {},
        }
    )
    payload["contains_checkpoint"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="self-hash mismatch"):
        ARTIFACT_MANIFEST._read_self_hashed_manifest(
            path,
            expected_schema="12-6.windows-application-artifact.v1",
            expected_keys=frozenset(payload),
        )


def test_validate_evidence_inputs_rejects_nonready_status(tmp_path: Path) -> None:
    status = tmp_path / "status.json"
    missing = tmp_path / "missing.json"
    app_root = tmp_path / "application"
    runtime_root = tmp_path / "runtime"
    app_root.mkdir()
    runtime_root.mkdir()
    app = app_root / "app-manifest.json"
    runtime = runtime_root / "runtime-manifest.json"
    status.write_text(
        json.dumps(
            {
                "schema_version": "12-6.windows-product-status.v1",
                "ready": False,
                "runtime": {
                    "profile_id": "windows-x86_64",
                    "python_actual": "3.11.9",
                    "profile_manifest_sha256": ARTIFACT_MANIFEST._WINDOWS_PROFILE_MANIFEST_SHA256,
                },
                "checkpoint": None,
                "errors": ["runtime failed"],
            }
        ),
        encoding="utf-8",
    )
    missing.write_text(
        json.dumps(
            {
                "schema_version": "12-6.windows-product-status.v1",
                "ready": False,
                "runtime": {},
                "checkpoint": None,
                "errors": ["checkpoint does not exist: fixture"],
            }
        ),
        encoding="utf-8",
    )
    app.write_text(
        json.dumps(
            _self_hashed_manifest(
                {
                    "schema_version": "12-6.windows-application-artifact.v1",
                    "source_sha": "a" * 40,
                    "contains_runtime_wheels": False,
                    "contains_checkpoint": False,
                    "files": {},
                }
            )
        ),
        encoding="utf-8",
    )
    runtime.write_text(
        json.dumps(
            _self_hashed_manifest(
                {
                    "schema_version": "12-6.windows-runtime-artifact.v1",
                    "profile_id": "windows-x86_64",
                    "python_version": "3.11.9",
                    "profile_manifest_sha256": ARTIFACT_MANIFEST._WINDOWS_PROFILE_MANIFEST_SHA256,
                    "contains_application_wheel": False,
                    "contains_checkpoint": False,
                    "files": {},
                }
            )
        ),
        encoding="utf-8",
    )
    args = ARTIFACT_MANIFEST.argparse.Namespace(
        status=status,
        missing_status=missing,
        app_manifest=app,
        runtime_manifest=runtime,
        source_sha="a" * 40,
        app_artifact_id="1",
        app_artifact_digest="sha256:" + "c" * 64,
        runtime_artifact_id="2",
        runtime_artifact_digest="sha256:" + "d" * 64,
    )

    with pytest.raises(ValueError, match="exact ready Windows runtime status"):
        ARTIFACT_MANIFEST._validate_evidence_inputs(args)


def test_artifact_manifest_pins_committed_windows_profile_authority() -> None:
    profile = json.loads(
        (ROOT / "requirements" / "locks" / "windows-x86_64" / "profile.json").read_text(
            encoding="utf-8"
        )
    )
    assert (
        profile["manifest_sha256"]
        == ARTIFACT_MANIFEST._WINDOWS_PROFILE_MANIFEST_SHA256
    )


def test_validate_manifest_files_rejects_post_manifest_mutation(tmp_path: Path) -> None:
    payload = tmp_path / "launcher.py"
    payload.write_text("original", encoding="utf-8")
    manifest = {
        "schema_version": "fixture",
        "files": ARTIFACT_MANIFEST._files(tmp_path, exclude={"app-manifest.json"}),
    }
    manifest_path = tmp_path / "app-manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    payload.write_text("tampered", encoding="utf-8")

    with pytest.raises(ValueError, match="file inventory/hash mismatch"):
        ARTIFACT_MANIFEST._validate_manifest_files(manifest_path, manifest)


def test_application_manifest_rejects_nested_runtime_wheel(tmp_path: Path) -> None:
    _write_wheel(tmp_path, ["twelve_six/__init__.py"])
    nested = tmp_path / "nested" / "numpy-2.4.6-py3-none-any.whl"
    nested.parent.mkdir()
    nested.write_bytes(b"dependency-wheel-fixture")

    with pytest.raises(RuntimeError, match="unexpected wheel bytes"):
        ARTIFACT_MANIFEST._application(tmp_path, "e" * 40)


def test_runtime_manifest_rejects_wheel_outside_wheelhouse(tmp_path: Path) -> None:
    lock = tmp_path / "12-6-lock"
    lock.mkdir()
    (lock / "profile.json").write_text(
        json.dumps(
            {
                "profile_id": "windows-x86_64",
                "python": {"version": "3.11.9"},
                "manifest_sha256": "f" * 64,
            }
        ),
        encoding="utf-8",
    )
    wheel = tmp_path / "cache" / "numpy-2.4.6-py3-none-any.whl"
    wheel.parent.mkdir()
    wheel.write_bytes(b"dependency-wheel-fixture")

    with pytest.raises(RuntimeError, match="outside canonical wheelhouse"):
        ARTIFACT_MANIFEST._runtime(tmp_path)
