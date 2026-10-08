"""Plan-1 Section-4 exact archive/restore/negative and semantic-update tests."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from twelve_six.environment_lock import (
    EnvironmentLockError,
    build_lock,
    inspect_wheel,
    validate_lock,
    validate_semantic_transition,
    validate_wheelhouse,
    verify_or_restore,
)

PROJECT = b'[project]\ndependencies = ["fixture>=1.0"]\n'


def make_wheel(tmp_path: Path, *, name: str = "fixture", version: str = "1.0") -> Path:
    root = tmp_path / "wheels"
    root.mkdir(exist_ok=True)
    path = root / f"{name}-{version}-py3-none-any.whl"
    prefix = f"{name}-{version}.dist-info"
    contents = {
        f"{name}/__init__.py": b"VALUE = 42\n",
        f"{prefix}/METADATA": (
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
        ).encode(),
        f"{prefix}/WHEEL": (
            b"Wheel-Version: 1.0\nGenerator: plan1-test\n"
            b"Root-Is-Purelib: true\nTag: py3-none-any\n"
        ),
    }
    record = []
    for filename, raw in sorted(contents.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b"=").decode()
        record.append(f"{filename},sha256={digest},{len(raw)}\n")
    record_name = f"{prefix}/RECORD"
    record.append(f"{record_name},,\n")
    contents[record_name] = "".join(record).encode()
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as wheel:
        for filename, raw in sorted(contents.items()):
            info = zipfile.ZipInfo(filename, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o100644 << 16
            wheel.writestr(info, raw)
    return path


def make_lock(tmp_path: Path, *, platform="linux_x86_64", python="3.13"):
    wheel = make_wheel(tmp_path)
    return wheel, build_lock(wheel.parent, platform=platform, python=python, pyproject=PROJECT)


@pytest.mark.parametrize("platform", ["linux_x86_64", "win_amd64"])
def test_cross_platform_wheel_lock_is_canonical(tmp_path, platform):
    wheel, raw = make_lock(tmp_path, platform=platform)
    assert raw == build_lock(wheel.parent, platform=platform, python="3.13", pyproject=PROJECT)
    doc = validate_lock(raw, platform=platform, python="3.13",
                        expected_project_sha256=hashlib.sha256(PROJECT).hexdigest())
    assert doc["wheels"][0]["name"] == "fixture"
    result = verify_or_restore(raw, pyproject=PROJECT, wheelhouse=wheel.parent,
                               platform=platform, python="3.13", target=tmp_path / "env")
    assert result == {"status": "VERIFIED_WHEELHOUSE", "wheels": 1}


def test_wrong_python_platform_or_project_is_denied(tmp_path):
    wheel, raw = make_lock(tmp_path)
    for platform, python, project in (
        ("win_amd64", "3.13", PROJECT),
        ("linux_x86_64", "3.11", PROJECT),
        ("linux_x86_64", "3.13", PROJECT + b"\n"),
    ):
        with pytest.raises(EnvironmentLockError):
            verify_or_restore(raw, pyproject=project, wheelhouse=wheel.parent,
                              platform=platform, python=python, target=tmp_path / "no")


def test_duplicate_keys_nonfinite_bad_version_and_noncanonical_json_denied(tmp_path):
    _, raw = make_lock(tmp_path)
    mutations = [
        raw.replace(b'"schema_version":1,', b'"schema_version":1,"schema_version":1,'),
        raw.replace(b'"schema_version":1', b'"schema_version":true'),
        raw.replace(b'"schema_version":1', b'"schema_version":1.0'),
        raw.replace(b'"schema_version":1', b'"schema_version":NaN'),
        raw.rstrip(b"\n"),
    ]
    for content in mutations:
        with pytest.raises(EnvironmentLockError):
            validate_lock(content, platform="linux_x86_64", python="3.13",
                          expected_project_sha256=hashlib.sha256(PROJECT).hexdigest())


def test_unsafe_duplicate_wrong_arch_or_unpinned_wheel_rejected(tmp_path):
    _, raw = make_lock(tmp_path)
    edits = [
        lambda doc: doc["wheels"][0].update(filename="../fixture.whl"),
        lambda doc: doc["wheels"][0].update(
            filename="fixture-1.0-cp313-cp313-win_amd64.whl"
        ),
        lambda doc: doc["wheels"][0].update(version="1.1"),
        lambda doc: doc["wheels"][0].update(sha256="unverified"),
        lambda doc: doc["wheels"].append(dict(doc["wheels"][0])),
    ]
    for edit in edits:
        doc = json.loads(raw)
        edit(doc)
        serialized = (json.dumps(doc, sort_keys=True, separators=(",", ":")) + "\n").encode()
        with pytest.raises(EnvironmentLockError):
            validate_lock(serialized, platform="linux_x86_64", python="3.13",
                          expected_project_sha256=hashlib.sha256(PROJECT).hexdigest())


def test_missing_extra_or_modified_wheel_fails_before_use(tmp_path):
    wheel, raw = make_lock(tmp_path)
    parsed = validate_lock(raw, platform="linux_x86_64", python="3.13",
                           expected_project_sha256=hashlib.sha256(PROJECT).hexdigest())
    wheel.write_bytes(wheel.read_bytes() + b"tamper")
    with pytest.raises(EnvironmentLockError, match="hash/size"):
        validate_wheelhouse(parsed, wheel.parent)
    wheel.unlink()
    with pytest.raises(EnvironmentLockError, match="missing or untracked"):
        validate_wheelhouse(parsed, wheel.parent)
    make_wheel(tmp_path)
    extra = wheel.parent / "unexpected.txt"
    extra.write_text("untrusted")
    with pytest.raises(EnvironmentLockError, match="missing or untracked"):
        validate_wheelhouse(parsed, wheel.parent)
    extra.unlink()
    validate_wheelhouse(parsed, wheel.parent)


def test_inner_wheel_record_tamper_denied_even_with_new_outer_sha(tmp_path):
    wheel = make_wheel(tmp_path)
    with zipfile.ZipFile(wheel) as z:
        contents = {i.filename: z.read(i.filename) for i in z.infolist()}
    contents["fixture/__init__.py"] = b"ATTACK\n"
    with zipfile.ZipFile(wheel, "w") as z:
        for name, raw in contents.items():
            z.writestr(name, raw)
    with pytest.raises(EnvironmentLockError, match="bytes changed|metadata digest/size"):
        inspect_wheel(wheel)


def test_missing_mandatory_package_or_below_minimum_never_generates_runtime_lock(tmp_path):
    wheel = make_wheel(tmp_path)
    with pytest.raises(EnvironmentLockError, match="missing mandatory wheel projects"):
        build_lock(wheel.parent, platform="linux_x86_64", python="3.13",
                   pyproject=b'[project]\ndependencies=["torch>=2.5"]\n')
    with pytest.raises(EnvironmentLockError, match="violates project minimum"):
        build_lock(wheel.parent, platform="linux_x86_64", python="3.13",
                   pyproject=b'[project]\ndependencies=["fixture>=2.0"]\n')


def test_clean_offline_restore_executes_and_never_overwrites(tmp_path):
    platform = "win_amd64" if os.name == "nt" else "linux_x86_64"
    python = f"{sys.version_info.major}.{sys.version_info.minor}"
    wheel, raw = make_lock(tmp_path, platform=platform, python=python)
    target = tmp_path / "isolated"
    result = verify_or_restore(raw, pyproject=PROJECT, wheelhouse=wheel.parent,
                               platform=platform, python=python, target=target,
                               restore=True)
    assert result["status"] == "RESTORED_OFFLINE"
    exe = target / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    check = subprocess.run([str(exe), "-c", "import fixture; print(fixture.VALUE)"],
                           check=True, capture_output=True, text=True)
    assert check.stdout.strip() == "42"
    assert (target / "plan1-environment-receipt.json").exists()
    with pytest.raises(EnvironmentLockError, match="must not exist"):
        verify_or_restore(raw, pyproject=PROJECT, wheelhouse=wheel.parent,
                          platform=platform, python=python, target=target, restore=True)


def test_cross_platform_restore_refused_before_environment_mutation(tmp_path):
    alternate = "win_amd64" if os.name != "nt" else "linux_x86_64"
    wheel, raw = make_lock(tmp_path, platform=alternate)
    target = tmp_path / "not-created"
    with pytest.raises(EnvironmentLockError, match="different host ABI"):
        verify_or_restore(raw, pyproject=PROJECT, wheelhouse=wheel.parent,
                          platform=alternate, python="3.13", target=target, restore=True)
    assert not target.exists()


def test_dependency_change_requires_five_versioned_independent_evidence_classes(tmp_path):
    _, old = make_lock(tmp_path)
    oldsha = hashlib.sha256(old).hexdigest()
    new = old.replace(b'"schema_version":1', b'"schema_version":2')
    newsha = hashlib.sha256(new).hexdigest()
    assert validate_semantic_transition(
        old, old, expected_old_sha256=oldsha, expected_new_sha256=oldsha,
        review=None, independently_pinned_review_sha256=None,
    )["status"] == "UNCHANGED"
    with pytest.raises(EnvironmentLockError, match="independent reviewed"):
        validate_semantic_transition(
            old, new, expected_old_sha256=oldsha, expected_new_sha256=newsha,
            review=None, independently_pinned_review_sha256=None,
        )
    evidence = {
        "schema_version": 1, "old_lock_sha256": oldsha,
        "new_lock_sha256": newsha, "contract_version": "ENV_REVISION_V2",
        "decision": "REVIEWED_APPROVED",
        "impact": {
            category: {"status": "UNCHANGED_VERIFIED", "evidence_sha256": oldsha}
            for category in ("numerical", "tokenizer", "data", "checkpoint", "inference")
        },
        "evidence_refs": ["evidence/independent-pass.json"],
    }
    wire = (json.dumps(evidence, sort_keys=True, separators=(",", ":")) + "\n").encode()
    receipt = validate_semantic_transition(
        old, new, expected_old_sha256=oldsha, expected_new_sha256=newsha,
        review=wire, independently_pinned_review_sha256=hashlib.sha256(wire).hexdigest(),
    )
    assert receipt["status"] == "REQUALIFIED_CHANGE"
    evidence["impact"].pop("checkpoint")
    broken = (json.dumps(evidence, sort_keys=True, separators=(",", ":")) + "\n").encode()
    with pytest.raises(EnvironmentLockError, match="incomplete change semantic impact"):
        validate_semantic_transition(
            old, new, expected_old_sha256=oldsha, expected_new_sha256=newsha,
            review=broken,
            independently_pinned_review_sha256=hashlib.sha256(broken).hexdigest(),
        )


def test_cli_never_overwrites_lock_or_accepts_unpinned_environment(tmp_path, capsys):
    from tools.plan1_environment import main

    archive = make_wheel(tmp_path)
    project = tmp_path / "pyproject.toml"
    project.write_bytes(PROJECT)
    lock_file = tmp_path / "candidate.lock.json"
    args = [
        "--platform", "linux_x86_64", "--python", "3.13",
        "--project", str(project), "--wheelhouse", str(archive.parent),
        "--lock", str(lock_file),
    ]
    assert main(["generate", *args]) == 0
    original = lock_file.read_bytes()
    assert main(["generate", *args]) == 2
    assert lock_file.read_bytes() == original
    assert main(["verify", *args, "--expected-lock-sha256", "0" * 64]) == 2
    assert main(["verify", *args]) == 2
    approved = hashlib.sha256(original).hexdigest()
    assert main(["verify", *args, "--expected-lock-sha256", approved]) == 0
    assert "VERIFIED_WHEELHOUSE" in capsys.readouterr().out
