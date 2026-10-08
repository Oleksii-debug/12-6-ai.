"""Plan 8 Section 4 operator and signing boundary regressions (LOCAL_FREE)."""
from __future__ import annotations

import base64
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

_PATH = Path(__file__).resolve().parents[1] / "tools" / "plan8_physical_operator.py"
_SPEC = importlib.util.spec_from_file_location("plan8_physical_operator", _PATH)
assert _SPEC and _SPEC.loader
operator = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(operator)


def test_status_not_run_is_accessible_text(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    result = operator.main(["status", "--receipt", str(tmp_path / "none.json")])
    text = json.loads(capsys.readouterr().out)
    assert result == 0
    assert text["state"] == "NOT_RUN"


def test_status_never_claims_trust_on_unverified_receipt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_text('{"verdict":"PASS"}', encoding="utf-8")
    assert operator.main(["status", "--receipt", str(receipt)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["claimed_verdict"] == "PASS"
    assert result["state"] == "UNVERIFIED_RECEIPT"
    assert result["verification"] == "NOT_CHECKED"


@pytest.mark.parametrize(
    "payload",
    ['{"verdict":"PASS","verdict":"FAIL"}', '{"verdict":"PASS"', '"PASS"'],
)
def test_status_denies_malformed_or_duplicate_json(tmp_path: Path, payload: str) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_text(payload, encoding="utf-8")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        operator.main(["status", "--receipt", str(receipt)])


def test_status_denies_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text('{"verdict":"PASS"}', encoding="utf-8")
    symlink = tmp_path / "redirect.json"
    try:
        symlink.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    with pytest.raises(ValueError, match="non-symlink"):
        operator.main(["status", "--receipt", str(symlink)])


def test_trust_store_ed25519_rejects_forgery_and_unknown_key(tmp_path: Path) -> None:
    crypto = pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    key = Ed25519PrivateKey.generate()
    public = key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    trust = tmp_path / "authority.json"
    trust.write_text(
        json.dumps(
            {
                "schema_version": "12-6.physical-trust-store.v1",
                "keys": {"authority-1": base64.b64encode(public).decode("ascii")},
            }
        ),
        encoding="utf-8",
    )
    verifier = operator._public_key_verifier(trust)
    signed = key.sign(b"exact candidate")
    assert verifier("authority-1", b"exact candidate", signed) is True
    assert verifier("authority-1", b"other candidate", signed) is False
    assert verifier("untrusted-id", b"exact candidate", signed) is False
    assert crypto.__name__ == "cryptography"


@pytest.mark.parametrize(
    "payload",
    [
        '{"schema_version":"12-6.physical-trust-store.v1","keys":{"x":"AA==","x":"AA=="}}',
        '{"schema_version":"something-else","keys":{"x":"AA=="}}',
        '{"schema_version":"12-6.physical-trust-store.v1","keys":{}}',
    ],
)
def test_trust_store_fails_closed(tmp_path: Path, payload: str) -> None:
    pytest.importorskip("cryptography")
    trust = tmp_path / "trust.json"
    trust.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError):
        operator._public_key_verifier(trust)


def test_host_private_key_outside_repo_and_owner_only(tmp_path: Path) -> None:
    pytest.importorskip("cryptography")
    root = tmp_path / "checkout"
    root.mkdir()
    inside = root / "secret.key"
    inside.write_bytes(b"a" * 32)
    inside.chmod(0o600)
    with pytest.raises(ValueError, match="outside"):
        operator._host_signer(inside, root)
    private = tmp_path / "external.key"
    private.write_bytes(b"a" * 32)
    private.chmod(0o644)
    if os.name != "nt":
        with pytest.raises(ValueError, match="owner-only"):
            operator._host_signer(private, root)
    private.chmod(0o600)
    signer = operator._host_signer(private, root)
    assert len(signer("host", b"receipt")) == 64


def test_run_denies_existing_receipt_before_any_effect(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    packet = tmp_path / "packet.json"
    authority = tmp_path / "authority.json"
    host = tmp_path / "host.json"
    receipt = tmp_path / "existing.json"
    log = tmp_path / "new.log"
    receipt.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(operator, "_public_key_verifier", lambda *_: lambda *_: True)
    mock_packet = SimpleNamespace(
        packet=SimpleNamespace(target_git_sha="a" * 40, execution_mode=operator.ExecutionMode.SIMULATION)
    )
    monkeypatch.setattr(operator, "load_verified_signed_packet", lambda *_, **__: mock_packet)
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("execution must not begin")
    monkeypatch.setattr(operator, "_host_signer", forbidden)
    monkeypatch.setattr(operator, "execute_qualification", forbidden)
    with pytest.raises(ValueError, match="already exists"):
        operator.main([
            "run",
            "--packet", str(packet), "--authority-keys", str(authority),
            "--host-keys", str(host), "--repo-root", str(tmp_path),
            "--receipt", str(receipt), "--log", str(log),
            "--host-private-key", str(tmp_path / "key"), "--host-key-id", "host-1",
        ])


def test_verify_requires_physical_pass_unless_simulation_is_explicit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(operator, "_public_key_verifier", lambda *_: lambda *_: True)
    mock_packet = SimpleNamespace(
        packet=SimpleNamespace(target_git_sha="a" * 40, execution_mode=operator.ExecutionMode.SIMULATION)
    )
    monkeypatch.setattr(operator, "load_verified_signed_packet", lambda *_, **__: mock_packet)
    observed: list[bool] = []

    def verify(*_a: object, **kw: object) -> dict[str, str]:
        observed.append(kw["require_real_pass"])
        return {"verdict": "SIMULATION_PASS"}
    monkeypatch.setattr(operator, "verify_qualification_evidence", verify)
    args = [
        "verify",
        "--packet", str(tmp_path / "packet"),
        "--authority-keys", str(tmp_path / "authority"),
        "--host-keys", str(tmp_path / "host"),
        "--repo-root", str(tmp_path),
        "--receipt", str(tmp_path / "receipt"),
        "--log", str(tmp_path / "log"),
    ]
    assert operator.main(args) == 0
    assert operator.main(args + ["--allow-simulation"]) == 0
    assert observed == [True, False]
    assert '"state": "VERIFIED"' in capsys.readouterr().out
