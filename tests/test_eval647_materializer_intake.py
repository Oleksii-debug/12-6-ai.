"""Fail-closed network and authority tests for the EVAL647 materializer."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/materialize_eval_code_reserve_v1.py"
MANIFEST = ROOT / "configs/evaluation/eval_code_reserve_v1.json"
spec = importlib.util.spec_from_file_location("eval647_bounded_materializer", SCRIPT)
assert spec is not None and spec.loader is not None
materializer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(materializer)


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


class FakeResponse:
    status = 200

    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.requested: list[int] = []
        self.closed = False

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        self.closed = True

    def read(self, size: int = -1) -> bytes:
        assert size >= 0, "a remote source must never be read without a bound"
        self.requested.append(size)
        return self.payload[:size]


@pytest.mark.parametrize("excess", [0, 1, 2048])
def test_bounded_source_fetch_closes_response(
    excess: int, monkeypatch: pytest.MonkeyPatch,
) -> None:
    size = 10438
    response = FakeResponse(b"x" * (size + excess))

    def fake_open(_request: object, *, timeout: int) -> FakeResponse:
        assert timeout == 30
        return response

    monkeypatch.setattr(materializer.urllib.request, "urlopen", fake_open)
    if excess:
        with pytest.raises(RuntimeError, match="remote source exceeds byte limit"):
            materializer._fetch("https://raw.githubusercontent.com/pinned", 30, size)
    else:
        assert materializer._fetch(
            "https://raw.githubusercontent.com/pinned", 30, size
        ) == response.payload
    assert response.requested == [size + 1]
    assert response.closed


def test_license_response_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    maximum = materializer.MAX_LICENSE_BYTES
    response = FakeResponse(b"L" * (maximum + 1))
    monkeypatch.setattr(
        materializer.urllib.request, "urlopen",
        lambda _request, *, timeout: response,
    )
    with pytest.raises(RuntimeError, match="remote source exceeds byte limit"):
        materializer._fetch(
            "https://raw.githubusercontent.com/pinned/LICENSE", 30, maximum
        )
    assert response.requested == [maximum + 1]
    assert response.closed


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_path", "wrong_repository", "wrong_revision",
        "training_allowed", "missing_gates", "extra_object",
    ],
)
def test_invalid_contract_fails_before_any_network_call(
    mutation: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest()
    if mutation == "wrong_path":
        manifest["objects"][0]["path"] = "unreserved.py"
    elif mutation == "wrong_repository":
        manifest["objects"][0]["repository"] = "attacker/repo"
    elif mutation == "wrong_revision":
        manifest["objects"][0]["revision"] = "main"
    elif mutation == "training_allowed":
        manifest["reservation"]["training_allowed"] = True
    elif mutation == "missing_gates":
        manifest["remaining_successor_gates"] = []
    else:
        manifest["objects"].append(manifest["objects"][0].copy())

    def forbidden_open(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("no network request before contract validation")

    monkeypatch.setattr(materializer.urllib.request, "urlopen", forbidden_open)
    with pytest.raises(ValueError):
        materializer.materialize(manifest)


@pytest.mark.parametrize("timeout", [0, -1, True])
def test_invalid_timeout_fails_before_fetch(
    timeout: object, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_open(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("invalid timeout reached the network")

    monkeypatch.setattr(materializer.urllib.request, "urlopen", forbidden_open)
    with pytest.raises(RuntimeError, match="timeout must be a positive integer"):
        materializer.materialize(_manifest(), timeout=timeout)


def test_valid_manifest_uses_exact_pinned_source_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int, int]] = []

    def fake_fetch(url: str, timeout: int, max_bytes: int) -> bytes:
        calls.append((url, timeout, max_bytes))
        return b"wrong"

    monkeypatch.setattr(materializer, "_fetch", fake_fetch)
    with pytest.raises(RuntimeError, match="raw byte-size drift"):
        materializer.materialize(_manifest())
    assert len(calls) == 1
    assert calls[0][1:] == (30, 10438)
    assert "/jd/tenacity/a2af454834c6bb5a1e39d67334031cdaf0f475b5/" in calls[0][0]


@pytest.mark.parametrize("case", ["forged", "oversize", "invalid_utf8"])
def test_cli_rejects_invalid_authority_without_network_or_output(
    case: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest_file = tmp_path / "контракт із пробілами.json"
    if case == "forged":
        forged = _manifest()
        forged["objects"][0]["repository"] = "attacker/repo"
        manifest_file.write_text(json.dumps(forged), encoding="utf-8")
    elif case == "oversize":
        manifest_file.write_bytes(b"{}" + b" " * materializer._VALIDATOR.MAX_INPUT_BYTES)
    else:
        manifest_file.write_bytes(b'{"worker_id":"\xff"}')
    destination = tmp_path / "не створювати.json"
    monkeypatch.setattr(
        sys, "argv",
        [str(SCRIPT), "--manifest", str(manifest_file), "--output", str(destination)],
    )

    def forbidden_open(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("invalid contract reached the network")

    monkeypatch.setattr(materializer.urllib.request, "urlopen", forbidden_open)
    assert materializer.main() == 2
    output = capsys.readouterr()
    assert output.err == ""
    assert len(output.out.splitlines()) == 1
    result = json.loads(output.out)
    assert result["status"] == "BLOCKED_INVALID_EVAL647_MATERIALIZATION"
    assert result["selection_validation_records_authorized"] == 0
    assert result["model_training_authorized"] is False
    assert result["final_test_outcomes_read"] is False
    assert not destination.exists()


@pytest.mark.parametrize("index", [0, 1])
def test_license_marker_is_not_provenance(index: int) -> None:
    row = _manifest()["objects"][index]
    forged = materializer.LICENSE_MARKERS[row["license_spdx"]] + b"\\nforged-license-bytes"
    with pytest.raises(RuntimeError, match="license SHA-256 drift"):
        materializer._check_pinned_license(row, forged)
