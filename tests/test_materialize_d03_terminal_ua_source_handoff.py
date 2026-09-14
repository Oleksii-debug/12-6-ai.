from __future__ import annotations

import hashlib
import importlib.util
import io
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
SPEC = importlib.util.spec_from_file_location(
    "materialize_d03_terminal_ua_source_handoff",
    ROOT / "tools" / "materialize_d03_terminal_ua_source_handoff.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def framed(rows):
    out = bytearray()
    for path, payload in rows:
        p = path.encode()
        out += len(p).to_bytes(4, "big") + p
        out += len(payload).to_bytes(8, "big") + payload
    return bytes(out)


def make_zip(rows):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, payload in rows:
            zf.writestr(name, payload)
    return buf.getvalue()


def test_parse_frame_round_trip():
    payload = framed([("a.txt", b"alpha\n"), ("b.txt", b"beta\n")])
    assert MODULE.parse_frame(payload) == [
        ("a.txt", b"alpha\n"),
        ("b.txt", b"beta\n"),
    ]


@pytest.mark.parametrize("payload", [b"", b"\x00", b"\x00\x00\x00\x05ab"])
def test_parse_frame_truncation_fails(payload):
    with pytest.raises(ValueError):
        MODULE.parse_frame(payload)


def test_parse_frame_duplicate_path_fails():
    payload = framed([("a.txt", b"one"), ("a.txt", b"two")])
    with pytest.raises(ValueError):
        MODULE.parse_frame(payload)


def test_exact_zip_verifies_digest_and_member_set(tmp_path):
    data = make_zip([("a.txt", b"alpha"), ("b.txt", b"beta")])
    path = tmp_path / "fixture.zip"
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    out = MODULE.read_exact_zip(path, digest, {"a.txt", "b.txt"})
    assert out == {"a.txt": b"alpha", "b.txt": b"beta"}


def test_exact_zip_digest_tamper_fails(tmp_path):
    data = make_zip([("a.txt", b"alpha")])
    path = tmp_path / "fixture.zip"
    path.write_bytes(data + b"tamper")
    digest = hashlib.sha256(data).hexdigest()
    with pytest.raises(ValueError):
        MODULE.read_exact_zip(path, digest, {"a.txt"})


def test_exact_zip_member_set_drift_fails(tmp_path):
    data = make_zip([("a.txt", b"alpha"), ("extra.txt", b"x")])
    path = tmp_path / "fixture.zip"
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    with pytest.raises(ValueError):
        MODULE.read_exact_zip(path, digest, {"a.txt"})


def test_build_record_is_zero_credit_and_deterministic():
    source = {
        "source_id": "source",
        "family": "family",
        "source_commit": "0" * 40,
    }
    first = MODULE.build_record(source, "a.txt", "Привіт\n".encode())
    second = MODULE.build_record(source, "a.txt", "Привіт\n".encode())
    assert first == second
    assert first["current_corpus_eligible"] is False
    assert first["training_eligible"] is False
    assert first["evaluation_eligible"] is False
