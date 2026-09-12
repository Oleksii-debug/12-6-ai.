from __future__ import annotations

import importlib.util
import io
import tarfile
from pathlib import Path

import pytest

TOOL = Path(__file__).parents[1] / "tools" / "materialize_d03_cpython_stdlib_v1.py"
SPEC = importlib.util.spec_from_file_location("cpython_stdlib_v1", TOOL)
assert SPEC is not None and SPEC.loader is not None
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def _archive(files: dict[str, bytes], *, links: dict[str, str] | None = None) -> bytes:
    output = io.BytesIO()
    prefix = f"cpython-{mod.UPSTREAM_COMMIT}/"
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for path, payload in files.items():
            info = tarfile.TarInfo(prefix + path)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        for path, target in (links or {}).items():
            info = tarfile.TarInfo(prefix + path)
            info.type = tarfile.SYMTYPE
            info.linkname = target
            archive.addfile(info)
    return output.getvalue()


def _good(name: str) -> bytes:
    return f'"""module {name}"""\nVALUE = {name!r}\n'.encode()


def _blobs(files: dict[str, bytes]) -> dict[str, str]:
    return {
        path: mod.git_blob_sha1(payload)
        for path, payload in files.items()
        if mod._allowed_path(path)
    }


def test_materializes_only_allowlisted_source_and_keeps_zero_credit() -> None:
    files = {
        "Lib/argparse.py": _good("argparse"),
        "Lib/json/decoder.py": _good("decoder"),
        "Lib/test/test_argparse.py": _good("test"),
        "Lib/tomllib/_parser.py": _good("tomllib"),
        "README.rst": b"not code\n",
    }
    data = _archive(files)
    rows, report = mod.materialize_archive_bytes(
        data,
        expected_blobs=_blobs(files),
        byte_cap=10_000,
    )
    assert [row["source_path"] for row in rows] == [
        "Lib/argparse.py",
        "Lib/json/decoder.py",
    ]
    assert all(row["training_eligible"] is False for row in rows)
    assert report["truth_boundary"]["training_authorized_bytes"] == 0
    assert report["truth_boundary"]["canonical_capacity_credit_bytes"] == 0
    assert report["truth_boundary"]["unique_causal_loss_positions"] == 0
    assert report["truth_boundary"]["model_training_executed"] is False
    assert report["rights"]["selected_objects_rights_admitted"] is False
    assert report["rights"]["file_level_incorporated_license_retest_required"] is True


def test_exact_empty_allowed_blob_is_verified_then_skipped() -> None:
    files = {
        "Lib/argparse.py": _good("argparse"),
        "Lib/email/mime/__init__.py": b"",
    }
    rows, report = mod.materialize_archive_bytes(
        _archive(files),
        expected_blobs=_blobs(files),
        byte_cap=10_000,
    )
    assert [row["source_path"] for row in rows] == ["Lib/argparse.py"]
    assert report["selection"]["eligible_archive_objects"] == 2
    assert report["selection"]["selected_objects"] == 1


def test_empty_substitution_for_nonempty_pinned_blob_fails_closed() -> None:
    files = {"Lib/argparse.py": b""}
    with pytest.raises(mod.CandidateError, match="Git blob does not match"):
        mod.materialize_archive_bytes(
            _archive(files),
            expected_blobs={"Lib/argparse.py": mod.git_blob_sha1(_good("argparse"))},
        )


def test_deterministic_identity_and_byte_cap() -> None:
    files = {
        "Lib/argparse.py": _good("a"),
        "Lib/ast.py": _good("b"),
        "Lib/base64.py": _good("c"),
    }
    one = _archive(files)
    two = _archive(dict(reversed(list(files.items()))))
    expected = _blobs(files)
    rows_a, report_a = mod.materialize_archive_bytes(
        one, expected_blobs=expected, byte_cap=60
    )
    rows_b, report_b = mod.materialize_archive_bytes(
        two, expected_blobs=expected, byte_cap=60
    )
    assert rows_a == rows_b
    assert report_a == report_b
    assert report_a["selection"]["selected_bytes"] <= 60


def test_rejects_selected_symlink() -> None:
    data = _archive(
        {"Lib/ast.py": _good("ast")},
        links={"Lib/argparse.py": "../../outside"},
    )
    with pytest.raises(mod.CandidateError, match="links forbidden"):
        mod.materialize_archive_bytes(
            data,
            expected_blobs={
                "Lib/ast.py": mod.git_blob_sha1(_good("ast")),
                "Lib/argparse.py": "0" * 40,
            },
        )


def test_rejects_unsafe_path_even_if_not_selected() -> None:
    data = _archive(
        {
            "Lib/ast.py": _good("ast"),
            "../escape": b"x",
        }
    )
    with pytest.raises(mod.CandidateError, match="unsafe archive path"):
        mod.materialize_archive_bytes(
            data,
            expected_blobs={"Lib/ast.py": mod.git_blob_sha1(_good("ast"))},
        )


def test_quarantines_policy_rejected_object_when_safe_object_survives() -> None:
    files = {
        "Lib/argparse.py": _good("safe"),
        "Lib/ast.py": b"# generated by tool\nVALUE = 1\n",
    }
    rows, report = mod.materialize_archive_bytes(
        _archive(files),
        expected_blobs=_blobs(files),
        byte_cap=10_000,
    )
    assert [row["source_path"] for row in rows] == ["Lib/argparse.py"]
    assert report["selection"]["rejected_by_source_policy"] == 1
    assert report["selection"]["selected_objects"] == 1
    assert report["truth_boundary"]["canonical_capacity_credit_bytes"] == 0
    assert report["rights"]["selected_objects_rights_admitted"] is False


@pytest.mark.parametrize(
    ("path", "payload", "message"),
    [
        ("Lib/argparse.py", b"# generated by tool\nVALUE=1\n", "generated-code"),
        ("Lib/argparse.py", b"if True print('x')\n", "parse/compile"),
        (
            "Lib/argparse.py",
            b'api_key = "abcdefghijklmnopqrstuvwxyz123456"\n',
            "secret-like",
        ),
        (
            "Lib/argparse.py",
            b"# SPDX-License-Identifier: MIT\nVALUE=1\n",
            "non-PSF SPDX",
        ),
    ],
)
def test_rejects_unsafe_source(path: str, payload: bytes, message: str) -> None:
    files = {path: payload}
    data = _archive(files)
    with pytest.raises(mod.CandidateError, match=message):
        mod.materialize_archive_bytes(data, expected_blobs=_blobs(files))


def test_rejects_exact_duplicate_payloads() -> None:
    payload = _good("same")
    files = {"Lib/argparse.py": payload, "Lib/ast.py": payload}
    data = _archive(files)
    with pytest.raises(mod.CandidateError, match="exact duplicate source payload"):
        mod.materialize_archive_bytes(data, expected_blobs=_blobs(files))


def test_rejects_invalid_cap() -> None:
    files = {"Lib/argparse.py": _good("argparse")}
    data = _archive(files)
    with pytest.raises(mod.CandidateError, match="byte_cap"):
        mod.materialize_archive_bytes(
            data,
            expected_blobs=_blobs(files),
            byte_cap=mod.MAX_SELECTED_BYTES + 1,
        )


def test_rejects_git_blob_substitution() -> None:
    files = {"Lib/argparse.py": _good("argparse")}
    data = _archive(files)
    with pytest.raises(mod.CandidateError, match="Git blob does not match"):
        mod.materialize_archive_bytes(
            data,
            expected_blobs={"Lib/argparse.py": "0" * 40},
        )


def test_rejects_wrong_archive_root() -> None:
    payload = _good("argparse")
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        info = tarfile.TarInfo("cpython-wrong/Lib/argparse.py")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    with pytest.raises(mod.CandidateError, match="archive root"):
        mod.materialize_archive_bytes(
            output.getvalue(),
            expected_blobs={"Lib/argparse.py": mod.git_blob_sha1(payload)},
        )


def test_report_self_hash_is_stable() -> None:
    files = {"Lib/argparse.py": _good("argparse")}
    data = _archive(files)
    _, report = mod.materialize_archive_bytes(data, expected_blobs=_blobs(files))
    claimed = report.pop("report_identity_sha256")
    assert mod.sha256_bytes(mod.canonical_json_bytes(report)) == claimed
