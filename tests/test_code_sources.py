from __future__ import annotations

from twelve_six.data.code_sources import (
    CodeSourceCandidate,
    LicenseObservation,
    filter_code_file,
    ingest_code_file,
    near_duplicate_analysis,
    strict_utf8_identity,
)


def candidate() -> CodeSourceCandidate:
    revision = "a" * 40
    return CodeSourceCandidate(
        source_id="github:example/repo",
        repository="example/repo",
        revision=revision,
        source_url="https://github.com/example/repo",
        d03_source_version=revision,
        license=LicenseObservation(
            spdx_id="MIT",
            path="LICENSE",
            git_blob_sha1="b" * 40,
            text_sha256="c" * 64,
        ),
    )


def test_strict_utf8_preserves_unicode_newlines_indentation_and_comments() -> None:
    payload = "\n\t# keep comment\r\nvalue = 'Ａ'\n\n".encode("utf-8")
    text = strict_utf8_identity(payload)
    assert text.encode("utf-8") == payload
    assert text.startswith("\n\t# keep comment\r\n")
    assert "Ａ" in text
    assert text.endswith("\n\n")


def test_conservative_artifact_filters() -> None:
    assert not filter_code_file("vendor/pkg/a.py", b"print('x')\n").accepted
    assert not filter_code_file("src/generated.py", b"# @generated\nprint('x')\n").accepted
    assert not filter_code_file("src/app.min.js", b"const x=1;\n").accepted
    assert not filter_code_file("src/a.py", b"x=\x00y").accepted
    assert not filter_code_file("assets/logo.png", b"not actually png").accepted


def test_ingestion_keeps_repository_revision_path_license_and_source_hash() -> None:
    payload = b"def add(a, b):\n    # comment stays\n    return a + b\n"
    record = ingest_code_file(
        candidate(),
        path="src/add.py",
        git_blob_sha1="d" * 40,
        payload=payload,
        rights_status="BLOCKED_NOT_IN_D03_APPROVED_REGISTRY",
        training_eligible=False,
    )
    assert record.mechanically_accepted
    assert record.repository == "example/repo"
    assert record.revision == "a" * 40
    assert record.path == "src/add.py"
    assert record.language == "Python"
    assert record.text.encode("utf-8") == payload
    assert record.license_spdx_id == "MIT"
    assert not record.training_eligible


def test_near_duplicate_diagnostic_finds_copy_with_small_identifier_change() -> None:
    base = candidate()
    left = ingest_code_file(
        base,
        path="src/a.py",
        git_blob_sha1="d" * 40,
        payload=(b"def transform(value):\n    result = value + 1\n    return result\n" * 8),
        rights_status="APPROVED_FOR_TRAINING",
        training_eligible=True,
    )
    right = ingest_code_file(
        base,
        path="src/b.py",
        git_blob_sha1="e" * 40,
        payload=(b"def transform(value):\n    result = value + 1\n    return result\n" * 7)
        + b"# retained comment\n",
        rights_status="APPROVED_FOR_TRAINING",
        training_eligible=True,
    )
    report = near_duplicate_analysis([left, right], threshold=0.80)
    assert report["pair_count"] == 1
    assert not report["semantic_cleanliness_claimed"]
