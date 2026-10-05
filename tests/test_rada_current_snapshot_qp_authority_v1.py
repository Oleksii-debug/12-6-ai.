from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from twelve_six.data import rada_current_snapshot_qp_authority_v1 as authority


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY_PATH = (
    ROOT / "evidence/d03_rada_bulk/current_snapshot_github_replay_authority_v1.json"
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _rehashed(value: dict[str, object]) -> bytes:
    core = dict(value)
    core.pop("authority_identity_sha256", None)
    updated = dict(value)
    updated["authority_identity_sha256"] = hashlib.sha256(_canonical(core)).hexdigest()
    return (
        json.dumps(
            updated,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def test_canonical_current_rada_replay_authority_binds_physical_result() -> None:
    raw = AUTHORITY_PATH.read_bytes()
    assert (
        hashlib.sha256(raw).hexdigest()
        == authority.CANONICAL_AUTHORITY_FILE_SHA256
    )
    value = authority.load_current_rada_replay_authority(
        AUTHORITY_PATH,
        expected_raw_sha256=authority.CANONICAL_AUTHORITY_FILE_SHA256,
    )
    binding = authority.accepted_payload_binding(value)

    assert binding.source_family == "ua.rada.open-data.laws-texts"
    assert binding.source_archive_sha256 == (
        "9f937b3283dd2d9508a0a92442fc007e7e6d373a17d3d50fa428e324cb2fa290"
    )
    assert binding.accepted_chunk_count == 101733
    assert binding.accepted_payload_utf8_bytes == 192393157
    assert binding.accepted_jsonl_file_bytes == 224897989
    assert binding.accepted_jsonl_sha256 == (
        "8d1343708b3ce1747d32c3b551d6fb8ab7123be5b37f74fff3c457b6439159a7"
    )
    assert binding.exact_duplicate_payload_hashes_observed_not_removed == 765
    assert binding.workflow_run_id == 37367953460


def test_current_product_normalizer_bytes_match_authority() -> None:
    value = authority.load_current_rada_replay_authority(AUTHORITY_PATH)
    authority.verify_current_product_mechanics(value, repository_root=ROOT)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("github_reexecution", "artifact_id"), True),
        (("quality_privacy", "accepted_payload_utf8_bytes"), 192078166),
        (("quality_privacy", "accepted_chunk_count"), 101559),
        (("quality_privacy", "accepted_jsonl_sha256"), "0" * 64),
        (("consumer_gate", "production_qp_authority_established"), True),
        (("consumer_gate", "current_source_global_dedup_executed"), True),
        (("truth_boundary", "canonical_capacity_credited"), False),
        (("truth_boundary", "training_authorized_bytes"), 1),
    ],
)
def test_rehashed_authority_tampering_fails_closed(
    tmp_path: Path,
    path: tuple[str, str],
    replacement: object,
) -> None:
    value = json.loads(AUTHORITY_PATH.read_text(encoding="utf-8"))
    value[path[0]][path[1]] = replacement
    candidate = tmp_path / "authority.json"
    candidate.write_bytes(_rehashed(value))

    with pytest.raises(authority.RadaCurrentSnapshotAuthorityError):
        authority.load_current_rada_replay_authority(candidate)


def test_authority_rejects_duplicate_json_keys() -> None:
    with pytest.raises(
        authority.RadaCurrentSnapshotAuthorityError,
        match="duplicate JSON key",
    ):
        authority._strict_json_bytes(
            b'{"schema_version":"a","schema_version":"b"}'
        )


@pytest.mark.parametrize("raw", [b'{"x":1.0}', b'{"x":NaN}', b'{"x":1e400}'])
def test_authority_rejects_float_and_nonfinite_json(raw: bytes) -> None:
    with pytest.raises(authority.RadaCurrentSnapshotAuthorityError):
        authority._strict_json_bytes(raw)


def test_authority_rejects_oversized_integer_token() -> None:
    raw = ('{"x":' + "9" * 65 + "}").encode("ascii")
    with pytest.raises(
        authority.RadaCurrentSnapshotAuthorityError,
        match="integer token exceeds bound",
    ):
        authority._strict_json_bytes(raw)


def test_authority_rejects_excessive_json_depth() -> None:
    raw = ('{"x":' + "[" * 40 + "0" + "]" * 40 + "}").encode("ascii")
    with pytest.raises(
        authority.RadaCurrentSnapshotAuthorityError,
        match="nesting limit exceeded",
    ):
        authority._strict_json_bytes(raw)


def test_authority_transport_hash_is_external_binding(tmp_path: Path) -> None:
    candidate = tmp_path / "authority.json"
    candidate.write_bytes(AUTHORITY_PATH.read_bytes())
    with pytest.raises(
        authority.RadaCurrentSnapshotAuthorityError,
        match="raw SHA-256 drift",
    ):
        authority.load_current_rada_replay_authority(
            candidate,
            expected_raw_sha256="0" * 64,
        )


def test_default_loader_rejects_semantically_equivalent_reserialization(
    tmp_path: Path,
) -> None:
    value = json.loads(AUTHORITY_PATH.read_text(encoding="utf-8"))
    compact = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    assert hashlib.sha256(compact).hexdigest() != (
        authority.CANONICAL_AUTHORITY_FILE_SHA256
    )
    candidate = tmp_path / "authority.json"
    candidate.write_bytes(compact)

    with pytest.raises(
        authority.RadaCurrentSnapshotAuthorityError,
        match="raw SHA-256 drift",
    ):
        authority.load_current_rada_replay_authority(candidate)


def test_current_product_mechanic_substitution_fails_closed(tmp_path: Path) -> None:
    value = authority.load_current_rada_replay_authority(AUTHORITY_PATH)
    mechanics = value["mechanics"]
    for key in ("normalizer_path", "normalization_config_path"):
        target = tmp_path / mechanics[key]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / mechanics[key]).read_bytes())
    normalizer = tmp_path / mechanics["normalizer_path"]
    normalizer.write_bytes(normalizer.read_bytes() + b"\n# tampered\n")

    with pytest.raises(
        authority.RadaCurrentSnapshotAuthorityError,
        match="Git blob drift",
    ):
        authority.verify_current_product_mechanics(
            value,
            repository_root=tmp_path,
        )


def test_binding_does_not_widen_scientific_authority() -> None:
    value = authority.load_current_rada_replay_authority(AUTHORITY_PATH)
    assert value["consumer_gate"] == {
        "allowed_next_consumer": "CURRENT_GLOBAL_CROSS_SOURCE_DEDUP_ONLY",
        "production_qp_authority_established": False,
        "current_source_global_dedup_executed": False,
        "current_source_eval_decontamination_executed": False,
    }
    assert value["truth_boundary"] == {
        "canonical_capacity_credited": 0,
        "training_authorized_bytes": 0,
        "authorized_optimized_target_exposure": 0,
        "tokenizer_fit_authorized": False,
        "training_executed": False,
        "learned_weights_created": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
    }
