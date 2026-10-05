"""Fail-closed binding for the current Rada laws normalize/QP replay.

This module does not admit corpus bytes. It pins the exact source snapshot,
normalization/QP mechanics, and successful GitHub re-execution needed by the
next current global cross-source dedup consumer.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "12-6.d03-rada-current-snapshot-replay-authority.v1"
STATUS = "PASS_REPRODUCIBLE_MECHANICS_ZERO_CREDIT_READY_FOR_GLOBAL_DEDUP_CONSUMER"
AUTHORITY_IDENTITY_SHA256 = (
    "8e6e3a37b49dc26f3de3f863d5535f49ae6c457330c91bebd05bd2b7d51bbdc9"
)
CANONICAL_AUTHORITY_FILE_SHA256 = (
    "862488ab49bffcfa0f1953a271021d079e478dc3627ae116f009571e28fdebab"
)
MAX_AUTHORITY_BYTES = 65_536
MAX_JSON_DEPTH = 32
MAX_JSON_NODES = 4_096

_EXPECTED_CORE: dict[str, Any] = {
    "schema_version": SCHEMA,
    "status": STATUS,
    "source": {
        "family": "ua.rada.open-data.laws-texts",
        "retained_artifact_id": 11281540677,
        "retained_artifact_run_id": 37136885886,
        "retained_artifact_outer_sha256": (
            "2d0351a7ae6bbcd6ca5bc5d84c115717a467dbad51bd9b17e71072aed1827aeb"
        ),
        "source_archive_bytes": 46774786,
        "source_archive_sha256": (
            "9f937b3283dd2d9508a0a92442fc007e7e6d373a17d3d50fa428e324cb2fa290"
        ),
        "canonical_entry_count": 3055,
        "canonical_raw_bytes": 353891824,
        "entry_identity_sha256": (
            "1fcc222a959d1dfc24e2b23b71a5412b1050e22a5004cbd36f0dc79998898cc0"
        ),
        "qualification_identity_sha256": (
            "ced7b9370925ba0aa17952efd2cc5ff917601d44fa301b8449ba135644e5a8b3"
        ),
        "successor_pin_identity_sha256": (
            "dcda0321145c03160cef435bc3d1ef5ac668c3415bc013750ed38cd2d891561e"
        ),
    },
    "mechanics": {
        "product_parent_sha": "019944d5fe12334791f05f1232d13de4a12e37d3",
        "normalizer_path": "tools/normalize_d03_rada_bulk_html.py",
        "normalizer_git_blob_sha1": "55ef5f5e7f09042489fa2642d099aedffaddb740",
        "normalization_config_path": "configs/data/d03_rada_bulk_normalization_v1.json",
        "normalization_config_git_blob_sha1": (
            "3e0ca093762242f18fadb21002946b4b733afea3"
        ),
        "qp_reference_commit": "43a7d62854255da4426d6a3dc4d5ad9b1e4898e9",
        "qp_tool_path": "tools/filter_d03_rada_bulk_quality_privacy.py",
        "qp_tool_git_blob_sha1": "69491874e60a1b846d683fff3c10214a869f18ff",
        "qp_config_path": "configs/data/d03_rada_bulk_quality_privacy_v1.json",
        "qp_config_git_blob_sha1": "e1798398a87893c5561afc6d9cd4f3d8fdce6d8b",
    },
    "normalization": {
        "record_count": 3055,
        "nonempty_record_count": 3055,
        "payload_utf8_bytes": 211492947,
        "jsonl_file_bytes": 214699202,
        "jsonl_sha256": (
            "da13b3cc036b19e44e212f3816a66ed261a743c41ed8545076c35eeb706ad511"
        ),
        "inventory_sha256": (
            "247aa5dde026f26aae4bc4587c7b0c9fb9496588d25bf8d58c79a198afc5fd76"
        ),
        "source_encoding_counts": {"utf-8": 887, "windows-1251": 2168},
    },
    "quality_privacy": {
        "total_chunks": 113252,
        "accepted_chunk_count": 101733,
        "rejected_chunk_count": 11519,
        "accepted_payload_utf8_bytes": 192393157,
        "accepted_jsonl_file_bytes": 224897989,
        "accepted_jsonl_sha256": (
            "8d1343708b3ce1747d32c3b551d6fb8ab7123be5b37f74fff3c457b6439159a7"
        ),
        "accepted_inventory_sha256": (
            "468a3132819527e1ce2b3aa375a55a15d55b7af53e66bf06ae1e5a69daf73e7a"
        ),
        "accepted_source_encoding_counts": {
            "utf-8": 80134,
            "windows-1251": 21599,
        },
        "exact_duplicate_payload_hashes_observed_not_removed": 765,
        "zero_chunk_parent_count": 0,
    },
    "github_reexecution": {
        "pr": 2806,
        "head_sha": "839be52926e07d69e8d5b9875101c65da1e475b0",
        "workflow_run_id": 37367953460,
        "job_id": 111957620578,
        "conclusion": "success",
        "artifact_id": 11372000691,
        "artifact_digest_sha256": (
            "c74bb46d81a816c9bacfc27773b14f6348cf8bf2d6c473e72f2cccd7053d092f"
        ),
        "local_measurement_identity_sha256": (
            "5fbcb694e005b40db37232132d84fefdaa2fe428700368d4d63365ef3886bf38"
        ),
        "fresh_replay_count": 2,
        "replays_byte_identical": True,
    },
    "consumer_gate": {
        "allowed_next_consumer": "CURRENT_GLOBAL_CROSS_SOURCE_DEDUP_ONLY",
        "production_qp_authority_established": False,
        "current_source_global_dedup_executed": False,
        "current_source_eval_decontamination_executed": False,
    },
    "truth_boundary": {
        "canonical_capacity_credited": 0,
        "training_authorized_bytes": 0,
        "authorized_optimized_target_exposure": 0,
        "tokenizer_fit_authorized": False,
        "training_executed": False,
        "learned_weights_created": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
    },
}


class RadaCurrentSnapshotAuthorityError(ValueError):
    """Raised when current Rada replay authority fails closed."""


@dataclass(frozen=True)
class RadaAcceptedPayloadBinding:
    """Immutable exact input contract for the next global-dedup consumer."""

    source_family: str
    source_archive_sha256: str
    source_archive_bytes: int
    product_parent_sha: str
    normalizer_git_blob_sha1: str
    normalization_config_git_blob_sha1: str
    qp_reference_commit: str
    qp_tool_git_blob_sha1: str
    qp_config_git_blob_sha1: str
    accepted_chunk_count: int
    accepted_payload_utf8_bytes: int
    accepted_jsonl_file_bytes: int
    accepted_jsonl_sha256: str
    accepted_inventory_sha256: str
    accepted_source_encoding_counts: tuple[tuple[str, int], ...]
    exact_duplicate_payload_hashes_observed_not_removed: int
    execution_head_sha: str
    workflow_run_id: int
    artifact_digest_sha256: str


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RadaCurrentSnapshotAuthorityError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _bounded_int(token: str) -> int:
    digits = token[1:] if token.startswith("-") else token
    if not digits or len(digits) > 64:
        raise RadaCurrentSnapshotAuthorityError("JSON integer token exceeds bound")
    return int(token)


def _reject_float(token: str) -> float:
    del token
    raise RadaCurrentSnapshotAuthorityError("JSON floats are not permitted")


def _reject_constant(token: str) -> float:
    del token
    raise RadaCurrentSnapshotAuthorityError(
        "non-finite JSON constants are not permitted"
    )


def _validate_shape(
    value: Any,
    *,
    depth: int = 0,
    counter: list[int] | None = None,
) -> None:
    if counter is None:
        counter = [0]
    counter[0] += 1
    if counter[0] > MAX_JSON_NODES:
        raise RadaCurrentSnapshotAuthorityError("authority JSON node limit exceeded")
    if depth > MAX_JSON_DEPTH:
        raise RadaCurrentSnapshotAuthorityError(
            "authority JSON nesting limit exceeded"
        )
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise RadaCurrentSnapshotAuthorityError(
                    "authority JSON key is not text"
                )
            _validate_shape(item, depth=depth + 1, counter=counter)
    elif type(value) is list:
        for item in value:
            _validate_shape(item, depth=depth + 1, counter=counter)


def _strict_json_bytes(raw: bytes) -> dict[str, Any]:
    if type(raw) is not bytes or not raw:
        raise RadaCurrentSnapshotAuthorityError("authority bytes are empty")
    if len(raw) > MAX_AUTHORITY_BYTES:
        raise RadaCurrentSnapshotAuthorityError("authority file exceeds byte limit")
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_int=_bounded_int,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except RadaCurrentSnapshotAuthorityError:
        raise
    except RecursionError as exc:
        raise RadaCurrentSnapshotAuthorityError(
            "authority JSON nesting limit exceeded"
        ) from exc
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RadaCurrentSnapshotAuthorityError(
            "authority is not strict UTF-8 JSON"
        ) from exc
    _validate_shape(value)
    if type(value) is not dict:
        raise RadaCurrentSnapshotAuthorityError(
            "authority JSON root must be object"
        )
    return value


def _exact_tree(observed: Any, expected: Any, *, path: str) -> None:
    if type(observed) is not type(expected):
        raise RadaCurrentSnapshotAuthorityError(f"authority type drift: {path}")
    if type(expected) is dict:
        if set(observed) != set(expected):
            raise RadaCurrentSnapshotAuthorityError(f"authority schema drift: {path}")
        for key, expected_value in expected.items():
            _exact_tree(observed[key], expected_value, path=f"{path}.{key}")
        return
    if observed != expected:
        raise RadaCurrentSnapshotAuthorityError(f"authority value drift: {path}")


def validate_current_rada_replay_authority(
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the exact zero-credit replay authority and return a detached copy."""

    if type(authority) is not dict:
        raise RadaCurrentSnapshotAuthorityError("authority must be exact object")
    expected_keys = set(_EXPECTED_CORE) | {"authority_identity_sha256"}
    if set(authority) != expected_keys:
        raise RadaCurrentSnapshotAuthorityError("authority root schema drift")
    identity = authority.get("authority_identity_sha256")
    if type(identity) is not str or identity != AUTHORITY_IDENTITY_SHA256:
        raise RadaCurrentSnapshotAuthorityError("authority identity drift")
    core = {key: authority[key] for key in _EXPECTED_CORE}
    _exact_tree(core, _EXPECTED_CORE, path="authority")
    if _sha256(_canonical(core)) != identity:
        raise RadaCurrentSnapshotAuthorityError("authority self-hash mismatch")
    return json.loads(json.dumps(authority, ensure_ascii=False))


def _read_regular_bytes(path: Path, *, label: str) -> bytes:
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise RadaCurrentSnapshotAuthorityError(
            f"cannot stat {label}: {path}"
        ) from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise RadaCurrentSnapshotAuthorityError(
            f"{label} is not a regular file: {path}"
        )
    try:
        return path.read_bytes()
    except OSError as exc:
        raise RadaCurrentSnapshotAuthorityError(
            f"cannot read {label}: {path}"
        ) from exc


def load_current_rada_replay_authority(
    path: Path,
    *,
    expected_raw_sha256: str | None = None,
) -> dict[str, Any]:
    """Load one regular authority file, optionally binding its exact transport bytes."""

    raw = _read_regular_bytes(path, label="Rada replay authority")
    if len(raw) > MAX_AUTHORITY_BYTES:
        raise RadaCurrentSnapshotAuthorityError("authority file exceeds byte limit")
    if expected_raw_sha256 is not None:
        if (
            type(expected_raw_sha256) is not str
            or len(expected_raw_sha256) != 64
            or any(char not in "0123456789abcdef" for char in expected_raw_sha256)
        ):
            raise RadaCurrentSnapshotAuthorityError(
                "expected raw SHA-256 is malformed"
            )
        if _sha256(raw) != expected_raw_sha256:
            raise RadaCurrentSnapshotAuthorityError("authority raw SHA-256 drift")
    return validate_current_rada_replay_authority(_strict_json_bytes(raw))


def _git_blob_sha1(payload: bytes) -> str:
    prefix = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(prefix + payload).hexdigest()  # noqa: S324 - Git identity


def verify_current_product_mechanics(
    authority: Mapping[str, Any],
    *,
    repository_root: Path,
) -> None:
    """Bind the current-product normalizer/config bytes before replay consumption."""

    validated = validate_current_rada_replay_authority(authority)
    mechanics = validated["mechanics"]
    for path_key, blob_key in (
        ("normalizer_path", "normalizer_git_blob_sha1"),
        ("normalization_config_path", "normalization_config_git_blob_sha1"),
    ):
        relative = mechanics[path_key]
        payload = _read_regular_bytes(
            repository_root / relative,
            label=f"current Rada mechanic {relative}",
        )
        if _git_blob_sha1(payload) != mechanics[blob_key]:
            raise RadaCurrentSnapshotAuthorityError(
                f"current Rada mechanic Git blob drift: {relative}"
            )


def accepted_payload_binding(
    authority: Mapping[str, Any],
) -> RadaAcceptedPayloadBinding:
    """Expose only immutable fields required to construct the next dedup consumer."""

    validated = validate_current_rada_replay_authority(authority)
    source = validated["source"]
    mechanics = validated["mechanics"]
    qp = validated["quality_privacy"]
    execution = validated["github_reexecution"]
    return RadaAcceptedPayloadBinding(
        source_family=source["family"],
        source_archive_sha256=source["source_archive_sha256"],
        source_archive_bytes=source["source_archive_bytes"],
        product_parent_sha=mechanics["product_parent_sha"],
        normalizer_git_blob_sha1=mechanics["normalizer_git_blob_sha1"],
        normalization_config_git_blob_sha1=mechanics[
            "normalization_config_git_blob_sha1"
        ],
        qp_reference_commit=mechanics["qp_reference_commit"],
        qp_tool_git_blob_sha1=mechanics["qp_tool_git_blob_sha1"],
        qp_config_git_blob_sha1=mechanics["qp_config_git_blob_sha1"],
        accepted_chunk_count=qp["accepted_chunk_count"],
        accepted_payload_utf8_bytes=qp["accepted_payload_utf8_bytes"],
        accepted_jsonl_file_bytes=qp["accepted_jsonl_file_bytes"],
        accepted_jsonl_sha256=qp["accepted_jsonl_sha256"],
        accepted_inventory_sha256=qp["accepted_inventory_sha256"],
        accepted_source_encoding_counts=tuple(
            sorted(qp["accepted_source_encoding_counts"].items())
        ),
        exact_duplicate_payload_hashes_observed_not_removed=qp[
            "exact_duplicate_payload_hashes_observed_not_removed"
        ],
        execution_head_sha=execution["head_sha"],
        workflow_run_id=execution["workflow_run_id"],
        artifact_digest_sha256=execution["artifact_digest_sha256"],
    )
