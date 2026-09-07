"""Fresh-process two-clean proof for canonical post-pack loss materialization.

The proof deliberately reuses :mod:`twelve_six.packing.loss_materialization`.
It does not tokenize, pack, or assign loss spans independently. Raw document
text exists only in the ephemeral input packet and is never copied into the
returned durable proof.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from twelve_six.tokenization import ByteTokenizer

from .loss_materialization import (
    MATERIALIZATION_SCHEMA,
    LossMaterializationDocument,
    build_postpack_loss_materialization,
)

INPUT_SCHEMA = "12-6.postpack-two-clean-input.v1"
PROOF_SCHEMA = "12-6.postpack-two-clean-proof.v1"
_REQUIRED_BINDINGS = (
    "normalization",
    "evaluation_reservations",
    "dedup",
    "split",
    "packing",
)
_HEX = frozenset("0123456789abcdef")
_DOCUMENT_KEYS = {
    "document_id",
    "text",
    "source_id",
    "language",
    "modality",
    "family_id",
    "normalized_payload_sha256",
    "source_bytes",
    "split",
    "dedup_cluster_id",
    "retained_after_dedup",
    "evaluation_reserved",
    "reserved_target_ranges",
}


class TwoCleanBuildError(ValueError):
    """Raised when independent clean materializations cannot be trusted."""


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_obj(value: Any) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _require_sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in _HEX for character in value)
    ):
        raise TwoCleanBuildError(f"{field} must be exact lowercase SHA-256")
    return value


def _normalize_bindings(value: Mapping[str, str]) -> dict[str, str]:
    if set(value) != set(_REQUIRED_BINDINGS):
        raise TwoCleanBuildError("stage_bindings contain an unexpected or missing stage")
    return {
        name: _require_sha256(value[name], f"stage_bindings.{name}")
        for name in _REQUIRED_BINDINGS
    }


def _document_row(document: LossMaterializationDocument) -> dict[str, Any]:
    return {
        "document_id": document.document_id,
        "text": document.text,
        "source_id": document.source_id,
        "language": document.language,
        "modality": document.modality,
        "family_id": document.family_id,
        "normalized_payload_sha256": document.normalized_payload_sha256,
        "source_bytes": document.source_bytes,
        "split": document.split,
        "dedup_cluster_id": document.dedup_cluster_id,
        "retained_after_dedup": document.retained_after_dedup,
        "evaluation_reserved": document.evaluation_reserved,
        "reserved_target_ranges": [list(item) for item in document.reserved_target_ranges],
    }


def make_input_packet(
    documents: Sequence[LossMaterializationDocument],
    *,
    terminal_corpus_authority_identity_sha256: str,
    stage_bindings: Mapping[str, str],
) -> dict[str, Any]:
    """Freeze one ephemeral build request and give it a deterministic identity."""
    if not documents:
        raise TwoCleanBuildError("documents must not be empty")
    ordered = sorted(documents, key=lambda item: item.document_id)
    if len({item.document_id for item in ordered}) != len(ordered):
        raise TwoCleanBuildError("document_id values must be unique")
    packet: dict[str, Any] = {
        "schema_version": INPUT_SCHEMA,
        "terminal_corpus_authority_identity_sha256": _require_sha256(
            terminal_corpus_authority_identity_sha256,
            "terminal_corpus_authority_identity_sha256",
        ),
        "stage_bindings": _normalize_bindings(stage_bindings),
        "documents": [_document_row(item) for item in ordered],
        "claim_boundary": {
            "ephemeral_input_contains_source_text": True,
            "durable_proof_contains_source_text": False,
            "authorizes_training": False,
            "authorizes_paid_compute": False,
        },
    }
    packet["input_packet_identity_sha256"] = _sha256_obj(packet)
    return packet


def _verify_input_packet(
    packet: Mapping[str, Any], *, expected_identity_sha256: str
) -> dict[str, Any]:
    expected = _require_sha256(expected_identity_sha256, "expected_input_packet_identity_sha256")
    value = dict(packet)
    if value.get("schema_version") != INPUT_SCHEMA:
        raise TwoCleanBuildError("unexpected input packet schema")
    observed = _require_sha256(
        value.get("input_packet_identity_sha256"), "input_packet_identity_sha256"
    )
    body = dict(value)
    body.pop("input_packet_identity_sha256", None)
    if _sha256_obj(body) != observed:
        raise TwoCleanBuildError("input packet self-identity mismatch")
    if observed != expected:
        raise TwoCleanBuildError("input packet does not match independently expected identity")
    _require_sha256(
        value.get("terminal_corpus_authority_identity_sha256"),
        "terminal_corpus_authority_identity_sha256",
    )
    bindings = value.get("stage_bindings")
    if not isinstance(bindings, Mapping):
        raise TwoCleanBuildError("stage_bindings must be an object")
    _normalize_bindings(bindings)
    boundary = value.get("claim_boundary")
    if boundary != {
        "ephemeral_input_contains_source_text": True,
        "durable_proof_contains_source_text": False,
        "authorizes_training": False,
        "authorizes_paid_compute": False,
    }:
        raise TwoCleanBuildError("input claim boundary drift")
    rows = value.get("documents")
    if not isinstance(rows, list) or not rows:
        raise TwoCleanBuildError("documents must be a non-empty list")
    return value


def _document_from_row(row: Mapping[str, Any]) -> LossMaterializationDocument:
    if set(row) != _DOCUMENT_KEYS:
        raise TwoCleanBuildError("document row has unexpected or missing fields")
    ranges = row["reserved_target_ranges"]
    if not isinstance(ranges, list):
        raise TwoCleanBuildError("reserved_target_ranges must be a list")
    normalized_ranges: list[tuple[int, int]] = []
    for item in ranges:
        if not isinstance(item, list) or len(item) != 2:
            raise TwoCleanBuildError("reserved_target_ranges entries must be two-item lists")
        normalized_ranges.append((item[0], item[1]))
    return LossMaterializationDocument(
        document_id=row["document_id"],
        text=row["text"],
        source_id=row["source_id"],
        language=row["language"],
        modality=row["modality"],
        family_id=row["family_id"],
        normalized_payload_sha256=row["normalized_payload_sha256"],
        source_bytes=row["source_bytes"],
        split=row["split"],
        dedup_cluster_id=row["dedup_cluster_id"],
        retained_after_dedup=row["retained_after_dedup"],
        evaluation_reserved=row["evaluation_reserved"],
        reserved_target_ranges=tuple(normalized_ranges),
    )


def _build_one(packet: Mapping[str, Any]) -> dict[str, Any]:
    rows = packet["documents"]
    documents = tuple(_document_from_row(row) for row in rows)
    return build_postpack_loss_materialization(
        documents,
        ByteTokenizer(),
        terminal_corpus_authority_identity_sha256=(
            packet["terminal_corpus_authority_identity_sha256"]
        ),
        stage_bindings=packet["stage_bindings"],
    )


def _verify_materialization_bytes(
    payload: bytes,
    *,
    expected_corpus_identity_sha256: str,
    expected_stage_bindings: Mapping[str, str],
) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TwoCleanBuildError("materialization is not canonical UTF-8 JSON") from exc
    if not isinstance(value, dict) or value.get("schema_version") != MATERIALIZATION_SCHEMA:
        raise TwoCleanBuildError("unexpected post-pack materialization schema")
    observed_identity = _require_sha256(
        value.get("materialization_identity_sha256"), "materialization_identity_sha256"
    )
    body = dict(value)
    body.pop("materialization_identity_sha256", None)
    if _sha256_obj(body) != observed_identity:
        raise TwoCleanBuildError("post-pack materialization self-identity mismatch")
    if _canonical_json_bytes(value) != payload:
        raise TwoCleanBuildError("post-pack materialization bytes are not canonical")
    expected_corpus = _require_sha256(
        expected_corpus_identity_sha256, "expected_corpus_identity_sha256"
    )
    if value.get("terminal_corpus_authority_identity_sha256") != expected_corpus:
        raise TwoCleanBuildError("terminal corpus authority drifted during clean build")
    expected_bindings = _normalize_bindings(expected_stage_bindings)
    if value.get("stage_bindings") != expected_bindings:
        raise TwoCleanBuildError("stage binding drifted during clean build")
    return value


def compare_clean_build_bytes(
    first: bytes,
    second: bytes,
    *,
    expected_corpus_identity_sha256: str,
    expected_stage_bindings: Mapping[str, str],
) -> dict[str, Any]:
    """Require literal equality of two independently produced canonical outputs."""
    first_value = _verify_materialization_bytes(
        first,
        expected_corpus_identity_sha256=expected_corpus_identity_sha256,
        expected_stage_bindings=expected_stage_bindings,
    )
    second_value = _verify_materialization_bytes(
        second,
        expected_corpus_identity_sha256=expected_corpus_identity_sha256,
        expected_stage_bindings=expected_stage_bindings,
    )
    if first != second:
        raise TwoCleanBuildError("independent post-pack materialization bytes differ")
    if first_value["materialization_identity_sha256"] != second_value[
        "materialization_identity_sha256"
    ]:
        raise TwoCleanBuildError("independent materialization identities differ")
    return first_value


def _run_child(input_path: Path, output_path: Path, expected_identity: str) -> None:
    packet = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(packet, dict):
        raise TwoCleanBuildError("input packet must be a JSON object")
    verified = _verify_input_packet(packet, expected_identity_sha256=expected_identity)
    output_path.write_bytes(_canonical_json_bytes(_build_one(verified)))


def prove_two_clean_build(
    packet: Mapping[str, Any],
    *,
    expected_input_packet_identity_sha256: str,
    python_executable: str | None = None,
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    """Run two fresh Python processes and emit a text-free equality proof."""
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int):
        raise TwoCleanBuildError("timeout_seconds must be a positive integer")
    if timeout_seconds <= 0:
        raise TwoCleanBuildError("timeout_seconds must be a positive integer")
    verified = _verify_input_packet(
        packet, expected_identity_sha256=expected_input_packet_identity_sha256
    )
    packet_identity = verified["input_packet_identity_sha256"]
    corpus_identity = verified["terminal_corpus_authority_identity_sha256"]
    bindings = verified["stage_bindings"]
    executable = python_executable or sys.executable
    if not executable:
        raise TwoCleanBuildError("python executable is unavailable")

    with tempfile.TemporaryDirectory(prefix="twelve-six-two-clean-") as directory:
        root = Path(directory)
        input_path = root / "input.json"
        input_path.write_bytes(_canonical_json_bytes(verified))
        outputs: list[bytes] = []
        output_hashes: list[str] = []
        for name in ("clean-a", "clean-b"):
            work = root / name
            work.mkdir()
            output_path = work / "postpack.json"
            command = [
                executable,
                "-m",
                "twelve_six.packing.two_clean_build",
                "--child",
                str(input_path),
                str(output_path),
                "--expected-input-identity",
                packet_identity,
            ]
            env = dict(os.environ)
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            completed = subprocess.run(
                command,
                cwd=work,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout).strip()[-1000:]
                raise TwoCleanBuildError(f"{name} fresh process failed: {detail}")
            if not output_path.is_file():
                raise TwoCleanBuildError(f"{name} did not produce postpack.json")
            payload = output_path.read_bytes()
            outputs.append(payload)
            output_hashes.append(_sha256_bytes(payload))

    materialization = compare_clean_build_bytes(
        outputs[0],
        outputs[1],
        expected_corpus_identity_sha256=corpus_identity,
        expected_stage_bindings=bindings,
    )
    proof: dict[str, Any] = {
        "schema_version": PROOF_SCHEMA,
        "input_packet_identity_sha256": packet_identity,
        "terminal_corpus_authority_identity_sha256": corpus_identity,
        "stage_bindings": dict(bindings),
        "fresh_process_count": 2,
        "byte_identical": True,
        "build_a_sha256": output_hashes[0],
        "build_b_sha256": output_hashes[1],
        "materialization_identity_sha256": materialization[
            "materialization_identity_sha256"
        ],
        "claim_boundary": {
            "contains_source_text": False,
            "authorizes_training": False,
            "authorizes_paid_compute": False,
            "creates_positive_unique_loss_authority": False,
        },
    }
    proof["proof_identity_sha256"] = _sha256_obj(proof)
    return proof


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", nargs=2, metavar=("INPUT", "OUTPUT"))
    parser.add_argument("--expected-input-identity")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.child is None or args.expected_input_identity is None:
        raise TwoCleanBuildError(
            "module CLI is child-only; use prove_two_clean_build() from the trusted parent"
        )
    _run_child(
        Path(args.child[0]),
        Path(args.child[1]),
        args.expected_input_identity,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
