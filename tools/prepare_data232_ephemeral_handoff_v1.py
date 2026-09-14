"""Prepare verified post-dedup payload rows for current DATA-232 execution.

The retained-inventory logic remains authoritative in
``expanded_postdedup_inventory_v1``.  This module is only an operator carrier: it
checks an independently expected checkout, streams physical payload input into the
canonical validator, serializes the canonical matcher projection deterministically,
and publishes one fresh execution workspace plus a hash-only receipt.
"""
from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from twelve_six.data.expanded_postdedup_inventory_v1 import (
    prepare_ephemeral_data232_rows,
)

RECEIPT_SCHEMA = "12-6.data232-ephemeral-handoff-run-receipt.v1"
TRAINING_RECORDS_NAME = "training_records.jsonl"
TRAINING_HANDOFF_NAME = "training_handoff.json"
RECEIPT_NAME = "execution_receipt.json"
UPSTREAM_MODULE = "src/twelve_six/data/expanded_postdedup_inventory_v1.py"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "carrier_implementation_git_sha",
        "retained_inventory_implementation_git_sha",
        "input_files_sha256",
        "output_files_sha256",
        "postdedup_inventory_identity_sha256",
        "input_survivor_authority_sha256",
        "training_handoff_identity_sha256",
        "retained_source_count",
        "retained_payload_bytes",
        "training_records_file_bytes",
        "payload_match_proven",
        "durable_receipt_hash_only",
        "raw_text_persisted_in_receipt",
        "record_ids_persisted_in_receipt",
        "authorized_optimized_target_exposure",
        "tokenizer_fit_authorized",
        "optimizer_updates_executed_on_real_targets",
        "training_executed",
        "learned_weights_created",
        "final_test_outcomes_read",
        "paid_compute_used",
        "foreign_pretrained_weights",
        "external_llm_or_api_used_for_data_or_intelligence",
        "receipt_identity_sha256",
    }
)


def _canonical_bytes(value: object, *, newline: bool = False) -> bytes:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return payload + (b"\n" if newline else b"")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be lowercase 64-hex SHA-256")
    return value


def _require_git_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or _GIT_SHA_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be lowercase 40-hex Git SHA")
    return value


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _strict_loads(raw: bytes, label: str) -> dict[str, Any]:
    try:
        decoded = raw.decode("utf-8")
        value = json.loads(
            decoded,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid UTF-8 JSON in {label}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain a JSON object")
    return value


def _load_json_with_sha(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    return _strict_loads(raw, str(path)), _sha256_bytes(raw)


class _StreamingPayloadRows(Iterable[dict[str, Any]]):
    """Single-use JSONL iterator that hashes exactly the bytes it validates."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.sha256: str | None = None
        self._iterated = False

    def __iter__(self) -> Iterable[dict[str, Any]]:
        if self._iterated:
            raise RuntimeError("payload JSONL iterator is single-use")
        self._iterated = True
        digest = hashlib.sha256()
        saw_row = False
        with self.path.open("rb") as handle:
            for line_number, raw in enumerate(handle, 1):
                digest.update(raw)
                if not raw.strip():
                    raise ValueError(
                        f"blank JSONL line {line_number} is forbidden: {self.path}"
                    )
                saw_row = True
                yield _strict_loads(
                    raw,
                    f"{self.path}:line {line_number}",
                )
        if not saw_row:
            raise ValueError(f"payload JSONL must be non-empty: {self.path}")
        self.sha256 = digest.hexdigest()


def _git(
    repo_root: Path,
    args: list[str],
    *,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=check,
        capture_output=True,
        text=True,
    )


def require_exact_checkout(
    repo_root: Path,
    *,
    expected_carrier_git_sha: str,
    expected_retained_inventory_git_sha: str,
) -> tuple[str, str]:
    """Require the independently expected carrier and exact PR940 source ancestry."""

    carrier = _require_git_sha(
        expected_carrier_git_sha,
        "expected carrier Git SHA",
    )
    upstream = _require_git_sha(
        expected_retained_inventory_git_sha,
        "expected retained-inventory Git SHA",
    )
    head = _git(repo_root, ["rev-parse", "--verify", "HEAD"], check=True).stdout.strip()
    if head != carrier:
        raise RuntimeError(
            "checked-out carrier head differs from independent expectation: "
            f"expected={carrier} actual={head}"
        )

    unstaged = _git(repo_root, ["diff", "--quiet", "HEAD", "--"])
    staged = _git(repo_root, ["diff", "--cached", "--quiet", "HEAD", "--"])
    if unstaged.returncode not in (0, 1) or staged.returncode not in (0, 1):
        raise RuntimeError("unable to verify tracked working-tree cleanliness")
    if unstaged.returncode != 0 or staged.returncode != 0:
        raise RuntimeError("tracked working tree differs from expected carrier head")

    ancestor = _git(repo_root, ["merge-base", "--is-ancestor", upstream, head])
    if ancestor.returncode == 1:
        raise RuntimeError("retained-inventory authority is not an ancestor of carrier")
    if ancestor.returncode != 0:
        raise RuntimeError("unable to verify retained-inventory ancestry")

    source_diff = _git(repo_root, ["diff", "--quiet", upstream, "--", UPSTREAM_MODULE])
    if source_diff.returncode == 1:
        raise RuntimeError("canonical retained-inventory implementation drifted")
    if source_diff.returncode != 0:
        raise RuntimeError("unable to verify retained-inventory implementation")
    return head, upstream


def build_receipt(
    *,
    carrier_git_sha: str,
    upstream_git_sha: str,
    inventory_file_sha256: str,
    payload_file_sha256: str,
    training_records_sha256: str,
    training_handoff_sha256: str,
    training_records_file_bytes: int,
    inventory: Mapping[str, Any],
    handoff: Mapping[str, Any],
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "carrier_implementation_git_sha": _require_git_sha(
            carrier_git_sha, "carrier_git_sha"
        ),
        "retained_inventory_implementation_git_sha": _require_git_sha(
            upstream_git_sha, "upstream_git_sha"
        ),
        "input_files_sha256": {
            "retained_inventory_json": _require_sha256(
                inventory_file_sha256, "inventory_file_sha256"
            ),
            "payload_rows_jsonl": _require_sha256(
                payload_file_sha256, "payload_file_sha256"
            ),
        },
        "output_files_sha256": {
            TRAINING_RECORDS_NAME: _require_sha256(
                training_records_sha256, "training_records_sha256"
            ),
            TRAINING_HANDOFF_NAME: _require_sha256(
                training_handoff_sha256, "training_handoff_sha256"
            ),
        },
        "postdedup_inventory_identity_sha256": _require_sha256(
            handoff.get("postdedup_inventory_identity_sha256"),
            "postdedup_inventory_identity_sha256",
        ),
        "input_survivor_authority_sha256": _require_sha256(
            handoff.get("input_survivor_authority_sha256"),
            "input_survivor_authority_sha256",
        ),
        "training_handoff_identity_sha256": _require_sha256(
            handoff.get("handoff_identity_sha256"),
            "handoff_identity_sha256",
        ),
        "retained_source_count": handoff.get("retained_source_count"),
        "retained_payload_bytes": inventory.get("retained_payload_bytes"),
        "training_records_file_bytes": training_records_file_bytes,
        "payload_match_proven": True,
        "durable_receipt_hash_only": True,
        "raw_text_persisted_in_receipt": False,
        "record_ids_persisted_in_receipt": False,
        "authorized_optimized_target_exposure": 0,
        "tokenizer_fit_authorized": False,
        "optimizer_updates_executed_on_real_targets": 0,
        "training_executed": False,
        "learned_weights_created": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
        "foreign_pretrained_weights": False,
        "external_llm_or_api_used_for_data_or_intelligence": False,
    }
    for key in ("retained_source_count", "retained_payload_bytes", "training_records_file_bytes"):
        value = receipt[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"receipt {key} must be a positive integer")
    receipt["receipt_identity_sha256"] = _sha256_bytes(_canonical_bytes(receipt))
    return receipt


def verify_receipt(receipt: Mapping[str, Any]) -> None:
    if set(receipt) != _RECEIPT_KEYS:
        raise ValueError("execution receipt key set drift")
    if receipt.get("schema_version") != RECEIPT_SCHEMA:
        raise ValueError("execution receipt schema drift")
    claimed = _require_sha256(
        receipt.get("receipt_identity_sha256"),
        "receipt_identity_sha256",
    )
    body = deepcopy(dict(receipt))
    body.pop("receipt_identity_sha256", None)
    if _sha256_bytes(_canonical_bytes(body)) != claimed:
        raise ValueError("execution receipt self-hash mismatch")
    _require_git_sha(
        receipt.get("carrier_implementation_git_sha"),
        "carrier_implementation_git_sha",
    )
    _require_git_sha(
        receipt.get("retained_inventory_implementation_git_sha"),
        "retained_inventory_implementation_git_sha",
    )
    for group in ("input_files_sha256", "output_files_sha256"):
        values = receipt.get(group)
        if not isinstance(values, Mapping):
            raise TypeError(f"receipt {group} must be an object")
        for name, value in values.items():
            _require_sha256(value, f"{group}.{name}")
    for key in (
        "postdedup_inventory_identity_sha256",
        "input_survivor_authority_sha256",
        "training_handoff_identity_sha256",
    ):
        _require_sha256(receipt.get(key), key)
    if receipt.get("payload_match_proven") is not True:
        raise ValueError("payload identity match is not proven")
    if receipt.get("durable_receipt_hash_only") is not True:
        raise ValueError("receipt is not declared hash-only")
    required_false = (
        "raw_text_persisted_in_receipt",
        "record_ids_persisted_in_receipt",
        "tokenizer_fit_authorized",
        "training_executed",
        "learned_weights_created",
        "final_test_outcomes_read",
        "paid_compute_used",
        "foreign_pretrained_weights",
        "external_llm_or_api_used_for_data_or_intelligence",
    )
    for key in required_false:
        if receipt.get(key) is not False:
            raise ValueError(f"receipt truth boundary widened: {key}")
    for key in (
        "authorized_optimized_target_exposure",
        "optimizer_updates_executed_on_real_targets",
    ):
        value = receipt.get(key)
        if isinstance(value, bool) or value != 0:
            raise ValueError(f"receipt truth boundary widened: {key}")


def _write_training_records(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
) -> tuple[str, int]:
    digest = hashlib.sha256()
    total_bytes = 0
    with path.open("xb") as handle:
        for row in rows:
            raw = _canonical_bytes(row, newline=True)
            handle.write(raw)
            digest.update(raw)
            total_bytes += len(raw)
        handle.flush()
        os.fsync(handle.fileno())
    if total_bytes == 0:
        raise ValueError("canonical training JSONL unexpectedly empty")
    return digest.hexdigest(), total_bytes


def _write_bytes(path: Path, raw: bytes) -> str:
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return _sha256_bytes(raw)


def _rename_directory_no_replace(source: Path, destination: Path) -> None:
    if os.name == "nt":
        os.rename(source, destination)
        return
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
    except AttributeError as exc:
        raise RuntimeError(
            "atomic no-replace directory publication is unsupported"
        ) from exc
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(error_number, os.strerror(error_number), destination)
    if error_number in (errno.ENOSYS, errno.EINVAL):
        raise RuntimeError(
            "atomic no-replace directory publication is unsupported"
        )
    raise OSError(error_number, os.strerror(error_number), destination)


def prepare_and_publish(
    *,
    inventory_path: Path,
    payload_path: Path,
    output_dir: Path,
    expected_inventory_identity_sha256: str,
    carrier_git_sha: str,
    upstream_git_sha: str,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"output workspace already exists: {output_dir}")
    expected_inventory = _require_sha256(
        expected_inventory_identity_sha256,
        "expected_inventory_identity_sha256",
    )
    inventory, inventory_file_sha256 = _load_json_with_sha(inventory_path)
    payload_rows = _StreamingPayloadRows(payload_path)
    training_rows, handoff = prepare_ephemeral_data232_rows(
        inventory,
        payload_rows,
        expected_inventory_identity_sha256=expected_inventory,
    )
    if payload_rows.sha256 is None:
        raise RuntimeError("payload file hash was not finalized")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        training_sha256, training_file_bytes = _write_training_records(
            temporary / TRAINING_RECORDS_NAME,
            training_rows,
        )
        handoff_bytes = _canonical_bytes(handoff, newline=True)
        handoff_file_sha256 = _write_bytes(
            temporary / TRAINING_HANDOFF_NAME,
            handoff_bytes,
        )
        receipt = build_receipt(
            carrier_git_sha=carrier_git_sha,
            upstream_git_sha=upstream_git_sha,
            inventory_file_sha256=inventory_file_sha256,
            payload_file_sha256=payload_rows.sha256,
            training_records_sha256=training_sha256,
            training_handoff_sha256=handoff_file_sha256,
            training_records_file_bytes=training_file_bytes,
            inventory=inventory,
            handoff=handoff,
        )
        verify_receipt(receipt)
        _write_bytes(
            temporary / RECEIPT_NAME,
            _canonical_bytes(receipt, newline=True),
        )
        if output_dir.exists():
            raise FileExistsError(
                f"output workspace appeared during publication: {output_dir}"
            )
        _rename_directory_no_replace(temporary, output_dir)
        return receipt
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Bind retained post-dedup inventory to physical payload rows and publish "
            "the exact ephemeral DATA-232 training handoff"
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--inventory-json", type=Path, required=True)
    parser.add_argument("--payload-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-inventory-identity-sha256", required=True)
    parser.add_argument("--expected-carrier-git-sha", required=True)
    parser.add_argument("--expected-retained-inventory-git-sha", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    carrier_sha, upstream_sha = require_exact_checkout(
        args.repo_root,
        expected_carrier_git_sha=args.expected_carrier_git_sha,
        expected_retained_inventory_git_sha=args.expected_retained_inventory_git_sha,
    )
    # Exact checkout and no-overwrite gates intentionally precede payload reads.
    if args.output_dir.exists():
        raise FileExistsError(f"output workspace already exists: {args.output_dir}")
    receipt = prepare_and_publish(
        inventory_path=args.inventory_json,
        payload_path=args.payload_jsonl,
        output_dir=args.output_dir,
        expected_inventory_identity_sha256=args.expected_inventory_identity_sha256,
        carrier_git_sha=carrier_sha,
        upstream_git_sha=upstream_sha,
    )
    print(receipt["receipt_identity_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
