#!/usr/bin/env python3
"""Execute the canonical balanced split application on exact authenticated inputs.

This is an operator carrier only. It does not define balance policy or split science,
and it grants no corpus, tokenizer, exposure, optimizer, or training authority.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NoReturn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from twelve_six.data.balanced_split_application_v1 import (  # noqa: E402
    CANONICAL_SPLIT_GIT_BLOB_SHA1,
    CANONICAL_SPLIT_VALIDATION_FRACTION,
    CANONICAL_SPLIT_VARIANT_SEEDS,
    BalancedSplitApplicationError,
    build_balanced_split_application,
    verify_balanced_split_application,
)


class BalancedSplitRunnerError(ValueError):
    """Raised when the execution carrier cannot establish a safe input/output boundary."""


def _blocked(message: str) -> NoReturn:
    raise BalancedSplitRunnerError(message)


def _strict_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _blocked(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> NoReturn:
    _blocked(f"non-finite JSON constant is forbidden: {value}")


def _strict_json_loads(text: str, *, source: str) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=_strict_object_pairs,
            parse_constant=_reject_nonfinite,
        )
    except BalancedSplitRunnerError:
        raise
    except json.JSONDecodeError as exc:
        raise BalancedSplitRunnerError(f"malformed JSON in {source}: {exc.msg}") from exc


def _read_regular_utf8(path: Path, *, role: str) -> str:
    if path.is_symlink():
        _blocked(f"{role} must not be a symlink: {path}")
    if not path.is_file():
        _blocked(f"{role} must be a regular file: {path}")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise BalancedSplitRunnerError(f"cannot read {role}: {path}") from exc
    if not payload:
        _blocked(f"{role} must not be empty: {path}")
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BalancedSplitRunnerError(f"{role} must be strict UTF-8: {path}") from exc


def read_json_object(path: Path, *, role: str) -> dict[str, Any]:
    value = _strict_json_loads(_read_regular_utf8(path, role=role), source=str(path))
    if not isinstance(value, dict):
        _blocked(f"{role} JSON root must be an object: {path}")
    return value


def read_jsonl_objects(path: Path, *, role: str) -> list[dict[str, Any]]:
    text = _read_regular_utf8(path, role=role)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            _blocked(f"{role} contains blank JSONL line {line_number}: {path}")
        value = _strict_json_loads(line, source=f"{path}:{line_number}")
        if not isinstance(value, dict):
            _blocked(f"{role} JSONL line {line_number} must be an object: {path}")
        rows.append(value)
    if not rows:
        _blocked(f"{role} must contain at least one JSON object: {path}")
    return rows


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise BalancedSplitRunnerError("output is not canonical-JSON serializable") from exc
    return (text + "\n").encode("utf-8")


def _atomic_create_only(path: Path, payload: bytes) -> None:
    if path.is_symlink() or path.exists():
        _blocked(f"output already exists or is a symlink: {path}")

    parent = path.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise BalancedSplitRunnerError(f"cannot create output directory: {parent}") from exc
    if parent.is_symlink() or not parent.is_dir():
        _blocked(f"output parent must be a real directory, not a symlink: {parent}")

    fd = -1
    temporary: Path | None = None
    try:
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=parent)
        temporary = Path(temp_name)
        with os.fdopen(fd, "wb") as handle:
            fd = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise BalancedSplitRunnerError(
                f"output appeared concurrently; refusing overwrite: {path}"
            ) from exc
        except OSError as exc:
            raise BalancedSplitRunnerError(
                f"atomic create-only output publication failed: {path}"
            ) from exc
    finally:
        if fd >= 0:
            os.close(fd)
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def execute_balanced_split_application(
    *,
    selection_path: Path,
    raw_records_path: Path,
    output_path: Path,
    expected_selection_identity_sha256: str,
    expected_retained_inventory_identity_sha256: str,
    expected_decontamination_authority_sha256: str,
    expected_dedup_authority_sha256: str,
    expected_balance_policy_identity_sha256: str,
    expected_balance_result_identity_sha256: str,
) -> dict[str, Any]:
    """Build, independently verify, then atomically publish one split application."""

    if output_path.is_symlink() or output_path.exists():
        _blocked(f"output already exists or is a symlink: {output_path}")

    selection = read_json_object(selection_path, role="balanced selection authority")
    raw_records = read_jsonl_objects(raw_records_path, role="physical raw records")
    kwargs = {
        "expected_selection_identity_sha256": expected_selection_identity_sha256,
        "expected_retained_inventory_identity_sha256": expected_retained_inventory_identity_sha256,
        "expected_decontamination_authority_sha256": expected_decontamination_authority_sha256,
        "expected_dedup_authority_sha256": expected_dedup_authority_sha256,
        "expected_balance_policy_identity_sha256": expected_balance_policy_identity_sha256,
        "expected_balance_result_identity_sha256": expected_balance_result_identity_sha256,
        "expected_split_git_blob_sha1": CANONICAL_SPLIT_GIT_BLOB_SHA1,
        "variant_seeds": CANONICAL_SPLIT_VARIANT_SEEDS,
        "validation_fraction": CANONICAL_SPLIT_VALIDATION_FRACTION,
    }
    application = build_balanced_split_application(selection, raw_records, **kwargs)
    verify_balanced_split_application(application, selection, raw_records, **kwargs)

    if application.get("status") != "PASS_ZERO_CREDIT":
        _blocked("canonical split application did not produce PASS_ZERO_CREDIT")
    claim_boundary = application.get("claim_boundary")
    if not isinstance(claim_boundary, Mapping):
        _blocked("canonical split application claim boundary is missing")
    if claim_boundary.get("authorized_optimized_target_exposure") != 0:
        _blocked("canonical split application unexpectedly widened optimized-target exposure")
    if claim_boundary.get("model_training_authorized") is not False:
        _blocked("canonical split application unexpectedly authorized model training")

    _atomic_create_only(output_path, _canonical_json_bytes(application))
    return application


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the canonical cluster-safe split to one terminal balanced selection. "
            "This command never grants training or optimized-target exposure."
        )
    )
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--raw-records", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-selection-identity-sha256", required=True)
    parser.add_argument("--expected-retained-inventory-identity-sha256", required=True)
    parser.add_argument("--expected-decontamination-authority-sha256", required=True)
    parser.add_argument("--expected-dedup-authority-sha256", required=True)
    parser.add_argument("--expected-balance-policy-identity-sha256", required=True)
    parser.add_argument("--expected-balance-result-identity-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        application = execute_balanced_split_application(
            selection_path=args.selection,
            raw_records_path=args.raw_records,
            output_path=args.output,
            expected_selection_identity_sha256=args.expected_selection_identity_sha256,
            expected_retained_inventory_identity_sha256=(
                args.expected_retained_inventory_identity_sha256
            ),
            expected_decontamination_authority_sha256=(
                args.expected_decontamination_authority_sha256
            ),
            expected_dedup_authority_sha256=args.expected_dedup_authority_sha256,
            expected_balance_policy_identity_sha256=args.expected_balance_policy_identity_sha256,
            expected_balance_result_identity_sha256=args.expected_balance_result_identity_sha256,
        )
    except (BalancedSplitRunnerError, BalancedSplitApplicationError, OSError) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2

    print("D03_BALANCED_SPLIT_APPLICATION_V1=PASS_ZERO_CREDIT")
    print(f"APPLICATION_IDENTITY_SHA256={application['application_identity_sha256']}")
    print("AUTHORIZED_OPTIMIZED_TARGET_EXPOSURE=0")
    print("TRAINING_EXECUTED=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
