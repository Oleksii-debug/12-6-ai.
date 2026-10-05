#!/usr/bin/env python3
"""Bind or verify the learned-20M canonical byte-tokenizer decision authority."""

from __future__ import annotations

import argparse
import json
import math
import os
import stat
import tempfile
from pathlib import Path
from typing import Any

from twelve_six.tokenization.decision_authority import (
    TokenizerDecisionError,
    bind_byte_baseline_decision,
    verify_byte_baseline_decision,
)


def _pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate_json_key:{key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non_finite_json_constant:{value}")


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non_finite_json_number:{value}")
    # A lexically nonzero external number must not silently become zero.
    # Preserve genuine positive/negative JSON zero, including 0e-9999.
    significand = value.split("e", 1)[0].split("E", 1)[0]
    if parsed == 0.0 and any(digit in "123456789" for digit in significand):
        raise ValueError(f"nonzero_json_number_underflowed_to_zero:{value}")
    return parsed


MAX_INPUT_BYTES = 1_048_576
MAX_JSON_DEPTH = 64
MAX_JSON_NODES = 10_000


def _load(path: Path) -> dict[str, Any]:
    # Read only a bounded prefix, including when the path is a network file.
    with path.open("rb") as source:
        raw = source.read(MAX_INPUT_BYTES + 1)
    if len(raw) > MAX_INPUT_BYTES:
        raise ValueError("tokenizer input exceeds byte limit")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs_without_duplicates,
            parse_constant=_reject_constant,
            parse_float=_parse_finite_float,
        )
    except RecursionError as exc:
        raise ValueError("tokenizer input JSON nesting limit exceeded") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")

    # The byte cap bounds parsing; the iterative walk bounds post-parse work.
    # Python's decoder may allow escaped lone surrogates: reject them explicitly.
    pending: list[tuple[Any, int]] = [(value, 0)]
    nodes = 0
    while pending:
        current, depth = pending.pop()
        nodes += 1
        if depth > MAX_JSON_DEPTH or nodes > MAX_JSON_NODES:
            raise ValueError("tokenizer input exceeds JSON structure limit")
        if isinstance(current, dict):
            for key, child in current.items():
                key.encode("utf-8")
                pending.append((child, depth + 1))
        elif isinstance(current, list):
            pending.extend((child, depth + 1) for child in current)
        elif isinstance(current, str):
            current.encode("utf-8")
    return value


def _serialize_report(value: dict[str, Any]) -> str:
    """One strict finite UTF-8 JSON contract for file and stdout reports."""
    try:
        rendered = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        # A JSON-escaped lone surrogate must not reach stdout or disk.
        rendered.encode("utf-8")
        return rendered + "\n"
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError("tokenizer report is not strict finite JSON") from exc


class PublicationIndeterminate(OSError):
    """A possible final exists but ownership/commit truth cannot be proved."""

    def __init__(self, message: str, *, staged: Path) -> None:
        super().__init__(message)
        self.staged = staged


class PublicationCleanupPending(OSError):
    """The exact final is committed; only the redundant staging alias remains."""

    def __init__(self, message: str, *, staged: Path) -> None:
        super().__init__(message)
        self.staged = staged


def _lstat_or_none(path: Path) -> os.stat_result | None:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None


def _write(path: Path, value: dict[str, Any]) -> None:
    """Create one exact report or preserve enough state for deterministic recovery."""
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    payload = _serialize_report(value).encode("utf-8")

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(name)
    committed = False
    indeterminate = False
    primary: BaseException | None = None
    try:
        with os.fdopen(descriptor, "wb") as handle:
            if handle.write(payload) != len(payload):
                raise OSError("incomplete tokenizer report staging write")
            handle.flush()
            os.fsync(handle.fileno())

        staged = temporary.stat(follow_symlinks=False)
        if not stat.S_ISREG(staged.st_mode) or staged.st_size != len(payload):
            raise OSError("tokenizer report staging identity changed")
        identity = (staged.st_dev, staged.st_ino)

        link_error: BaseException | None = None
        try:
            # Same-directory hard link is create-only: an existing target wins.
            os.link(temporary, path)
        except (OSError, KeyboardInterrupt, SystemExit) as exc:
            link_error = exc

        try:
            final = _lstat_or_none(path)
        except (OSError, KeyboardInterrupt, SystemExit) as inspect_error:
            indeterminate = True
            raise PublicationIndeterminate(
                "PUBLICATION_INDETERMINATE: cannot inspect tokenizer output after "
                f"possible publication; retained stage {temporary}",
                staged=temporary,
            ) from inspect_error

        if (
            final is not None
            and stat.S_ISREG(final.st_mode)
            and (final.st_dev, final.st_ino) == identity
            and final.st_size == len(payload)
        ):
            # This also resolves wrappers that create the hard link and then raise.
            committed = True
        elif link_error is not None and final is None:
            raise link_error
        elif final is not None:
            indeterminate = True
            raise PublicationIndeterminate(
                "PUBLICATION_INDETERMINATE: output exists but is not the staged "
                f"authority inode; retained stage {temporary}",
                staged=temporary,
            ) from link_error
        else:
            indeterminate = True
            raise PublicationIndeterminate(
                "PUBLICATION_INDETERMINATE: link returned without an inspectable "
                f"output; retained stage {temporary}",
                staged=temporary,
            ) from link_error
    except BaseException as exc:
        primary = exc
        raise
    finally:
        if indeterminate:
            pass
        else:
            try:
                temporary.unlink(missing_ok=True)
            except (OSError, KeyboardInterrupt, SystemExit) as cleanup_error:
                if committed and primary is None:
                    raise PublicationCleanupPending(
                        "tokenizer authority is COMMITTED_AND_VERIFIED; staged cleanup "
                        f"is pending at {temporary}; remove only that staging alias "
                        "after confirming the final output remains unchanged",
                        staged=temporary,
                    ) from cleanup_error
                raise PublicationIndeterminate(
                    "STAGING_CLEANUP_INDETERMINATE: tokenizer authority was not "
                    f"reported committed; retained stage {temporary}; reconcile "
                    "the stage and final before retry",
                    staged=temporary,
                ) from (primary if primary is not None else cleanup_error)


def _emit_input_error(exc: Exception) -> None:
    print(
        json.dumps(
            {"contract_valid": False, "error": str(exc)},
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--balanced-selection", type=Path, required=True)
    parser.add_argument("--split-application", type=Path, required=True)
    parser.add_argument("--expected-selection-identity-sha256", required=True)
    parser.add_argument("--expected-application-identity-sha256", required=True)
    parser.add_argument("--expected-retained-inventory-identity-sha256", required=True)
    parser.add_argument("--expected-decontamination-authority-sha256", required=True)
    parser.add_argument("--expected-dedup-authority-sha256", required=True)
    parser.add_argument("--expected-balance-policy-identity-sha256", required=True)
    parser.add_argument("--expected-balance-result-identity-sha256", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-report", type=Path)
    return parser.parse_args()


def _kwargs(args: argparse.Namespace) -> dict[str, str]:
    return {
        "expected_selection_identity_sha256": args.expected_selection_identity_sha256,
        "expected_application_identity_sha256": (
            args.expected_application_identity_sha256
        ),
        "expected_retained_inventory_identity_sha256": (
            args.expected_retained_inventory_identity_sha256
        ),
        "expected_decontamination_authority_sha256": (
            args.expected_decontamination_authority_sha256
        ),
        "expected_dedup_authority_sha256": args.expected_dedup_authority_sha256,
        "expected_balance_policy_identity_sha256": (
            args.expected_balance_policy_identity_sha256
        ),
        "expected_balance_result_identity_sha256": (
            args.expected_balance_result_identity_sha256
        ),
    }


def main() -> int:
    args = parse_args()
    try:
        selection = _load(args.balanced_selection)
        application = _load(args.split_application)
        verified_report = (
            _load(args.verify_report) if args.verify_report is not None else None
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        _emit_input_error(exc)
        return 2

    try:
        if verified_report is not None:
            report = verified_report
            verify_byte_baseline_decision(report, selection, application, **_kwargs(args))
        else:
            report = bind_byte_baseline_decision(selection, application, **_kwargs(args))
    except TokenizerDecisionError as exc:
        _emit_input_error(exc)
        return 2

    if args.output is not None:
        try:
            _write(args.output, report)
        except PublicationCleanupPending as exc:
            print(
                json.dumps(
                    {
                        "contract_valid": True,
                        "output_committed": True,
                        "cleanup_pending": True,
                        "recovery": str(exc),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            return 0
        except (OSError, ValueError) as exc:
            _emit_input_error(exc)
            return 2
    else:
        try:
            serialized = _serialize_report(report)
        except ValueError as exc:
            _emit_input_error(exc)
            return 2
        print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
