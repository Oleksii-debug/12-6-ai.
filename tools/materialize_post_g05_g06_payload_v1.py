#!/usr/bin/env python3
"""Materialize a deterministic post-G05/G06 payload from exact current authorities."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from twelve_six.data.post_g05_g06_materialization_v1 import (
    canonical_record_bytes,
    materialize_post_g05_g06,
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"{path}:{number} must contain a JSON object")
        records.append(value)
    return records


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _ensure_new(paths: tuple[Path, ...]) -> None:
    for path in paths:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to overwrite output: {path}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--composition-preflight", required=True, type=Path)
    parser.add_argument("--expected-composition-preflight-identity", required=True)
    parser.add_argument("--g05-authority", required=True, type=Path)
    parser.add_argument("--expected-g05-execution-identity", required=True)
    parser.add_argument("--g06-execution-envelope", required=True, type=Path)
    parser.add_argument("--expected-g06-envelope-identity", required=True)
    parser.add_argument("--expected-g06-execution-identity", required=True)
    parser.add_argument("--privacy-source", required=True, type=Path)
    parser.add_argument("--expected-privacy-implementation-git-blob-sha1", required=True)
    parser.add_argument("--expected-privacy-policy-sha256", required=True)
    parser.add_argument("--execution-head-sha", required=True)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--output-inventory", required=True, type=Path)
    parser.add_argument("--output-evidence", required=True, type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    outputs = (args.output_jsonl, args.output_inventory, args.output_evidence)
    _ensure_new(outputs)
    kwargs = {
        "records": _load_jsonl(args.input_jsonl),
        "composition_preflight": _load_json(args.composition_preflight),
        "expected_composition_preflight_identity_sha256": (
            args.expected_composition_preflight_identity
        ),
        "g05_authority": _load_json(args.g05_authority),
        "expected_g05_execution_identity_sha256": args.expected_g05_execution_identity,
        "g06_execution_envelope": _load_json(args.g06_execution_envelope),
        "expected_g06_envelope_identity_sha256": args.expected_g06_envelope_identity,
        "expected_g06_execution_identity_sha256": args.expected_g06_execution_identity,
        "privacy_source_path": args.privacy_source,
        "expected_privacy_implementation_git_blob_sha1": (
            args.expected_privacy_implementation_git_blob_sha1
        ),
        "expected_privacy_policy_sha256": args.expected_privacy_policy_sha256,
        "execution_head_sha": args.execution_head_sha,
    }
    records_a, inventory_a, evidence_a = materialize_post_g05_g06(**kwargs)
    records_b, inventory_b, evidence_b = materialize_post_g05_g06(**kwargs)
    if canonical_record_bytes(records_a) != canonical_record_bytes(records_b):
        raise RuntimeError("repeat post-G05/G06 payload materialization is not byte-identical")
    if inventory_a != inventory_b or evidence_a != evidence_b:
        raise RuntimeError("repeat post-G05/G06 evidence materialization is not byte-identical")
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.write_bytes(canonical_record_bytes(records_a))
    _write_json(args.output_inventory, inventory_a)
    repeat_evidence = dict(evidence_a)
    repeat_evidence["repeat_materialization_byte_identical"] = True
    core = dict(repeat_evidence)
    core.pop("materialization_identity_sha256", None)
    canonical = (
        json.dumps(core, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    repeat_evidence["materialization_identity_sha256"] = hashlib.sha256(canonical).hexdigest()
    _write_json(args.output_evidence, repeat_evidence)
    print(repeat_evidence["materialization_identity_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
