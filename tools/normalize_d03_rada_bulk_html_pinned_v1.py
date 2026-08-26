#!/usr/bin/env python3
"""Pinned production entrypoint for D03 Rada bulk normalization v1.

The generic materializer remains useful for unit fixtures, but production
materialization must first prove the exact #618 parent and exact v1 policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from twelve_six.d03_rada_normalization_contract import (
    bind_manifest_to_contract,
    canonical_config_sha256,
)
from tools.normalize_d03_rada_bulk_html import (
    DEFAULT_CONFIG,
    NormalizationError,
    materialize_normalized_records,
)


def _load_object(path: Path) -> tuple[dict, bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NormalizationError(f"cannot load JSON: {path}") from exc
    if not isinstance(value, dict):
        raise NormalizationError(f"JSON root must be an object: {path}")
    return value, raw


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--probe-report", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config, _ = _load_object(args.config)
    contract_sha256 = canonical_config_sha256(config)
    probe, probe_bytes = _load_object(args.probe_report)
    try:
        archive = args.archive.read_bytes()
    except OSError as exc:
        raise NormalizationError(f"cannot read archive: {args.archive}") from exc

    jsonl, manifest = materialize_normalized_records(
        archive,
        probe,
        config,
        probe_report_sha256=hashlib.sha256(probe_bytes).hexdigest(),
    )
    manifest = bind_manifest_to_contract(
        manifest,
        normalization_contract_sha256=contract_sha256,
    )
    manifest_bytes = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    ).encode("utf-8") + b"\n"

    _atomic_write(args.output_jsonl, jsonl)
    _atomic_write(args.output_manifest, manifest_bytes)
    print(
        json.dumps(
            {
                "status": "PASS_PINNED_NORMALIZATION_CONTRACT",
                "normalization_contract_sha256": contract_sha256,
                "manifest_identity_sha256": manifest["manifest_identity_sha256"],
                "record_count": manifest["normalization"]["record_count"],
                "normalized_bytes_observed_not_credited": manifest["normalization"][
                    "normalized_bytes_observed_not_credited"
                ],
                "training_authorized_bytes": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
