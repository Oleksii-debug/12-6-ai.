#!/usr/bin/env python3
"""Rebuild corpus V0.1 and emit a compact, manifest-bound privacy scan summary."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from twelve_six.data.corpus_v01 import build_corpus
from twelve_six.data.privacy_filter import scan_record
from twelve_six.data.privacy_reporting import build_corpus_scan_summary


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/data/corpus_v01.json"))
    parser.add_argument(
        "--retained-manifest",
        type=Path,
        default=Path("data/corpus/v0.1/manifest.json"),
    )
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _args()
    retained = json.loads(args.retained_manifest.read_text(encoding="utf-8"))
    retained_identity = retained["corpus_identity_sha256"]

    with tempfile.TemporaryDirectory(prefix="data33-pii-corpus-v01-") as directory:
        output_dir = Path(directory) / "corpus"
        rebuilt = build_corpus(args.config, output_dir)
        if rebuilt["corpus_identity_sha256"] != retained_identity:
            raise RuntimeError("corpus V0.1 identity differs from retained manifest")

        results = []
        for shard in rebuilt["shards"]:
            shard_path = output_dir / shard["path"]
            if _sha256_file(shard_path) != shard["sha256"]:
                raise RuntimeError("rebuilt shard hash mismatch")
            with shard_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    results.append(
                        scan_record(
                            record_id=row["record_id"],
                            source_id=row["source_id"],
                            source_version=row["source_version"],
                            modality=row["modality"],
                            text=row["text"],
                        )
                    )

    summary = build_corpus_scan_summary(
        results,
        corpus_identity_sha256=retained_identity,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0 if summary["records_train_eligible_after_privacy"] == summary["records_total"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
