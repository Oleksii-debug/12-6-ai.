from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

from twelve_six.data.loc_books_intake import (
    LocBooksIntakeError,
    canonical_json,
    materialize_from_gzip_stream,
    validate_config,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def _git_blob_sha1(data: bytes) -> str:
    import hashlib

    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def _verify_registry(cfg: dict[str, object]) -> None:
    audit = cfg["common_pile_audit"]
    assert isinstance(audit, dict)
    registry_path = Path(str(audit["registry_path"]))
    raw = registry_path.read_bytes()
    if _git_blob_sha1(raw) != audit["registry_blob_sha1"]:
        raise LocBooksIntakeError("local Common Pile registry blob mismatch")


def main() -> int:
    parser = argparse.ArgumentParser(description="Bounded zero-credit LoC candidate materializer")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-gzip", type=Path)
    parser.add_argument("--candidate-jsonl", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(cfg)
    _verify_registry(cfg)
    if args.input_gzip is not None:
        with args.input_gzip.open("rb") as stream:
            candidates, report = materialize_from_gzip_stream(cfg, stream)
    else:
        request = urllib.request.Request(
            cfg["upstream"]["pinned_url"],
            headers={"User-Agent": "12-6-ai-d03-loc-intake/1"},
        )
        with urllib.request.urlopen(request, timeout=60) as stream:
            candidates, report = materialize_from_gzip_stream(cfg, stream)
    _write_jsonl(args.candidate_jsonl, candidates)
    _write_json(args.report_json, report)
    print(report["report_identity_sha256"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
