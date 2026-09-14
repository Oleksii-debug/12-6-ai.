#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

from validate_d03_terminal_ua_source_seals import validate as validate_seal

ROOT = Path(__file__).resolve().parents[1]
PHP_ZIP_SHA256 = "f13ff28bd9bd998bbb8ebfb4a50893778f7edaea3f09ea052771bd1ba8b2c4e6"
RUST_ZIP_SHA256 = "87b9d3299f863450bff915f780de9f43f41428e6937147a2522cb7dfa7371e4d"
PHP_EVIDENCE_ID = "a591e8af0aa6aa9e3040c002087ea316485f38ed7d43ef30e127346498c02204"
RUST_MANIFEST_SHA = "df9ab2943e140b8735c692eb4ee40c3f10dd03ca843ec9800e0bdf4d7ff664af"
PHP_ENTRIES = {
    "ATTRIBUTION.txt",
    "authority.json",
    "license.xml",
    "normalized.bundle.bin",
    "raw/language/basic-syntax.xml",
    "raw/language/context.xml",
    "raw/language/errors.xml",
    "raw/language/oop5.xml",
    "raw/language/types/boolean.xml",
    "raw/language/types/callable.xml",
    "raw/language/types/float.xml",
    "raw/language/types/integer.xml",
    "raw/language/types.xml",
    "raw/language/wrappers.xml",
    "raw.bundle.bin",
}
RUST_ENTRIES = {
    "manifest.json",
    "source_report.json",
    "normalized/ch01-01-installation.txt",
    "normalized/ch01-02-hello-world.txt",
    "raw/ch01-01-installation.md",
    "raw/ch01-02-hello-world.md",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(obj: Any) -> bytes:
    return (
        json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()


def read_exact_zip(path: Path, expected_sha256: str, expected_names: set[str]) -> dict[str, bytes]:
    data = path.read_bytes()
    observed = sha256(data)
    if observed != expected_sha256:
        raise ValueError(f"artifact digest mismatch: {path}: {observed}")
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = [info.filename for info in zf.infolist()]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate ZIP member: {path}")
        if set(names) != expected_names:
            raise ValueError(f"ZIP member-set drift: {path}")
        out: dict[str, bytes] = {}
        for info in zf.infolist():
            if info.is_dir() or info.filename.startswith(("/", "../")):
                raise ValueError(f"unsafe ZIP member: {info.filename}")
            if ".." in Path(info.filename).parts:
                raise ValueError(f"unsafe ZIP member: {info.filename}")
            out[info.filename] = zf.read(info)
    return out


def parse_frame(data: bytes) -> list[tuple[str, bytes]]:
    rows: list[tuple[str, bytes]] = []
    offset = 0
    seen: set[str] = set()
    while offset < len(data):
        if len(data) - offset < 4:
            raise ValueError("truncated frame path length")
        path_len = int.from_bytes(data[offset : offset + 4], "big")
        offset += 4
        if path_len <= 0 or len(data) - offset < path_len + 8:
            raise ValueError("invalid frame path length")
        path_raw = data[offset : offset + path_len]
        offset += path_len
        try:
            path = path_raw.decode("utf-8", "strict")
        except UnicodeDecodeError as exc:
            raise ValueError("non-UTF8 frame path") from exc
        payload_len = int.from_bytes(data[offset : offset + 8], "big")
        offset += 8
        if payload_len <= 0 or len(data) - offset < payload_len:
            raise ValueError("invalid frame payload length")
        payload = data[offset : offset + payload_len]
        offset += payload_len
        if path in seen:
            raise ValueError(f"duplicate framed path: {path}")
        seen.add(path)
        rows.append((path, payload))
    if not rows:
        raise ValueError("empty frame")
    return rows


def build_record(source: dict[str, Any], path: str, payload: bytes) -> dict[str, Any]:
    text = payload.decode("utf-8", "strict")
    digest = sha256(payload)
    record_id = sha256(
        (source["source_id"] + "\0" + path + "\0" + digest).encode("utf-8")
    )
    return {
        "record_id": record_id,
        "source_id": source["source_id"],
        "family": source["family"],
        "language": "uk",
        "source_commit": source["source_commit"],
        "path": path,
        "normalized_bytes": len(payload),
        "normalized_sha256": digest,
        "text": text,
        "current_corpus_eligible": False,
        "training_eligible": False,
        "evaluation_eligible": False,
    }


def materialize_php(entries: dict[str, bytes], source: dict[str, Any]) -> list[dict[str, Any]]:
    authority = json.loads(entries["authority.json"])
    if authority.get("terminal") is not True or authority.get("terminal_state") != "ADMIT":
        raise ValueError("PHP authority is not terminal ADMIT")
    if authority.get("evidence_identity_sha256") != PHP_EVIDENCE_ID:
        raise ValueError("PHP authority identity drift")
    snapshot = authority.get("snapshot", {})
    framed = entries["normalized.bundle.bin"]
    if sha256(framed) != source["normalized_bundle_sha256"]:
        raise ValueError("PHP normalized bundle drift")
    if snapshot.get("normalized_bundle_sha256") != source["normalized_bundle_sha256"]:
        raise ValueError("PHP authority bundle binding drift")
    expected = {
        row["path"]: (row["normalized_bytes"], row["normalized_sha256"])
        for row in authority.get("files", [])
    }
    rows = parse_frame(framed)
    if len(rows) != source["selected_objects"] or set(dict(rows)) != set(expected):
        raise ValueError("PHP framed object set drift")
    records: list[dict[str, Any]] = []
    for path, payload in rows:
        if expected[path] != (len(payload), sha256(payload)):
            raise ValueError(f"PHP framed payload drift: {path}")
        records.append(build_record(source, path, payload))
    if sum(r["normalized_bytes"] for r in records) != source["normalized_bytes"]:
        raise ValueError("PHP normalized byte total drift")
    return records


def materialize_rust(entries: dict[str, bytes], source: dict[str, Any]) -> list[dict[str, Any]]:
    report = json.loads(entries["source_report.json"])
    manifest = json.loads(entries["manifest.json"])
    if report.get("terminal") is not True or report.get("verdict") != "ADMIT":
        raise ValueError("Rust authority is not terminal ADMIT")
    if report.get("manifest_sha256") != RUST_MANIFEST_SHA:
        raise ValueError("Rust report manifest binding drift")
    if manifest.get("manifest_sha256") != RUST_MANIFEST_SHA or manifest.get("verdict") != "ADMIT":
        raise ValueError("Rust terminal manifest drift")
    if manifest.get("normalized_bundle_sha256") != source["normalized_bundle_sha256"]:
        raise ValueError("Rust normalized bundle binding drift")
    selection = report.get("selection", {})
    expected = {
        row["path"]: (row["normalized_bytes"], row["normalized_sha256"])
        for row in selection.get("files", [])
    }
    records: list[dict[str, Any]] = []
    for source_path in sorted(expected):
        member = "normalized/" + Path(source_path).with_suffix(".txt").name
        payload = entries.get(member)
        if payload is None:
            raise ValueError(f"missing Rust normalized payload: {member}")
        if expected[source_path] != (len(payload), sha256(payload)):
            raise ValueError(f"Rust normalized payload drift: {source_path}")
        records.append(build_record(source, source_path, payload))
    if len(records) != source["selected_objects"]:
        raise ValueError("Rust object-count drift")
    if sum(r["normalized_bytes"] for r in records) != source["normalized_bytes"]:
        raise ValueError("Rust normalized byte total drift")
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--php-artifact", type=Path, required=True)
    ap.add_argument("--rust-artifact", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/data/d03_terminal_ua_source_seals_v1.json",
    )
    args = ap.parse_args()

    seal = json.loads(args.config.read_text(encoding="utf-8"))
    validate_seal(seal)
    by_id = {src["source_id"]: src for src in seal["sources"]}
    php_entries = read_exact_zip(args.php_artifact, PHP_ZIP_SHA256, PHP_ENTRIES)
    rust_entries = read_exact_zip(args.rust_artifact, RUST_ZIP_SHA256, RUST_ENTRIES)
    records = materialize_php(php_entries, by_id["next100-028-php-doc-uk"])
    records += materialize_rust(rust_entries, by_id["next100-030-rustbook-ua-oer"])
    records.sort(key=lambda row: (row["source_id"], row["path"]))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = b"".join(canonical(row) for row in records)
    (args.out_dir / "ua_terminal_source_handoff.jsonl").write_bytes(jsonl)
    manifest = {
        "schema_version": "12-6.d03-terminal-ua-source-handoff.v1",
        "seal_identity_sha256": seal["authority_identity_sha256"],
        "record_count": len(records),
        "independent_family_count": len({row["family"] for row in records}),
        "normalized_source_bytes": sum(row["normalized_bytes"] for row in records),
        "jsonl_sha256": sha256(jsonl),
        "canonical_capacity_credit_bytes": 0,
        "training_authorized_bytes": 0,
        "authorized_unique_loss_positions": 0,
        "requires_current_global_dedup": True,
        "requires_fresh_eval_decontamination": True,
        "training_executed": False,
        "final_test_accessed": False,
        "paid_compute_used": False,
    }
    manifest["manifest_identity_sha256"] = sha256(canonical(manifest))
    (args.out_dir / "manifest.json").write_bytes(canonical(manifest))
    print("D03_UA_HANDOFF=PASS")
    print(f"D03_UA_HANDOFF_RECORDS={len(records)}")
    print(f"D03_UA_HANDOFF_BYTES={manifest['normalized_source_bytes']}")
    print(f"D03_UA_HANDOFF_JSONL_SHA256={manifest['jsonl_sha256']}")
    print("D03_UA_HANDOFF_TRAINING_AUTHORIZED_BYTES=0")


if __name__ == "__main__":
    main()
