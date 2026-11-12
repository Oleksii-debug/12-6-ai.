"""Materialize the accepted Section 16 corpus with the frozen 32K tokenizer."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from tools import plan2_public_domain_books_v1 as books
from tools import plan2_s17_exposure_stream_v1 as exposure
from tools import plan2_s17_frozen_tokenizer_v1 as frozen
from tools import plan2_s17_reserved_roles_v1 as reservations
from tools import plan2_s17_train_packing_v1 as packing
from tools.plan2_physical_materialization_v1 import _atomic_write, _read_destination

S16_ACCEPTED_MAIN = "3aed42cbb9234783ef7b72a6cb6b082d6f709a28"
S16_MANIFEST = "evidence/plan2_s16/physical_corpus_manifest_v1.json"
S16_MANIFEST_FILE_SHA256 = "5ff1316a2697254996d443a33619a24572bbc73b4c1c0b7ab920229a11c5f88b"
SOURCE_AUTHORITY_BLOBS = {
    "configs/data/d03_common_pile_arxiv_abstracts_source_admission_v1.json": "2a812fceb29dc273d75eec46f1957e445738f526",
    "configs/data/d03_franko1901_exact_materialization_v1.json": "90d6f1f56ad572bbd81f1e3cf87098e2304f7784",
    "configs/data/d03_languk_supreme_court_postexecution_rights_v1.json": "693bf6d83dbf4cbf402693f806573ffdeba2ea7d",
    "configs/data/next100_037_python_docs_source_authority_v1.json": "b15abac8744ccda9fe58d1351f7925b6ab328034",
    "configs/data/plan2_public_domain_books_v1.json": "f7328b939e6e5a44c18d50c2ab0c10c84b3f5d13",
    "evidence/data_bulk_code1/permissive_python_bundle_v1_terminal.json": "b2b9b361dbf0e6c883dca4d7f006812c69cdd4e0",
}
MANIFEST = "bundle-manifest.json"
SCHEMA = "12-6.plan2-s17-physical-bundle.v1"
ROW_FIELDS = {
    "record_id_sha256", "source_id", "source_family", "stratum", "utf8_bytes", "content_sha256", "text",
}
OUTPUT_MEMBERS = {
    MANIFEST, "split-manifest.json", "tokenizer-manifest.json", "frozen-tokenizer.json.gz",
    "train-sources.json", "validation-manifest.json", "test-manifest.json", "packing-manifest.json",
    "exposure-ledger.json", "shards", "exposures",
}


class BundleDenied(ValueError):
    """Accepted data, tokenizer, or physical bundle changed."""


def need(ok: bool, reason: str) -> None:
    if not ok:
        raise BundleDenied(reason)


def _checked_source_terms(root: Path, manifest: dict[str, Any]) -> None:
    authorities = {row["evidence"] for row in manifest["selected_source_terms"].values()}
    need(authorities == SOURCE_AUTHORITY_BLOBS.keys(), "accepted source authority set changed")
    for path in sorted(authorities):
        need(books.git_blob(books.read_checked(root, path).replace(b"\r\n", b"\n")) == SOURCE_AUTHORITY_BLOBS[path],
             "source authority changed since accepted Section 16 admission")


def read_admission(root: Path, train: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw = books.read_checked(root, S16_MANIFEST).replace(b"\r\n", b"\n")
    need(books.sha(raw) == S16_MANIFEST_FILE_SHA256, "accepted Section 16 manifest changed")
    manifest = json.loads(raw)
    need(manifest["physical_s3_s9_admission_passed"] is True
         and manifest["heldout_payloads_exported"] is False,
         "Section 16 admission or reservation boundary failed")
    _checked_source_terms(root, manifest)
    need(not any(path.is_symlink() for path in (train, *train.parents))
         and train.is_file() and train.stat().st_size == manifest["payload_bytes"],
         "invalid Section 16 physical training file")
    payload = _read_destination(train)
    need(books.sha(payload) == manifest["payload_sha256"], "training payload differs from accepted bytes")
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_contents: set[str] = set()
    stratum_bytes: Counter[str] = Counter()
    for line in payload.splitlines(keepends=True):
        row = json.loads(line)
        need(type(row) is dict and row.keys() == ROW_FIELDS and line == books.canonical(row),
             "invalid accepted record shape or encoding")
        identity, content = row["record_id_sha256"], row["content_sha256"]
        text = row["text"].encode("utf-8", "strict")
        need(identity not in seen_ids and content not in seen_contents
             and len(text) == row["utf8_bytes"] and books.sha(text) == content
             and row["source_family"] in manifest["selected_source_terms"],
             "training record identity or source terms changed")
        seen_ids.add(identity)
        seen_contents.add(content)
        stratum_bytes[row["stratum"]] += len(text)
        rows.append(row)
    need(len(rows) == manifest["records"]
         and dict(stratum_bytes) == manifest["s9_mixture"]["selected_bytes_by_stratum"],
         "training membership or language accounting changed")
    return manifest, rows


def _sealed(core: dict[str, Any]) -> dict[str, Any]:
    return {**core, "manifest_sha256": books.sha(books.canonical(core))}


def _publish(destination: Path, name: str, raw: bytes) -> str:
    path = destination / name
    if not path.exists() and not path.is_symlink():
        _atomic_write(destination, path, raw)
    need(_read_destination(path) == raw, "immutable bundle member differs from verified bytes")
    return books.sha(raw)


def stage(root: Path, train: Path, destination: Path) -> dict[str, Any]:
    need(not any(path.is_symlink() for path in (destination, *destination.parents)),
         "unsafe bundle destination")
    if destination.exists():
        need(destination.is_dir() and all(
            member.name in OUTPUT_MEMBERS and not member.is_symlink() for member in destination.iterdir()
        ), "unexpected or unsafe bundle member")
    admission, rows = read_admission(root, train)
    tokenizer, tokenizer_proof = frozen.load_frozen(root)
    role_manifests = reservations.reserved_role_manifests(root, admission)
    need(set(role_manifests) == {"validation", "test"}, "missing reserved evaluation roles")
    roles = {
        "train": {
            "records": admission["records"],
            "physical_payload_sha256": admission["payload_sha256"],
            "source_membership_sha256": admission["manifest_sha256"],
        },
        **{role: {"manifest_sha256": value["manifest_sha256"], "records": value["records"]}
           for role, value in role_manifests.items()},
    }
    split = _sealed({
        "schema": "12-6.plan2-s17-physical-split.v1",
        "s16_admission_manifest_sha256": admission["manifest_sha256"],
        "roles": roles,
    })
    destination.mkdir(parents=True, exist_ok=True)
    files = {
        "split-manifest.json": books.canonical(split),
        "tokenizer-manifest.json": books.canonical(tokenizer_proof),
        "frozen-tokenizer.json.gz": books.read_checked(root, frozen.ARTIFACT),
        "train-sources.json": books.canonical(
            [{key: value for key, value in row.items() if key != "text"} for row in rows],
        ),
        **{f"{role}-manifest.json": books.canonical(value) for role, value in role_manifests.items()},
    }
    member_hashes = {name: _publish(destination, name, raw) for name, raw in files.items()}
    packed = packing.stage_train_shards(destination, tokenizer, rows)
    packed = _sealed({
        **packed,
        "cluster_split_sha256": split["manifest_sha256"],
        "tokenizer_manifest_sha256": tokenizer_proof["source_manifest_sha256"],
    })
    member_hashes["packing-manifest.json"] = _publish(
        destination, "packing-manifest.json", books.canonical(packed),
    )
    ledger = exposure.stage_exposure(destination, packed, tokenizer.identity.to_dict())
    need(ledger["target_count"] == packed["target_count"], "packing and exposure target totals differ")
    member_hashes["exposure-ledger.json"] = books.sha(_read_destination(destination / "exposure-ledger.json"))
    core = {
        "schema": SCHEMA,
        "s16_accepted_main": S16_ACCEPTED_MAIN,
        "s16_admission_manifest_sha256": admission["manifest_sha256"],
        "split_manifest_sha256": split["manifest_sha256"],
        "tokenizer_identity": tokenizer.identity.to_dict(),
        "packing_manifest_sha256": packed["manifest_sha256"],
        "exposure_manifest_sha256": ledger["manifest_sha256"],
        "train_records": admission["records"],
        "train_target_count": packed["target_count"],
        "packing_shards": len(packed["shards"]),
        "tokenizer_scale_profiles": tokenizer_proof["compatible_scale_profiles"],
        "source_coverage": admission["coverage"],
        "source_bytes_by_stratum": admission["s9_mixture"]["selected_bytes_by_stratum"],
        "member_file_sha256": member_hashes,
        "production_release_authorized": False,
        "model_training_executed": False,
    }
    manifest = _sealed(core)
    _publish(destination, MANIFEST, books.canonical(manifest))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = stage(args.root.resolve(strict=True), args.train, args.out_dir)
    except BundleDenied as exc:
        print(json.dumps({"decision": "DENIED", "reason": str(exc)}))
        return 2
    except (KeyError, OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"decision": "DENIED", "error_type": type(exc).__name__}))
        return 2
    print(json.dumps({"decision": "PHYSICAL_BUNDLE_VERIFIED", "manifest_sha256": result["manifest_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
