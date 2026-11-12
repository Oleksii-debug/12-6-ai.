"""Stage and open the verified Plan 2 dataset for a separate Plan 9 campaign."""
from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from tools import plan2_public_domain_books_v1 as books
from tools import plan2_s17_physical_bundle_v1 as s17
from tools import plan2_s18_release_integrity_v1 as integrity
from tools import plan2_s18_release_readback_v1 as readback
from tools import plan2_s18_source_notices_v1 as notices
from tools.plan2_physical_materialization_v1 import _atomic_write, _read_destination

AUTHORITY = "evidence/plan2_s18/release_authority_v1.json"
AUTHORITY_SCHEMA = "12-6.plan2-s18-local-plan9-release-authority.v1"
AUTHORITY_FILE_SHA256 = "e90cd3a2052a6808da7208fe9a304b4f8bbbb67c46329016210f9fda239eca78"
S17_ACCEPTED_MAIN = "613fede87d0da2fa051e51cd7041aacc796f7c9b"
S17_ACCEPTED_TREE = "ed2f479102665542cfe263fbcdad2b397c59a146"


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise integrity.IntegrityDenied(reason)


def _read(root: Path, name: str) -> bytes:
    path = root / name
    _require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 15_000_000,
             "missing or oversized physical release input")
    return _read_destination(path)


def _write(root: Path, name: str, raw: bytes) -> None:
    path = root / name
    _require(not any(part.is_symlink() for part in (path, *path.parents)),
             "unsafe release publication member")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        _atomic_write(path.parent, path, raw)
    _require(_read_destination(path) == raw, "immutable release member changed")


def stage_candidate(
    root: Path, bundle: Path, destination: Path, notice_cache: Path | None = None,
) -> dict[str, Any]:
    """Produce a complete local release candidate from the accepted data."""
    from tools import plan2_s18_plan9_consumer_v1 as consumer

    _require(not any(path.is_symlink() for path in (destination, *destination.parents)),
             "unsafe release destination")
    bundle = bundle.resolve(strict=True)
    target = destination.resolve()
    _require(bundle != target and not bundle.is_relative_to(target)
             and not target.is_relative_to(bundle), "source and release must use independent directories")
    evidence = readback.verify_physical_bundle(root, bundle)
    bundle_manifest = evidence["bundle_manifest"]
    plan9 = consumer.qualify_plan9_consumer(root, bundle, bundle_manifest)
    source_notices = notices.build_source_notices(root, notice_cache)
    admission = books.read_checked(root, s17.S16_MANIFEST).replace(b"\r\n", b"\n")
    _require(books.sha(admission) == s17.S16_MANIFEST_FILE_SHA256, "admitted dataset manifest changed")
    admission_doc = json.loads(admission)
    tokenizer = _read(bundle, "tokenizer-manifest.json")
    packing = json.loads(_read(bundle, "packing-manifest.json"))
    exposure = json.loads(_read(bundle, "exposure-ledger.json"))
    dataset_sha, tokenizer_sha = books.sha(admission), books.sha(tokenizer)
    shards = books.canonical({
        "dataset_sha256": dataset_sha,
        "tokenizer_sha256": tokenizer_sha,
        "packing_manifest_sha256": packing["manifest_sha256"],
        "target_count": packing["target_count"],
        "shards": packing["shards"],
    })
    exposure_manifest = books.canonical({
        "shards_sha256": books.sha(shards),
        "exposure_ledger_sha256": exposure["manifest_sha256"],
        "chain_head_sha256": exposure["chain_head_sha256"],
        "target_count": exposure["target_count"],
        "exposure_shards": exposure["exposure_shards"],
    })
    rights = books.canonical({
        "dataset_sha256": dataset_sha,
        "selected_source_terms": admission_doc["selected_source_terms"],
        "notice_sha256": {name: books.sha(raw) for name, raw in sorted(source_notices.items())},
        "scope": "LOCAL_PLAN9_TRAINING_DATA_HANDOFF",
        "evaluation_payloads_included": False,
        "public_redistribution_authorized": False,
    })
    binding = books.canonical({
        "dataset_sha256": dataset_sha,
        "tokenizer_sha256": tokenizer_sha,
        "shards_sha256": books.sha(shards),
        "exposure_sha256": books.sha(exposure_manifest),
        "rights_sha256": books.sha(rights),
        "physical_consumer": plan9,
        "campaign_authorized": False,
    })
    descriptors = {
        "dataset-manifest.json": admission,
        "shards-manifest.json": shards,
        "exposure-manifest.json": exposure_manifest,
        "rights-attribution.json": rights,
        "plan9-binding.json": binding,
        **source_notices,
    }
    physical = evidence["member_sha256"]
    _require(not physical.keys() & descriptors.keys(), "release metadata overlaps physical bundle members")
    inventory = {**physical, **{name: books.sha(raw) for name, raw in descriptors.items()}}
    manifest = {
        "schema_version": integrity.SCHEMA,
        "producer_commit_sha1": S17_ACCEPTED_MAIN,
        "producer_tree_sha1": S17_ACCEPTED_TREE,
        "members": dict(sorted(inventory.items())),
    }
    raw_manifest = books.canonical(manifest)
    expected = set(inventory) | {integrity.MANIFEST}
    allowed_directories = {
        folder.as_posix() for name in expected for folder in Path(name).parents if folder != Path(".")
    }
    for member in destination.rglob("*"):
        _require(not member.is_symlink(), "linked preexisting release member")
        name = member.relative_to(destination).as_posix()
        _require((member.is_file() and name in expected)
                 or (member.is_dir() and name in allowed_directories),
                 "unlisted preexisting release member")
    for name, digest in physical.items():
        data = _read(bundle, name)
        _require(books.sha(data) == digest, "physical bundle member changed during release")
        _write(destination, name, data)
    for name, data in descriptors.items():
        _write(destination, name, data)
    _write(destination, integrity.MANIFEST, raw_manifest)
    pin = books.sha(raw_manifest)
    check = integrity.verify(destination, pin)
    return {
        "release_manifest_sha256": pin,
        "inventory_sha256": books.sha(books.canonical(inventory)),
        "bundle_manifest_sha256": bundle_manifest["manifest_sha256"],
        "s17_qualification_receipt_sha256": evidence["qualification_receipt_sha256"],
        "members": check["member_count"],
        "training_targets": bundle_manifest["train_target_count"],
        "plan9_binding_sha256": books.sha(binding),
        "rights_attribution_sha256": books.sha(rights),
    }


def open_release(root: Path, destination: Path) -> dict[str, Any]:
    """Consume a locally materialized release against the repository's approved pin."""
    raw = books.read_checked(root, AUTHORITY).replace(b"\r\n", b"\n")
    authority = json.loads(raw)
    core = {key: value for key, value in authority.items() if key != "authority_sha256"}
    _require(books.sha(raw) == AUTHORITY_FILE_SHA256
             and raw == books.canonical(authority)
             and authority["authority_sha256"] == books.sha(books.canonical(core))
             and authority["schema"] == AUTHORITY_SCHEMA
             and authority["decision"] == "ACCEPT_LOCAL_PLAN9_DATA_RELEASE"
             and authority["s17_accepted_main"] == S17_ACCEPTED_MAIN
             and authority["campaign_authorized"] is False,
             "missing accepted Plan 2 release authority")
    admitted = books.read_checked(root, s17.S16_MANIFEST).replace(b"\r\n", b"\n")
    _require(books.sha(admitted) == s17.S16_MANIFEST_FILE_SHA256, "admitted dataset manifest changed")
    s17._checked_source_terms(root, json.loads(admitted))
    check = integrity.verify(destination, authority["release_manifest_sha256"])
    _require(books.sha(books.canonical(check["member_sha256"])) == authority["inventory_sha256"]
             and check["member_sha256"]["plan9-binding.json"] == authority["plan9_binding_sha256"]
             and check["member_sha256"]["rights-attribution.json"] == authority["rights_attribution_sha256"],
             "trusted release authority changed")
    return {
        "release_manifest_sha256": authority["release_manifest_sha256"],
        "authority_sha256": authority["authority_sha256"],
        "training_targets": authority["training_targets"],
        "plan9_data_release_verified": True,
        "campaign_authorized": False,
    }


def iter_release_batches(
    root: Path, destination: Path, *, batch_size: int = 16,
) -> Iterator[dict[str, Any]]:
    """Provide aligned Trainer inputs from the approved local release."""
    from tools import plan2_s18_plan9_consumer_v1 as consumer

    open_release(root, destination)
    packed = json.loads(_read(destination, "packing-manifest.json"))
    yield from consumer.iter_plan9_batches(destination, packed, batch_size=batch_size)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--notice-cache", type=Path)
    parser.add_argument("--verify-release", action="store_true")
    args = parser.parse_args()
    try:
        root = args.root.resolve(strict=True)
        if args.verify_release:
            result = open_release(root, args.out_dir)
        else:
            _require(args.bundle is not None, "physical Section 17 bundle is required")
            result = stage_candidate(root, args.bundle, args.out_dir, args.notice_cache)
    except integrity.IntegrityDenied as exc:
        print(json.dumps({"decision": "DENIED", "reason": str(exc)}))
        return 2
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"decision": "DENIED", "error_type": type(exc).__name__}))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
