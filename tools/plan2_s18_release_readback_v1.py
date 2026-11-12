"""Read the accepted S17 training bundle against repository-pinned evidence."""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any

from tools import plan2_public_domain_books_v1 as books
from tools import plan2_s17_clean_rebuild_witness_v1 as witness
from tools import plan2_s17_frozen_tokenizer_v1 as frozen
from tools import plan2_s17_physical_bundle_v1 as physical
from tools import plan2_s17_reserved_roles_v1 as reservations
from tools import plan2_s18_release_integrity_v1 as integrity

S17_ACCEPTED_MAIN = "613fede87d0da2fa051e51cd7041aacc796f7c9b"
QUALIFICATION = "evidence/plan2_s17/physical_bundle_receipt_v1.json"
QUALIFICATION_FILE_SHA256 = "0d1e229cf80e8fb4186d9d0080a7f7b395f953e4611bcd018329167459b4e0bd"
SCHEMA = "12-6.plan2-s18-physical-release-readback.v1"
MAX_METADATA_BYTES = 16 * 1024 * 1024
EXPECTED_METADATA = {
    "exposure-ledger.json", "frozen-tokenizer.json.gz", "packing-manifest.json",
    "split-manifest.json", "test-manifest.json", "tokenizer-manifest.json",
    "train-sources.json", "validation-manifest.json",
}


def _need(condition: bool, reason: str) -> None:
    if not condition:
        raise integrity.IntegrityDenied(reason)


def _repository_receipt(root: Path, name: str, expected_sha256: str) -> dict[str, Any]:
    # Git for Windows may check canonical repository JSON out with CRLF.
    raw = books.read_checked(root, name).replace(b"\r\n", b"\n")
    _need(witness.digest(raw) == expected_sha256, "accepted repository receipt changed")
    return json.loads(raw)


def _plain(path: Path, *, directory: bool = False) -> bool:
    entry = path.lstat()
    reparse = getattr(entry, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if directory:
        return stat.S_ISDIR(entry.st_mode) and not reparse
    return stat.S_ISREG(entry.st_mode) and not reparse and entry.st_nlink == 1


def _metadata(bundle: Path, name: str, expected_sha256: str) -> dict[str, Any]:
    path = bundle / name
    _need(_plain(path), "unsafe physical bundle metadata member")
    raw = integrity._read_regular(path, limit=MAX_METADATA_BYTES)
    _need(witness.digest(raw) == expected_sha256, "accepted bundle metadata changed: " + name)
    return json.loads(raw)


def _inventory_shape(bundle: Path, members: set[str]) -> None:
    # Reject stray raw/heldout payloads before opening any shard or exposure file.
    directories = {"shards", "exposures"}
    for folder in (bundle, *(bundle / name for name in sorted(directories))):
        _need(_plain(folder, directory=True), "unsafe physical bundle directory")
        prefix = "" if folder == bundle else folder.name + "/"
        allowed = ({name for name in members if "/" not in name} | directories
                   if folder == bundle else {name[len(prefix):] for name in members if name.startswith(prefix)})
        observed = set()
        with os.scandir(folder) as entries:
            for entry in entries:
                _need(entry.name in allowed, "unlisted physical bundle member")
                _need(_plain(Path(entry.path), directory=folder == bundle and entry.name in directories),
                      "unsafe physical bundle member")
                observed.add(entry.name)
        _need(observed == allowed, "missing physical bundle member")


def verify_physical_bundle(root: Path, bundle: Path) -> dict[str, Any]:
    """Check all S17 files against the accepted witness before S18 staging."""
    root, bundle = Path(root).absolute(), Path(bundle).absolute()
    try:
        _need(all(_plain(path, directory=True) for path in (bundle, *bundle.parents)),
              "unsafe physical bundle root or ancestor")
        qualified = _repository_receipt(root, QUALIFICATION, QUALIFICATION_FILE_SHA256)
        admission = _repository_receipt(root, physical.S16_MANIFEST, physical.S16_MANIFEST_FILE_SHA256)
        physical._checked_source_terms(root, admission)
        expected_bundle = qualified["bundle"]
        byte_witness = qualified["byte_witness"]
        _need(qualified["independent_clean_checkout_verified"] is True
              and qualified["independent_shard_and_exposure_readback_verified"] is True
              and byte_witness["byte_identical"] is True
              and byte_witness["expected_inventory_sha256"] == byte_witness["observed_inventory_sha256"],
              "accepted independent S17 checks missing")
        _need(admission["physical_s3_s9_admission_passed"] is True
              and expected_bundle["s16_admission_manifest_sha256"] == admission["manifest_sha256"]
              and expected_bundle["train_records"] == admission["records"],
              "accepted S16 admission and S17 bundle do not agree")
        expected = dict(expected_bundle["member_file_sha256"])
        _need(set(expected) == EXPECTED_METADATA, "unexpected accepted metadata members")
        expected[physical.MANIFEST] = witness.digest(witness.canonical(expected_bundle))
        manifest = _metadata(bundle, physical.MANIFEST, expected[physical.MANIFEST])
        packing = _metadata(bundle, "packing-manifest.json", expected["packing-manifest.json"])
        exposure = _metadata(bundle, "exposure-ledger.json", expected["exposure-ledger.json"])
        split = _metadata(bundle, "split-manifest.json", expected["split-manifest.json"])
        tokenizer = _metadata(bundle, "tokenizer-manifest.json", expected["tokenizer-manifest.json"])
        _, expected_tokenizer = frozen.load_frozen(root)
        _need(tokenizer == expected_tokenizer and tokenizer["tokenizer_identity"] == manifest["tokenizer_identity"]
              and tokenizer["source_manifest_sha256"] == packing["tokenizer_manifest_sha256"]
              == exposure["tokenizer_manifest_sha256"],
              "accepted frozen tokenizer lineage changed")
        _need(split["manifest_sha256"] == packing["cluster_split_sha256"] == exposure["cluster_split_sha256"]
              == manifest["split_manifest_sha256"]
              and split["roles"]["train"]["physical_payload_sha256"] == admission["payload_sha256"]
              and packing["manifest_sha256"] == exposure["packing_manifest_sha256"]
              == manifest["packing_manifest_sha256"]
              and exposure["manifest_sha256"] == manifest["exposure_manifest_sha256"]
              and exposure["chain_head_sha256"] == qualified["exposure_chain_sha256"],
              "accepted split, packing, or exposure lineage changed")
        for role, accepted in reservations.reserved_role_manifests(root, admission).items():
            name = role + "-manifest.json"
            reserved = _metadata(bundle, name, expected[name])
            _need(reserved == accepted and split["roles"][role]["manifest_sha256"] == reserved["manifest_sha256"]
                  and split["roles"][role]["records"] == reserved["records"],
                  "independent hash-only reservation changed")
        shards, exposures = packing["shards"], exposure["exposure_shards"]
        _need(len(shards) == len(exposures) == manifest["packing_shards"] == exposure["source_shard_count"],
              "accepted physical shard count changed")
        targets, blocks = 0, 0
        for index, (shard, events) in enumerate(zip(shards, exposures, strict=True)):
            name = f"shards/shard-{index:06d}.json"
            event_name = f"exposures/shard-{index:06d}.json"
            _need(shard["path"] == name == events["source_shard_path"] and events["path"] == event_name
                  and shard["target_count"] == events["exposure_count"],
                  "accepted physical shard or exposure mapping changed")
            expected[name], expected[event_name] = shard["sha256"], events["sha256"]
            targets += shard["target_count"]
            blocks += shard["block_count"]
        _need(targets == manifest["train_target_count"] == packing["target_count"] == exposure["target_count"]
              and blocks == packing["block_count"] == exposure["block_count"],
              "accepted target or block accounting changed")
        inventory_sha256 = witness.digest(witness.canonical(expected))
        _need(inventory_sha256 == byte_witness["expected_inventory_sha256"]
              and len(expected) == byte_witness["member_count"],
              "computed inventory differs from the accepted clean rebuild")
        _inventory_shape(bundle, set(expected))
        before = witness._physical_tree_pin(bundle)
        observed, total = witness.inventory(bundle)
        _need(observed == expected and total == byte_witness["total_member_bytes"],
              "physical release bytes differ from the accepted inventory")
        _inventory_shape(bundle, set(expected))
        _need(witness._physical_tree_pin(bundle) == before,
              "physical release members changed during readback")
    except integrity.IntegrityDenied:
        raise
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise integrity.IntegrityDenied("unverifiable accepted physical release bundle") from exc
    return {
        "schema": SCHEMA,
        "s17_accepted_main": S17_ACCEPTED_MAIN,
        "qualification_receipt_sha256": qualified["receipt_sha256"],
        "qualification_file_sha256": QUALIFICATION_FILE_SHA256,
        "bundle_manifest": manifest,
        "inventory_sha256": inventory_sha256,
        "member_sha256": observed,
        "member_count": len(observed),
        "total_member_bytes": total,
        "training_records": manifest["train_records"],
        "training_shards": len(shards),
        "exposure_shards": len(exposures),
        "causal_targets": targets,
        "blocks": blocks,
        "reserved_role_counts": {role: split["roles"][role]["records"] for role in ("validation", "test")},
        "release_authority_granted_by_readback": False,
    }
