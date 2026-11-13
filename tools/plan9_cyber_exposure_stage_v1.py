"""Stage a local train-only development bundle and seal every target identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from tools import plan2_s17_exposure_stream_v1 as exposure
from tools import plan2_s17_frozen_tokenizer_v1 as frozen
from tools import plan9_scale500m_launch_preparation_v1 as preparation

FROZEN_TOKENIZER_FILE_SHA256 = "eb9d4981271e153cb091889c022d76a17a14cd9337bef7268e06af805ba1d21e"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode()


def write(root: Path, name: str, body: dict) -> str:
    raw = canonical(body)
    (root / name).write_bytes(raw)
    return sha(raw)


def sealed(body: dict) -> dict:
    return {**body, "manifest_sha256": sha(canonical(body))}


def stage(root: Path, candidate: Path, out: Path) -> dict:
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    checked = preparation.prepare_cyber_gpu_profiles(root, candidate)
    receipt_bytes = (candidate / "receipt.json").read_bytes()
    if sha(receipt_bytes) != checked["evidence_sha256"]["corpus"]:
        raise ValueError("candidate receipt changed after inspection")
    receipt = json.loads(receipt_bytes)
    sources = receipt["files"]
    _, tokenizer = frozen.load_frozen(root)
    frozen_payload = (root / frozen.ARTIFACT).read_bytes()
    if sha(frozen_payload) != FROZEN_TOKENIZER_FILE_SHA256:
        raise ValueError("frozen tokenizer file bytes differ from the verified candidate")
    if tokenizer["tokenizer_identity"] != receipt["tokenizer"]:
        raise ValueError("frozen tokenizer differs from candidate")
    split = sealed({
        "schema": "12-6.plan9-cyber-development-train-split.v1",
        "roles": {
            "train": {"candidate_text_sha256": sources["train.jsonl"],
                      "records": receipt["records"]["cyber_train"] + receipt["records"]["general_train"]},
            "validation": {"payload_included": False, "candidate_text_sha256": sources["validation.jsonl"]},
            "test": {"payload_included": False, "candidate_text_sha256": sources["test.jsonl"]},
        },
        "screened_holdout_receipt_sha256": checked["evidence_sha256"]["holdout"],
        "independent_evaluation_receipt_sha256": checked["evidence_sha256"]["attack_eval"],
    })
    raw_pack = (candidate / "packing-manifest.json").read_bytes()
    if sha(raw_pack) != sources["packing-manifest.json"]:
        raise ValueError("candidate packing changed after inspection")
    pack = sealed({
        **json.loads(raw_pack),
        "cluster_split_sha256": split["manifest_sha256"],
        "tokenizer_manifest_sha256": tokenizer["source_manifest_sha256"],
    })
    with tempfile.TemporaryDirectory(dir=out.parent) as temp:
        dest = Path(temp) / "bundle"
        dest.mkdir()
        (dest / "shards").mkdir()
        for ref in pack["shards"]:
            path = ref["path"]
            shutil.copyfile(candidate / path, dest / path)
        rights = {}
        for name, expected in sources.items():
            if name in {"train.jsonl", "validation.jsonl", "test.jsonl", "packing-manifest.json"}:
                continue
            raw = (candidate / name).read_bytes()
            if sha(raw) != expected:
                raise ValueError(f"rights member changed: {name}")
            target = dest / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            rights[name] = expected
        (dest / "frozen-tokenizer.json.gz").write_bytes(frozen_payload)
        files = {
            "split-manifest.json": write(dest, "split-manifest.json", split),
            "tokenizer-manifest.json": write(dest, "tokenizer-manifest.json", tokenizer),
            "packing-manifest.json": write(dest, "packing-manifest.json", pack),
            "frozen-tokenizer.json.gz": FROZEN_TOKENIZER_FILE_SHA256,
            **rights,
        }
        ledger = exposure.stage_exposure(dest, pack, tokenizer["tokenizer_identity"])
        if ledger["target_count"] != sum(receipt["causal_targets"].values()):
            raise ValueError("ledger target total differs from the candidate")
        files[exposure.MANIFEST] = sha((dest / exposure.MANIFEST).read_bytes())
        manifest = sealed({
            "schema": "12-6.plan9-cyber-development-train-exposure.v1",
            "scope": "LOCAL_DEVELOPMENT_ONLY",
            "candidate_receipt_sha256": checked["evidence_sha256"]["corpus"],
            "files": files,
            "exposure_manifest_sha256": ledger["manifest_sha256"],
            "exposure_chain_head_sha256": ledger["chain_head_sha256"],
            "shards": ledger["source_shard_count"],
            "targets": ledger["target_count"],
            "blocks": ledger["block_count"],
            "raw_training_text_included": False,
            "evaluation_payload_opened": False,
            "external_data_transfer_authorized": False,
            "campaign_authorized": False,
            "training_executed": False,
        })
        write(dest, "receipt.json", manifest)
        dest.rename(out)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(stage(Path.cwd(), args.candidate, args.out), sort_keys=True))
