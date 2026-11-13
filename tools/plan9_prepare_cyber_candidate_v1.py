"""Prepare a physical cyber-majority corpus candidate without launching training."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from tempfile import TemporaryDirectory

from tools import plan2_s15_physical_tokenizer_candidate_v1 as segmenter
from tools import plan2_s17_frozen_tokenizer_v1 as frozen
from tools import plan2_s17_physical_bundle_v1 as accepted
from tools import plan2_s17_train_packing_v1 as packing
from tools.plan2_s18_physical_release_v1 import open_release

PINS = {
    "cisa-kev.json": "41c8091ad62d508d6f8906ad5cd26b20efe551f7d51b95ecc8a6981fa640b4ea",
    "cisa-license.txt": "8dfcf2d0cba33bede386eb470daf2ac08caf6540fc2516baab10dd546f90938a",
    "nist-800-53.json": "1645df6a370dcb931db2e2d5d70c2f77bc89c38499a416c23a70eb2c0e595bcc",
    "nist-license.txt": "9f0ddd1ed71f691bb483965a7c46e1315ab6cc28dac0426d4c3d27108aafde13",
}
SOURCES = {
    "cisa": {
        "commit": "6b8d7ba579672e4f84e2b1f255724aa97c8455b4",
        "source": "https://github.com/cisagov/kev-data/tree/6b8d7ba579672e4f84e2b1f255724aa97c8455b4",
        "license": "CC0-1.0",
    },
    "nist": {
        "commit": "bc8a528770033611df899b3d52703fb3dc91a20d",
        "source": "https://github.com/usnistgov/oscal-content/tree/bc8a528770033611df899b3d52703fb3dc91a20d",
        "license": "CC0-1.0",
    },
}
CYBER_PERCENT = 75
GENERAL_PERCENT = 100 - CYBER_PERCENT
NIST_TEXT_PARTS = {"statement", "guidance", "item"}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def pinned(directory: Path, name: str) -> bytes:
    raw = (directory / name).read_bytes()
    if sha(raw) != PINS[name]:
        raise ValueError(f"source bytes changed: {name}")
    return raw


def general_attribution(release: Path, expected_manifest_sha256: str) -> dict[str, bytes]:
    manifest = (release / "release-manifest.json").read_bytes()
    if sha(manifest) != expected_manifest_sha256:
        raise ValueError("accepted general release manifest changed")
    raw = (release / "rights-attribution.json").read_bytes()
    if sha(raw) != json.loads(manifest)["members"]["rights-attribution.json"]:
        raise ValueError("accepted general corpus attribution changed")
    rights = json.loads(raw)
    if rights["scope"] != "LOCAL_PLAN9_TRAINING_DATA_HANDOFF" or rights["public_redistribution_authorized"] is not False:
        raise ValueError("general corpus rights scope differs from the accepted release")
    result = {"general-rights/rights-attribution.json": raw}
    for name, expected in rights["notice_sha256"].items():
        path = Path(name)
        if path.is_absolute() or path.parts[0] != "notices" or ".." in path.parts or "\\" in name:
            raise ValueError("invalid general corpus notice path")
        member = (release / path).read_bytes()
        if sha(member) != expected:
            raise ValueError("accepted general corpus notice changed")
        result[f"general-rights/{name}"] = member
    return result


def row(family: str, item: str, text: str) -> dict:
    raw = text.encode()
    return {
        "record_id_sha256": sha(f"{family}:{item}".encode()),
        "source_id": f"{family}:{item}",
        "source_family": family,
        "stratum": "cyber",
        "utf8_bytes": len(raw),
        "content_sha256": sha(raw),
        "text": text,
    }


def group_role(key: str) -> str:
    bucket = int(sha(key.encode())[:8], 16) % 100
    if bucket < 10:
        return "test"
    if bucket < 20:
        return "validation"
    return "train"


def prose(parts) -> list[str]:
    texts = []
    for part in parts:
        if part["name"] in NIST_TEXT_PARTS:
            if part.get("prose"):
                texts.append(part["prose"])
            texts.extend(prose(part.get("parts", [])))
    return texts


def nist_controls(controls, group: str):
    for control in controls:
        paragraphs = prose(control.get("parts", []))
        if paragraphs:
            prefix = f"NIST SP 800-53 Rev. 5, {control['id'].upper()}: {control['title']}"
            yield group, row("nist-sp800-53-r5", control["id"], "\n\n".join([prefix, *paragraphs]))
        yield from nist_controls(control.get("controls", []), group)


def cyber_rows(directory: Path):
    cisa = json.loads(pinned(directory, "cisa-kev.json"))
    cves = cisa["vulnerabilities"]
    cve_ids = {item["cveID"] for item in cves}
    if cisa["catalogVersion"] != "2026.11.10" or len(cves) != 1779 or cisa["count"] != 1779:
        raise ValueError("unexpected CISA release")
    if len(cve_ids) != 1779 or not all(re.fullmatch(r"CVE-\d{4}-\d+", item) for item in cve_ids):
        raise ValueError("invalid or duplicate CISA identifiers")
    for entry in cves:
        cve = entry["cveID"]
        vendor = entry["vendorProject"]
        title = f"{cve} | {vendor} | {entry['product']} | {entry['vulnerabilityName']}"
        body = "\n".join((title, entry["shortDescription"], entry["requiredAction"]))
        yield "cisa:" + normalized(vendor), row("cisa-kev", cve, body)
    nist = json.loads(pinned(directory, "nist-800-53.json"))["catalog"]
    if nist["metadata"]["version"] != "5.2.0" or len(nist["groups"]) != 20:
        raise ValueError("unexpected NIST release")
    for family in nist["groups"]:
        yield from nist_controls(family["controls"], "nist:" + family["id"])


def target_count(tokenizer, item: dict) -> int:
    return sum(
        len(tokenizer.encode(part["text"], add_bos=True, add_eos=True)) - 1
        for part in segmenter.segments(item["record_id_sha256"], item["text"])
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", required=True, type=Path)
    parser.add_argument("--general-train", required=True, type=Path)
    parser.add_argument("--general-release", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    root = Path.cwd()
    authority = open_release(root, args.general_release)
    manifest, general = accepted.read_admission(root, args.general_train)
    tokenizer, tokenizer_receipt = frozen.load_frozen(root)
    groups = {"train": [], "validation": [], "test": []}
    protected = {
        item["content_sha256"]
        for role in ("validation", "test")
        for item in json.loads((args.general_release / f"{role}-manifest.json").read_bytes())["members"]
    }
    used_content = {normalized(item["text"]) for item in general}
    ids = set()
    for group, item in sorted(cyber_rows(args.sources), key=lambda row: (row[0], row[1]["source_id"])):
        key = normalized(item["text"])
        if item["record_id_sha256"] in ids:
            raise ValueError(f"duplicate source record {item['source_id']}")
        ids.add(item["record_id_sha256"])
        if key in used_content or item["content_sha256"] in protected:
            continue
        used_content.add(key)
        groups[group_role(group)].append(item)
    if any({item["source_family"] for item in group} != {"cisa-kev", "nist-sp800-53-r5"} for group in groups.values()):
        raise ValueError("a cyber split is missing a source family")
    cyber = groups["train"]
    cyber_targets = sum(target_count(tokenizer, item) for item in cyber)
    general_budget = cyber_targets * GENERAL_PERCENT // CYBER_PERCENT
    general_targets = 0
    selected_general = []
    for item in sorted(general, key=lambda item: sha(("cyber-general-v1:" + item["record_id_sha256"]).encode())):
        targets = target_count(tokenizer, item)
        if general_targets + targets <= general_budget:
            selected_general.append({**item, "stratum": "general"})
            general_targets += targets
        if general_targets == general_budget:
            break
    if not selected_general or general_targets < general_budget * 0.98:
        raise ValueError("general target share is outside the preparation band")
    train = cyber + selected_general
    all_ids = [item["record_id_sha256"] for rows in (train, groups["validation"], groups["test"]) for item in rows]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("duplicate record ID across partitions")
    with TemporaryDirectory(dir=args.out.parent) as temporary:
        staged = Path(temporary) / "candidate"
        staged.mkdir()
        pack = packing.stage_train_shards(staged, tokenizer, train)
        if pack["actual_loss_targets_by_stratum"] != {"cyber": cyber_targets, "general": general_targets}:
            raise ValueError("packed causal target counts changed")
        files = {
            "train.jsonl": b"".join(map(canonical, sorted(train, key=lambda item: item["record_id_sha256"]))),
            "validation.jsonl": b"".join(map(canonical, groups["validation"])),
            "test.jsonl": b"".join(map(canonical, groups["test"])),
            "packing-manifest.json": canonical(pack),
            "LICENSE-cisa.txt": pinned(args.sources, "cisa-license.txt"),
            "LICENSE-nist.txt": pinned(args.sources, "nist-license.txt"),
            **general_attribution(args.general_release, authority["release_manifest_sha256"]),
        }
        for name, raw in files.items():
            path = staged / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        receipt = {
            "schema": "12-6.cyber-majority-corpus-candidate.v1",
            "scope": "SOURCE_VERIFIED_DEVELOPMENT_CANDIDATE",
            "general_authority_sha256": authority["authority_sha256"],
            "general_admission_sha256": manifest["manifest_sha256"],
            "general_rights_scope": "LOCAL_PLAN9_TRAINING_DATA_HANDOFF",
            "public_redistribution_authorized": False,
            "sources": SOURCES,
            "source_files_sha256": PINS,
            "tokenizer": tokenizer_receipt["tokenizer_identity"],
            "group_split_policy": "sha256-10pct-test-10pct-validation-by-cisa-vendor-or-nist-control-family",
            "exact_decontamination": "NFKC-casefold-whitespace text and released general held-out content SHA-256",
            "minimum_cyber_percent_of_loss_positions": CYBER_PERCENT,
            "records": {
                "cyber_train": len(cyber), "general_train": len(selected_general),
                "cyber_validation": len(groups["validation"]), "cyber_test": len(groups["test"]),
            },
            "blocks": pack["block_count"],
            "causal_targets": pack["actual_loss_targets_by_stratum"],
            "cyber_share_ppm": cyber_targets * 1_000_000 // (cyber_targets + general_targets),
            "trained": False,
            "campaign_authorized": False,
            "files": {name: sha(raw) for name, raw in files.items()},
        }
        (staged / "receipt.json").write_bytes(canonical(receipt))
        staged.rename(args.out)
    print(json.dumps({"receipt_sha256": sha(canonical(receipt)), **receipt}, sort_keys=True))


if __name__ == "__main__":
    main()
