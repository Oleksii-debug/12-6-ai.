"""Screen the current cyber corpus holdout against its exact selected training."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

SCHEMA = "12-6.cyber-development-holdout-screen.v1"
PARTITIONS = ("train", "validation", "test")
FAMILIES = ("cisa-kev", "nist-sp800-53-r5")
WORD = re.compile(r"\w+", re.UNICODE)
CVE = re.compile(r"\bCVE-\d{4}-\d{4,}\b", re.IGNORECASE)
NIST_CONTROL = re.compile(r"\b[A-Z]{2}-\d{1,3}(?:\(\d{1,3}\))?(?![\w(])")
REASONS = ("identifier_seen_in_train", "substantive_paragraph_seen_in_train", "eightgram_coverage_ge_80_percent")


def canonical(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n").encode()


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def checked_read(path: Path, pin: str) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"missing or linked input: {path}")
    raw = path.read_bytes()
    if sha(raw) != pin:
        raise ValueError(f"input digest mismatch: {path}")
    return raw


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def words(text: str) -> list[str]:
    return WORD.findall(unicodedata.normalize("NFKC", text).casefold())


def eightgrams(text: str) -> set[tuple[str, ...]]:
    tokens = words(text)
    return {tuple(tokens[i:i + 8]) for i in range(len(tokens) - 7)}


def content_parts(row: dict) -> list[str]:
    if row["source_family"] == "cisa-kev":
        return row["text"].splitlines()[1:]
    if row["source_family"] == "nist-sp800-53-r5":
        return row["text"].split("\n\n")[1:]
    return row["text"].split("\n\n")


def groups_and_identifiers(source_dir: Path, receipt: dict) -> dict[str, tuple[str, str]]:
    pins = receipt["source_files_sha256"]
    cisa = json.loads(checked_read(source_dir / "cisa-kev.json", pins["cisa-kev.json"]))
    nist = json.loads(checked_read(source_dir / "nist-800-53.json", pins["nist-800-53.json"]))
    result = {}

    def put(source_id: str, group: str, identifier: str) -> None:
        if source_id in result:
            raise ValueError(f"duplicate upstream source ID: {source_id}")
        result[source_id] = (group, identifier)

    def nist_controls(controls: list[dict], group: str) -> None:
        for control in controls:
            control_id = control["id"]
            base, separator, enhancement = control_id.partition(".")
            identifier = base.upper() + (f"({enhancement})" if separator else "")
            if NIST_CONTROL.fullmatch(identifier) is None:
                raise ValueError(f"unexpected NIST control ID: {control_id}")
            put("nist-sp800-53-r5:" + control_id, group, identifier)
            nist_controls(control.get("controls", []), group)

    for item in cisa["vulnerabilities"]:
        put("cisa-kev:" + item["cveID"], "cisa:" + normalized(item["vendorProject"]), item["cveID"])
    for family in nist["catalog"]["groups"]:
        nist_controls(family["controls"], "nist:" + family["id"])
    return result


def load_candidate(
    candidate_dir: Path, source_dir: Path, expected_receipt_sha256: str,
) -> tuple[dict[str, list[dict]], dict[str, tuple[str, str]], dict]:
    raw = checked_read(candidate_dir / "receipt.json", expected_receipt_sha256)
    receipt = json.loads(raw)
    if raw != canonical(receipt) or receipt["schema"] != "12-6.cyber-majority-corpus-candidate.v1":
        raise ValueError("unexpected candidate receipt")
    upstream = groups_and_identifiers(source_dir, receipt)
    partitions = {}
    seen_ids = set()
    seen_text = set()
    group_roles = {}
    for role in PARTITIONS:
        name = role + ".jsonl"
        data = checked_read(candidate_dir / name, receipt["files"][name])
        rows = []
        for line in data.splitlines(keepends=True):
            row = json.loads(line)
            record_id, text = row["record_id_sha256"], row["text"]
            text_key = normalized(text)
            if line != canonical(row) or not text_key or sha(text.encode()) != row["content_sha256"]:
                raise ValueError(f"invalid {role} record")
            if record_id in seen_ids or text_key in seen_text:
                raise ValueError(f"duplicate record across candidate partitions: {record_id}")
            seen_ids.add(record_id)
            seen_text.add(text_key)
            if row["stratum"] == "cyber":
                if row["source_family"] not in FAMILIES:
                    raise ValueError(f"unexpected cyber family: {row['source_family']}")
                group, _ = upstream[row["source_id"]]
                bucket = int(sha(group.encode())[:8], 16) % 100
                expected = "test" if bucket < 10 else "validation" if bucket < 20 else "train"
                if group_roles.setdefault(group, role) != role or expected != role:
                    raise ValueError(f"upstream group moved between partitions: {group}")
            elif role != "train" or row["stratum"] != "general":
                raise ValueError(f"unsupported {role} stratum")
            rows.append(row)
        partitions[role] = rows
    return partitions, upstream, receipt


def screen(partitions: dict[str, list[dict]], upstream: dict[str, tuple[str, str]]) -> tuple[dict, dict, list]:
    paragraph_index = set()
    gram_index = set()
    identifiers = set()
    for row in partitions["train"]:
        text = unicodedata.normalize("NFKC", row["text"])
        identifiers.update(match.group().upper() for match in CVE.finditer(text))
        identifiers.update(match.group() for match in NIST_CONTROL.finditer(text.upper()))
        parts = content_parts(row)
        paragraph_index.update(tuple(tokens) for part in parts if len(tokens := words(part)) >= 12)
        gram_index.update(eightgrams(" ".join(parts)))

    summary = {}
    retained = {}
    exclusions = []
    for role in ("validation", "test"):
        retained[role] = []
        family_counts = defaultdict(Counter)
        original_groups = defaultdict(set)
        retained_groups = defaultdict(set)
        for row in partitions[role]:
            source_id = row["source_id"]
            family = row["source_family"]
            group, identifier = upstream[source_id]
            family_counts[family]["input"] += 1
            original_groups[family].add(group)
            parts = content_parts(row)
            grams = eightgrams(" ".join(parts))
            matched = len(grams & gram_index)
            reasons = []
            if identifier in identifiers:
                reasons.append(REASONS[0])
            if any(len(tokens := words(part)) >= 12 and tuple(tokens) in paragraph_index for part in parts):
                reasons.append(REASONS[1])
            if grams and matched * 5 >= len(grams) * 4:
                reasons.append(REASONS[2])
            if reasons:
                family_counts[family].update(reasons)
                exclusions.append({
                    "split": role,
                    "source_family": family,
                    "source_id": source_id,
                    "group": group,
                    "reasons": reasons,
                    "matched_train_8grams": matched,
                    "unique_body_8grams": len(grams),
                })
                continue
            retained[role].append(row)
            family_counts[family]["retained"] += 1
            retained_groups[family].add(group)
        summary[role] = {
            family: {
                "input": family_counts[family]["input"],
                "retained": family_counts[family]["retained"],
                "excluded": family_counts[family]["input"] - family_counts[family]["retained"],
                "reason_counts": {reason: family_counts[family][reason] for reason in REASONS},
                "original_group_count": len(original_groups[family]),
                "retained_groups": sorted(retained_groups[family]),
            }
            for family in FAMILIES
        }
    return summary, retained, exclusions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--expected-candidate-receipt-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.out.is_symlink():
        raise FileExistsError(args.out)
    partitions, upstream, candidate = load_candidate(
        args.candidate, args.sources, args.expected_candidate_receipt_sha256,
    )
    summary, retained, exclusions = screen(partitions, upstream)
    files = {role + ".jsonl": b"".join(map(canonical, rows)) for role, rows in retained.items()}
    files["exclusions.jsonl"] = b"".join(map(canonical, exclusions))
    report = {
        "schema": SCHEMA,
        "candidate_receipt_sha256": args.expected_candidate_receipt_sha256,
        "source_files_sha256": candidate["source_files_sha256"],
        "partition_sha256": {role: candidate["files"][role + ".jsonl"] for role in PARTITIONS},
        "training_records_screened": len(partitions["train"]),
        "scope": "SCREENED_SAME_SOURCE_DEVELOPMENT_HOLDOUT",
        "independent_evaluation_suite": False,
        "campaign_authorized": False,
        "criteria": {
            "identifiers": "held-out record ID as a whole CVE or exact NIST control/enhancement identifier in training text",
            "substantive_paragraph_min_words": 12,
            "word_normalization": "NFKC-casefold-unicode-word-tokens",
            "eightgram_overlap_percent": 80,
            "cyber_titles_excluded_from_text_matching": True,
            "reason_counts_can_overlap": True,
            "group_role_unchanged": True,
        },
        "splits": summary,
        "files": {name: sha(raw) for name, raw in files.items()},
    }
    raw_report = canonical(report)
    with TemporaryDirectory(dir=args.out.parent) as temporary:
        staged = Path(temporary) / "holdout"
        staged.mkdir()
        for name, raw in files.items():
            (staged / name).write_bytes(raw)
        (staged / "receipt.json").write_bytes(raw_report)
        staged.rename(args.out)
    print(json.dumps({"receipt_sha256": sha(raw_report), **report}, sort_keys=True))


if __name__ == "__main__":
    main()
