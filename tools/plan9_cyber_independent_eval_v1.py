"""Build an evaluation-only ATT&CK mitigation-to-technique retrieval suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory

SOURCE_COMMIT = "f59b03ecf828579d5f9e93683ce25b7e2e5fd9d7"
SOURCE_SHA = "539a843a7b53ed9749740b2123dddae3726b1a4831841c32aca740b011b910b5"
LICENSE_SHA = "738144f7fb054722a4ef9d3367c51710341dc12fc574c6ac3a41daaaecd8bf5e"
TRAIN_SHA = "5a2c727ad5146470ed30de552df349caf7960a7e40cdd1923b25dd4069fff140"
CORPUS_RECEIPT_SHA = "aa49940f5419e5cda564c35baea3427b9c44c5037ed4b85807a4111f0c10563a"
WORD = re.compile(r"\w+", re.UNICODE)
ATTACK_ID = re.compile(r"\b[TM]\d{4}(?:\.\d{3})?\b", re.IGNORECASE)
LIMIT_PER_TACTIC = 12


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def verified(path: Path, expected: str) -> bytes:
    raw = path.read_bytes()
    if digest(raw) != expected:
        raise ValueError(f"pinned input changed: {path.name}")
    return raw


def words(text: str) -> tuple[str, ...]:
    return tuple(WORD.findall(unicodedata.normalize("NFKC", text).casefold()))


def grams(tokens: tuple[str, ...]) -> set[tuple[str, ...]]:
    return {tokens[i:i + 8] for i in range(len(tokens) - 7)}


def identifier(obj: dict) -> str | None:
    ids = {
        ref["external_id"] for ref in obj.get("external_references", [])
        if ref.get("source_name") == "mitre-attack" and ATTACK_ID.fullmatch(ref.get("external_id", ""))
    }
    return next(iter(ids)) if len(ids) == 1 else None


def active(obj: dict) -> bool:
    return obj.get("revoked", False) is False and obj.get("x_mitre_deprecated", False) is False


def strict_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON field: {key}")
        value[key] = item
    return value


def score(cases_path: Path, predictions_path: Path, expected_cases_sha: str) -> dict:
    cases = [json.loads(line) for line in verified(cases_path, expected_cases_sha).splitlines()]
    by_id = {item["id"]: item for item in cases}
    if not cases or len(by_id) != len(cases):
        raise ValueError("missing or duplicate benchmark cases")
    predictions = {}
    prediction_bytes = predictions_path.read_bytes()
    for line in prediction_bytes.splitlines():
        item = json.loads(line, object_pairs_hook=strict_object)
        if (type(item) is not dict or set(item) != {"id", "prediction"} or type(item["id"]) is not str
                or item["id"] in predictions or type(item["prediction"]) is not str):
            raise ValueError("invalid or duplicate prediction record")
        predictions[item["id"]] = item["prediction"]
    if set(predictions) != set(by_id):
        raise ValueError("prediction IDs must exactly match benchmark case IDs")
    per_case = []
    for case_id, item in by_id.items():
        ids = item["expected_mitigation_ids"]
        if (not isinstance(ids, list) or not ids
                or any(not isinstance(value, str) or not re.fullmatch(r"M\d{4}", value) for value in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError("invalid mitigation reference")
        expected = set(ids)
        predicted = {value.upper() for value in ATTACK_ID.findall(predictions[case_id]) if value.upper().startswith("M")}
        count = len(expected & predicted)
        precision = count / len(predicted) if predicted else 0
        recall = count / len(expected)
        f1 = 2 * count / (len(expected) + len(predicted))
        per_case.append((item["category"], precision, recall, f1, predicted == expected))

    def aggregate(rows):
        return {
            "cases": len(rows),
            "precision": mean(row[1] for row in rows),
            "recall": mean(row[2] for row in rows),
            "f1": mean(row[3] for row in rows),
            "exact_match_rate": mean(row[4] for row in rows),
        }

    return {
        "schema": "12-6.cyber-independent-attack-score.v1",
        "case_file_sha256": expected_cases_sha,
        "predictions_sha256": digest(prediction_bytes),
        "aggregation": "macro_average_over_cases",
        "prediction_parsing": "extract_and_deduplicate_mitigation_ids_case_insensitively",
        "all": aggregate(per_case),
        "by_tactic": {tactic: aggregate([row for row in per_case if row[0] == tactic])
                      for tactic in sorted({row[0] for row in per_case})},
    }


def build(source: Path, corpus: Path, destination: Path) -> dict:
    training = verified(corpus / "train.jsonl", TRAIN_SHA)
    verified(corpus / "receipt.json", CORPUS_RECEIPT_SHA)
    train_grams = set()
    train_identifiers = set()
    for line in training.splitlines():
        text = json.loads(line)["text"]
        train_grams.update(grams(words(text)))
        train_identifiers.update(item.upper() for item in ATTACK_ID.findall(text))
    stix = json.loads(verified(source / "enterprise-attack.json", SOURCE_SHA))
    license_bytes = verified(source / "LICENSE.txt", LICENSE_SHA)
    objects = {obj["id"]: obj for obj in stix["objects"] if active(obj)}
    techniques = {
        uid: obj for uid, obj in objects.items()
        if obj["type"] == "attack-pattern" and identifier(obj) and not obj.get("x_mitre_is_subtechnique", False)
    }
    mitigations = {
        uid: obj for uid, obj in objects.items()
        if obj["type"] == "course-of-action" and identifier(obj)
    }
    by_technique = defaultdict(dict)
    for relationship in objects.values():
        if relationship.get("relationship_type") == "mitigates":
            attack, mitigation = relationship["target_ref"], relationship["source_ref"]
            if attack in techniques and mitigation in mitigations:
                key = identifier(mitigations[mitigation])
                current = by_technique[attack].get(key)
                if current is None or relationship["id"] < current["id"]:
                    by_technique[attack][key] = relationship
    by_tactic = defaultdict(list)
    excluded = Counter()
    for uid, links in sorted(by_technique.items()):
        obj = techniques[uid]
        attack_id = identifier(obj)
        tactics = sorted({phase["phase_name"] for phase in obj.get("kill_chain_phases", [])
                          if phase["kill_chain_name"] == "mitre-attack"})
        if not tactics:
            excluded["missing_tactic"] += 1
            continue
        if attack_id in train_identifiers:
            excluded["technique_identifier_seen"] += 1
            continue
        evidence = [grams(words(rel.get("description", ""))) for rel in links.values()]
        if (set(links) & train_identifiers or not any(len(part) >= 5 for part in evidence)
                or any(len(part) >= 5 and len(part & train_grams) * 5 >= len(part) * 4 for part in evidence)):
            excluded["mitigation_evidence_screen"] += 1
            continue
        by_tactic[tactics[0]].append((uid, links))
    items = []
    for tactic, cases in sorted(by_tactic.items()):
        cases.sort(key=lambda case: digest(case[0].encode()))
        for uid, links in cases[:LIMIT_PER_TACTIC]:
            technique = techniques[uid]
            ids = sorted(links)
            item = {
                "id": f"attack-mitigation:{identifier(technique)}",
                "category": tactic,
                "prompt": f"List MITRE Enterprise ATT&CK mitigation IDs associated with "
                          f"{technique['name']} ({identifier(technique)}). Return IDs only.",
                "expected_mitigation_ids": ids,
                "metric": "reported_set_precision_recall_f1_and_exact_match",
                "references": [f"https://attack.mitre.org/techniques/{identifier(technique)}/"],
                "stix_relationship_ids": [links[key]["id"] for key in ids],
            }
            items.append(item)
    counts = Counter(item["category"] for item in items)
    if len(items) < 50 or len(counts) < 7 or len({item["id"] for item in items}) != len(items):
        raise ValueError("independent evaluation coverage is too small or has duplicate IDs")
    cases = b"".join(map(canonical, items))
    receipt = {
        "schema": "12-6.cyber-independent-attack-evaluation.v1",
        "evaluation_only": True,
        "source_commit": SOURCE_COMMIT,
        "source_data_sha256": SOURCE_SHA,
        "license_sha256": LICENSE_SHA,
        "terms": "https://attack.mitre.org/resources/terms-of-use/",
        "verified_training_sha256": TRAIN_SHA,
        "candidate_receipt_sha256": CORPUS_RECEIPT_SHA,
        "cases": len(items),
        "by_primary_tactic": dict(counts),
        "excluded_techniques": dict(excluded),
        "case_file_sha256": digest(cases),
        "training_uses_evaluation": False,
        "qualification_scope": "DEVELOPMENT_RETRIEVAL_METRIC_NOT_FINAL_CHAMPION_ACCEPTANCE",
    }
    with TemporaryDirectory(dir=destination.parent) as scratch:
        staged = Path(scratch) / "eval"
        staged.mkdir()
        (staged / "cases.jsonl").write_bytes(cases)
        (staged / "LICENSE.txt").write_bytes(license_bytes)
        (staged / "receipt.json").write_bytes(canonical(receipt))
        staged.rename(destination)
    return {**receipt, "receipt_sha256": digest(canonical(receipt))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    materialize = modes.add_parser("build")
    materialize.add_argument("--source", type=Path, required=True)
    materialize.add_argument("--corpus", type=Path, required=True)
    materialize.add_argument("--out", type=Path, required=True)
    evaluate = modes.add_parser("score")
    evaluate.add_argument("--cases", type=Path, required=True)
    evaluate.add_argument("--cases-sha256", required=True)
    evaluate.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "score":
        print(json.dumps(score(args.cases, args.predictions, args.cases_sha256), sort_keys=True))
        return
    if args.out.exists():
        raise FileExistsError(args.out)
    print(json.dumps(build(args.source, args.corpus, args.out), sort_keys=True))


if __name__ == "__main__":
    main()
