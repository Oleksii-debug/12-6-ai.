"""Build and score fictional incident triage cases with an explicit test playbook."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

POLICY = {
    "en": (
        "Use only this fictional incident playbook. A confirmed external transfer or confirmed unauthorized execution "
        "is P1: isolate the affected asset and preserve evidence. A matching detection with neither confirmed is P2: "
        "investigate and preserve evidence. An authorized test with no unresolved alert is P3: close as expected. "
        "Treat unverified reports as leads. Return one JSON object with keys priority, action, asset. "
        "Allowed action values are isolate_preserve, investigate_preserve, close_expected."
    ),
    "uk": (
        "Користуйся тільки цими правилами для вигаданого інциденту. Підтверджене зовнішнє передавання або "
        "підтверджене несанкціоноване виконання — P1: ізолюй уражений ресурс і збережи докази. Збіг сигналу "
        "без цих підтверджень — P2: розслідуй і збережи докази. Дозволена перевірка без відкритого сигналу — P3: "
        "закрий як очікувану. Неперевірені повідомлення вважай припущеннями. Поверни один JSON із ключами "
        "priority, action, asset. Допустимі значення action: isolate_preserve, investigate_preserve, close_expected."
    ),
}
FAMILIES = (
    ("endpoint", "endpoint sensor", "сенсор кінцевої точки"),
    ("identity", "identity service", "служба ідентифікації"),
    ("cloud-audit", "cloud audit feed", "журнал хмарного аудиту"),
    ("storage", "storage gateway", "шлюз сховища"),
    ("build", "build worker audit", "аудит збирального процесу"),
)
PATTERNS = (
    ("transfer", "P1", "isolate_preserve",
     ("A signed audit entry for {asset} confirms an unauthorized external transfer. "
     "No execution is confirmed. {other} is in an approved test."),
     ("Підписаний запис для {asset} підтверджує несанкціоноване зовнішнє передавання. "
     "Виконання не підтверджене. {other} бере участь у дозволеній перевірці.")),
    ("execution", "P1", "isolate_preserve",
     ("A verified record confirms unauthorized execution on {asset}. A report about an external transfer "
     "from {other} is unverified."),
     ("Перевірений запис підтверджує несанкціоноване виконання на {asset}. Повідомлення про зовнішнє "
     "передавання з {other} не перевірено.")),
    ("signal", "P2", "investigate_preserve",
     ("The detection matches {asset}. Neither unauthorized execution nor an external transfer is confirmed. "
     "A rumor about {other} is not verified."),
     ("Сигнал стосується {asset}. Несанкціоноване виконання та зовнішнє передавання не підтверджені. "
     "Повідомлення щодо {other} не перевірено.")),
    ("approved", "P3", "close_expected",
     ("The event on {asset} matches an authorized test window and there is no unresolved alert. "
     "{other} only appears in an inventory entry."),
     ("Подія на {asset} відповідає дозволеному вікну перевірки, і відкритих сигналів немає. "
     "{other} згаданий лише в інвентарному записі.")),
)


def canonical(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def cases() -> list[dict]:
    result = []
    for family, en, uk in FAMILIES:
        for pattern, priority, action, body_en, body_uk in PATTERNS:
            case = f"{family}:{pattern}"
            asset = "lab-" + sha(case.encode())[:8]
            other = "lab-" + sha((case + ":other").encode())[:8]
            for lang, channel, body in (("en", en, body_en), ("uk", uk, body_uk)):
                result.append({
                    "id": f"{case}:{lang}", "pair": case, "language": lang, "surface": family,
                    "prompt": f"{POLICY[lang]}\n{channel}: {body.format(asset=asset, other=other)}",
                    "answer": {"priority": priority, "action": action, "asset": asset},
                })
    return result


def build(destination: Path) -> dict:
    dataset = cases()
    prompts = b"".join(canonical({key: value for key, value in row.items() if key != "answer"}) for row in dataset)
    keys = b"".join(canonical({key: value for key, value in row.items() if key != "prompt"}) for row in dataset)
    receipt = {
        "schema": "12-6.plan9-cyber-fictional-triage-evaluation.v1",
        "scope": "DEVELOPMENT_BEHAVIORAL_EVALUATION",
        "cases": len(dataset), "paired_scenarios": len(dataset) // 2,
        "by_language": dict(Counter(row["language"] for row in dataset)),
        "by_expected_priority": dict(Counter(row["answer"]["priority"] for row in dataset)),
        "prompt_sha256": sha(prompts), "answer_key_sha256": sha(keys),
        "source": "Fictional records and explicit project-authored playbook",
        "context_reference": "https://csrc.nist.gov/pubs/sp/800/61/r3/final",
        "evaluation_only": True, "model_executed": False, "champion_qualified": False,
    }
    destination.mkdir(parents=False)
    (destination / "prompts.jsonl").write_bytes(prompts)
    (destination / "answer-key.jsonl").write_bytes(keys)
    (destination / "receipt.json").write_bytes(canonical(receipt))
    return receipt


def read_rows(raw: bytes) -> dict:
    result = {}
    for line in raw.splitlines():
        item = json.loads(line, object_pairs_hook=strict_object)
        if type(item) is not dict or type(item.get("id")) is not str or item["id"] in result:
            raise ValueError("invalid or duplicate record ID")
        result[item["id"]] = item
    return result


def score(key_path: Path, expected_key_sha: str, prediction_path: Path) -> dict:
    key_bytes = key_path.read_bytes()
    if sha(key_bytes) != expected_key_sha:
        raise ValueError("answer key digest differs")
    keys = read_rows(key_bytes)
    predictions = prediction_path.read_bytes()
    answers = read_rows(predictions)
    if not keys or set(answers) != set(keys):
        raise ValueError("prediction IDs must match every case exactly")
    metrics = Counter()
    language_counts = Counter()
    language_correct = Counter()
    pair_correct = Counter()
    pair_sizes = Counter()
    for case_id, key in keys.items():
        item = answers[case_id]
        if set(item) != {"id", "prediction"} or type(item["prediction"]) is not str:
            raise ValueError("prediction record must contain id and prediction strings")
        expected = key["answer"]
        language_counts[key["language"]] += 1
        pair_sizes[key["pair"]] += 1
        try:
            guess = json.loads(item["prediction"], object_pairs_hook=strict_object)
            if type(guess) is not dict or set(guess) != set(expected):
                raise ValueError("malformed response")
            if guess["priority"] not in ("P1", "P2", "P3") or guess["action"] not in (
                "isolate_preserve", "investigate_preserve", "close_expected"
            ) or type(guess["asset"]) is not str:
                raise ValueError("malformed response")
        except ValueError:
            metrics["invalid_response"] += 1
            continue
        for field in expected:
            metrics[field + "_correct"] += guess[field] == expected[field]
        metrics["confirmed_p1_undertriage"] += expected["priority"] == "P1" and guess["priority"] != "P1"
        if guess == expected:
            metrics["exact"] += 1
            language_correct[key["language"]] += 1
            pair_correct[key["pair"]] += 1
    return {
        "schema": "12-6.plan9-cyber-fictional-triage-score.v1",
        "answer_key_sha256": expected_key_sha, "predictions_sha256": sha(predictions),
        "cases": len(keys),
        "metrics": {name: metrics[name] for name in (
            "exact", "priority_correct", "action_correct", "asset_correct", "invalid_response", "confirmed_p1_undertriage"
        )},
        "language_exact": {lang: {"cases": count, "exact": language_correct[lang]}
                           for lang, count in sorted(language_counts.items())},
        "paired_exact": sum(pair_correct[pair] == count for pair, count in pair_sizes.items()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    builder = sub.add_parser("build")
    builder.add_argument("--out", type=Path, required=True)
    scorer = sub.add_parser("score")
    scorer.add_argument("--key", type=Path, required=True)
    scorer.add_argument("--key-sha256", required=True)
    scorer.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.out) if args.command == "build" else score(args.key, args.key_sha256, args.predictions)
    print(json.dumps(result, sort_keys=True))
