#!/usr/bin/env python3
"""Run the real Lingua 2.2.0 probe; never report success when the exact package is absent."""

from __future__ import annotations

import argparse
import json
import platform
import time
from importlib.metadata import PackageNotFoundError, version

EXPECTED_VERSION = "2.2.0"


def write_record(path: str, record: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    record = {
        "schema_version": 1,
        "component_id": "LINGUA",
        "expected_version": EXPECTED_VERSION,
        "python": platform.python_version(),
        "executed": False,
        "status": "NOT_EXECUTED",
        "cases": [],
    }
    try:
        installed = version("lingua-language-detector")
    except PackageNotFoundError:
        record["reason"] = "exact package missing"
        write_record(args.out, record)
        return 2
    if installed != EXPECTED_VERSION:
        record["reason"] = f"version drift: installed={installed!r}"
        write_record(args.out, record)
        return 3

    from lingua import Language, LanguageDetectorBuilder

    detector = (
        LanguageDetectorBuilder.from_languages(
            Language.UKRAINIAN,
            Language.ENGLISH,
            Language.SLOVAK,
            Language.RUSSIAN,
        )
        .build()
    )
    fixtures = [
        ("ua_sentence", "Це тест українського тексту."),
        ("en_sentence", "This is a deterministic English test."),
        ("sk_sentence", "Toto je deterministický test v slovenčine."),
        ("ru_sentence", "Это детерминированный тест русского текста."),
        ("code_like", "def hello(name): return name.strip()"),
        ("mixed", "Україна and English in one string."),
        ("noise", "x9__42 !!!"),
    ]
    for name, text in fixtures:
        start = time.perf_counter()
        language = detector.detect_language_of(text)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        record["cases"].append(
            {"name": name, "detected": str(language), "latency_ms": round(elapsed_ms, 6)}
        )
    record["executed"] = True
    record["status"] = "EXECUTED"
    write_record(args.out, record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
