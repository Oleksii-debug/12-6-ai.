"""Run deterministic R01-E10 efficiency calibration for the incumbent byte tokenizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.tokenization.byte import ByteTokenizer
from twelve_six.tokenization.calibration import calibrate_tokenizer_efficiency


def _load_samples(path: Path) -> dict[str, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("sample JSON must be an object mapping strata to arrays of strings")
    result: dict[str, list[str]] = {}
    for stratum, texts in payload.items():
        if not isinstance(stratum, str) or not isinstance(texts, list):
            raise ValueError("sample JSON must map string strata to arrays")
        result[stratum] = texts
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-json", type=Path, required=True)
    parser.add_argument("--context-tokens", type=int, default=1024)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = calibrate_tokenizer_efficiency(
        ByteTokenizer(),
        _load_samples(args.sample_json),
        context_tokens=args.context_tokens,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report.canonical_json() + "\n", encoding="utf-8")
    print(report.report_sha256)


if __name__ == "__main__":
    main()
