#!/usr/bin/env python3
"""Execute M100 materialization with the exact DATA-21/22 50k-char normalization cap."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "tools/materialize_milestone100_real_corpus.py"
    spec = importlib.util.spec_from_file_location("m100_materializer", target)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load milestone materializer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    incumbent = module.run_bounded_intake

    def exact_data21_intake(registry, output_dir, **kwargs):
        kwargs["max_normalized_chars"] = 50_000
        return incumbent(registry, output_dir, **kwargs)

    module.run_bounded_intake = exact_data21_intake
    report = module.materialize(repo_root)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
