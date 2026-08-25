#!/usr/bin/env python3
"""Finalize M100 and correct the exact materializer reproduction command."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path


def _hash(value):
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    target = root / "tools/finalize_milestone100_evidence.py"
    spec = importlib.util.spec_from_file_location("m100_finalizer", target)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load milestone finalizer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rc = module.main()
    if rc != 0:
        return rc
    path = root / "evidence/milestone100/milestone100_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["reproduction"]["command"] = (
        "python tools/run_milestone100_materializer_exact.py && "
        "python -m twelve_six.scaling_experiment run --repo-root . --source-sha $(git rev-parse HEAD) "
        "--output evidence/milestone100/research41_real_corpus_baseline.json --torch-threads 2 && "
        "python -m twelve_six.scaling_500k_evidence run --repo-root . --source-sha $(git rev-parse HEAD) "
        "--baseline evidence/milestone100/research41_real_corpus_baseline.json "
        "--output evidence/milestone100/learned_468k_real_corpus.json "
        "--checkpoint-root artifacts/milestone100/checkpoints --seeds 1337 1338 "
        "--token-budgets 4096 16384 65536 --torch-threads 2 && "
        "python tools/prove_milestone100_fresh_resume.py --repo-root . --source-sha $(git rev-parse HEAD) "
        "--report evidence/milestone100/learned_468k_real_corpus.json "
        "--output evidence/milestone100/fresh_process_resume.json && "
        "python tools/finalize_milestone100_evidence_v2.py"
    )
    summary["reproduction"]["data21_normalization_cap_chars"] = 50_000
    summary.pop("summary_sha256", None)
    summary["summary_sha256"] = _hash(summary)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "summary_sha256": summary["summary_sha256"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
