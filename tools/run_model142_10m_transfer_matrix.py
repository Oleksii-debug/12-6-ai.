#!/usr/bin/env python3
"""Execute MODEL-142 one-run or complete fresh-process transfer evidence."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from twelve_six.architecture_transfer_10m import (
    load_experiment_config,
    run_candidate,
    summarize_matrix,
    validate_summary,
)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _run_one(args: argparse.Namespace) -> int:
    repo_root = args.repo_root.resolve()
    report = run_candidate(
        repo_root=repo_root,
        source_sha=args.source_sha,
        config_path=(repo_root / args.config).resolve(),
        candidate_id=args.candidate,
        seed=args.seed,
        torch_threads=args.torch_threads,
    )
    _write(args.output, report)
    print(
        json.dumps(
            {
                "candidate": report["candidate"],
                "seed": report["seed"],
                "parameters": report["model"]["parameters"],
                "final_heldout_bpb": report["metrics"]["final_heldout_bpb"],
                "final_train_bpb": report["metrics"]["final_train_bpb"],
                "tokens_per_s": report["metrics"]["optimized_tokens_per_s"],
                "bf16_full_context_kv_bytes": report["kv_cache"]["bf16_bytes"],
            },
            sort_keys=True,
        )
    )
    return 0


def _matrix(args: argparse.Namespace) -> int:
    repo_root = args.repo_root.resolve()
    config_path = (repo_root / args.config).resolve()
    config = load_experiment_config(config_path)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_paths: list[Path] = []
    script = Path(__file__).resolve()
    for candidate in config["candidates"]:
        candidate_id = str(candidate["id"])
        for seed in config["controls"]["seeds"]:
            run_path = output_dir / f"{candidate_id}-seed{int(seed)}.json"
            command = [
                sys.executable,
                str(script),
                "run-one",
                "--repo-root",
                str(repo_root),
                "--source-sha",
                args.source_sha,
                "--config",
                str(Path(args.config)),
                "--candidate",
                candidate_id,
                "--seed",
                str(int(seed)),
                "--torch-threads",
                str(args.torch_threads),
                "--output",
                str(run_path),
            ]
            subprocess.run(command, cwd=repo_root, check=True)
            run_paths.append(run_path)
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in run_paths]
    summary = summarize_matrix(config, runs)
    validate_summary(summary)
    summary["source_sha"] = args.source_sha
    summary["run_files"] = [path.name for path in run_paths]
    # Re-hash after adding collection-only source/run metadata.
    unsigned = dict(summary)
    unsigned.pop("report_sha256")
    from twelve_six.architecture_transfer_10m import _canonical_hash

    summary["report_sha256"] = _canonical_hash(unsigned)
    _write(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("run-one")
    one.add_argument("--repo-root", type=Path, default=Path("."))
    one.add_argument("--source-sha", required=True)
    one.add_argument(
        "--config",
        default="configs/experiments/model142_10m_transfer_matrix.v1.json",
    )
    one.add_argument("--candidate", required=True)
    one.add_argument("--seed", type=int, required=True)
    one.add_argument("--torch-threads", type=int, default=2)
    one.add_argument("--output", type=Path, required=True)
    matrix = sub.add_parser("matrix")
    matrix.add_argument("--repo-root", type=Path, default=Path("."))
    matrix.add_argument("--source-sha", required=True)
    matrix.add_argument(
        "--config",
        default="configs/experiments/model142_10m_transfer_matrix.v1.json",
    )
    matrix.add_argument("--torch-threads", type=int, default=2)
    matrix.add_argument("--output-dir", type=Path, required=True)
    matrix.add_argument("--summary", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run-one":
        return _run_one(args)
    return _matrix(args)


if __name__ == "__main__":
    raise SystemExit(main())
