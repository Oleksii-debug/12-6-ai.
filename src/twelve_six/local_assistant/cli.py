from __future__ import annotations

import argparse
import json
import sys

from .authority import CapabilityGate, CapabilityUnavailableError
from .orchestrator import LocalAssistantOrchestrator, RunOptions, write_trace


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m twelve_six.local_assistant",
        description="LOCAL_FREE plain-text post-Base orchestration shell.",
    )
    parser.add_argument("--task", help="Plain-text task. If omitted, one line is read from stdin.")
    parser.add_argument("--checkpoint", help="Verified learned Base checkpoint directory.")
    parser.add_argument("--expected-model-spec-sha256")
    parser.add_argument("--mock-model", action="store_true", help="Use deterministic mechanics fixture.")
    parser.add_argument("--hypothesis-search", action="store_true")
    parser.add_argument("--memory-db")
    parser.add_argument("--mock-tools", action="store_true")
    parser.add_argument("--trace", help="Write machine trace JSON to this path.")
    parser.add_argument("--authorities", action="store_true", help="Print capability authority JSON and exit.")
    parser.add_argument("--probe", action="store_true", help="Run a deterministic fixture probe.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    gate = CapabilityGate()
    if args.authorities:
        print(json.dumps(gate.snapshot(), indent=2, sort_keys=True))
        return 0

    task = args.task
    expected = None
    if args.probe:
        task = "LOCAL_FREE_PROBE"
        expected = "LOCAL_FREE_PROBE"
        args.mock_model = True
    if task is None:
        task = sys.stdin.readline().rstrip("\n")
    if not task.strip():
        print("ERROR: task must be non-empty", file=sys.stderr)
        return 2

    try:
        options = RunOptions(
            checkpoint=args.checkpoint,
            expected_model_spec_sha256=args.expected_model_spec_sha256,
            mock_model=args.mock_model,
            use_hypothesis_search=args.hypothesis_search,
            memory_db=args.memory_db,
            use_mock_tools=args.mock_tools,
            expected_answer_fixture=expected,
        )
        result = LocalAssistantOrchestrator(gate).run(task, options)
    except (CapabilityUnavailableError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.trace:
        write_trace(args.trace, result.trace)
    print(result.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
