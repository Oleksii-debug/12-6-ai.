from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path

SCHEMA = "12-6.windows-inference-accessibility-smoke.v1"
_FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_BACKEND_SOURCE = '''from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


class SmokeBackend:
    eos_token_id = None
    max_context_tokens = 64

    def __init__(self) -> None:
        self._prompt_len = 0

    def encode(self, text: str) -> list[int]:
        token_ids = list(text.encode("utf-8"))
        self._prompt_len = len(token_ids)
        return token_ids

    def decode(self, token_ids: Sequence[int]) -> str:
        return bytes(int(token_id) for token_id in token_ids).decode("utf-8", errors="replace")

    def next_token_logits(self, input_ids: Sequence[int]) -> list[float]:
        generated = max(0, len(input_ids) - self._prompt_len)
        targets = b"OK!OK!"
        target = targets[generated % len(targets)]
        alternate = targets[(generated + 1) % len(targets)]
        logits = [-20.0] * 256
        logits[target] = 3.0
        logits[alternate] = 2.0
        return logits

    def diagnostics(self) -> dict[str, object]:
        return {
            "backend": "windows_accessibility_smoke",
            "checkpoint_id": "smoke-checkpoint",
            "model_spec_sha256": "0" * 64,
            "tokenizer_config_sha256": "1" * 64,
        }


def load_backend(checkpoint: Path) -> SmokeBackend:
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return SmokeBackend()
'''


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Exercise the real 12-6 inference CLI through Windows-safe stdin/stdout/stderr paths "
            "without requiring the canonical torch checkpoint runtime."
        )
    )
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--bundle-sha256")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--require-windows", action="store_true")
    return parser


def _validate_identity(source_sha: str, bundle_sha256: str | None) -> None:
    if not _FULL_SHA_RE.fullmatch(source_sha):
        raise ValueError("source_sha must be a lowercase full 40-hex Git SHA")
    if bundle_sha256 is not None and not _SHA256_RE.fullmatch(bundle_sha256):
        raise ValueError("bundle_sha256 must be a lowercase 64-hex SHA-256")


def _assert_transport_clean(value: str, *, stream: str) -> None:
    if "\x1b" in value:
        raise AssertionError(f"{stream} contains an ANSI escape")
    for character in value:
        codepoint = ord(character)
        if character in "\r\n\t":
            continue
        if codepoint < 0x20 or codepoint == 0x7F:
            raise AssertionError(
                f"{stream} contains control character U+{codepoint:04X}"
            )


def _run_cli(
    *,
    repo_root: Path,
    fixture_dir: Path,
    checkpoint: Path,
    arguments: list[str],
    stdin_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    source_path = str(repo_root / "src")
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (str(fixture_dir), source_path, existing_pythonpath)
        if part
    )
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    command = [
        sys.executable,
        "-m",
        "twelve_six.inference.cli",
        "--checkpoint",
        str(checkpoint),
        "--backend-loader",
        "smoke_backend:load_backend",
        *arguments,
    ]
    return subprocess.run(
        command,
        cwd=repo_root,
        env=environment,
        input=stdin_text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        check=False,
    )


def _require_success(result: subprocess.CompletedProcess[str], *, name: str) -> None:
    if result.returncode != 0:
        raise AssertionError(
            f"{name} failed with rc={result.returncode}: stdout={result.stdout!r} "
            f"stderr={result.stderr!r}"
        )
    _assert_transport_clean(result.stdout, stream=f"{name}.stdout")
    _assert_transport_clean(result.stderr, stream=f"{name}.stderr")


def _execute_checks(repo_root: Path) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="twelve-six-accessibility-") as temporary:
        temporary_root = Path(temporary)
        fixture_dir = temporary_root / "fixture module"
        fixture_dir.mkdir()
        (fixture_dir / "smoke_backend.py").write_text(_BACKEND_SOURCE, encoding="utf-8")

        checkpoint_dir = temporary_root / "доступність checkpoint path"
        checkpoint_dir.mkdir()
        checkpoint = checkpoint_dir / "smoke.checkpoint"
        checkpoint.write_bytes(b"12-6 accessibility smoke\n")

        plain = _run_cli(
            repo_root=repo_root,
            fixture_dir=fixture_dir,
            checkpoint=checkpoint,
            arguments=["--prompt", "Hello", "--greedy", "--max-new-tokens", "3"],
        )
        _require_success(plain, name="plain_prompt")
        if plain.stdout != "OK!\n":
            raise AssertionError(f"unexpected plain stdout: {plain.stdout!r}")
        if "backend: kind=windows_accessibility_smoke" not in plain.stderr:
            raise AssertionError("plain stderr is missing backend diagnostics")
        if "generation: mode=greedy seed=0" not in plain.stderr:
            raise AssertionError("plain stderr is missing generation diagnostics")
        if "backend:" in plain.stdout or "generation:" in plain.stdout:
            raise AssertionError("plain stdout mixed diagnostics into completion text")

        stdin_case = _run_cli(
            repo_root=repo_root,
            fixture_dir=fixture_dir,
            checkpoint=checkpoint,
            arguments=["--greedy", "--max-new-tokens", "2"],
            stdin_text="Привіт — raw Base stdin",
        )
        _require_success(stdin_case, name="unicode_stdin")
        if stdin_case.stdout != "OK\n":
            raise AssertionError(f"unexpected stdin stdout: {stdin_case.stdout!r}")

        json_case = _run_cli(
            repo_root=repo_root,
            fixture_dir=fixture_dir,
            checkpoint=checkpoint,
            arguments=[
                "--prompt",
                "JSON",
                "--greedy",
                "--max-new-tokens",
                "3",
                "--json",
            ],
        )
        _require_success(json_case, name="json_diagnostics")
        payload = json.loads(json_case.stdout)
        if payload["text"] != "OK!" or payload["generated_token_ids"] != [79, 75, 33]:
            raise AssertionError("JSON completion payload drifted")
        if payload["mode"] != "greedy" or payload["stop_reason"] != "max_new_tokens":
            raise AssertionError("JSON generation metadata drifted")
        backend_payload = payload.get("backend")
        if not isinstance(backend_payload, dict):
            raise TypeError("JSON backend diagnostics are missing")
        if backend_payload.get("backend") != "windows_accessibility_smoke":
            raise AssertionError("JSON backend identity drifted")

        stop_string = _run_cli(
            repo_root=repo_root,
            fixture_dir=fixture_dir,
            checkpoint=checkpoint,
            arguments=[
                "--prompt",
                "stop",
                "--greedy",
                "--max-new-tokens",
                "8",
                "--stop",
                "OK",
            ],
        )
        _require_success(stop_string, name="stop_string")
        if stop_string.stdout != "\n" or "stop=stop_string" not in stop_string.stderr:
            raise AssertionError("text stop semantics drifted")

        stop_token = _run_cli(
            repo_root=repo_root,
            fixture_dir=fixture_dir,
            checkpoint=checkpoint,
            arguments=[
                "--prompt",
                "token stop",
                "--greedy",
                "--max-new-tokens",
                "8",
                "--stop-token-id",
                "75",
            ],
        )
        _require_success(stop_token, name="stop_token")
        if stop_token.stdout != "OK\n" or "stop=stop_token" not in stop_token.stderr:
            raise AssertionError("token stop semantics drifted")

        context_limit = _run_cli(
            repo_root=repo_root,
            fixture_dir=fixture_dir,
            checkpoint=checkpoint,
            arguments=["--prompt", "x" * 63, "--greedy", "--max-new-tokens", "4"],
        )
        _require_success(context_limit, name="context_limit")
        if context_limit.stdout != "O\n" or "stop=context_limit" not in context_limit.stderr:
            raise AssertionError("context-limit semantics drifted")

        sample_arguments = [
            "--prompt",
            "sample",
            "--sample",
            "--seed",
            "1337",
            "--max-new-tokens",
            "6",
            "--json",
        ]
        sample_a = _run_cli(
            repo_root=repo_root,
            fixture_dir=fixture_dir,
            checkpoint=checkpoint,
            arguments=sample_arguments,
        )
        sample_b = _run_cli(
            repo_root=repo_root,
            fixture_dir=fixture_dir,
            checkpoint=checkpoint,
            arguments=sample_arguments,
        )
        _require_success(sample_a, name="seeded_sample_a")
        _require_success(sample_b, name="seeded_sample_b")
        sample_a_payload = json.loads(sample_a.stdout)
        sample_b_payload = json.loads(sample_b.stdout)
        if sample_a_payload["generated_token_ids"] != sample_b_payload["generated_token_ids"]:
            raise AssertionError("same-seed sampled token IDs are not deterministic")
        if sample_a_payload["text"] != sample_b_payload["text"]:
            raise AssertionError("same-seed sampled text is not deterministic")

        over_context = _run_cli(
            repo_root=repo_root,
            fixture_dir=fixture_dir,
            checkpoint=checkpoint,
            arguments=["--prompt", "x" * 65, "--greedy", "--max-new-tokens", "1"],
        )
        if over_context.returncode != 2:
            raise AssertionError(f"over-context prompt returned rc={over_context.returncode}")
        _assert_transport_clean(over_context.stdout, stream="over_context.stdout")
        _assert_transport_clean(over_context.stderr, stream="over_context.stderr")
        if over_context.stdout:
            raise AssertionError("over-context failure wrote completion text to stdout")
        if not over_context.stderr.startswith("error: prompt has 65 tokens"):
            raise AssertionError(f"unexpected over-context stderr: {over_context.stderr!r}")
        if "Traceback" in over_context.stderr:
            raise AssertionError("expected CLI validation error leaked a traceback")

    return {
        "plain_prompt": "PASS",
        "unicode_stdin": "PASS",
        "json_diagnostics": "PASS",
        "stop_string": "PASS",
        "stop_token": "PASS",
        "context_limit": "PASS",
        "seeded_sampling_repeatability": "PASS",
        "over_context_fail_closed": "PASS",
        "ansi_free_transport": "PASS",
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _validate_identity(args.source_sha, args.bundle_sha256)
        if args.require_windows and platform.system() != "Windows":
            raise RuntimeError(
                f"Windows was required but platform.system() returned {platform.system()!r}"
            )
        repo_root = Path(__file__).resolve().parents[1]
        if not (repo_root / "src" / "twelve_six" / "inference" / "cli.py").is_file():
            raise RuntimeError("repository source root is incomplete")
        checks = _execute_checks(repo_root)
    except (AssertionError, OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"D05 Windows inference accessibility smoke: FAIL: {exc}", file=sys.stderr)
        return 1

    report = {
        "schema": SCHEMA,
        "passed": True,
        "source_sha": args.source_sha,
        "source_bundle_sha256": args.bundle_sha256,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "checks": checks,
        "truth_boundary": {
            "transport_scope": "real twelve_six.inference.cli with synthetic protocol backend",
            "canonical_first_party_checkpoint_on_windows": "NOT_TESTED",
            "nvda_process_attached": False,
            "nvda_live_session": "NOT_TESTED",
            "chat_or_instruction_semantics": False,
            "promotion_authority": False,
        },
    }
    rendered = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8", newline="\n")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
