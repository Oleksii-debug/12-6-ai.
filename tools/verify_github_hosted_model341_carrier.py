"""Fail-closed preflight for the canonical MODEL-341 GitHub-hosted carrier."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKER_PATH = ROOT / "src" / "twelve_six" / "training" / "external_worker.py"
LAUNCHER_PATH = ROOT / "tools" / "run_external_training_worker.py"

SCHEMA = "twelve-six-github-hosted-carrier-preflight-v1"
SUPPORTED_MODE = "preflight"
EXPECTED_REPOSITORY = "Oleksii-debug/12-6-ai."
EXPECTED_VISIBILITY = "public"
EXPECTED_RUNNER_OS = "Linux"
EXPECTED_RUNNER_ARCH = "X64"
EXPECTED_WORKER_BLOB_SHA = "e29ee141062aab2e104bcf28d79deefea2306260"
EXPECTED_LAUNCHER_BLOB_SHA = "a580bf35f68c1bbddaaee293accddab50247e84a"
EXPECTED_WORKER_CONSTANTS = {
    "PROTOCOL_VERSION": 1,
    "STATE_SCHEMA": "twelve-six-external-worker-state-v1",
    "MODE": "synthetic-mechanics",
    "MODEL_CARRIER_GIT_SHA": "61aa37b340565dd1ba791adc16ce430f9b17fbaf",
    "MODEL_SPEC_SHA256": "fbff24d561a2818453554d58ca23fc6ace3303b078f1935a8576c4565bd92441",
    "INIT_SPEC_SHA256": "86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5",
    "EXPECTED_PARAMETERS": 20_613_440,
    "MODEL_SEED": 341,
    "SEQUENCE_LENGTH": 32,
}
FORBIDDEN_WORKER_AUTHORITY_ENV = (
    "TWELVE_SIX_EXTERNAL_TRAINING_ROOT",
    "TWELVE_SIX_TRAINER_AUTHORITY_SHA256",
)
FUTURE_REAL_TARGET_REQUIREMENTS = (
    "CANONICAL_TRAINING_RUN_LEASE_TERMINAL",
    "AUTHENTICATED_LAUNCH_MANIFEST_TERMINAL",
    "CORPUS_AUTHORITY_TERMINAL",
    "TOKENIZER_FIT_AUTHORIZED",
    "AUTHORIZED_OPTIMIZED_TARGET_EXPOSURE_POSITIVE",
)
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


class HostedCarrierPreflightError(RuntimeError):
    """Fail-closed carrier preflight error."""


def _git_blob_sha(path: Path) -> str:
    payload = path.read_bytes()
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload, usedforsecurity=False).hexdigest()


def _literal_constants(path: Path) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())
    constants: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            constants[target.id] = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
    return constants


def _require_environment(env: Mapping[str, str]) -> None:
    required = {
        "GITHUB_ACTIONS": "true",
        "GITHUB_REPOSITORY": EXPECTED_REPOSITORY,
        "GITHUB_REPOSITORY_VISIBILITY": EXPECTED_VISIBILITY,
        "RUNNER_OS": EXPECTED_RUNNER_OS,
        "RUNNER_ARCH": EXPECTED_RUNNER_ARCH,
    }
    for name, expected in required.items():
        actual = env.get(name)
        if actual != expected:
            raise HostedCarrierPreflightError(
                f"{name} must be {expected!r}; observed {actual!r}"
            )
    for name in FORBIDDEN_WORKER_AUTHORITY_ENV:
        if env.get(name):
            raise HostedCarrierPreflightError(
                f"{name} must be absent during fail-closed carrier preflight"
            )


def _require_worker_identity(root: Path) -> dict[str, object]:
    worker_path = root / WORKER_PATH.relative_to(ROOT)
    launcher_path = root / LAUNCHER_PATH.relative_to(ROOT)
    if not worker_path.is_file() or not launcher_path.is_file():
        raise HostedCarrierPreflightError("canonical external-worker files are unavailable")

    worker_blob_sha = _git_blob_sha(worker_path)
    launcher_blob_sha = _git_blob_sha(launcher_path)
    if worker_blob_sha != EXPECTED_WORKER_BLOB_SHA:
        raise HostedCarrierPreflightError(
            "external-worker Git blob identity drifted; re-audit carrier binding"
        )
    if launcher_blob_sha != EXPECTED_LAUNCHER_BLOB_SHA:
        raise HostedCarrierPreflightError(
            "external-worker launcher Git blob identity drifted; re-audit carrier binding"
        )

    constants = _literal_constants(worker_path)
    observed = {name: constants.get(name) for name in EXPECTED_WORKER_CONSTANTS}
    if observed != EXPECTED_WORKER_CONSTANTS:
        raise HostedCarrierPreflightError(
            "external-worker protocol/model identity drifted; re-audit carrier binding"
        )

    return {
        "git_blob_sha1": worker_blob_sha,
        "launcher_git_blob_sha1": launcher_blob_sha,
        **observed,
    }


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def build_preflight_evidence(
    *,
    source_sha: str,
    mode: str,
    env: Mapping[str, str] | None = None,
    root: Path = ROOT,
) -> dict[str, object]:
    """Validate the zero-effect hosted carrier boundary and return bound evidence."""

    if mode != SUPPORTED_MODE:
        raise HostedCarrierPreflightError(
            f"unsupported carrier mode {mode!r}; only {SUPPORTED_MODE!r} is permitted"
        )
    if _GIT_SHA.fullmatch(source_sha) is None:
        raise HostedCarrierPreflightError("source_sha must be a full lowercase Git SHA")

    active_env = os.environ if env is None else env
    _require_environment(active_env)
    worker_identity = _require_worker_identity(root)

    evidence: dict[str, object] = {
        "schema": SCHEMA,
        "source_sha": source_sha,
        "result": "PASS",
        "carrier_mode": SUPPORTED_MODE,
        "repository": EXPECTED_REPOSITORY,
        "repository_visibility": EXPECTED_VISIBILITY,
        "runner": {
            "provider": "github-hosted",
            "label": "ubuntu-24.04",
            "os": EXPECTED_RUNNER_OS,
            "arch": EXPECTED_RUNNER_ARCH,
        },
        "worker_identity": worker_identity,
        "scientific_effects": {
            "worker_invoked": False,
            "real_target_execution_supported": False,
            "authorized_optimized_target_exposure": 0,
            "optimizer_updates_executed_on_real_targets": 0,
            "training_executed": False,
            "learned_weights_created": False,
        },
        "authority_boundary": {
            "worker_authority_environment_present": False,
            "paid_compute_authorized": False,
            "foreign_pretrained_weights_permitted": False,
            "external_llm_or_api_for_data_or_intelligence_permitted": False,
        },
        "future_real_target_requirements": list(FUTURE_REAL_TARGET_REQUIREMENTS),
    }
    evidence["evidence_sha256"] = hashlib.sha256(_canonical_bytes(evidence)).hexdigest()
    return evidence


def _write_evidence(path: Path, evidence: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fail-closed MODEL-341 GitHub-hosted carrier preflight."
    )
    parser.add_argument("--mode", default=SUPPORTED_MODE)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--evidence-out", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        evidence = build_preflight_evidence(
            source_sha=args.source_sha,
            mode=args.mode,
        )
    except HostedCarrierPreflightError as exc:
        print(f"hosted carrier preflight: FAIL: {exc}", file=sys.stderr)
        return 2

    _write_evidence(args.evidence_out, evidence)
    print(json.dumps(evidence, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
