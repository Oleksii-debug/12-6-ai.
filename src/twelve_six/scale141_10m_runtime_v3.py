"""SCALE-141 authoritative entrypoint with stable identity and memorization probes.

The v2 runtime owns actual-token accounting and memory-bounded evaluation. This
entrypoint adds two fail-closed corrections found during pre-execution audit:

* dataclass configuration is normalized to JSON-native types before run-manifest
  hashing/comparison, so tuple-valued AdamW fields survive the fresh process;
* the train-vs-heldout gap remains only a generalization proxy, while a separate
  hash-only training-passage continuation probe records memorization progression
  without changing DATA-25 or emitting source text.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from twelve_six.checkpoint import hash_json
from twelve_six import scale141_10m_runtime_v2 as v2
from twelve_six import scale141_memorization as memorization

SCHEMA = v2.SCHEMA
_V2_RUN_MANIFEST = v2._run_manifest_v2
_BASE_EVAL_POINT = v2.core._eval_point


def _json_normalize(value: Any) -> Any:
    return json.loads(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def _json_stable_run_manifest(*args, **kwargs) -> dict[str, Any]:
    value = _V2_RUN_MANIFEST(*args, **kwargs)
    value["diagnostics"] = {
        "memorization": {
            "probe_id": memorization.PROBE_ID,
            "type": "hash-only short-continuation recovery/NLL on deterministic project-owned training passages",
            "samples_per_modality": memorization.SAMPLES_PER_MODALITY,
            "continuation_tokens": memorization.WIDTH,
            "text_emitted": False,
            "canary_injection": False,
            "canary_injection_reason": "preserve exact reconstructed DATA-25 corpus identity",
            "privacy_leakage_claim": "NONE",
        }
    }
    value.pop("identity_sha256", None)
    value = _json_normalize(value)
    value["identity_sha256"] = hash_json(value)
    return value


def _eval_point_with_memorization(
    model,
    checkpoint: Path,
    corpus: Path,
    manifest: dict[str, Any],
    tok,
    trainer,
    interval_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    value = _BASE_EVAL_POINT(
        model, checkpoint, corpus, manifest, tok, trainer, interval_rows
    )
    rows_by_modality = {
        modality: list(v2.core._rows(corpus, manifest, "train", modality))
        for modality in ("uk", "en", "code")
    }
    probe = memorization.hashed_training_probe(
        model,
        tok,
        rows_by_modality,
        seed=v2.core.SEED,
        context_tokens=v2.SEQ,
    )
    value["memorization"]["metric_role"] = (
        "TRAIN_HELDOUT_GAP_IS_GENERALIZATION_PROXY_NOT_MEMORIZATION_ALONE"
    )
    value["memorization"]["hash_only_training_passage_probe"] = probe
    value["memorization"]["privacy_leakage_claim"] = "NONE"
    return value


def _install() -> None:
    # v2._install_runtime_contract resolves _run_manifest_v2 at call time, so
    # replacing it here makes phase1 and the separate resume process install the
    # same JSON-stable builder. _eval_point is not reset by v2 and is patched here.
    v2._run_manifest_v2 = _json_stable_run_manifest
    v2.core._eval_point = _eval_point_with_memorization


def phase1(repo: Path, source_sha: str, out: Path) -> dict[str, Any]:
    _install()
    return v2.phase1(repo, source_sha, out)


def resume(repo: Path, source_sha: str, out: Path) -> dict[str, Any]:
    _install()
    return v2.resume(repo, source_sha, out)


def validate(path: Path, expected_source_sha: str | None = None) -> dict[str, Any]:
    report = v2.validate(path, expected_source_sha)
    for target in v2.EVAL_TOKEN_TARGETS:
        point = report["scheduled"].get(str(target))
        if point is None:
            raise v2.Scale141RuntimeError(
                f"missing scheduled evaluation at optimized-token target {target}"
            )
        probe = point["memorization"].get("hash_only_training_passage_probe")
        if not isinstance(probe, dict):
            raise v2.Scale141RuntimeError("missing hash-only memorization probe")
        if probe.get("text_emitted") is not False:
            raise v2.Scale141RuntimeError("memorization probe emitted source text")
        if probe.get("model_non_mutation_passed") is not True:
            raise v2.Scale141RuntimeError("memorization probe mutation gate failed")
        if probe.get("canary_injection") is not False:
            raise v2.Scale141RuntimeError("SCALE-141 corpus identity was altered by canaries")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("phase1", "resume"):
        p = sub.add_parser(name)
        p.add_argument("--repo-root", type=Path, default=Path("."))
        p.add_argument("--source-sha", required=True)
        p.add_argument("--output-dir", type=Path, required=True)
    p = sub.add_parser("validate")
    p.add_argument("report", type=Path)
    p.add_argument("--expected-source-sha")
    args = parser.parse_args()
    if args.command == "phase1":
        value = phase1(
            args.repo_root.resolve(), args.source_sha, args.output_dir.resolve()
        )
    elif args.command == "resume":
        value = resume(
            args.repo_root.resolve(), args.source_sha, args.output_dir.resolve()
        )
    else:
        value = validate(args.report, args.expected_source_sha)
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
