"""Fail-closed bridge from VERIFY-218 science to the RUNTIME-225 source contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from twelve_six.checkpoint import verify_checkpoint

SCHEMA = "12-6.verify218-runtime225-authority.v1"
WORKER_ID = "VERIFY-218-LEARNED-10M-INDEPENDENT"
STATUS = "VERIFIED_LEARNED_10M"
PRODUCER_SHA = "c02c8aa38e691521ae2ab6a4ff3ea1d643efd6ef"
SCIENTIFIC_ARTIFACT_ID = 9602650341
SCIENTIFIC_ARTIFACT_NAME = "learn217-terminal-10m-learned-base"
SCIENTIFIC_ARTIFACT_DIGEST = (
    "sha256:8631e90417e40365b3fc0d6bc98ee6adda5a4ed24530e675d9a91c93219537ee"
)
SCIENTIFIC_RUN_ID = 32952787070
RUNTIME_ARTIFACT_ID = 9602907196
RUNTIME_ARTIFACT_NAME = "scale141-10m-learned-fallback"
RUNTIME_ARTIFACT_DIGEST = (
    "sha256:d2abd029f64207567a1d6b4ce9943ff15bfd211acdd05e9ff84156ce66607218"
)
RUNTIME_RUN_ID = 32952786715
MODEL_SHA = "61caa5469123e23b9b72fc2024140bfca84c4c480dcb0a7e712ba800a4f22998"
PARAMETERS = 10_000_640
TOKENIZER_VERSION = "s0-byte-v1"
TOKENIZER_CONFIG = "b04055c1061dd641dcab7cb9d62a931f09b8d1a070140a926ceb4e91d73ca8e1"
TOKENIZER_VOCAB = "905ed40bb42cc4d550e228ff5f24158d504b38e8ed5974dfa3077bd5867ad571"
CORPUS_SHA = "422f545da93526ceba2c0ff738e0b9deac65f35bfb31d87d2aab068cda091fc8"
BEST_ID = "12f9edd88bf5e596ae6f985564a5dcff96033922100ba91678ef9a76c0df3156"
FINAL_ID = "20fbb9ffe0e0ecb2b0098dd6f7c18e23cd6cfcc0a0e48cb25c73c26d2f50926d"
COMMON_EVAL = "7189e6df053574beb686727c94e684cdbaf08a34ef33aa953eff7cdae0320113"


class Verify218BridgeError(RuntimeError):
    """Fail-closed bridge error."""


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise Verify218BridgeError(message)


def _obj(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def _read(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Verify218BridgeError(f"{label} is not readable JSON") from exc
    _require(isinstance(value, dict), f"{label} must be an object")
    return value


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def _validate_transport(
    artifact: Mapping[str, Any],
    run: Mapping[str, Any],
    *,
    artifact_id: int,
    artifact_name: str,
    artifact_digest: str,
    run_id: int,
) -> None:
    _require(artifact.get("id") == artifact_id, "artifact id mismatch")
    _require(artifact.get("name") == artifact_name, "artifact name mismatch")
    _require(artifact.get("digest") == artifact_digest, "artifact digest mismatch")
    _require(artifact.get("expired") is False, "artifact is expired")
    arun = _obj(artifact.get("workflow_run"), "artifact workflow_run")
    _require(arun.get("id") == run_id, "artifact workflow run mismatch")
    _require(arun.get("head_sha") == PRODUCER_SHA, "artifact head SHA mismatch")
    _require(run.get("id") == run_id, "workflow run id mismatch")
    _require(run.get("head_sha") == PRODUCER_SHA, "workflow run head SHA mismatch")
    _require(run.get("status") == "completed", "workflow run is not completed")
    _require(run.get("conclusion") == "success", "workflow run is not SUCCESS")


def _validate_science(science: Mapping[str, Any]) -> None:
    _require(science.get("schema") == "12-6.verify218-learned-10m-independent.v1", "science schema")
    _require(science.get("worker") == WORKER_ID, "science worker")
    _require(science.get("state") == STATUS, "science state")
    producer = _obj(science.get("producer"), "science producer")
    _require(producer.get("git_sha") == PRODUCER_SHA, "science producer SHA")
    _require(producer.get("artifact_id") == SCIENTIFIC_ARTIFACT_ID, "science artifact id")
    _require(
        producer.get("artifact_zip_sha256") == SCIENTIFIC_ARTIFACT_DIGEST.split(":", 1)[1],
        "science artifact digest",
    )
    model = _obj(science.get("model"), "science model")
    _require(model.get("model_spec_sha256") == MODEL_SHA, "science ModelSpec")
    _require(model.get("parameter_count") == PARAMETERS, "science parameter count")
    _require(model.get("tokenizer_version") == TOKENIZER_VERSION, "science tokenizer")
    _require(model.get("tokenizer_config_sha256") == TOKENIZER_CONFIG, "science tokenizer config")
    _require(model.get("tokenizer_vocab_sha256") == TOKENIZER_VOCAB, "science tokenizer vocab")
    data = _obj(science.get("data_and_eval"), "science data")
    _require(data.get("corpus_identity_sha256") == CORPUS_SHA, "science corpus")
    _require(data.get("ladder_common_evaluation_identity_sha256") == COMMON_EVAL, "science eval id")
    _require(data.get("best_improved_over_reconstructed_random_init") is True, "best vs init")
    _require(data.get("final_improved_over_reconstructed_random_init") is True, "final vs init")
    bounds = _obj(science.get("boundaries"), "science boundaries")
    _require(bounds.get("training_executed") is False, "verifier trained")
    _require(bounds.get("optimizer_updates") == 0, "verifier optimized")
    _require(bounds.get("foreign_pretrained_weights") is False, "foreign weights")
    _require(bounds.get("paid_compute") is False, "paid compute")
    _require(bounds.get("evaluation_mutated_model") is False, "evaluation mutation")


def _validate_resume_probe(probe: Mapping[str, Any], science: Mapping[str, Any]) -> None:
    phase1 = _obj(_obj(science.get("recovery"), "science recovery").get("phase1"), "phase1")
    _require(probe.get("checkpoint_id") == phase1.get("checkpoint_id"), "probe checkpoint")
    _require(int(probe.get("optimizer_step", -1)) == int(phase1.get("step", -2)), "probe step")
    _require(int(probe.get("tokens_seen", -1)) == int(phase1.get("tokens_seen", -2)), "probe tokens")
    _require(probe.get("checkpoint_safe_after_restore") is True, "probe safety")
    _require(probe.get("rng_restore_requested") is True, "probe RNG restore")
    _require(probe.get("training_executed") is False, "probe trained")
    _require(probe.get("optimizer_updates") == 0, "probe optimized")
    state = _obj(probe.get("optimizer_state"), "probe optimizer")
    for field in ("populated_parameters", "tensor_leaves", "scalar_step_tensors"):
        _require(int(state.get(field, 0)) > 0, f"probe {field}")


def _role_evidence(role: Mapping[str, Any], label: str) -> None:
    logits = _obj(_obj(role.get("first_party_logits"), f"{label} logits").get("outputs"), "logits outputs")
    _require(bool(logits), f"{label} logits missing")
    for row in logits.values():
        item = _obj(row, f"{label} logits row")
        digest = item.get("logits_float32_sha256")
        _require(isinstance(digest, str) and len(digest) == 64, f"{label} logits digest")
        token = item.get("argmax_token_id")
        _require(isinstance(token, int) and not isinstance(token, bool) and 0 <= token < 256, f"{label} argmax")
    generation = _obj(role.get("greedy_generation"), f"{label} generation")
    _require(generation.get("decoding") == "greedy", f"{label} decoding")
    _require(bool(_obj(generation.get("outputs"), "generation outputs")), f"{label} generation")
    evaluation = _obj(role.get("common_evaluation"), f"{label} evaluation")
    _require(evaluation.get("non_mutation_passed") is True, f"{label} mutation")
    for field in ("loss", "bits_per_byte"):
        try:
            value = float(evaluation.get(field))
        except (TypeError, ValueError) as exc:
            raise Verify218BridgeError(f"{label} {field} nonnumeric") from exc
        _require(math.isfinite(value) and value > 0.0, f"{label} {field} invalid")


def _runtime_roles(science: Mapping[str, Any], root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    fresh = _read(root / "fresh-verification.json", "runtime fresh")
    index = _read(root / "retained" / "index.json", "runtime retained")
    _require(fresh.get("schema") == "12-6.scale141-fresh-verification.v2", "fresh schema")
    _require(fresh.get("source_sha") == PRODUCER_SHA, "fresh source")
    _require(fresh.get("corpus_identity_sha256") == CORPUS_SHA, "fresh corpus")
    _require(_obj(fresh.get("fresh_verification"), "fresh gates").get("status") == "PASS", "fresh status")
    _require(index.get("schema") == "12-6.scale141-retained-checkpoints.v1", "retained schema")
    _require(index.get("source_sha") == PRODUCER_SHA, "retained source")
    sroles = _obj(science.get("checkpoints"), "science checkpoints")
    roles = _obj(index.get("roles"), "runtime roles")
    evidence = _obj(fresh.get("evidence"), "runtime evidence")
    resolved: dict[str, dict[str, Any]] = {}
    for label, expected in (("best", BEST_ID), ("final", FINAL_ID)):
        sr = _obj(sroles.get(label), f"science {label}")
        rr = _obj(roles.get(label), f"runtime {label}")
        ev = _obj(evidence.get(label), f"runtime evidence {label}")
        checked = verify_checkpoint(root / "retained" / label)
        identity = _obj(checked.get("identity"), f"{label} identity")
        for observed in (sr.get("checkpoint_id"), rr.get("checkpoint_id"), ev.get("checkpoint_id"), checked.get("checkpoint_id")):
            _require(observed == expected, f"{label} checkpoint divergence")
        _require(identity.get("git_sha") == PRODUCER_SHA, f"{label} source")
        _require(identity.get("model_spec_hash") == MODEL_SHA, f"{label} ModelSpec")
        _require(identity.get("dataset_manifest_hash") == CORPUS_SHA, f"{label} corpus")
        step, tokens = identity.get("step"), identity.get("tokens_seen")
        _require(isinstance(step, int) and step > 0, f"{label} step")
        _require(isinstance(tokens, int) and tokens > 0, f"{label} tokens")
        resolved[label] = {"checkpoint_id": expected, "step": step, "tokens_seen": tokens}
        _role_evidence(sr, label)
    common = _obj(_obj(fresh.get("ladder_common_evaluation"), "common eval").get("identity"), "eval identity")
    _require(common.get("identity_sha256") == COMMON_EVAL, "runtime common eval")
    return resolved["best"], resolved["final"]


def _producer_resume(root: Path) -> dict[str, Any]:
    report = _read(root / "report.json", "runtime report")
    _require(report.get("source_sha") == PRODUCER_SHA, "report source")
    corpus = _obj(report.get("corpus"), "report corpus")
    _require(corpus.get("identity_sha256") == CORPUS_SHA, "report corpus id")
    _require(int(corpus.get("optimized_tokens", 0)) > 1_000_000, "smoke-only source")
    resume = _obj(report.get("fresh_process_resume"), "producer resume")
    _require(resume.get("phase1_pid") != resume.get("resume_pid"), "producer not fresh process")
    _require(resume.get("metric_recheck_passed") is True, "producer resume metric")
    loaded, first = resume.get("loaded_step"), resume.get("first_resumed_step")
    _require(isinstance(loaded, int) and loaded > 0, "producer loaded step")
    _require(isinstance(first, int) and first > loaded, "producer continuation")
    if "recovery_resolution" in resume:
        _require(resume.get("recovery_resolution") == "EXACT_PHASE1_REFERENCE", "recovery resolution")
    return dict(resume)


def build_authority(*, science: Mapping[str, Any], scientific_artifact: Mapping[str, Any], scientific_run: Mapping[str, Any], runtime_artifact: Mapping[str, Any], runtime_run: Mapping[str, Any], runtime_root: Path, resume_probe: Mapping[str, Any]) -> dict[str, Any]:
    _validate_transport(scientific_artifact, scientific_run, artifact_id=SCIENTIFIC_ARTIFACT_ID, artifact_name=SCIENTIFIC_ARTIFACT_NAME, artifact_digest=SCIENTIFIC_ARTIFACT_DIGEST, run_id=SCIENTIFIC_RUN_ID)
    _validate_transport(runtime_artifact, runtime_run, artifact_id=RUNTIME_ARTIFACT_ID, artifact_name=RUNTIME_ARTIFACT_NAME, artifact_digest=RUNTIME_ARTIFACT_DIGEST, run_id=RUNTIME_RUN_ID)
    _validate_science(science)
    _validate_resume_probe(resume_probe, science)
    best, final = _runtime_roles(science, runtime_root)
    producer_resume = _producer_resume(runtime_root)
    result: dict[str, Any] = {
        "schema": SCHEMA, "worker_id": WORKER_ID, "status": STATUS,
        "verified_learned_10m": True, "foreign_pretrained_weights": False,
        "mechanics_only_checkpoint": False, "one_step_smoke": False,
        "gates": {name: True for name in ("checkpoint_integrity", "fresh_process_resume", "finite_first_party_logits", "heldout_bpb", "evaluation_non_mutation", "greedy_generation", "best_final_role_resolution")},
        "model": {"model_spec_sha256": MODEL_SHA, "parameter_count": PARAMETERS},
        "tokenizer": {"version": TOKENIZER_VERSION, "config_sha256": TOKENIZER_CONFIG, "vocab_sha256": TOKENIZER_VOCAB},
        "corpus_identity_sha256": CORPUS_SHA,
        "source": {"artifact_id": RUNTIME_ARTIFACT_ID, "artifact_name": RUNTIME_ARTIFACT_NAME, "artifact_digest": RUNTIME_ARTIFACT_DIGEST, "workflow_run_id": RUNTIME_RUN_ID, "source_sha": PRODUCER_SHA},
        "checkpoint": {"role": "best", **best},
        "final_checkpoint": {"role": "final", **final},
        "scientific_authority": {"artifact_id": SCIENTIFIC_ARTIFACT_ID, "artifact_name": SCIENTIFIC_ARTIFACT_NAME, "artifact_digest": SCIENTIFIC_ARTIFACT_DIGEST, "workflow_run_id": SCIENTIFIC_RUN_ID, "source_sha": PRODUCER_SHA, "identity_sha256": science.get("identity_sha256")},
        "fresh_process_resume_evidence": {"producer_report": producer_resume, "independent_restore_probe": dict(resume_probe)},
        "truth_boundary": {"runtime_source_cross_bound_to_scientific_checkpoint_ids": True, "source_substitution_allowed": False, "campaign_retrained_by_verify218": False, "optimizer_updates_by_verify218": 0, "paid_compute": False, "external_real_corpus_claim": False, "learned_20m_claim": False, "direct_3m_vs_10m_ranking": False},
    }
    result["identity_sha256"] = _canonical_sha256(result)
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("scientific-authority", "scientific-artifact-metadata", "scientific-run-metadata", "runtime-artifact-metadata", "runtime-run-metadata", "runtime-root", "resume-probe", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    a = p.parse_args()
    result = build_authority(
        science=_read(a.scientific_authority, "science authority"),
        scientific_artifact=_read(a.scientific_artifact_metadata, "science artifact"),
        scientific_run=_read(a.scientific_run_metadata, "science run"),
        runtime_artifact=_read(a.runtime_artifact_metadata, "runtime artifact"),
        runtime_run=_read(a.runtime_run_metadata, "runtime run"),
        runtime_root=a.runtime_root.resolve(),
        resume_probe=_read(a.resume_probe, "resume probe"),
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "identity_sha256": result["identity_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
