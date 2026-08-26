"""Independent scientific admission verifier for the terminal LEARN-191 3M artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import subprocess
from pathlib import Path
from typing import Any, Mapping

import torch

from twelve_six import milestone100_first_learned as m100
from twelve_six.checkpoint import hash_json, verify_checkpoint
from twelve_six.inference.contracts import GenerationConfig
from twelve_six.inference.first_party import load_first_party_backend
from twelve_six.inference.generation import generate
from twelve_six.model import InitSpec, ModelSpec, TwelveSixDecoder
from twelve_six.tokenization import ByteTokenizer

WORKER = "VERIFY-219-LEARNED-3M-INDEPENDENT"
STATE = "VERIFIED_LEARNED_3M"
SCHEMA = "12-6.verify219-learned-3m-independent.v1"
PRODUCER_SHA = "a75920cef8bde37a8c590e34095be83c97b75f1d"
PRODUCER_ARTIFACT_ID = 9597788382
PRODUCER_ARTIFACT_ZIP_SHA256 = "f57bf36113a68fffd4bfcf877bf08762393479b9c09e6fd0fd613fbb91f044ee"
EXPECTED_MODEL_SPEC_SHA256 = "462c85da80a3c0d7d6a4f1a570b87d208b1847d8a57b12a4d9be7e36846b65dc"
EXPECTED_PARAMETER_COUNT = 3_213_120
EXPECTED_CORPUS_ID = "422f545da93526ceba2c0ff738e0b9deac65f35bfb31d87d2aab068cda091fc8"
EXPECTED_M150_EVAL_ID = "7189e6df053574beb686727c94e684cdbaf08a34ef33aa953eff7cdae0320113"
TARGETS = (16_632, 65_772, 131_292)
LIMITS = {"uk": 256, "en": 192, "code": 128}
SOURCE_FAMILY = {
    "uk": "project-authored:uk:corpus-v01",
    "en": "project-authored:en:corpus-v01",
    "code": "project-authored:code:corpus-v01",
}
PROMPTS = {"uk": "Українська мова ", "en": "The training corpus ", "code": "def stable_"}
ABS_TOL = 1e-7


class Verify219Error(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Verify219Error(message)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Verify219Error(f"cannot read {path}: {exc}") from exc
    _require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def _self_hash(value: Mapping[str, Any], field: str, label: str) -> str:
    supplied = value.get(field)
    _require(isinstance(supplied, str) and len(supplied) == 64, f"{label} hash missing")
    unsigned = dict(value)
    unsigned.pop(field, None)
    _require(hash_json(unsigned) == supplied, f"{label} self-hash mismatch")
    return supplied


def _tree_hash(root: Path) -> str:
    _require(root.is_dir(), f"checkpoint missing: {root}")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    _require(bool(files), f"checkpoint empty: {root}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _close(actual: Any, expected: Any, label: str) -> None:
    try:
        a = float(actual)
        e = float(expected)
    except (TypeError, ValueError) as exc:
        raise Verify219Error(f"{label} not numeric") from exc
    _require(math.isfinite(a) and math.isfinite(e), f"{label} not finite")
    _require(abs(a - e) <= ABS_TOL, f"{label} mismatch: {a} != {e}")


def _selection_eval(model, corpus: Path, manifest: Mapping[str, Any], tok: ByteTokenizer) -> dict[str, Any]:
    before = m100._state_hash(model)
    mode = bool(model.training)
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    by: dict[str, Any] = {}
    try:
        with torch.no_grad():
            for stratum in ("uk", "en", "code"):
                examples = []
                for example in m100._packed(corpus, manifest, tok, "validation", stratum):
                    examples.append(example)
                    if len(examples) == LIMITS[stratum]:
                        break
                _require(len(examples) == LIMITS[stratum], f"validation/{stratum} subset exhausted")
                nll = 0.0
                tokens = 0
                for start in range(0, len(examples), 32):
                    part_nll, part_tokens = m100._eval_examples(model, examples[start : start + 32])
                    nll += part_nll
                    tokens += part_tokens
                _require(tokens > 0, f"validation/{stratum} has no target bytes")
                by[stratum] = {
                    "bits_per_byte": nll / math.log(2.0) / tokens,
                    "loss_nats_per_byte": nll / tokens,
                    "predicted_byte_tokens": tokens,
                }
                total_nll += nll
                total_tokens += tokens
    finally:
        model.train(mode)
    after = m100._state_hash(model)
    _require(after == before, "selection evaluation mutated model")
    return {
        "split": "validation",
        "bits_per_byte": total_nll / math.log(2.0) / total_tokens,
        "loss_nats_per_byte": total_nll / total_tokens,
        "predicted_byte_tokens": total_tokens,
        "by_stratum": by,
        "by_source_family": {
            SOURCE_FAMILY[stratum]: {**by[stratum], "stratum": stratum}
            for stratum in ("uk", "en", "code")
        },
        "model_state_sha256_before": before,
        "model_state_sha256_after": after,
        "non_mutation_passed": True,
    }


def _compare_selection(actual: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    for field in ("bits_per_byte", "loss_nats_per_byte"):
        _close(actual.get(field), expected.get(field), f"{label}.{field}")
    _require(
        int(actual.get("predicted_byte_tokens", -1)) == int(expected.get("predicted_byte_tokens", -2)),
        f"{label}.predicted_byte_tokens mismatch",
    )
    for stratum in ("uk", "en", "code"):
        a = actual.get("by_stratum", {}).get(stratum)
        e = expected.get("by_stratum", {}).get(stratum)
        _require(isinstance(a, Mapping) and isinstance(e, Mapping), f"{label}.{stratum} missing")
        for field in ("bits_per_byte", "loss_nats_per_byte"):
            _close(a.get(field), e.get(field), f"{label}.{stratum}.{field}")
        _require(
            int(a.get("predicted_byte_tokens", -1)) == int(e.get("predicted_byte_tokens", -2)),
            f"{label}.{stratum}.predicted_byte_tokens mismatch",
        )
    _require(actual.get("non_mutation_passed") is True, f"{label} mutation flag failed")
    _require(
        actual.get("model_state_sha256_before") == actual.get("model_state_sha256_after"),
        f"{label} state changed",
    )


def _m150_identity(tok: ByteTokenizer, manifest: Mapping[str, Any]) -> dict[str, Any]:
    identity = tok.identity
    value: dict[str, Any] = {
        "schema": "12-6.learned-base-ladder-evaluation-identity.v1",
        "corpus_identity_sha256": manifest["corpus_identity_sha256"],
        "split": "validation",
        "strata_order": ["uk", "en", "code"],
        "metric": "autoregressive_cross_entropy_nats_and_bits_per_raw_utf8_byte",
        "target_mask": "labels[:,1:] != -100",
        "tokenizer": {
            "version": identity.version,
            "config_sha256": identity.config_sha256,
            "vocab_sha256": identity.vocab_sha256,
            "vocab_size": identity.vocab_size,
            "normalization": identity.normalization,
            "encoding": identity.encoding,
            "special_tokens": dict(identity.special_tokens),
        },
        "packing": {"version": m100.PACKING_VERSION, "sequence_length": m100.SEQ, "cross_document": False},
    }
    value["identity_sha256"] = hash_json(value)
    return value


def _checkpoint_identity(checked: Mapping[str, Any], run: Mapping[str, Any], tok: ByteTokenizer, role: str) -> dict[str, Any]:
    identity = checked.get("identity")
    _require(isinstance(identity, Mapping), f"{role} identity missing")
    _require(identity.get("git_sha") == PRODUCER_SHA, f"{role} source SHA mismatch")
    _require(identity.get("model_spec_hash") == EXPECTED_MODEL_SPEC_SHA256, f"{role} ModelSpec mismatch")
    _require(int(identity.get("parameter_count", -1)) == EXPECTED_PARAMETER_COUNT, f"{role} parameter mismatch")
    _require(identity.get("dataset_manifest_hash") == EXPECTED_CORPUS_ID, f"{role} corpus mismatch")
    _require(identity.get("run_manifest_hash") == run.get("identity_sha256"), f"{role} run mismatch")
    _require(identity.get("tokenizer_hash") == tok.identity.config_sha256, f"{role} tokenizer mismatch")
    _require(identity.get("tokenizer_vocab_hash") == tok.identity.vocab_sha256, f"{role} vocab mismatch")
    _require(int(identity.get("step", -1)) > 0 and int(identity.get("tokens_seen", -1)) > 0, f"{role} is not learned")
    return dict(identity)


def _logits(checkpoint: Path) -> dict[str, Any]:
    backend = load_first_party_backend(checkpoint)
    outputs = {}
    for name, prompt in PROMPTS.items():
        ids = list(backend.encode(prompt))
        values = list(backend.next_token_logits(ids))
        _require(len(values) == 256 and all(math.isfinite(float(v)) for v in values), f"{name} logits invalid")
        packed = b"".join(struct.pack("<f", float(v)) for v in values)
        ranked = sorted(range(256), key=lambda i: (-float(values[i]), i))[:8]
        outputs[name] = {
            "prompt": prompt,
            "input_ids": ids,
            "logits_float32_sha256": hashlib.sha256(packed).hexdigest(),
            "argmax_token_id": ranked[0],
            "top8_token_ids": ranked,
        }
    return {"backend_diagnostics": backend.diagnostics(), "outputs": outputs}


def _generation(checkpoint: Path) -> dict[str, Any]:
    backend = load_first_party_backend(checkpoint)
    config = GenerationConfig(max_new_tokens=48, sample=False)
    outputs = {}
    for name, prompt in PROMPTS.items():
        result = generate(backend, prompt, config)
        outputs[name] = {
            "prompt": prompt,
            "generated_token_ids": list(result.generated_token_ids),
            "text": result.text,
            "stop_reason": result.stop_reason,
        }
    return {"backend_diagnostics": backend.diagnostics(), "decoding": "greedy", "outputs": outputs}


def _head(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Verify219Error(f"cannot resolve HEAD: {exc}") from exc


def verify(repo: Path, artifact_root: Path, verifier_head_sha: str, output: Path) -> dict[str, Any]:
    repo = repo.resolve()
    _require(_head(repo) == verifier_head_sha, "verifier exact-head mismatch")
    _require(verifier_head_sha != PRODUCER_SHA, "verifier must differ from producer")
    root = artifact_root.resolve() / "learn191-evidence"
    report = _read_json(root / "learn191-real-3m-report.json")
    run = _read_json(root / "run-manifest.json")
    truth = _read_json(root / "truth.json")
    phase1 = _read_json(root / "3m" / "phase1.json")
    resume = _read_json(root / "3m" / "resume.json")
    fresh = _read_json(root / "3m" / "final-fresh-load-proof.json")

    hashes = {
        "report": _self_hash(report, "identity_sha256", "report"),
        "run_manifest": _self_hash(run, "identity_sha256", "run manifest"),
        "truth": _self_hash(truth, "identity_sha256", "truth"),
        "phase1": _self_hash(phase1, "identity_sha256", "phase1"),
        "resume": _self_hash(resume, "identity_sha256", "resume"),
        "final_fresh_load": _self_hash(fresh, "identity_sha256", "final fresh load"),
    }
    for label, value in (
        ("report", report.get("source_sha")),
        ("run", run.get("source_sha")),
        ("truth", truth.get("source_sha")),
        ("phase1", phase1.get("source_sha")),
        ("resume", resume.get("source_sha")),
        ("fresh", fresh.get("source_sha")),
    ):
        _require(value == PRODUCER_SHA, f"{label} source SHA mismatch")
    _require(report.get("worker_id") == "LEARN-191-REAL-3M", "producer worker mismatch")
    _require(int(report.get("model", {}).get("parameter_count", -1)) == EXPECTED_PARAMETER_COUNT, "parameter count mismatch")
    _require(report.get("model", {}).get("spec_sha256") == EXPECTED_MODEL_SPEC_SHA256, "ModelSpec mismatch")
    _require(report.get("corpus_identity_sha256") == EXPECTED_CORPUS_ID, "corpus identity mismatch")
    _require(report.get("retained_m150_evaluation_identity_sha256") == EXPECTED_M150_EVAL_ID, "retained M150 identity mismatch")
    boundary = report.get("truth_boundary", {})
    for field in ("foreign_pretrained_weights", "sft", "paid_compute"):
        _require(boundary.get(field) is False, f"truth boundary {field} violated")
    _require(resume.get("fresh_process_resume_passed") is True, "producer fresh resume failed")
    _require(int(resume.get("process_pid", -1)) != int(phase1.get("process_pid", -1)), "resume PID not fresh")

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    scratch = output.parent / "verify219-corpus"
    manifest = m100._build_corpus(repo, scratch)
    _require(manifest["corpus_identity_sha256"] == EXPECTED_CORPUS_ID, "rebuilt DATA-25 drift")
    tok = ByteTokenizer()
    m150_identity = _m150_identity(tok, manifest)
    _require(m150_identity["identity_sha256"] == EXPECTED_M150_EVAL_ID, "reconstructed M150 identity drift")

    spec = ModelSpec.from_dict(report["model"]["spec"])
    init = InitSpec.from_dict(report["model"]["init_spec"])
    _require(spec.identity_sha256() == EXPECTED_MODEL_SPEC_SHA256, "report ModelSpec semantic drift")
    torch.manual_seed(int(report["seed"]))
    random_model = TwelveSixDecoder(spec, init)
    random_selection = _selection_eval(random_model, scratch / "corpus-a", manifest, tok)
    evaluations = {int(item["target_optimized_tokens"]): item for item in report["evaluations"]}
    _require(set(evaluations) == {0, *TARGETS}, "producer evaluation target set mismatch")
    _compare_selection(random_selection, evaluations[0]["selection_validation"], "random_init")
    random_common = m100._evaluate(random_model, scratch / "corpus-a", manifest, tok)

    checkpoints = {int(item["target_optimized_tokens"]): item for item in report["checkpoints"]}
    _require(set(checkpoints) == set(TARGETS), "producer checkpoint target set mismatch")
    results: dict[str, Any] = {}
    for target in TARGETS:
        path = root / "3m" / f"checkpoint-t{target:06d}"
        before = _tree_hash(path)
        checked = verify_checkpoint(path)
        identity = _checkpoint_identity(checked, run, tok, f"target_{target}")
        expected_cp = checkpoints[target]
        _require(expected_cp.get("checkpoint_id") == checked.get("checkpoint_id"), f"target {target} checkpoint ID mismatch")
        _require(int(expected_cp.get("optimizer_step", -1)) == int(identity["step"]), f"target {target} step mismatch")
        _require(int(expected_cp.get("actual_optimized_tokens", -1)) == int(identity["tokens_seen"]), f"target {target} tokens mismatch")
        backend = load_first_party_backend(path)
        measured = _selection_eval(backend.model, scratch / "corpus-a", manifest, tok)
        _compare_selection(measured, evaluations[target]["selection_validation"], f"target_{target}")
        after = _tree_hash(path)
        _require(before == after, f"target {target} checkpoint tree mutated")
        results[str(target)] = {
            "checkpoint_id": checked["checkpoint_id"],
            "identity": identity,
            "selection_validation": measured,
            "checkpoint_tree_sha256_before": before,
            "checkpoint_tree_sha256_after": after,
        }

    measured_best = min(TARGETS, key=lambda target: (float(results[str(target)]["selection_validation"]["bits_per_byte"]), target))
    _require(int(report["best_checkpoint"]["target_optimized_tokens"]) == measured_best, "producer best checkpoint selection mismatch")
    _require(int(report["final_checkpoint"]["target_optimized_tokens"]) == TARGETS[-1], "producer final checkpoint target mismatch")
    final_path = root / "3m" / f"checkpoint-t{TARGETS[-1]:06d}"
    final_backend = load_first_party_backend(final_path)
    final_common = m100._evaluate(final_backend.model, scratch / "corpus-a", manifest, tok)
    _require(float(results[str(measured_best)]["selection_validation"]["bits_per_byte"]) < float(random_selection["bits_per_byte"]), "best 3M did not improve over random init")
    _require(float(results[str(TARGETS[-1])]["selection_validation"]["bits_per_byte"]) < float(random_selection["bits_per_byte"]), "final 3M did not improve over random init")
    _require(float(final_common["bits_per_byte"]) < float(random_common["bits_per_byte"]), "final 3M M150-common metric did not improve over random init")

    logits_a = _logits(final_path)
    logits_b = _logits(final_path)
    _require(logits_a == logits_b, "final logits are not reproducible")
    generation = _generation(final_path)
    _require(generation == report.get("generation"), "final greedy generation mismatch")
    _require(generation == fresh.get("generation"), "final fresh-load generation mismatch")
    _require(fresh.get("checkpoint_id") == checkpoints[TARGETS[-1]].get("checkpoint_id"), "fresh-load checkpoint ID mismatch")

    result: dict[str, Any] = {
        "schema": SCHEMA,
        "worker": WORKER,
        "state": STATE,
        "verifier_head_sha": verifier_head_sha,
        "producer": {
            "git_sha": PRODUCER_SHA,
            "artifact_id": PRODUCER_ARTIFACT_ID,
            "artifact_zip_sha256": PRODUCER_ARTIFACT_ZIP_SHA256,
            "evidence_hashes": hashes,
        },
        "model": {
            "parameter_count": EXPECTED_PARAMETER_COUNT,
            "model_spec_sha256": EXPECTED_MODEL_SPEC_SHA256,
            "tokenizer_version": tok.identity.version,
            "tokenizer_config_sha256": tok.identity.config_sha256,
            "tokenizer_vocab_sha256": tok.identity.vocab_sha256,
        },
        "data_and_eval": {
            "corpus_identity_sha256": manifest["corpus_identity_sha256"],
            "m150_common_evaluation_identity_sha256": m150_identity["identity_sha256"],
            "random_init_selection_validation": random_selection,
            "random_init_m150_common": random_common,
            "final_m150_common": final_common,
            "final_m150_common_improved_over_random_init": float(final_common["bits_per_byte"]) < float(random_common["bits_per_byte"]),
        },
        "checkpoints": results,
        "measured_best_target_optimized_tokens": measured_best,
        "final_first_party_logits": logits_a,
        "final_greedy_generation": generation,
        "fresh_process_resume": {
            "phase1_pid": phase1["process_pid"],
            "resume_pid": resume["process_pid"],
            "passed": True,
        },
        "boundaries": {
            "training_executed": False,
            "optimizer_updates": 0,
            "foreign_pretrained_weights": False,
            "paid_compute": False,
            "instruction_or_behavioral_alignment": False,
            "external_llm_used": False,
            "evaluation_mutated_model": False,
            "claim_scope": "Independent admission of the exact LEARN-191 3M artifact under DATA-25; not a production, external-representativeness, instruction-following, or broad-capability claim.",
        },
    }
    result["identity_sha256"] = hash_json(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--verifier-head-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = verify(args.repo_root, args.artifact_root, args.verifier_head_sha, args.output)
    print(json.dumps({"worker": value["worker"], "state": value["state"], "identity_sha256": value["identity_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
