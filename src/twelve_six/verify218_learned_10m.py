"""Independent scientific verifier for the terminal LEARN-217 learned 10M Base artifact.

This module is intentionally an artifact consumer, not a training runtime. It
rebuilds the canonical DATA-25 evaluation corpus from repository sources,
reconstructs the random-init baseline, independently reloads retained learned
checkpoints, reruns the common ladder evaluation, and binds all claims to exact
immutable producer identities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from twelve_six import milestone100_first_learned as m100
from twelve_six.checkpoint import hash_json, verify_checkpoint
from twelve_six.inference.contracts import GenerationConfig
from twelve_six.inference.first_party import load_first_party_backend
from twelve_six.inference.generation import generate
from twelve_six.model import InitSpec, ModelSpec, TwelveSixDecoder
from twelve_six.tokenization import ByteTokenizer

WORKER = "VERIFY-218-LEARNED-10M-INDEPENDENT"
STATE = "VERIFIED_LEARNED_10M"
SCHEMA = "12-6.verify218-learned-10m-independent.v1"

PRODUCER_SHA = "c02c8aa38e691521ae2ab6a4ff3ea1d643efd6ef"
PRODUCER_ARTIFACT_ID = 9602650341
PRODUCER_ARTIFACT_ZIP_SHA256 = (
    "8631e90417e40365b3fc0d6bc98ee6adda5a4ed24530e675d9a91c93219537ee"
)
EXPECTED_MODEL_SPEC_SHA256 = (
    "61caa5469123e23b9b72fc2024140bfca84c4c480dcb0a7e712ba800a4f22998"
)
EXPECTED_PARAMETER_COUNT = 10_000_640
EXPECTED_CORPUS_ID = "422f545da93526ceba2c0ff738e0b9deac65f35bfb31d87d2aab068cda091fc8"
EXPECTED_LADDER_EVAL_ID = (
    "7189e6df053574beb686727c94e684cdbaf08a34ef33aa953eff7cdae0320113"
)
EXPECTED_TOKENIZER_VERSION = "s0-byte-v1"
PROMPTS = {
    "uk": "Українська мова ",
    "en": "The training corpus ",
    "code": "def stable_",
}
ABS_TOL = 1e-7


class Verify218Error(RuntimeError):
    """Fail-closed verifier error."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Verify218Error(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise Verify218Error(f"{path} must contain a JSON object")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Verify218Error(message)


def _verify_self_hash(value: Mapping[str, Any], field: str, label: str) -> str:
    supplied = value.get(field)
    _require(
        isinstance(supplied, str) and len(supplied) == 64,
        f"{label} {field} missing",
    )
    unsigned = dict(value)
    unsigned.pop(field, None)
    _require(hash_json(unsigned) == supplied, f"{label} self-hash mismatch")
    return supplied


def _tree_sha256(root: Path) -> str:
    _require(root.is_dir(), f"checkpoint directory missing: {root}")
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    _require(bool(files), f"checkpoint directory is empty: {root}")
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _assert_close_scalar(actual: Any, expected: Any, label: str) -> None:
    try:
        a = float(actual)
        e = float(expected)
    except (TypeError, ValueError) as exc:
        raise Verify218Error(f"{label} is not numeric") from exc
    _require(math.isfinite(a) and math.isfinite(e), f"{label} is not finite")
    _require(abs(a - e) <= ABS_TOL, f"{label} mismatch: {a} != {e}")


def _compare_common_eval(
    actual: Mapping[str, Any], expected: Mapping[str, Any], *, label: str
) -> None:
    for field in ("loss", "bits_per_byte"):
        _assert_close_scalar(actual.get(field), expected.get(field), f"{label}.{field}")
    _require(
        int(actual.get("predicted_byte_tokens", -1))
        == int(expected.get("predicted_byte_tokens", -2)),
        f"{label}.predicted_byte_tokens mismatch",
    )
    actual_by = actual.get("by_stratum")
    expected_by = expected.get("by_stratum")
    _require(isinstance(actual_by, Mapping), f"{label}.by_stratum missing")
    _require(isinstance(expected_by, Mapping), f"{label}.expected by_stratum missing")
    for stratum in ("uk", "en", "code"):
        a = actual_by.get(stratum)
        e = expected_by.get(stratum)
        _require(isinstance(a, Mapping), f"{label}.{stratum} missing")
        _require(isinstance(e, Mapping), f"{label}.expected {stratum} missing")
        for field in ("loss", "bits_per_byte"):
            _assert_close_scalar(
                a.get(field), e.get(field), f"{label}.{stratum}.{field}"
            )
        _require(
            int(a.get("predicted_byte_tokens", -1))
            == int(e.get("predicted_byte_tokens", -2)),
            f"{label}.{stratum}.predicted_byte_tokens mismatch",
        )
    _require(actual.get("non_mutation_passed") is True, f"{label} mutated model state")
    _require(
        actual.get("model_state_sha256_before")
        == actual.get("model_state_sha256_after"),
        f"{label} state hash changed",
    )


def _model_state_sha256(model: TwelveSixDecoder) -> str:
    """Hash model state without delegating to the producer evaluator."""

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii") + b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _independent_eval_batch(
    model: TwelveSixDecoder, examples: list[Any]
) -> tuple[float, int]:
    """Compute one validation batch directly from logits and masked CE targets."""

    ids = torch.tensor([example.input_ids for example in examples], dtype=torch.long)
    labels = torch.tensor([example.labels for example in examples], dtype=torch.long)
    logits = model(ids).logits[:, :-1, :].contiguous()
    targets = labels[:, 1:].contiguous()
    tokens = int(targets.ne(-100).sum().item())
    nll = F.cross_entropy(
        logits.reshape(-1, model.spec.vocab_size),
        targets.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    return float(nll.item()), tokens


def _independent_common_eval(
    model: TwelveSixDecoder,
    corpus: Path,
    manifest: dict[str, Any],
    tokenizer: ByteTokenizer,
) -> dict[str, Any]:
    """Recompute the common held-out metric without calling producer ``_evaluate``.

    The packing contract is shared by design, but metric accumulation, masked
    cross-entropy, state non-mutation, and aggregation are implemented here so
    a systematic bug in the producer evaluator cannot simply self-confirm.
    """

    before = _model_state_sha256(model)
    training = model.training
    total_nll = 0.0
    total_tokens = 0
    by_stratum: dict[str, Any] = {}
    model.eval()
    try:
        with torch.no_grad():
            for stratum in ("uk", "en", "code"):
                nll_sum = 0.0
                tokens = 0
                pending: list[Any] = []
                for example in m100._packed(
                    corpus, manifest, tokenizer, "validation", stratum
                ):
                    pending.append(example)
                    if len(pending) == 32:
                        nll, count = _independent_eval_batch(model, pending)
                        nll_sum += nll
                        tokens += count
                        pending = []
                if pending:
                    nll, count = _independent_eval_batch(model, pending)
                    nll_sum += nll
                    tokens += count
                _require(tokens > 0, f"no held-out target bytes for {stratum}")
                loss = nll_sum / tokens
                by_stratum[stratum] = {
                    "loss": loss,
                    "bits_per_byte": nll_sum / math.log(2.0) / tokens,
                    "predicted_byte_tokens": tokens,
                }
                total_nll += nll_sum
                total_tokens += tokens
    finally:
        model.train(training)
    after = _model_state_sha256(model)
    _require(after == before, "independent common evaluation mutated model state")
    loss = total_nll / total_tokens
    return {
        "loss": loss,
        "bits_per_byte": total_nll / math.log(2.0) / total_tokens,
        "predicted_byte_tokens": total_tokens,
        "by_stratum": by_stratum,
        "model_state_sha256_before": before,
        "model_state_sha256_after": after,
        "non_mutation_passed": True,
    }


def _ladder_evaluation_identity(
    tokenizer: ByteTokenizer, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    identity = tokenizer.identity
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
        "packing": {
            "version": m100.PACKING_VERSION,
            "sequence_length": m100.SEQ,
            "cross_document": False,
        },
    }
    value["identity_sha256"] = hash_json(value)
    return value


def _checkpoint_identity_checks(
    checked: Mapping[str, Any],
    *,
    run_manifest: Mapping[str, Any],
    tokenizer: ByteTokenizer,
    role: str,
) -> dict[str, Any]:
    identity = checked.get("identity")
    _require(isinstance(identity, Mapping), f"{role} checkpoint identity missing")
    _require(identity.get("git_sha") == PRODUCER_SHA, f"{role} producer SHA mismatch")
    _require(
        identity.get("model_spec_hash") == EXPECTED_MODEL_SPEC_SHA256,
        f"{role} ModelSpec mismatch",
    )
    _require(
        int(identity.get("parameter_count", -1)) == EXPECTED_PARAMETER_COUNT,
        f"{role} parameter count mismatch",
    )
    _require(
        identity.get("tokenizer_hash") == tokenizer.identity.config_sha256,
        f"{role} tokenizer config mismatch",
    )
    _require(
        identity.get("tokenizer_vocab_hash") == tokenizer.identity.vocab_sha256,
        f"{role} tokenizer vocabulary mismatch",
    )
    _require(
        identity.get("dataset_manifest_hash") == EXPECTED_CORPUS_ID,
        f"{role} corpus identity mismatch",
    )
    _require(
        identity.get("run_manifest_hash") == run_manifest.get("identity_sha256"),
        f"{role} run-manifest identity mismatch",
    )
    _require(int(identity.get("step", -1)) > 0, f"{role} checkpoint is not learned")
    _require(
        int(identity.get("tokens_seen", -1)) > 0,
        f"{role} tokens_seen is not positive",
    )
    return dict(identity)


def _logits_snapshot(checkpoint: Path) -> dict[str, Any]:
    backend = load_first_party_backend(checkpoint)
    outputs: dict[str, Any] = {}
    for name, prompt in PROMPTS.items():
        input_ids = list(backend.encode(prompt))
        logits = list(backend.next_token_logits(input_ids))
        _require(len(logits) == 256, f"{name} logits vocabulary mismatch")
        _require(
            all(math.isfinite(float(value)) for value in logits),
            f"{name} logits are non-finite",
        )
        packed = b"".join(struct.pack("<f", float(value)) for value in logits)
        ranked = sorted(
            range(len(logits)), key=lambda idx: (-float(logits[idx]), idx)
        )[:8]
        outputs[name] = {
            "prompt": prompt,
            "input_ids": input_ids,
            "logits_float32_sha256": hashlib.sha256(packed).hexdigest(),
            "argmax_token_id": ranked[0],
            "top8_token_ids": ranked,
        }
    return {"backend_diagnostics": backend.diagnostics(), "outputs": outputs}


def _generation_snapshot(checkpoint: Path) -> dict[str, Any]:
    backend = load_first_party_backend(checkpoint)
    config = GenerationConfig(max_new_tokens=64, sample=False)
    outputs: dict[str, Any] = {}
    for name, prompt in PROMPTS.items():
        result = generate(backend, prompt, config)
        outputs[name] = {
            "prompt": prompt,
            "generated_token_ids": list(result.generated_token_ids),
            "text": result.text,
            "stop_reason": result.stop_reason,
        }
    return {"decoding": "greedy", "outputs": outputs}


def _reconstruct_random_init(run_manifest: Mapping[str, Any]) -> TwelveSixDecoder:
    spec_raw = run_manifest.get("model_spec")
    init_raw = run_manifest.get("init_spec")
    trainer_raw = run_manifest.get("trainer_config")
    _require(isinstance(spec_raw, dict), "run manifest model_spec missing")
    _require(isinstance(init_raw, dict), "run manifest init_spec missing")
    _require(isinstance(trainer_raw, Mapping), "run manifest trainer_config missing")
    spec = ModelSpec.from_dict(spec_raw)
    init = InitSpec.from_dict(init_raw)
    _require(
        spec.identity_sha256() == EXPECTED_MODEL_SPEC_SHA256,
        "random-init ModelSpec drift",
    )
    _require(
        spec.parameter_count() == EXPECTED_PARAMETER_COUNT,
        "random-init parameter drift",
    )
    seed = trainer_raw.get("seed")
    _require(
        isinstance(seed, int) and not isinstance(seed, bool),
        "training seed missing",
    )
    torch.manual_seed(seed)
    model = TwelveSixDecoder(spec, init)
    _require(
        sum(parameter.numel() for parameter in model.parameters())
        == EXPECTED_PARAMETER_COUNT,
        "random-init instantiated parameter count mismatch",
    )
    return model


def _current_head(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Verify218Error(f"cannot resolve verifier HEAD: {exc}") from exc


def verify(
    *,
    repo: Path,
    artifact_root: Path,
    verifier_head_sha: str,
    output_path: Path,
) -> dict[str, Any]:
    repo = repo.resolve()
    artifact_root = artifact_root.resolve()
    _require(_current_head(repo) == verifier_head_sha, "verifier exact-head mismatch")
    _require(verifier_head_sha != PRODUCER_SHA, "verifier must not be producer commit")

    evidence_root = artifact_root / "scale141-evidence"
    summary = _read_json(evidence_root / "learn217-terminal-summary.json")
    report = _read_json(evidence_root / "report.json")
    fresh = _read_json(evidence_root / "fresh-verification.json")
    phase1 = _read_json(evidence_root / "phase1.json")
    run_manifest = _read_json(evidence_root / "run-manifest.json")
    retained_index = _read_json(evidence_root / "retained" / "index.json")

    report_hash = _verify_self_hash(report, "report_sha256", "report")
    fresh_hash = _verify_self_hash(fresh, "identity_sha256", "fresh verification")
    phase1_hash = _verify_self_hash(phase1, "identity_sha256", "phase1")
    run_hash = _verify_self_hash(run_manifest, "identity_sha256", "run manifest")

    _require(
        summary.get("status") == "TERMINAL_LEARNED_10M_PASS",
        "producer status mismatch",
    )
    for label, value in (
        ("summary source SHA", summary.get("source_sha")),
        ("report source SHA", report.get("source_sha")),
        ("fresh source SHA", fresh.get("source_sha")),
        ("phase1 source SHA", phase1.get("source_sha")),
        ("run source SHA", run_manifest.get("source_sha")),
    ):
        _require(value == PRODUCER_SHA, f"{label} mismatch")

    _require(
        int(summary.get("parameter_count", -1)) == EXPECTED_PARAMETER_COUNT,
        "summary parameter count mismatch",
    )
    _require(
        summary.get("model_spec_sha256") == EXPECTED_MODEL_SPEC_SHA256,
        "summary ModelSpec mismatch",
    )
    report_model = report.get("model", {})
    _require(
        report_model.get("model_spec_sha256") == EXPECTED_MODEL_SPEC_SHA256,
        "report ModelSpec mismatch",
    )
    _require(
        int(report_model.get("parameter_count", -1)) == EXPECTED_PARAMETER_COUNT,
        "report parameter count mismatch",
    )
    _require(
        report.get("corpus", {}).get("identity_sha256") == EXPECTED_CORPUS_ID,
        "report corpus identity mismatch",
    )
    _require(
        fresh.get("corpus_identity_sha256") == EXPECTED_CORPUS_ID,
        "fresh corpus identity mismatch",
    )
    _require(
        fresh.get("fresh_verification", {}).get("status") == "PASS",
        "producer fresh verification was not PASS",
    )
    producer_ladder = fresh.get("ladder_common_evaluation", {})
    _require(
        producer_ladder.get("identity", {}).get("identity_sha256")
        == EXPECTED_LADDER_EVAL_ID,
        "producer common evaluation identity mismatch",
    )
    _require(
        run_manifest.get("foreign_pretrained_weights") is False,
        "foreign/pretrained weights boundary violated",
    )
    _require(
        run_manifest.get("instruction_tuning") is False,
        "instruction tuning entered Base",
    )
    _require(run_manifest.get("paid_compute") is False, "producer claims paid compute")
    _require(
        summary.get("paid_compute") is False,
        "summary paid-compute boundary mismatch",
    )
    _require(
        summary.get("foreign_weights") is False,
        "summary foreign-weight boundary mismatch",
    )
    _require(
        summary.get("sft_rlhf_dpo") is False,
        "summary post-training boundary mismatch",
    )

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    scratch = output_path.parent / "verify218-corpus"
    manifest = m100._build_corpus(repo, scratch)
    _require(
        manifest["corpus_identity_sha256"] == EXPECTED_CORPUS_ID,
        "rebuilt corpus drift",
    )
    tokenizer = ByteTokenizer()
    _require(
        tokenizer.identity.version == EXPECTED_TOKENIZER_VERSION,
        "tokenizer version drift",
    )
    ladder_identity = _ladder_evaluation_identity(tokenizer, manifest)
    _require(
        ladder_identity["identity_sha256"] == EXPECTED_LADDER_EVAL_ID,
        "independently reconstructed ladder evaluation identity mismatch",
    )

    all_scheduled = producer_ladder.get("all_scheduled")
    _require(
        isinstance(all_scheduled, Mapping),
        "producer common scheduled evidence missing",
    )
    initial_expected = all_scheduled.get("0")
    _require(
        isinstance(initial_expected, Mapping),
        "producer random-init common evidence missing",
    )
    initial_expected_eval = initial_expected.get("evaluation")
    _require(
        isinstance(initial_expected_eval, Mapping),
        "producer random-init common evaluation missing",
    )
    random_model = _reconstruct_random_init(run_manifest)
    random_eval = _independent_common_eval(
        random_model, scratch / "corpus-a", manifest, tokenizer
    )
    _compare_common_eval(
        random_eval,
        initial_expected_eval,
        label="random_init_common_eval",
    )

    role_results: dict[str, Any] = {}
    for role in ("best", "final"):
        checkpoint = evidence_root / "retained" / role
        before_tree = _tree_sha256(checkpoint)
        checked = verify_checkpoint(checkpoint)
        identity = _checkpoint_identity_checks(
            checked,
            run_manifest=run_manifest,
            tokenizer=tokenizer,
            role=role,
        )

        retained_role = retained_index.get("roles", {}).get(role)
        _require(isinstance(retained_role, Mapping), f"retained {role} index missing")
        _require(
            retained_role.get("checkpoint_id") == checked.get("checkpoint_id"),
            f"retained {role} checkpoint ID mismatch",
        )

        producer_role = fresh.get("evidence", {}).get(role)
        _require(
            isinstance(producer_role, Mapping),
            f"producer fresh {role} evidence missing",
        )
        _require(
            producer_role.get("checkpoint_id") == checked.get("checkpoint_id"),
            f"producer {role} checkpoint ID mismatch",
        )

        backend = load_first_party_backend(checkpoint)
        common_eval = _independent_common_eval(
            backend.model,
            scratch / "corpus-a",
            manifest,
            tokenizer,
        )
        producer_common = producer_role.get("ladder_common_evaluation")
        _require(
            isinstance(producer_common, Mapping),
            f"producer {role} common evaluation missing",
        )
        _compare_common_eval(
            common_eval,
            producer_common,
            label=f"{role}_common_eval",
        )

        logits_a = _logits_snapshot(checkpoint)
        logits_b = _logits_snapshot(checkpoint)
        _require(logits_a == logits_b, f"{role} logits are not reproducible")
        producer_logits = producer_role.get("first_party_logits")
        _require(
            isinstance(producer_logits, Mapping) and logits_a == producer_logits,
            f"{role} logits fingerprint mismatch",
        )

        generation = _generation_snapshot(checkpoint)
        producer_generation = producer_role.get("generation")
        _require(
            isinstance(producer_generation, Mapping)
            and generation == producer_generation,
            f"{role} greedy generation mismatch",
        )

        after_tree = _tree_sha256(checkpoint)
        _require(
            before_tree == after_tree,
            f"{role} checkpoint tree mutated during verification",
        )
        role_results[role] = {
            "checkpoint_id": checked["checkpoint_id"],
            "identity": identity,
            "checkpoint_tree_sha256_before": before_tree,
            "checkpoint_tree_sha256_after": after_tree,
            "common_evaluation": common_eval,
            "first_party_logits": logits_a,
            "greedy_generation": generation,
        }

    best_bpb = float(role_results["best"]["common_evaluation"]["bits_per_byte"])
    final_bpb = float(role_results["final"]["common_evaluation"]["bits_per_byte"])
    initial_bpb = float(random_eval["bits_per_byte"])
    _require(
        best_bpb < initial_bpb,
        "learned best did not improve over reconstructed random init",
    )
    _require(
        final_bpb < initial_bpb,
        "learned final did not improve over reconstructed random init",
    )

    recovery_results: dict[str, Any] = {}
    for label, directory, index_key in (
        ("phase1", "recovery-phase1", "phase1"),
        ("current", "recovery-current", "current"),
    ):
        checkpoint = evidence_root / "retained" / directory
        checked = verify_checkpoint(checkpoint)
        identity = _checkpoint_identity_checks(
            checked,
            run_manifest=run_manifest,
            tokenizer=tokenizer,
            role=f"recovery_{label}",
        )
        expected = retained_index.get("recovery", {}).get(index_key)
        _require(isinstance(expected, Mapping), f"recovery {label} index missing")
        _require(
            expected.get("checkpoint_id") == checked.get("checkpoint_id"),
            f"recovery {label} checkpoint ID mismatch",
        )
        _require(
            int(expected.get("optimizer_step", -1)) == int(identity["step"]),
            f"recovery {label} optimizer step mismatch",
        )
        _require(
            int(expected.get("tokens_seen", -1)) == int(identity["tokens_seen"]),
            f"recovery {label} tokens_seen mismatch",
        )
        recovery_results[label] = {
            "checkpoint_id": checked["checkpoint_id"],
            "step": int(identity["step"]),
            "tokens_seen": int(identity["tokens_seen"]),
        }
    _require(
        recovery_results["current"]["step"]
        >= recovery_results["phase1"]["step"],
        "current recovery step predates phase1 recovery",
    )
    _require(
        recovery_results["current"]["tokens_seen"]
        >= recovery_results["phase1"]["tokens_seen"],
        "current recovery tokens predate phase1 recovery",
    )

    result: dict[str, Any] = {
        "schema": SCHEMA,
        "worker": WORKER,
        "state": STATE,
        "verifier_head_sha": verifier_head_sha,
        "producer": {
            "git_sha": PRODUCER_SHA,
            "artifact_id": PRODUCER_ARTIFACT_ID,
            "artifact_zip_sha256": PRODUCER_ARTIFACT_ZIP_SHA256,
            "report_sha256": report_hash,
            "fresh_verification_sha256": fresh_hash,
            "phase1_identity_sha256": phase1_hash,
            "run_manifest_identity_sha256": run_hash,
        },
        "model": {
            "parameter_count": EXPECTED_PARAMETER_COUNT,
            "model_spec_sha256": EXPECTED_MODEL_SPEC_SHA256,
            "tokenizer_version": tokenizer.identity.version,
            "tokenizer_config_sha256": tokenizer.identity.config_sha256,
            "tokenizer_vocab_sha256": tokenizer.identity.vocab_sha256,
        },
        "data_and_eval": {
            "corpus_identity_sha256": manifest["corpus_identity_sha256"],
            "ladder_common_evaluation_identity_sha256": ladder_identity[
                "identity_sha256"
            ],
            "random_init_common_evaluation": random_eval,
            "best_common_evaluation": role_results["best"]["common_evaluation"],
            "final_common_evaluation": role_results["final"]["common_evaluation"],
            "best_improved_over_reconstructed_random_init": best_bpb < initial_bpb,
            "final_improved_over_reconstructed_random_init": final_bpb < initial_bpb,
            "metric_implementation": "verify218_independent_masked_cross_entropy_v1",
            "producer_evaluator_reused": False,
        },
        "checkpoints": role_results,
        "recovery": recovery_results,
        "boundaries": {
            "training_executed": False,
            "optimizer_updates": 0,
            "foreign_pretrained_weights": False,
            "paid_compute": False,
            "instruction_or_behavioral_alignment": False,
            "external_llm_used": False,
            "evaluation_mutated_model": False,
            "claim_scope": (
                "Independent admission evidence for the exact LEARN-217 learned-10M "
                "artifact on DATA-25/common ladder evaluation only; not a production, "
                "representativeness, instruction-following, or broad capability claim."
            ),
            "data_truth_boundary": (
                "DATA-25 is project-authored and is not claimed to be an external-real "
                "or representative production pretraining corpus."
            ),
        },
    }
    result["identity_sha256"] = hash_json(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--verifier-head-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(
        repo=args.repo_root,
        artifact_root=args.artifact_root,
        verifier_head_sha=args.verifier_head_sha,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "worker": result["worker"],
                "state": result["state"],
                "identity_sha256": result["identity_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
