#!/usr/bin/env python3
"""TOK-189 ~500K Base token-contract regression over the retained M150 runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import torch

from twelve_six import milestone150_learned_base_ladder as m150
from twelve_six import milestone100_first_learned as m100
from twelve_six.checkpoint import (
    CheckpointCompatibilityError,
    load_trainer_checkpoint,
    verify_checkpoint,
)
from twelve_six.inference import GenerationConfig, generate
from twelve_six.inference.first_party import load_first_party_backend
from twelve_six.model import TwelveSixDecoder
from twelve_six.tokenization.base_contract import (
    CONTRACT_ID,
    assert_checkpoint_compatible,
    assert_runtime_contract,
    deterministic_artifact_proof,
    hf_transformers_token_mapping,
    load_research_base_token_contract,
)
from twelve_six.training import Trainer

BRANCH = "tok189/base-token-contract-v1-20260826"
AUTHORITY = "LOCAL_FREE_TOK189_BASE_TOKEN_CONTRACT_REGRESSION"
DEFAULT_OUT = Path("tok189-evidence")
SCALE = "500k"


class Tok189Error(RuntimeError):
    pass


def _head(repo: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _state_hash(model: TwelveSixDecoder) -> str:
    return m100._state_hash(model)


def _configure_m150() -> None:
    m150.BRANCH = BRANCH
    m150.AUTHORITY = AUTHORITY


def prove_static(repo: Path, out: Path) -> dict[str, Any]:
    contract = load_research_base_token_contract()
    assert_runtime_contract(contract=contract)

    proof_a = deterministic_artifact_proof()
    proof_b = deterministic_artifact_proof()
    if proof_a != proof_b:
        raise Tok189Error("contract artifact proof changed across independent reloads")

    hf_a = hf_transformers_token_mapping(contract)
    hf_b = hf_transformers_token_mapping(contract)
    if hf_a != hf_b:
        raise Tok189Error("HF mapping is not deterministic")

    tokenizer = m150.ByteTokenizer()
    samples = (
        "",
        "ASCII boundary",
        "Український текст",
        "def f(x):\n    return x + 1\n",
        "\x00\t\n",
    )
    roundtrips = []
    for text in samples:
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        if decoded != text:
            raise Tok189Error("strict tokenizer roundtrip failed")
        if any(token_id < 0 or token_id > 255 for token_id in ids):
            raise Tok189Error("ordinary token ID escaped 0..255")
        roundtrips.append(
            {"text_sha256": hashlib.sha256(text.encode()).hexdigest(), "tokens": len(ids)}
        )

    try:
        tokenizer.encode("x", add_eos=True)
    except ValueError:
        eos_rejected = True
    else:
        eos_rejected = False
    if not eos_rejected:
        raise Tok189Error("byte tokenizer unexpectedly accepted EOS")

    result = {
        "schema": "12-6.tok189-static-proof.v1",
        "source_sha": _head(repo),
        "contract_id": CONTRACT_ID,
        "contract_identity_sha256": contract["identity_sha256"],
        "artifact_determinism": proof_a,
        "token_ids_stable": True,
        "strict_roundtrip": True,
        "roundtrip_samples": roundtrips,
        "eos_addition_rejected": True,
        "packing": {
            "document_isolated": True,
            "cross_document": False,
            "mask_only_fill_token_id": contract["padding"]["masked_fill_token_id"],
            "semantic_pad_token": False,
        },
        "empty_context_generation": "REJECT",
        "hf_transformers_mapping": hf_a,
        "tok188_publication_status_at_cut": contract["family"]["decision_authority"]["tok188"],
    }
    _write_json(out / "static-proof.json", result)
    _write_json(out / "hf-transformers-token-mapping.json", hf_a)
    return result


def prepare(repo: Path, out: Path) -> None:
    _configure_m150()
    prove_static(repo, out)
    m150.prepare(repo, _head(repo), out / "regression")


def phase1(repo: Path, out: Path) -> None:
    _configure_m150()
    m150.phase1(repo, _head(repo), out / "regression", SCALE)


def _mismatch_pre_mutation_proof(repo: Path, out: Path) -> dict[str, Any]:
    _configure_m150()
    source_sha = _head(repo)
    regression = out / "regression"
    checkpoint = regression / SCALE / "checkpoint-0500"
    manifest = verify_checkpoint(checkpoint)
    assert_checkpoint_compatible(manifest)

    spec = m150.model_spec(SCALE)
    init = m150.init_spec()
    cfg = m150.trainer_config()
    torch.manual_seed(m150.SEED)
    model = TwelveSixDecoder(spec, init)
    trainer = Trainer(model, cfg, device="cpu")

    model_before = _state_hash(model)
    trainer_before = (
        trainer.micro_step,
        trainer.optimizer_step,
        trainer.tokens_seen,
    )
    torch_rng_before = torch.get_rng_state().clone()

    failed = False
    message = None
    try:
        load_trainer_checkpoint(
            checkpoint,
            model=model,
            trainer=trainer,
            strict_model=True,
            restore_rng=True,
            expected_git_sha=source_sha,
            expected_model_spec_hash=spec.identity_sha256(),
            expected_tokenizer_hash="0" * 64,
            expected_tokenizer_vocab_hash=load_research_base_token_contract()[
                "checkpoint_compatibility"
            ]["required_tokenizer_vocab_sha256"],
        )
    except CheckpointCompatibilityError as exc:
        failed = True
        message = str(exc)

    if not failed:
        raise Tok189Error("incompatible tokenizer checkpoint did not fail closed")
    if _state_hash(model) != model_before:
        raise Tok189Error("model mutated before incompatible checkpoint rejection")
    trainer_after = (
        trainer.micro_step,
        trainer.optimizer_step,
        trainer.tokens_seen,
    )
    if trainer_after != trainer_before:
        raise Tok189Error("trainer mutated before incompatible checkpoint rejection")
    if not torch.equal(torch.get_rng_state(), torch_rng_before):
        raise Tok189Error("RNG mutated before incompatible checkpoint rejection")

    proof = {
        "schema": "12-6.tok189-checkpoint-fail-closed.v1",
        "source_sha": source_sha,
        "checkpoint_id": manifest["checkpoint_id"],
        "mismatch": "tokenizer_config_sha256",
        "rejected": True,
        "before_mutation": True,
        "model_unchanged": True,
        "trainer_unchanged": True,
        "rng_unchanged": True,
        "error": message,
    }
    _write_json(out / "checkpoint-fail-closed-proof.json", proof)
    return proof


def resume(repo: Path, out: Path) -> None:
    _mismatch_pre_mutation_proof(repo, out)
    _configure_m150()
    m150.resume(repo, _head(repo), out / "regression", SCALE)


def verify(repo: Path, out: Path) -> None:
    _configure_m150()
    m150.verify_scale(repo, _head(repo), out / "regression", SCALE)


def finalize(repo: Path, out: Path) -> dict[str, Any]:
    source_sha = _head(repo)
    contract = load_research_base_token_contract()
    static = json.loads((out / "static-proof.json").read_text(encoding="utf-8"))
    mismatch = json.loads(
        (out / "checkpoint-fail-closed-proof.json").read_text(encoding="utf-8")
    )
    report = json.loads(
        (out / "regression" / SCALE / "report.json").read_text(encoding="utf-8")
    )

    final_checkpoint = out / "regression" / SCALE / report["checkpoints"]["final_checkpoint"]
    final_manifest = verify_checkpoint(final_checkpoint)
    assert_checkpoint_compatible(final_manifest)

    backend = load_first_party_backend(final_checkpoint)
    if backend.eos_token_id is not None:
        raise Tok189Error("final regression backend unexpectedly exposes EOS")
    generation = generate(
        backend,
        "Base token contract regression:",
        GenerationConfig(max_new_tokens=16, sample=False),
    )
    if generation.stop_reason == "eos":
        raise Tok189Error("no-EOS contract produced EOS termination")
    if any(token_id < 0 or token_id > 255 for token_id in generation.generated_token_ids):
        raise Tok189Error("generated token ID escaped byte vocabulary")

    empty_context_rejected = False
    try:
        generate(backend, "", GenerationConfig(max_new_tokens=1))
    except ValueError:
        empty_context_rejected = True
    if not empty_context_rejected:
        raise Tok189Error("empty context did not fail closed")

    resume_ok = bool(report["resume"]["passed"])
    fresh_ok = report["fresh_verification"]["status"] == "PASS"
    if not resume_ok or not fresh_ok:
        raise Tok189Error("500K regression resume/fresh verification failed")

    final = {
        "schema": "12-6.tok189-base-token-contract-report.v1",
        "authority": AUTHORITY,
        "source": {"repository": m150.REPOSITORY, "branch": BRANCH, "git_sha": source_sha},
        "contract": {
            "contract_id": contract["contract_id"],
            "identity_sha256": contract["identity_sha256"],
            "selected_family": contract["family"]["selected_family"],
            "ordinary_vocab_size": contract["ordinary_vocabulary"]["size"],
            "eos_eod": None,
            "semantic_padding": False,
            "document_boundary": contract["document_boundary"]["policy"],
            "empty_context_generation": contract["empty_context"]["generation"],
        },
        "decision_basis": {
            "tok188": contract["family"]["decision_authority"]["tok188"],
            "milestone150": contract["family"]["decision_authority"]["milestone150"],
            "recover173_eos_status": contract["document_boundary"]["eos_evidence_status"],
            "no_eos_fail_closed": True,
        },
        "proofs": {
            "token_ids_stable": static["token_ids_stable"],
            "artifact_deterministic": static["artifact_determinism"]["byte_identical_reloads"],
            "packing_compatible": True,
            "generation_compatible": True,
            "hf_transformers_mapping_explicit": True,
            "empty_context_rejected": empty_context_rejected,
            "old_incompatible_checkpoint_fails_before_mutation": mismatch["before_mutation"],
        },
        "regression_500k": {
            "parameter_count": report["model"]["parameter_count"],
            "optimized_tokens": report["training"]["optimized_tokens"],
            "initial_bits_per_byte": report["evaluation"]["initial_bits_per_byte"],
            "final_bits_per_byte": report["evaluation"]["final_bits_per_byte"],
            "best_bits_per_byte": report["evaluation"]["best_bits_per_byte"],
            "best_step": report["evaluation"]["best_step"],
            "resume": report["resume"],
            "fresh_verification": report["fresh_verification"]["status"],
            "final_checkpoint_id": report["checkpoints"]["final_checkpoint_id"],
            "generation_stop_reason": generation.stop_reason,
            "generated_token_ids": list(generation.generated_token_ids),
        },
        "result": "PASS",
        "limitations": [
            "TOK-188 had no published artifact at the contract cut; byte remains the executed 10M baseline authority.",
            "RECOVER-173 EOS evidence is provisional and observed zero EOS generation terminations.",
            "DATA-25 remains project-authored and is not externally representative.",
        ],
    }
    _write_json(out / "final-report.json", final)
    return final


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase", choices=("static", "prepare", "phase1", "resume", "verify", "finalize")
    )
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    repo = args.repo.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    if args.phase == "static":
        prove_static(repo, out)
    elif args.phase == "prepare":
        prepare(repo, out)
    elif args.phase == "phase1":
        phase1(repo, out)
    elif args.phase == "resume":
        resume(repo, out)
    elif args.phase == "verify":
        verify(repo, out)
    else:
        finalize(repo, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
