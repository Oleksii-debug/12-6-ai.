from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from twelve_six.checkpoint import (
    CheckpointIdentity,
    load_verified_checkpoint,
    prepare_checkpoint_load,
    save_checkpoint,
)
from twelve_six.cloze import conditional_log_likelihood
from twelve_six.code_diagnostic import load_suite as load_code_suite
from twelve_six.code_diagnostic import score_probe
from twelve_six.en_raw_diagnostic import load_suite as load_en_suite
from twelve_six.evaluation_ua_v1 import generate_items as generate_ua_items
from twelve_six.evaluation_ua_v1 import score_completion as score_ua_completion
from twelve_six.model import ModelSpec, TwelveSixDecoder
from twelve_six.tokenization import BYTE_TOKENIZER_HASH, BYTE_VOCAB_HASH, ByteTokenizer

M150_PRODUCER_SHA = "5838cd16869dcfcf762368d8673eddf52d51b7e3"
LEARN191_PRODUCER_SHA = "a75920cef8bde37a8c590e34095be83c97b75f1d"
DATA25_IDENTITY = "422f545da93526ceba2c0ff738e0b9deac65f35bfb31d87d2aab068cda091fc8"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _find_checkpoint(root: Path, candidates: tuple[str, ...]) -> Path:
    for suffix in candidates:
        direct = root / suffix
        if (direct / "manifest.json").is_file():
            return direct
        matches = sorted(root.glob(f"**/{suffix}/manifest.json"))
        if matches:
            return matches[0].parent
    raise FileNotFoundError(f"no D05 checkpoint matching {candidates!r} below {root}")


def _load_model(checkpoint: Path, *, expected_sha: str) -> tuple[TwelveSixDecoder, dict[str, Any]]:
    verified = prepare_checkpoint_load(checkpoint)
    manifest = verified.manifest
    identity = manifest["identity"]
    if identity["git_sha"] != expected_sha:
        raise RuntimeError(
            f"producer SHA mismatch for {checkpoint}: {identity['git_sha']} != {expected_sha}"
        )
    if identity["tokenizer_hash"] != BYTE_TOKENIZER_HASH:
        raise RuntimeError("checkpoint tokenizer identity is not the canonical byte tokenizer")
    if identity["tokenizer_vocab_hash"] != BYTE_VOCAB_HASH:
        raise RuntimeError("checkpoint byte vocabulary identity drift")
    if identity["dataset_manifest_hash"] != DATA25_IDENTITY:
        raise RuntimeError(
            "checkpoint DATA-25 identity mismatch: "
            f"{identity['dataset_manifest_hash']} != {DATA25_IDENTITY}"
        )
    spec = ModelSpec.from_dict(dict(identity["model_spec"]))
    model = TwelveSixDecoder(spec)
    load_verified_checkpoint(
        verified,
        model=model,
        restore_rng=False,
        expected_git_sha=expected_sha,
        expected_tokenizer_hash=BYTE_TOKENIZER_HASH,
        expected_tokenizer_vocab_hash=BYTE_VOCAB_HASH,
        expected_dataset_manifest_hash=DATA25_IDENTITY,
    )
    if spec.parameter_count() != int(identity["parameter_count"]):
        raise RuntimeError("checkpoint parameter count disagrees with ModelSpec")
    return model, manifest


def _tiny_d05_roundtrip() -> dict[str, Any]:
    spec = ModelSpec(
        schema_version=1,
        vocab_size=256,
        max_seq_len=32,
        d_model=16,
        n_layers=1,
        n_heads=2,
        n_kv_heads=2,
        head_dim=8,
        d_ff=32,
        rope_rotary_dim=8,
    )
    model = TwelveSixDecoder(spec)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ids = torch.tensor([[65, 66, 67, 68, 69, 70]], dtype=torch.long)
    logits = model(ids).logits
    loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, spec.vocab_size), ids[:, 1:].reshape(-1))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    before_hash = _state_sha256(model)
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    identity = CheckpointIdentity(
        git_sha=git_sha,
        model_spec=spec.to_dict(),
        parameter_count=spec.parameter_count(),
        tokenizer_hash=BYTE_TOKENIZER_HASH,
        tokenizer_vocab_hash=BYTE_VOCAB_HASH,
        dataset_manifest_hash=_sha256_bytes(b"integrate222-d05-smoke-dataset"),
        run_manifest_hash=_sha256_bytes(b"integrate222-d05-smoke-run"),
        training_config={"purpose": "integrate222_d05_roundtrip"},
        seed=1337,
        precision="fp32",
        step=1,
        tokens_seen=5,
        optimizer={"name": "AdamW", "lr": 1e-3},
        scheduler=None,
    )
    with tempfile.TemporaryDirectory(prefix="integrate222-d05-") as tmp:
        checkpoint = Path(tmp) / "checkpoint"
        saved = save_checkpoint(
            checkpoint,
            model=model,
            identity=identity,
            optimizer=optimizer,
            trainer_state={"optimized_tokens": 5, "step": 1},
        )
        verified = prepare_checkpoint_load(checkpoint)
        reloaded = TwelveSixDecoder(spec)
        reloaded_optimizer = torch.optim.AdamW(reloaded.parameters(), lr=1e-3)
        result = load_verified_checkpoint(
            verified,
            model=reloaded,
            optimizer=reloaded_optimizer,
            restore_rng=True,
            expected_git_sha=git_sha,
            expected_tokenizer_hash=BYTE_TOKENIZER_HASH,
            expected_tokenizer_vocab_hash=BYTE_VOCAB_HASH,
        )
        if _state_sha256(reloaded) != before_hash:
            raise RuntimeError("D05 roundtrip changed model bytes")
        if result.trainer_state.get("optimized_tokens") != 5:
            raise RuntimeError("D05 roundtrip lost optimized-token state")
        if not reloaded_optimizer.state:
            raise RuntimeError("D05 roundtrip lost optimizer state")
        return {
            "status": "PASS",
            "checkpoint_id": saved["checkpoint_id"],
            "model_state_sha256": before_hash,
            "optimizer_state_entries": len(reloaded_optimizer.state),
            "optimized_tokens": 5,
        }


def _evaluator_smoke(model: TwelveSixDecoder, repo_root: Path) -> dict[str, Any]:
    tokenizer = ByteTokenizer()
    before_hash = _state_sha256(model)
    before_mode = model.training
    model.eval()
    try:
        ua_item = generate_ua_items()[0]
        ua_preferred = score_ua_completion(model, ua_item["context"], ua_item["preferred"])
        ua_contrast = score_ua_completion(model, ua_item["context"], ua_item["contrast"])

        en_item = load_en_suite(repo_root)[0]
        en_preferred = conditional_log_likelihood(
            model, tokenizer, en_item["context"], en_item["preferred"]
        )
        en_contrast = conditional_log_likelihood(
            model, tokenizer, en_item["context"], en_item["dispreferred"]
        )

        code_probe = load_code_suite(repo_root / "eval/reserved/code_diag_v1/probes.jsonl")[0]
        code_result = score_probe(model, tokenizer, code_probe)
    finally:
        model.train(before_mode)
    after_hash = _state_sha256(model)
    if before_hash != after_hash:
        raise RuntimeError("UA/EN/code evaluator smoke mutated model state")
    numeric = [
        float(ua_preferred["conditional_bpb"]),
        float(ua_contrast["conditional_bpb"]),
        float(en_preferred.log_likelihood),
        float(en_contrast.log_likelihood),
        float(code_result.raw_log_likelihood_margin_nats),
    ]
    if not all(math.isfinite(value) for value in numeric):
        raise RuntimeError("UA/EN/code evaluator smoke produced non-finite values")
    return {
        "status": "PASS",
        "model_state_sha256": before_hash,
        "ua": {
            "item_id": ua_item["item_id"],
            "preferred_bpb": ua_preferred["conditional_bpb"],
            "contrast_bpb": ua_contrast["conditional_bpb"],
        },
        "en": {
            "item_id": en_item["id"],
            "preferred_log_likelihood": en_preferred.log_likelihood,
            "contrast_log_likelihood": en_contrast.log_likelihood,
        },
        "code": {
            "probe_id": code_probe.id,
            "raw_margin_nats": code_result.raw_log_likelihood_margin_nats,
            "byte_normalized_margin_nats_per_byte": code_result.byte_normalized_margin_nats_per_byte,
        },
    }


def _m150_100k(m150_root: Path, repo_root: Path) -> dict[str, Any]:
    checkpoint = _find_checkpoint(
        m150_root,
        (
            "retained/100k/best",
            "retained/100k/final",
        ),
    )
    model, manifest = _load_model(checkpoint, expected_sha=M150_PRODUCER_SHA)
    if int(manifest["identity"]["parameter_count"]) != 95_568:
        raise RuntimeError("M150 100K checkpoint parameter count is not 95,568")
    evaluator = _evaluator_smoke(model, repo_root)
    tokenizer = ByteTokenizer()
    text = "Привіт. Hello. def f(x): return x + 1"
    token_ids = tokenizer.encode(text)[: model.spec.max_seq_len]
    if len(token_ids) < 2:
        raise RuntimeError("tiny continuation sample is too short")
    ids = torch.tensor([token_ids], dtype=torch.long)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    before = next(model.parameters()).detach().clone()
    optimizer.zero_grad(set_to_none=True)
    logits = model(ids).logits
    loss = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, model.spec.vocab_size),
        ids[:, 1:].reshape(-1),
    )
    if not torch.isfinite(loss):
        raise RuntimeError("M150 100K continuation loss is non-finite")
    loss.backward()
    optimizer.step()
    if torch.equal(before, next(model.parameters()).detach()):
        raise RuntimeError("M150 100K continuation produced no parameter update")
    return {
        "status": "PASS",
        "checkpoint": str(checkpoint),
        "checkpoint_id": manifest["checkpoint_id"],
        "producer_sha": manifest["identity"]["git_sha"],
        "parameter_count": manifest["identity"]["parameter_count"],
        "tokens_seen": manifest["identity"]["tokens_seen"],
        "tiny_continuation_loss": float(loss.item()),
        "tiny_continuation_tokens": len(token_ids) - 1,
        "evaluator_smoke": evaluator,
    }


def _learn191_3m(learn191_root: Path) -> dict[str, Any]:
    checkpoint = _find_checkpoint(
        learn191_root,
        (
            "learn191-evidence/3m/checkpoint-t131292",
            "3m/checkpoint-t131292",
            "checkpoint-t131292",
        ),
    )
    model, manifest = _load_model(checkpoint, expected_sha=LEARN191_PRODUCER_SHA)
    if int(manifest["identity"]["parameter_count"]) != 3_213_120:
        raise RuntimeError("LEARN-191 checkpoint parameter count is not 3,213,120")
    tokenizer = ByteTokenizer()
    ids = tokenizer.encode("Привіт. Hello. def f(x): return x")[: model.spec.max_seq_len]
    tensor = torch.tensor([ids], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        logits = model(tensor).logits
    if not torch.isfinite(logits).all():
        raise RuntimeError("LEARN-191 3M produced non-finite first-party logits")
    fingerprint = _sha256_bytes(logits.detach().cpu().contiguous().numpy().tobytes())
    return {
        "status": "PASS",
        "checkpoint": str(checkpoint),
        "checkpoint_id": manifest["checkpoint_id"],
        "producer_sha": manifest["identity"]["git_sha"],
        "parameter_count": manifest["identity"]["parameter_count"],
        "tokens_seen": manifest["identity"]["tokens_seen"],
        "logits_shape": list(logits.shape),
        "logits_sha256": fingerprint,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m150-root", type=Path, required=True)
    parser.add_argument("--learn191-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = args.output
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    report: dict[str, Any] = {
        "schema": "12-6.integrate222-composed-head-proof.v1",
        "head_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "execution_mode": "LOCAL_FREE_GITHUB_HOSTED_CPU_ONLY",
        "d05_roundtrip": _tiny_d05_roundtrip(),
        "m150_100k": _m150_100k(args.m150_root, repo_root),
        "learn191_3m": _learn191_3m(args.learn191_root),
        "learned_10m": {
            "status": "NOT_RUN_NO_INDEPENDENT_TERMINAL_10M",
            "reason": "VERIFY-218 had no published terminal independently verified learned-10M authority at composition time.",
        },
    }
    report["status"] = "PASS"
    path = output_dir / "proof.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
