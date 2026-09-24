from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import torch

from twelve_six.checkpoint import load_verified_checkpoint, prepare_checkpoint_load
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
LEARN217_PRODUCER_SHA = "c02c8aa38e691521ae2ab6a4ff3ea1d643efd6ef"
DATA25_IDENTITY = "422f545da93526ceba2c0ff738e0b9deac65f35bfb31d87d2aab068cda091fc8"
EXPECTED_PRODUCERS = {
    "learned_1m": M150_PRODUCER_SHA,
    "learned_3m": LEARN191_PRODUCER_SHA,
    "learned_10m": LEARN217_PRODUCER_SHA,
}
EXPECTED_PARAMETERS = {
    "learned_3m": 3_213_120,
    "learned_10m": 10_000_640,
}


def _parse_checkpoint(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or label not in EXPECTED_PRODUCERS or not raw_path:
        raise argparse.ArgumentTypeError(
            "checkpoint must be one of learned_1m=PATH, learned_3m=PATH, learned_10m=PATH"
        )
    return label, Path(raw_path)


def _state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _load(label: str, checkpoint: Path) -> tuple[TwelveSixDecoder, dict[str, Any]]:
    verified = prepare_checkpoint_load(checkpoint)
    manifest = verified.manifest
    identity = manifest["identity"]
    expected_sha = EXPECTED_PRODUCERS[label]
    if identity["git_sha"] != expected_sha:
        raise RuntimeError(
            f"{label}: producer SHA mismatch: {identity['git_sha']} != {expected_sha}"
        )
    if identity["tokenizer_hash"] != BYTE_TOKENIZER_HASH:
        raise RuntimeError(f"{label}: tokenizer identity drift")
    if identity["tokenizer_vocab_hash"] != BYTE_VOCAB_HASH:
        raise RuntimeError(f"{label}: tokenizer vocabulary identity drift")
    if identity["dataset_manifest_hash"] != DATA25_IDENTITY:
        raise RuntimeError(f"{label}: DATA-25 corpus identity drift")
    spec = ModelSpec.from_dict(dict(identity["model_spec"]))
    if spec.parameter_count() != int(identity["parameter_count"]):
        raise RuntimeError(f"{label}: ModelSpec parameter count disagrees with checkpoint")
    expected_parameters = EXPECTED_PARAMETERS.get(label)
    if expected_parameters is not None and spec.parameter_count() != expected_parameters:
        raise RuntimeError(
            f"{label}: parameter count {spec.parameter_count()} != {expected_parameters}"
        )
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
    return model, manifest


def _finite_logits(model: TwelveSixDecoder) -> dict[str, Any]:
    tokenizer = ByteTokenizer()
    ids = tokenizer.encode("Привіт. Hello. def f(x): return x + 1")[: model.spec.max_seq_len]
    if len(ids) < 2:
        raise RuntimeError("finite-logits probe is too short")
    tensor = torch.tensor([ids], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        logits = model(tensor).logits
    if not torch.isfinite(logits).all():
        raise RuntimeError("first-party forward produced non-finite logits")
    digest = hashlib.sha256(logits.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    return {
        "status": "PASS",
        "shape": list(logits.shape),
        "sha256": digest,
        "min": float(logits.min().item()),
        "max": float(logits.max().item()),
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
        float(code_result.byte_normalized_margin_nats_per_byte),
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
            "byte_normalized_margin_nats_per_byte": (
                code_result.byte_normalized_margin_nats_per_byte
            ),
        },
    }


def _verify_one(label: str, checkpoint: Path, repo_root: Path) -> dict[str, Any]:
    model, manifest = _load(label, checkpoint)
    identity = manifest["identity"]
    return {
        "status": "PASS",
        "checkpoint": str(checkpoint),
        "checkpoint_id": manifest["checkpoint_id"],
        "producer_sha": identity["git_sha"],
        "parameter_count": identity["parameter_count"],
        "tokens_seen": identity["tokens_seen"],
        "model_spec": identity["model_spec"],
        "finite_logits": _finite_logits(model),
        "evaluator_smoke": _evaluator_smoke(model, repo_root),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", type=_parse_checkpoint, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoints = dict(args.checkpoint)
    missing = sorted(set(EXPECTED_PRODUCERS) - set(checkpoints))
    if missing:
        raise RuntimeError(f"missing required learned checkpoints: {missing}")

    repo_root = Path(__file__).resolve().parents[1]
    models = {
        label: _verify_one(label, checkpoints[label], repo_root)
        for label in ("learned_1m", "learned_3m", "learned_10m")
    }
    result = {
        "schema": "12-6.integrate282-independent-learned-consumer.v1",
        "worker_id": "INTEGRATE-282-POST217-LEARNED-SPINE",
        "head_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "execution_mode": "LOCAL_FREE_GITHUB_HOSTED_CPU_ONLY",
        "producer_self_check_relabelled_independent": False,
        "independent_consumer_source": "INTEGRATE-282",
        "models": models,
        "status": "PASS",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
