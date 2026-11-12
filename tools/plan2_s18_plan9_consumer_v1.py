"""Bind the accepted Plan 2 release to the existing aligned-target Trainer API."""
from __future__ import annotations

import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch

from tools import plan2_public_domain_books_v1 as books
from tools import plan2_s17_exposure_stream_v1 as exposure
from tools import plan2_s17_frozen_tokenizer_v1 as frozen
from tools import plan2_s17_train_packing_v1 as packing
from tools.inference_runtime import GenerationConfig, ReferenceInference
from twelve_six.model import ModelSpec, TwelveSixDecoder, load_stage_config
from twelve_six.tokenization.migration_compatibility import assess_tokenizer_checkpoint_migration
from twelve_six.training.loss import causal_pair_loss

SCHEMA = "12-6.plan2-s18-plan9-consumer.v1"
BUNDLE_SHA256 = "3c357304e4db661bb74398ba4aa75c61c19b7727bca5817f8ec51296ee8765cd"
BUNDLE_FILE_SHA256 = "d0a225d9821a83336b5d6562feaf5fecfac5844e85a51bdbbc57104ffa9766d4"
PACKING_SHA256 = "db78dfa3e585510364f5f3f998a2d2b25bbf66f78ddc522bd6aa9eccb40aaaf7"
PACKING_FILE_SHA256 = "17e263d07f3cf143a2e76bc4e4ba858bc9fc1181d158fbb27fe465bf5d77042c"
TOKENIZER_SHA256 = "798322c30e9a5396d218fbb696313996dd963827ca7d6c8124fea39f0269bb8e"
SCALES = ("500M", "750M", "1B", "2B", "3B", "4B", "5B")
SPECIALS = {"pad": 256, "bos": 257, "eos": 258, "unk": 259}
VOCAB_SIZE = 32768
FIELDS = ("input_ids", "target_ids", "loss_mask")


class Plan9ConsumerDenied(ValueError):
    """The accepted Plan 2 training input cannot be read or bound."""


def _need(condition: bool, reason: str) -> None:
    if not condition:
        raise Plan9ConsumerDenied(reason)


def _read_manifest(bundle: Path, name: str, file_sha256: str) -> dict[str, Any]:
    raw = exposure._read(bundle / name)
    _need(books.sha(raw) == file_sha256, f"{name} differs from accepted S17 bytes")
    manifest = json.loads(raw)
    _need(type(manifest) is dict and books.canonical(manifest) == raw,
          f"{name} is not the accepted canonical object")
    return manifest


def _check_identity(manifest: dict[str, Any], expected: str) -> bool:
    return (manifest.get("manifest_sha256") == expected and
            books.sha(books.canonical({
                key: value for key, value in manifest.items() if key != "manifest_sha256"
            })) == expected)


def _batch(blocks: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    return {name: torch.tensor([block[name] for block in blocks], dtype=torch.long) for name in FIELDS}


def iter_plan9_batches(
    bundle: Path, packing_manifest: dict[str, Any], *, batch_size: int = 16,
) -> Iterator[dict[str, torch.Tensor]]:
    """Yield CPU batches in sealed S13 order for `Trainer.run`."""
    _need(type(batch_size) is int and batch_size > 0, "batch_size must be a positive integer")
    bundle = bundle.absolute()
    exposure._check_directory(bundle / "shards")
    _need(_check_identity(packing_manifest, PACKING_SHA256)
          and packing_manifest["schema_version"] == packing.SCHEMA
          and packing_manifest["split"] == "train"
          and packing_manifest["tokenizer_manifest_sha256"] == TOKENIZER_SHA256,
          "training packing manifest differs from the accepted release")
    block_size = packing_manifest["block_size"]
    shards = packing_manifest["shards"]
    segments = packing_manifest["source_segment_byte_map"]
    expected_names = {f"shard-{index:06d}.json" for index in range(len(shards))}
    _need({file.name for file in (bundle / "shards").iterdir()} == expected_names,
          "training shard membership differs from the accepted release")
    segment_index = offset = total_targets = total_blocks = 0
    previous_token = None
    pending: list[dict[str, Any]] = []
    for index, ref in enumerate(shards):
        path = f"shards/shard-{index:06d}.json"
        _need(ref["path"] == path, "training shard order changed")
        raw = exposure._read(bundle / path)
        _need(len(raw) == ref["byte_count"] and books.sha(raw) == ref["sha256"],
              "training shard bytes differ from the accepted release")
        shard = json.loads(raw)
        blocks = shard["blocks"]
        _need(shard["schema_version"] == packing.SHARD_SCHEMA and shard["shard_index"] == index
              and shard["block_size"] == block_size and len(blocks) == ref["block_count"],
              "training shard schema, width or block count changed")
        shard_targets = 0
        for block in blocks:
            _need(segment_index < len(segments), "training shard contains extra targets")
            segment = segments[segment_index]
            count, active = exposure._validate_block(
                block, segment, offset, previous_token, block_size, VOCAB_SIZE, SPECIALS,
            )
            shard_targets += count
            offset += count
            previous_token = active[-1]
            if offset == segment["target_count"]:
                segment_index += 1
                offset, previous_token = 0, None
        _need(shard_targets == ref["target_count"], "training shard target count changed")
        total_targets += shard_targets
        total_blocks += len(blocks)
        for block in blocks:
            pending.append(block)
            if len(pending) == batch_size:
                yield _batch(pending)
                pending.clear()
    _need(segment_index == len(segments) and offset == 0
          and total_targets == packing_manifest["target_count"]
          and total_blocks == packing_manifest["block_count"],
          "training shard stream is incomplete")
    if pending:
        yield _batch(pending)


def qualify_plan9_consumer(
    root: Path, bundle: Path, bundle_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Bind every configured scale and read a real aligned batch without training."""
    bundle = bundle.absolute()
    exposure._check_directory(bundle)
    accepted = _read_manifest(bundle, "bundle-manifest.json", BUNDLE_FILE_SHA256)
    _need(accepted == bundle_manifest and _check_identity(accepted, BUNDLE_SHA256),
          "Plan9 bundle identity differs from the accepted release")
    packed = _read_manifest(bundle, "packing-manifest.json", PACKING_FILE_SHA256)
    _need(_check_identity(packed, PACKING_SHA256)
          and packed["manifest_sha256"] == accepted["packing_manifest_sha256"]
          and packed["cluster_split_sha256"] == accepted["split_manifest_sha256"]
          and packed["target_count"] == accepted["train_target_count"],
          "Plan9 packing, split or target identity differs from the accepted release")
    tokenizer = accepted["tokenizer_identity"]
    _need(tokenizer["vocab_size"] == VOCAB_SIZE and tokenizer["special_tokens"] == SPECIALS
          and tuple(accepted["tokenizer_scale_profiles"]) == SCALES,
          "Plan9 tokenizer or scale contract changed")
    profiles = {}
    for scale in SCALES:
        profile_path = root / "configs" / "stages" / f"scale_{scale.lower()}.json"
        stage = load_stage_config(profile_path)
        spec = stage.model
        migration = assess_tokenizer_checkpoint_migration(
            target_tokenizer_identity=tokenizer, target_model_spec=spec.to_dict(),
            reuse_checkpoint_weights=False,
        )
        _need(stage.stage == f"SCALE-{scale}" and spec.max_seq_len >= packed["block_size"]
              and migration["status"] == "FRESH_INITIALIZATION_TARGET_TOKENIZER_BOUND"
              and migration["checkpoint_weight_reuse_allowed_by_tokenizer_contract"] is False,
              "Plan3/4 scale profile cannot consume this frozen tokenizer and shard width")
        profiles[stage.stage] = {
            "model_spec_sha256": spec.identity_sha256(),
            "init_spec_sha256": stage.init.identity_sha256(),
            "parameters": spec.parameter_count(),
            "max_seq_len": spec.max_seq_len,
            "vocab_size": spec.vocab_size,
            "migration_decision_sha256": books.sha(books.canonical(migration)),
        }
    batch = next(iter_plan9_batches(bundle, packed, batch_size=2))
    with torch.inference_mode():
        logits = torch.zeros((*batch["input_ids"].shape, VOCAB_SIZE), dtype=torch.float32)
        loss = causal_pair_loss(logits, batch["target_ids"], loss_mask=batch["loss_mask"])
    _need(math.isclose(loss.item(), math.log(VOCAB_SIZE), rel_tol=1e-6),
          "real shard batch does not satisfy the existing aligned-target loss contract")
    encoder, _ = frozen.load_frozen(root)
    _need(encoder.identity.to_dict() == tokenizer, "Plan4 tokenizer differs from the frozen bundle")
    smoke_spec = ModelSpec(
        schema_version=1, vocab_size=VOCAB_SIZE, max_seq_len=32,
        d_model=16, n_layers=1, n_heads=2, n_kv_heads=1, head_dim=8, d_ff=32, rope_rotary_dim=8,
    )
    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        torch.manual_seed(42)
        model = TwelveSixDecoder(smoke_spec)
        model.eval()
        generation = ReferenceInference(model, encoder).generate("Дані", GenerationConfig(max_new_tokens=1))
    _need(generation.status == "COMPLETED" and len(generation.output_token_ids) == 1,
          "frozen tokenizer failed Plan4 reference generation")
    receipt = {
        "schema_version": SCHEMA,
        "bundle_manifest_sha256": BUNDLE_SHA256,
        "packing_manifest_sha256": PACKING_SHA256,
        "tokenizer_manifest_sha256": TOKENIZER_SHA256,
        "split_manifest_sha256": accepted["split_manifest_sha256"],
        "exposure_manifest_sha256": accepted["exposure_manifest_sha256"],
        "ordered_target_count": packed["target_count"],
        "primary_profile": "SCALE-500M",
        "profile_compatibility": profiles,
        "weights_origin": "random_init",
        "consumer_api": "tools.plan2_s18_plan9_consumer_v1.iter_plan9_batches",
        "trainer_api": "twelve_six.training.trainer.Trainer.run",
        "plan4_reference_inference_verified": True,
        "real_shard_batch": {
            "first_shard_sha256": packed["shards"][0]["sha256"],
            "tensor_fields": list(FIELDS),
            "shape": list(batch["input_ids"].shape),
            "unmasked_targets": int(batch["loss_mask"].sum().item()),
            "aligned_target_loss_verified": True,
        },
        "model_training_executed": False,
        "optimizer_effect_authorized": False,
        "paid_compute_authorized": False,
    }
    return {**receipt, "manifest_sha256": books.sha(books.canonical(receipt))}
