"""Train and reopen a scratch English–Ukrainian 12-6 translation pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from tools.plan2_s15_real_32k_bpe_v1 import ByteBPE32k, fit_incremental
from twelve_six.checkpoint import (
    CheckpointIdentity,
    detect_git_sha,
    hash_json,
    load_trainer_checkpoint,
    load_verified_checkpoint,
    prepare_checkpoint_load,
    save_trainer_checkpoint,
)
from twelve_six.checkpoint.recovery_lock import exclusive_recovery_lock
from twelve_six.data.translation_parallel import load_translation_split
from twelve_six.model import (
    BaseInitManifest,
    InitSpec,
    ModelSpec,
    TwelveSixDecoder,
    build_canonical_base,
)
from twelve_six.training import Trainer, TrainerConfig
from twelve_six.training.loss import causal_lm_loss
from twelve_six.translation_evaluation import TranslationExample, evaluate_translation

SCHEMA = "12-6.translation-pilot.v1"
DIRECTIONS = {"en-uk": ("en", "uk"), "uk-en": ("uk", "en")}


def prompt_ids(tokenizer: ByteBPE32k, direction: str, source: str) -> list[int]:
    if direction not in DIRECTIONS or not source.strip():
        raise ValueError("a supported direction and nonempty source are required")
    return tokenizer.encode(f"{direction}\n{source}\n", add_bos=True)


def _fit_tokenizer(rows: Sequence[dict], vocab_size: int, seed: int) -> ByteBPE32k:
    sample = random.Random(seed).sample(list(rows), min(500, len(rows)))
    documents = [
        {"record_id": f"{row['pair_id']}:{lang}", "text": row[lang]}
        for row in sample for lang in ("en", "uk")
    ]
    merges, _ = fit_incremental(documents, vocab_size=vocab_size)
    return ByteBPE32k(merges, required_vocab_size=vocab_size)


def _examples(rows: Sequence[dict]) -> Iterator[TranslationExample]:
    for row in rows:
        for direction, (source_lang, target_lang) in DIRECTIONS.items():
            yield TranslationExample(
                f"{row['pair_id']}:{direction}", direction, row[source_lang], row[target_lang],
            )


def _encode(tokenizer: ByteBPE32k, example: TranslationExample) -> tuple[list[int], int]:
    prompt = prompt_ids(tokenizer, example.direction, example.source)
    return prompt + tokenizer.encode(example.reference, add_eos=True), len(prompt)


def _encoded_pairs(
    rows: Sequence[dict], tokenizer: ByteBPE32k, context: int,
) -> Iterator[tuple[TranslationExample, tuple[list[int], int]]]:
    for example in _examples(rows):
        encoded = _encode(tokenizer, example)
        if len(encoded[0]) <= context:
            yield example, encoded


def collate(examples: Sequence[tuple[list[int], int]], pad_id: int) -> dict[str, torch.Tensor]:
    width = max(len(tokens) for tokens, _ in examples)
    inputs = torch.full((len(examples), width), pad_id, dtype=torch.long)
    labels = torch.full_like(inputs, -100)
    for index, (tokens, prompt_length) in enumerate(examples):
        inputs[index, :len(tokens)] = torch.tensor(tokens)
        labels[index, prompt_length:len(tokens)] = inputs[index, prompt_length:len(tokens)]
    return {"input_ids": inputs, "labels": labels}


def _batches(
    examples: Sequence, batch_size: int, seed: int, start: int, length_window_batches: int = 1,
) -> Iterator[list]:
    per_epoch = _epoch_batches(len(examples), batch_size)
    epoch, first = divmod(start, per_epoch)
    while True:
        order = list(range(len(examples)))
        random.Random(seed + epoch).shuffle(order)
        if length_window_batches > 1:
            window = batch_size * length_window_batches
            for offset in range(0, len(order), window):
                order[offset:offset + window] = sorted(
                    order[offset:offset + window], key=lambda index: len(examples[index][0]),
                )
        batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
        if length_window_batches > 1:
            last = batches.pop() if len(batches[-1]) < batch_size else None
            random.Random(seed ^ (epoch + 1)).shuffle(batches)
            if last is not None:
                batches.append(last)
        for indexes in batches[first:]:
            yield [examples[index] for index in indexes]
        epoch += 1
        first = 0


def _epoch_batches(examples: int, batch_size: int) -> int:
    return (examples + batch_size - 1) // batch_size


@torch.inference_mode()
def _loss(model: TwelveSixDecoder, samples: Sequence, pad_id: int, batch_size: int) -> float:
    model.eval()
    loss_sum, positions = 0.0, 0
    for start in range(0, len(samples), batch_size):
        batch = collate(samples[start:start + batch_size], pad_id)
        count = int(batch["labels"][:, 1:].ne(-100).sum())
        loss = causal_lm_loss(model(batch["input_ids"]).logits, batch["labels"])
        loss_sum += count * loss.item()
        positions += count
    return loss_sum / positions


@torch.inference_mode()
def translate(
    model: TwelveSixDecoder, tokenizer: ByteBPE32k, direction: str, source: str,
    max_new_tokens: int = 40,
) -> str:
    model.eval()
    prompt = prompt_ids(tokenizer, direction, source)
    available = model.spec.max_seq_len - len(prompt)
    if available <= 0:
        raise ValueError("source exceeds pilot context")
    generated = model.generate(
        torch.tensor([prompt]), max_new_tokens=min(max_new_tokens, available), do_sample=False,
    )[0, len(prompt):].tolist()
    return _decode(tokenizer, generated)


def _decode(tokenizer: ByteBPE32k, generated: list[int]) -> str:
    if tokenizer.eos_id in generated:
        generated = generated[:generated.index(tokenizer.eos_id)]
    return tokenizer.decode(generated, errors="replace")


@torch.inference_mode()
def _translation_lookup(
    model: TwelveSixDecoder, tokenizer: ByteBPE32k,
    examples: Sequence[TranslationExample], max_new_tokens: int, batch_size: int,
) -> Callable[[str, str], str]:
    model.eval()
    groups = defaultdict(list)
    seen = set()
    for example in examples:
        key = example.direction, example.source
        if key not in seen:
            seen.add(key)
            prompt = prompt_ids(tokenizer, *key)
            groups[len(prompt)].append((key, prompt))
    translations = {}
    for length, prompts in sorted(groups.items()):
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start:start + batch_size]
            generated = model.generate(
                torch.tensor([prompt for _, prompt in batch]),
                max_new_tokens=min(max_new_tokens, model.spec.max_seq_len - length),
                do_sample=False,
            )[:, length:].tolist()
            for (key, _), ids in zip(batch, generated, strict=True):
                translations[key] = _decode(tokenizer, ids)

    def lookup(direction: str, source: str) -> str:
        return translations[direction, source]

    return lookup


def _source_only_examples(rows: Sequence[dict], tokenizer: ByteBPE32k, context: int):
    return [
        example for example in _examples(rows)
        if len(prompt_ids(tokenizer, example.direction, example.source)) < context
    ]


def _model(manifest: dict) -> TwelveSixDecoder:
    return build_canonical_base(
        ModelSpec.from_dict(manifest["model_spec"]),
        InitSpec.from_dict(manifest["init_spec"]),
        BaseInitManifest.from_dict(manifest["initialization"]),
    )


def _tokenizer(manifest: dict) -> ByteBPE32k:
    tokenizer = ByteBPE32k(
        manifest["merges"], required_vocab_size=manifest["model_spec"]["vocab_size"],
    )
    if tokenizer.identity.to_dict() != manifest["tokenizer"]:
        raise ValueError("recorded tokenizer identity differs from its merge rules")
    return tokenizer


def _identity(manifest: dict, trainer: Trainer) -> CheckpointIdentity:
    spec = ModelSpec.from_dict(manifest["model_spec"])
    return CheckpointIdentity(
        git_sha=manifest["git_sha"], model_spec=spec.to_dict(),
        parameter_count=spec.parameter_count(),
        tokenizer_hash=manifest["tokenizer"]["config_sha256"],
        tokenizer_vocab_hash=manifest["tokenizer"]["vocab_sha256"],
        dataset_manifest_hash=manifest["data_sha256"], run_manifest_hash=hash_json(manifest),
        training_config=manifest["training"], seed=trainer.config.seed,
        precision=trainer.config.precision, step=trainer.optimizer_step,
        tokens_seen=trainer.tokens_seen,
        optimizer={"name": "AdamW", "lr": trainer.config.learning_rate},
        scheduler={"name": trainer.config.scheduler, "warmup_steps": trainer.config.warmup_steps},
    )


def _expected(manifest: dict) -> dict[str, str]:
    return {
        "expected_git_sha": manifest["git_sha"],
        "expected_model_spec_hash": ModelSpec.from_dict(manifest["model_spec"]).identity_sha256(),
        "expected_tokenizer_hash": manifest["tokenizer"]["config_sha256"],
        "expected_tokenizer_vocab_hash": manifest["tokenizer"]["vocab_sha256"],
        "expected_dataset_manifest_hash": manifest["data_sha256"],
        "expected_run_manifest_hash": hash_json(manifest),
    }


def _restore_model(
    root: Path, step: int, manifest: dict, expected_checkpoint_id: str | None = None,
) -> TwelveSixDecoder:
    verified = prepare_checkpoint_load(_checkpoint(root, step))
    saved = verified.manifest
    if saved["identity"]["step"] != step:
        raise ValueError("verified checkpoint step differs from the requested step")
    if expected_checkpoint_id is not None and saved["checkpoint_id"] != expected_checkpoint_id:
        raise ValueError("checkpoint ID differs on readback")
    model = _model(manifest)
    load_verified_checkpoint(verified, model=model, restore_rng=False, **_expected(manifest))
    return model


def _write_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def _checkpoint(run: Path, step: int) -> Path:
    return run / "checkpoints" / f"step-{step:07d}"


def _sample(rows: Sequence[dict], count: int, seed: int) -> list[dict]:
    return random.Random(seed).sample(list(rows), min(count, len(rows)))


def _new_manifest(args: argparse.Namespace, train_rows: Sequence[dict]) -> dict:
    tokenizer_source = (
        json.loads(args.tokenizer_from.read_bytes()) if args.tokenizer_from is not None else None
    )
    tokenizer = (
        _tokenizer(tokenizer_source)
        if tokenizer_source is not None
        else _fit_tokenizer(train_rows, args.vocab_size, args.seed)
    )
    spec = ModelSpec(
        schema_version=1, vocab_size=tokenizer.vocab_size, max_seq_len=args.context,
        d_model=args.width, n_layers=args.layers, n_heads=4, n_kv_heads=2,
        head_dim=args.width // 4, d_ff=((args.width * 8 + 47) // 48) * 16,
        rope_rotary_dim=args.width // 4,
    )
    init = InitSpec()
    base = BaseInitManifest(1, spec.identity_sha256(), init.identity_sha256(), args.seed)
    config = TrainerConfig(
        learning_rate=args.lr, max_steps=args.steps, seed=args.seed,
        scheduler="cosine", warmup_steps=min(40, args.steps // 10),
    )
    head = detect_git_sha(Path(__file__).resolve().parents[1])
    if head is None:
        raise ValueError("translation training requires a Git checkout")
    return {
        "schema": SCHEMA, "git_sha": head, "data_sha256": args.data_sha256,
        "model_spec": spec.to_dict(), "init_spec": init.to_dict(), "initialization": base.to_dict(),
        "training": asdict(config), "merges": [list(pair) for pair in tokenizer.merges],
        "tokenizer": tokenizer.identity.to_dict(), "batch_size": args.batch_size,
        "tokenizer_origin_run_sha256": (
            hash_json(tokenizer_source) if tokenizer_source is not None else None
        ),
        "eval_pairs": args.eval_pairs, "threads": args.threads,
        "length_window_batches": args.length_window_batches,
    }


def train(args: argparse.Namespace) -> None:
    root = args.run.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with exclusive_recovery_lock(root / "writer"):
        _train_locked(args, root)


def _train_locked(args: argparse.Namespace, root: Path) -> None:
    manifest_path = root / "run.json"
    if manifest_path.exists() and args.resume is None:
        raise ValueError("run already exists; specify a verified checkpoint step with --resume")
    if args.resume is not None and not manifest_path.is_file():
        raise ValueError("resume requires the original run manifest")
    train_rows = load_translation_split(args.data, expected_manifest_sha256=args.data_sha256, split="train")
    validation_rows = load_translation_split(
        args.data, expected_manifest_sha256=args.data_sha256, split="validation",
    )
    manifest = (
        json.loads(manifest_path.read_bytes()) if args.resume is not None else _new_manifest(args, train_rows)
    )
    if manifest["schema"] != SCHEMA or manifest["data_sha256"] != args.data_sha256:
        raise ValueError("run and dataset identities differ")
    torch.set_num_threads(manifest["threads"])
    tokenizer = _tokenizer(manifest)
    context, batch_size = manifest["model_spec"]["max_seq_len"], manifest["batch_size"]
    config = TrainerConfig(**{**manifest["training"], "betas": tuple(manifest["training"]["betas"])})
    stop = args.stop_after if args.stop_after is not None else config.max_steps
    if not (args.resume or 0) < stop <= config.max_steps or args.checkpoint_every < 1:
        raise ValueError("stop step must advance this run within its recorded training horizon")
    encoded = [tokens for _, tokens in _encoded_pairs(train_rows, tokenizer, context)]
    validation = [
        tokens for _, tokens in _encoded_pairs(
            _sample(validation_rows, manifest["eval_pairs"], config.seed + 1), tokenizer, context,
        )
    ]
    if not encoded or not validation:
        raise ValueError("pilot needs nonempty train and validation sets within context")
    model = _model(manifest)
    trainer = Trainer(model, config, device="cpu")
    if args.resume is None:
        _write_json(root / "initial.json", {"validation_loss": _loss(model, validation, tokenizer.pad_id, batch_size)})
        _write_json(manifest_path, manifest)
    else:
        load_trainer_checkpoint(
            _checkpoint(root, args.resume), model=model, trainer=trainer,
            expected_step=args.resume, **_expected(manifest),
        )
    model.train()
    stream = _batches(
        encoded, batch_size, config.seed, trainer.optimizer_step,
        manifest.get("length_window_batches", 1),
    )
    checkpoint_id = None
    while trainer.optimizer_step < stop:
        metrics = trainer.train_microbatch(collate(next(stream), tokenizer.pad_id))
        step = metrics.optimizer_step
        if step % 50 == 0 or step == stop:
            print(json.dumps({"step": step, "loss": metrics.loss, "tokens": trainer.tokens_seen}), flush=True)
        if step % args.checkpoint_every == 0 or step == stop:
            path = _checkpoint(root, step)
            path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_id = save_trainer_checkpoint(
                path, model=model, trainer=trainer, identity=_identity(manifest, trainer),
            )["checkpoint_id"]
    validation_loss = _loss(model, validation, tokenizer.pad_id, batch_size)
    restored = _restore_model(root, stop, manifest, checkpoint_id)
    readback_loss = _loss(restored, validation, tokenizer.pad_id, batch_size)
    if readback_loss != validation_loss:
        raise ValueError("restored checkpoint validation loss differs")
    epochs, remaining = divmod(stop, _epoch_batches(len(encoded), batch_size))
    summary: dict[str, Any] = {
        "checkpoint_id": checkpoint_id, "step": stop, "target_tokens": trainer.tokens_seen,
        "parameters": model.spec.parameter_count(), "train_examples_in_context": len(encoded),
        "training_examples_seen": epochs * len(encoded) + remaining * batch_size,
        "train_examples_excluded": len(train_rows) * 2 - len(encoded),
        "validation_examples": len(validation), "validation_loss": validation_loss,
        "initial_validation_loss": json.loads((root / "initial.json").read_bytes())["validation_loss"],
    }
    summary["training_complete"] = stop == config.max_steps
    _write_json(root / f"evaluation-{stop:07d}.json", summary)


def _external_evaluation(args: argparse.Namespace, manifest: dict, tokenizer: ByteBPE32k) -> None:
    raw = args.pairs.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.pairs_sha256:
        raise ValueError("evaluation pair file SHA-256 mismatch")
    rows = [json.loads(line) for line in raw.splitlines()]
    context = manifest["model_spec"]["max_seq_len"]
    eligible = _source_only_examples(rows, tokenizer, context)
    restored = _restore_model(args.run, args.step, manifest)
    initial = _model(manifest)
    _write_json(args.output, {
        "schema": "12-6.translation-external-evaluation.v1",
        "pairs_sha256": args.pairs_sha256,
        "run_manifest_sha256": hash_json(manifest),
        "step": args.step,
        "generation_limit_tokens": args.max_new_tokens,
        "evaluation_batch_size": args.eval_batch_size,
        "source_context_exclusions": len(rows) * 2 - len(eligible),
        "source_copy": evaluate_translation(eligible, lambda _, source: source)["by_direction"],
        "initial": evaluate_translation(
            eligible, _translation_lookup(initial, tokenizer, eligible, args.max_new_tokens, args.eval_batch_size),
        ),
        "trained": evaluate_translation(
            eligible, _translation_lookup(restored, tokenizer, eligible, args.max_new_tokens, args.eval_batch_size),
        ),
    })


def _evaluate_tatoeba(args: argparse.Namespace, manifest: dict, tokenizer: ByteBPE32k) -> None:
    rows = load_translation_split(
        args.data, expected_manifest_sha256=manifest["data_sha256"], split=args.split,
    )
    sample = _sample(rows, args.pair_count, manifest["training"]["seed"] + (
        1 if args.split == "validation" else 2
    ))
    model = _restore_model(args.run, args.step, manifest)
    examples = _source_only_examples(sample, tokenizer, model.spec.max_seq_len)
    summary = evaluate_translation(
        examples, _translation_lookup(model, tokenizer, examples, args.max_new_tokens, args.eval_batch_size),
    )
    _write_json(args.output, {
        "run_manifest_sha256": hash_json(manifest),
        "step": args.step,
        "split": args.split,
        "generation_limit_tokens": args.max_new_tokens,
        "evaluation_batch_size": args.eval_batch_size,
        "source_context_exclusions": len(sample) * 2 - len(examples),
        **summary,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit = subparsers.add_parser("train")
    fit.add_argument("--data", type=Path, required=True)
    fit.add_argument("--data-sha256", required=True)
    fit.add_argument("--run", type=Path, required=True)
    fit.add_argument("--steps", type=int, default=2400)
    fit.add_argument("--stop-after", type=int)
    fit.add_argument("--resume", type=int)
    fit.add_argument("--checkpoint-every", type=int, default=300)
    fit.add_argument("--batch-size", type=int, default=16)
    fit.add_argument("--eval-pairs", type=int, default=48)
    fit.add_argument("--context", type=int, default=96)
    fit.add_argument("--vocab-size", type=int, default=2048)
    fit.add_argument("--tokenizer-from", type=Path, help="Reuse the tokenizer from a recorded training run")
    fit.add_argument("--threads", type=int, default=3)
    fit.add_argument("--seed", type=int, default=1701)
    fit.add_argument("--lr", type=float, default=0.001)
    fit.add_argument("--width", type=int, default=96)
    fit.add_argument("--layers", type=int, default=4)
    fit.add_argument("--length-window-batches", type=int, default=1)
    infer = subparsers.add_parser("translate")
    infer.add_argument("--run", type=Path, required=True)
    infer.add_argument("--step", type=int, required=True)
    infer.add_argument("--direction", choices=DIRECTIONS, required=True)
    infer.add_argument("--text", required=True)
    infer.add_argument("--max-new-tokens", type=int, default=64)
    evaluate = subparsers.add_parser("evaluate", help="Evaluate a pinned external bitext sample")
    evaluate.add_argument("--run", type=Path, required=True)
    evaluate.add_argument("--step", type=int, required=True)
    evaluate.add_argument("--pairs", type=Path, required=True)
    evaluate.add_argument("--pairs-sha256", required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--max-new-tokens", type=int, default=64)
    evaluate.add_argument("--eval-batch-size", type=int, default=16)
    tatoeba = subparsers.add_parser("evaluate-tatoeba", help="Score one frozen Tatoeba split")
    tatoeba.add_argument("--run", type=Path, required=True)
    tatoeba.add_argument("--step", type=int, required=True)
    tatoeba.add_argument("--data", type=Path, required=True)
    tatoeba.add_argument("--split", choices=("validation", "test"), required=True)
    tatoeba.add_argument("--pair-count", type=int, default=100)
    tatoeba.add_argument("--output", type=Path, required=True)
    tatoeba.add_argument("--max-new-tokens", type=int, default=40)
    tatoeba.add_argument("--eval-batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.command == "train":
        if min(args.batch_size, args.eval_pairs, args.threads, args.layers, args.length_window_batches) < 1:
            parser.error("batch, layer, evaluation, thread and window counts must be positive")
        if args.width < 8 or args.width % 8:
            parser.error("model width must be a positive multiple of eight")
        train(args)
        return
    manifest = json.loads((args.run / "run.json").read_bytes())
    if args.max_new_tokens < 1:
        parser.error("generation token budget must be positive")
    if getattr(args, "eval_batch_size", 1) < 1:
        parser.error("evaluation batch size must be positive")
    torch.set_num_threads(manifest["threads"])
    tokenizer = _tokenizer(manifest)
    if args.command == "evaluate":
        _external_evaluation(args, manifest, tokenizer)
        return
    if args.command == "evaluate-tatoeba":
        if args.pair_count < 1:
            parser.error("pair count must be positive")
        _evaluate_tatoeba(args, manifest, tokenizer)
        return
    result = translate(
        _restore_model(args.run, args.step, manifest), tokenizer, args.direction, args.text,
        args.max_new_tokens,
    )
    sys.stdout.buffer.write((result + "\n").encode("utf-8"))


if __name__ == "__main__":
    main()
