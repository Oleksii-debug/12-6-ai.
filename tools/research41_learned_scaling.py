"""RESEARCH41 learned-base LOCAL_FREE scaling runner.

Extends the incumbent 95K/268K/468K/1.04M controlled family with the exact
DATA-10 repeatable BPE candidate and project-authored UK/EN/code micro-corpus.
This is not representative-corpus or model-quality evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from twelve_six.checkpoint import (
    CheckpointIdentity, hash_json, load_trainer_checkpoint,
    save_trainer_checkpoint, sha256_file,
)
from twelve_six.model import InitSpec, ModelSpec, TwelveSixDecoder
from twelve_six.packing.scale_contracts import MixturePlan, MixtureSource
from twelve_six.scaling_experiment import _fit_log_plane
from twelve_six.tokenization.experiments import (
    CorpusFileIdentity, TokenizerTrainingManifest, train_hf_tokenizer,
)
from twelve_six.training import Trainer, TrainerConfig

SCHEMA = "12-6.research41-learned-scaling.v2"
CELL_SCHEMA = "12-6.research41-learned-cell.v2"
PROGRESS_SCHEMA = "12-6.research41-learned-progress.v2"
AUTHORITY = "LOCAL_FREE_PROJECT_AUTHORED_MICROCORPUS_EVIDENCE_NOT_PROMOTION"
REPO = "Oleksii-debug/12-6-ai."
DATA10_SHA = "077205ef2b1662a5029bc77b8fc762078cabeb17"
S2_MECHANICS_SHA = "003e268655b672df9df00afb8a32dbec4db5d2e1"
TOK_VERSION = "0.23.1"
TOK_VOCAB = 472
BUDGETS = (8_192, 32_768, 131_072, 524_288, 1_048_576)
COUNTS = (95_568, 267_912, 467_808, 1_038_464)
BATCH = 4
SEQ = 64
SEED = 1337
PACKING_VERSION = "research41-data10-bpe-weighted-cyclic-v2"
LN2 = math.log(2.0)

TRAIN = (
    ("uk-1", "Українська мова має відмінки, дієвідмінювання і словотвір. Ці дані потрібні для базового передтренування моделі.", "uk"),
    ("uk-2", "Дослідники працюють із текстами різних жанрів, щоб модель бачила слова у називному, родовому, давальному, знахідному та орудному відмінках.", "uk"),
    ("uk-3", "Київ, Львів і Ужгород мають різні мовні контексти; ґрунтовний корпус повинен містити літери ґ, ї, є, і та природні апострофи.", "uk"),
    ("en-1", "The training corpus contains English prose with varied syntax and vocabulary so the base model learns next-token statistics rather than instructions.", "en"),
    ("en-2", "These records test deterministic data selection, source provenance, deduplication, and restart behavior for a universal language model.", "en"),
    ("en-3", "Data quality includes valid encoding, stable normalization, explicit source rights, and strict separation from held-out evaluation material.", "en"),
    ("code-1", "def stable_hash(value: str) -> str:\n    return hashlib.sha256(value.encode('utf-8')).hexdigest()", "code"),
    ("code-2", "class Counter:\n    def __init__(self):\n        self.value = 0\n    def increment(self):\n        self.value += 1\n        return self.value", "code"),
    ("code-3", "SELECT source_id, COUNT(*) FROM records\nWHERE split = 'train'\nGROUP BY source_id ORDER BY source_id;", "code"),
)
HELDOUT = (
    ("uk-cases", "книга книги книзі книгу книгою; учень учня учневі учнем", "uk"),
    ("uk-verbs", "працювати працюю працюєш працює працюємо працюють; прочитати прочитають", "uk"),
    ("uk-orthography", "п'ять, об'єкт, м'який, під'їзд, ґанок, їжак, Європа, Україна", "uk"),
    ("en", "The multilingual base model compares token fertility on unseen English.", "en"),
    ("code", "for index, item in enumerate(records):\n    assert item.split == 'train'\n", "code"),
    ("unicode", "Україна — Kyiv — naïve café — λ = 3.14 — 😀", "multi"),
)
GEOMETRY = (
    (48, 3, 4, 4, 12, 104),
    (72, 4, 6, 6, 12, 174),
    (96, 4, 6, 6, 16, 238),
    (128, 5, 8, 8, 16, 338),
)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def git_head(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def snapshot_file(root: Path) -> tuple[str, str]:
    text = "\n".join(row[1] for row in TRAIN) + "\n"
    path = root / "data/synthetic/data10/uk-en-code-train.txt"
    if path.read_text(encoding="utf-8") != text:
        raise RuntimeError("DATA-10 training snapshot drift")
    return text, sha256_file(path)


def tokenizer(root: Path):
    text, corpus_sha = snapshot_file(root)
    manifest_sha = sha(json.dumps(
        {"records": [row[0] for row in TRAIN], "corpus_sha256": corpus_sha},
        sort_keys=True, separators=(",", ":"),
    ))
    manifest = TokenizerTrainingManifest(
        experiment_id="data10-bpe-512-v1",
        algorithm="bpe",
        tokenizers_version=TOK_VERSION,
        dataset_id="data10-project-authored-uk-en-code-v1",
        dataset_manifest_sha256=manifest_sha,
        corpus_files=(CorpusFileIdentity(
            "data/synthetic/data10/uk-en-code-train.txt", corpus_sha, len(text.encode())
        ),),
        vocab_size=512,
        min_frequency=2,
    )
    texts = tuple(row[1] for row in TRAIN)
    first = train_hf_tokenizer(manifest, texts)
    second = train_hf_tokenizer(manifest, texts)
    if first.vocab_size != TOK_VOCAB or first.identity != second.identity:
        raise RuntimeError("DATA-10 BPE repeatability/vocabulary drift")
    if any(first.unk_id in first.encode(row[1]) for row in HELDOUT):
        raise RuntimeError("held-out probe produced BPE unknown token")
    return first, manifest


def spec(row: Sequence[int]) -> ModelSpec:
    d, layers, heads, kv, hd, ff = row
    return ModelSpec(
        schema_version=1, vocab_size=TOK_VOCAB, max_seq_len=256,
        d_model=d, n_layers=layers, n_heads=heads, n_kv_heads=kv,
        head_dim=hd, d_ff=ff, activation="swiglu", norm_kind="rmsnorm",
        norm_placement="pre", norm_eps=1e-5, position_embedding="rope",
        rope_theta=10_000.0, rope_rotary_dim=hd, attention_bias=False,
        mlp_bias=False, attention_dropout=0.0, final_norm=True,
        tie_word_embeddings=True, lm_head_bias=False,
    )


def specs() -> tuple[ModelSpec, ...]:
    result = tuple(spec(row) for row in GEOMETRY)
    if tuple(x.parameter_count() for x in result) != COUNTS:
        raise RuntimeError("BPE controlled-family parameter drift")
    return result


def trainer_config() -> TrainerConfig:
    max_steps = math.ceil(BUDGETS[-1] / (BATCH * (SEQ - 1)))
    return TrainerConfig(
        learning_rate=3e-4, weight_decay=0.0, betas=(0.9, 0.95), eps=1e-8,
        max_steps=max_steps, warmup_steps=0, scheduler="constant",
        gradient_accumulation_steps=1, gradient_clip_norm=1.0,
        precision="fp32", seed=SEED, deterministic_algorithms=True,
        deterministic_warn_only=False,
    )


def control(root: Path) -> dict[str, Any]:
    tok, tok_manifest = tokenizer(root)
    packing_hash = hash_json({
        "version": PACKING_VERSION, "batch": BATCH, "sequence": SEQ,
        "batch_source": "MixturePlan.source_for_sample(global_optimizer_step)",
        "within_source": "cyclic-contiguous-token-stream",
    })
    source_hashes = {
        lang: sha(json.dumps(sorted(row[0] for row in TRAIN if row[2] == lang), separators=(",", ":")))
        for lang in ("uk", "en", "code")
    }
    plan = MixturePlan(
        plan_id="uk-en-code-token-budget-v1",
        tokenizer_config_sha256=tok.identity.config_sha256,
        tokenizer_vocab_sha256=tok.identity.vocab_sha256,
        packing_config_sha256=packing_hash,
        sources=tuple(MixtureSource(n, source_hashes[n], w) for n, w in (("uk", 45), ("en", 35), ("code", 20))),
        seed=SEED, num_shards=32,
    )
    _, train_sha = snapshot_file(root)
    train_fp = {sha(row[1]) for row in TRAIN}
    held_fp = {sha(row[1]) for row in HELDOUT}
    overlap = sorted(train_fp & held_fp)
    if overlap:
        raise RuntimeError("train/held-out overlap")
    data = {
        "dataset_id": "data10-project-authored-uk-en-code-v1",
        "training_authority": "PROJECT_AUTHORED_SYNTHETIC_ONLY",
        "representative_corpus": False,
        "external_training_sources_approved": 0,
        "train_snapshot_sha256": train_sha,
        "train_records": len(TRAIN), "heldout_records": len(HELDOUT),
        "heldout_identity_sha256": hash_json({row[0]: sha(row[1]) for row in HELDOUT}),
        "train_heldout_exact_fingerprint_overlap": overlap,
        "mixture_plan_sha256": plan.sha256,
        "mixture_weights": {"uk": 45, "en": 35, "code": 20},
    }
    env_parts = {
        "canonical_lock_index_sha256": sha256_file(root / "requirements/locks/index.json"),
        "tokenizer_overlay_sha256": sha256_file(root / "requirements/experiments/tokenizers-linux-x86_64.lock.txt"),
    }
    env_hash = hash_json(env_parts)
    cfg = trainer_config()
    descriptor = {
        "tokenizer": {
            "algorithm": "bpe", "library_version": TOK_VERSION,
            "requested_vocab_size": 512, "actual_vocab_size": tok.vocab_size,
            "training_manifest_sha256": tok_manifest.sha256,
            "config_sha256": tok.identity.config_sha256,
            "vocab_sha256": tok.identity.vocab_sha256,
            "repeatability_verified_by_retraining": True, "frozen": False,
            "selection_rationale": (
                "BPE is used because DATA-10 found it repeatable with zero held-out unknowns; "
                "the slightly smaller Unigram token count was not repeatable. No tokenizer is frozen."
            ),
        },
        "data": data, "mixture_plan_sha256": plan.sha256,
        "packing": {"version": PACKING_VERSION, "sha256": packing_hash},
        "context": 256, "training_sequence_length": SEQ, "batch_size": BATCH,
        "tokens_per_optimizer_step": BATCH * (SEQ - 1),
        "token_budgets": list(BUDGETS), "seed": SEED,
        "init": InitSpec().to_dict(), "optimizer": asdict(cfg), "precision": "fp32",
        "parameter_counts": list(COUNTS), "environment": {**env_parts, "composite_sha256": env_hash},
    }
    return {
        "tok": tok, "plan": plan, "data": data, "env_hash": env_hash,
        "cfg": cfg, "specs": specs(), "descriptor": descriptor,
        "identity": hash_json(descriptor),
    }


def streams(tok) -> dict[str, list[int]]:
    nl = tok.encode("\n")
    out = {}
    for lang in ("uk", "en", "code"):
        ids: list[int] = []
        for _, text, row_lang in TRAIN:
            if row_lang == lang:
                ids.extend(tok.encode(text)); ids.extend(nl)
        out[lang] = ids
    return out


def schedule(plan: MixturePlan, steps: int) -> tuple[list[str], list[int]]:
    names, offsets, counts = [], [], Counter()
    for i in range(steps):
        name = plan.source_for_sample(i)
        names.append(name); offsets.append(counts[name]); counts[name] += 1
    return names, offsets


def batch(stream: Sequence[int], occurrence: int) -> torch.Tensor:
    width = BATCH * SEQ
    base = occurrence * width % len(stream)
    rows = [[stream[(base + b * SEQ + j) % len(stream)] for j in range(SEQ)] for b in range(BATCH)]
    return torch.tensor(rows, dtype=torch.long)


@torch.no_grad()
def evaluate(model: TwelveSixDecoder, tok, rows) -> dict[str, float | int]:
    was = model.training; model.eval()
    nll = 0.0; targets = 0; byte_count = 0
    for _, text, _ in rows:
        ids = tok.encode(text); byte_count += len(text.encode())
        start = 0
        while start < len(ids) - 1:
            chunk = ids[start:start + model.spec.max_seq_len]
            if len(chunk) < 2: break
            x = torch.tensor(chunk, dtype=torch.long).unsqueeze(0)
            logits = model(x).logits
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, model.spec.vocab_size),
                x[:, 1:].reshape(-1), reduction="sum",
            )
            nll += float(loss.item()); targets += len(chunk) - 1
            start += model.spec.max_seq_len - 1
    model.train(was)
    if targets <= 0 or byte_count <= 0: raise RuntimeError("empty evaluation")
    return {
        "loss": nll / targets,
        "bpb": nll / (LN2 * byte_count),
        "target_tokens": targets, "utf8_bytes": byte_count,
    }


@torch.no_grad()
def generate(model: TwelveSixDecoder, tok) -> list[dict[str, Any]]:
    was = model.training; model.eval(); out = []
    for prompt in ("Україна", "The model", "def "):
        prompt_ids = tok.encode(prompt); ids = list(prompt_ids); made = []
        for _ in range(12):
            x = torch.tensor(ids[-model.spec.max_seq_len:], dtype=torch.long).unsqueeze(0)
            nxt = int(torch.argmax(model(x).logits[0, -1]).item())
            ids.append(nxt); made.append(nxt)
        out.append({"prompt": prompt, "prompt_token_ids": prompt_ids, "generated_token_ids": made,
                    "decoded_full": tok.decode(ids, skip_special_tokens=False), "greedy": True})
    model.train(was); return out


def snap(model) -> dict[str, torch.Tensor]:
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def ratio(now: Mapping[str, torch.Tensor], before: Mapping[str, torch.Tensor]) -> dict[str, float | int]:
    d2 = w2 = 0.0; changed = total = 0
    for name, tensor in now.items():
        ref = before[name]; d = tensor.float() - ref.float()
        d2 += float(torch.sum(d * d)); w2 += float(torch.sum(ref.float() * ref.float()))
        changed += int(d.ne(0).sum()); total += d.numel()
    dl2, wl2 = math.sqrt(d2), math.sqrt(w2)
    return {"delta_l2": dl2, "reference_weight_l2": wl2,
            "update_to_weight_ratio": dl2 / wl2 if wl2 else math.inf,
            "changed_parameter_elements": changed, "trainable_parameter_elements": total}


def tensor_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor): return value.numel() * value.element_size()
    if isinstance(value, Mapping): return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)): return sum(tensor_bytes(v) for v in value)
    return 0


def directory_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def paths(work: Path, params: int) -> tuple[Path, Path, Path]:
    root = work / "cells" / str(params)
    return root, root / "progress.json", root / "cell-report.json"


def binding(source_sha: str, s: ModelSpec, init: InitSpec, c: Mapping[str, Any], trainer: Trainer):
    tc = {
        "schema": "12-6.research41-checkpoint-binding.v2",
        "authority": AUTHORITY, "control_identity_sha256": c["identity"],
        "init_spec_sha256": init.identity_sha256(), "training": c["descriptor"]["optimizer"],
        "data": {
            "dataset_manifest_sha256": hash_json(c["data"]),
            "split_identity": f"data10-project-authored-train:{c['data']['train_snapshot_sha256']}",
            "tokenizer_sha256": c["tok"].identity.config_sha256,
            "tokenizer_vocab_sha256": c["tok"].identity.vocab_sha256,
            "packing_sha256": c["descriptor"]["packing"]["sha256"],
            "packing_version": PACKING_VERSION,
        },
        "environment": c["descriptor"]["environment"],
    }
    run_hash = hash_json({
        "schema": CELL_SCHEMA, "source_sha": source_sha, "control_identity_sha256": c["identity"],
        "model_spec_sha256": s.identity_sha256(), "parameters": s.parameter_count(),
    })
    ident = CheckpointIdentity(
        git_sha=source_sha, model_spec=s.to_dict(), parameter_count=s.parameter_count(),
        tokenizer_hash=c["tok"].identity.config_sha256,
        tokenizer_vocab_hash=c["tok"].identity.vocab_sha256,
        dataset_manifest_hash=hash_json(c["data"]), run_manifest_hash=run_hash,
        training_config=tc, seed=SEED, precision=trainer.config.precision,
        step=trainer.optimizer_step, tokens_seen=trainer.tokens_seen,
        optimizer={"name": "AdamW", "lr": trainer.config.learning_rate,
                   "betas": list(trainer.config.betas), "eps": trainer.config.eps,
                   "weight_decay": trainer.config.weight_decay},
        scheduler={"name": trainer.config.scheduler}, environment_lock_hash=c["env_hash"],
    )
    return ident, tc, run_hash


def run_cell(root: Path, source_sha: str, work: Path, model_index: int,
             threads: int, stop_after: int | None) -> dict[str, Any]:
    if git_head(root) != source_sha: raise RuntimeError("exact-checkout mismatch")
    torch.set_num_threads(threads); torch.use_deterministic_algorithms(True)
    c = control(root); tok = c["tok"]; s = c["specs"][model_index]; init = InitSpec()
    cell_root, progress_path, report_path = paths(work, s.parameter_count())
    cell_root.mkdir(parents=True, exist_ok=True)
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report.get("control_identity_sha256") == c["identity"]: return report
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else None
    if progress and (progress.get("schema") != PROGRESS_SCHEMA or
                     progress.get("control_identity_sha256") != c["identity"] or
                     progress.get("model_spec_sha256") != s.identity_sha256()):
        raise RuntimeError("progress identity mismatch")

    random.seed(SEED); torch.manual_seed(SEED)
    model = TwelveSixDecoder(s, init); trainer = Trainer(model, c["cfg"], device="cpu")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED); initial = snap(TwelveSixDecoder(s, init))
    prior = initial; pid = os.getpid(); resumes = []; observations = []; generations = []
    source_steps = Counter(); train_wall = train_cpu = eval_wall = ckpt_wall = 0.0
    latest_budget = 0

    if progress is None:
        t = time.perf_counter(); tr = evaluate(model, tok, TRAIN); va = evaluate(model, tok, HELDOUT)
        eval_wall += time.perf_counter() - t
        observations.append({
            "parameters": s.parameter_count(), "requested_token_budget": 0,
            "optimized_tokens": 0, "optimizer_steps": 0, "compute_proxy": 0,
            "train_loss": tr["loss"], "train_bpb": tr["bpb"],
            "validation_loss": va["loss"], "validation_bpb": va["bpb"],
            "gradient_norm": None, "clip_frequency": None,
            "parameter_update_ratio_interval": 0.0,
            "parameter_update_ratio_from_init": 0.0,
            "optimizer_tensor_bytes": 0, "checkpoint": None,
        })
        generations.append({"requested_token_budget": 0, "optimized_tokens": 0, "samples": generate(model, tok)})
    else:
        observations = progress["observations"]; generations = progress["generations"]
        resumes = progress["resumes"]; source_steps = Counter(progress["source_steps"])
        train_wall = progress["train_wall"]; train_cpu = progress["train_cpu"]
        eval_wall = progress["eval_wall"]; ckpt_wall = progress["ckpt_wall"]
        latest_budget = progress["latest_budget"]; prior_pid = progress["last_process_id"]
        checkpoint = Path(progress["latest_checkpoint"])
        ident, tc, run_hash = binding(source_sha, s, init, c, trainer)
        t = time.perf_counter()
        result = load_trainer_checkpoint(
            checkpoint, model=model, trainer=trainer, restore_rng=True,
            expected_git_sha=source_sha, expected_model_spec_hash=s.identity_sha256(),
            expected_init_spec_hash=init.identity_sha256(),
            expected_tokenizer_hash=tok.identity.config_sha256,
            expected_tokenizer_vocab_hash=tok.identity.vocab_sha256,
            expected_dataset_manifest_hash=hash_json(c["data"]),
            expected_split_identity=tc["data"]["split_identity"],
            expected_packing_hash=tc["data"]["packing_sha256"],
            expected_packing_version=PACKING_VERSION,
            expected_run_manifest_hash=run_hash,
            expected_training_config_hash=hash_json(tc),
            expected_environment_lock_hash=c["env_hash"], expected_seed=SEED,
        )
        load_wall = time.perf_counter() - t
        resumed_va = evaluate(model, tok, HELDOUT)["loss"]
        reproduced = math.isclose(resumed_va, observations[-1]["validation_loss"], rel_tol=0, abs_tol=1e-10)
        if not reproduced: raise RuntimeError("resume validation loss drift")
        resumes.append({
            "checkpoint_id": result.manifest.get("checkpoint_id"),
            "requested_token_budget": latest_budget, "optimized_tokens": trainer.tokens_seen,
            "previous_process_id": prior_pid, "current_process_id": pid,
            "fresh_process": prior_pid != pid, "load_wall_seconds": load_wall,
            "validation_loss_reproduced": reproduced,
        })
        prior = snap(model)

    token_streams = streams(tok)
    names, offsets = schedule(c["plan"], c["cfg"].max_steps)
    next_idx = sum(b <= latest_budget for b in BUDGETS)
    grads: list[float] = []; losses: list[float] = []; clipped = 0
    interval_wall = interval_cpu = 0.0; interval_sources = Counter()
    for step in range(trainer.optimizer_step, c["cfg"].max_steps):
        name = names[step]; x = batch(token_streams[name], offsets[step])
        tw = time.perf_counter(); tcpu = time.process_time()
        m = trainer.train_microbatch({"input_ids": x})
        cpu = time.process_time() - tcpu; wall = time.perf_counter() - tw
        if not m.optimizer_stepped or m.grad_norm is None: raise RuntimeError("uncommitted update")
        grads.append(float(m.grad_norm)); losses.append(float(m.update_loss or m.loss))
        clipped += int(m.grad_norm > float(trainer.config.gradient_clip_norm or math.inf))
        train_wall += wall; train_cpu += cpu; interval_wall += wall; interval_cpu += cpu
        source_steps[name] += 1; interval_sources[name] += 1

        while next_idx < len(BUDGETS) and trainer.tokens_seen >= BUDGETS[next_idx]:
            budget = BUDGETS[next_idx]
            te = time.perf_counter(); tr = evaluate(model, tok, TRAIN); va = evaluate(model, tok, HELDOUT)
            eval_wall += time.perf_counter() - te
            now = snap(model); interval_ratio = ratio(now, prior); init_ratio = ratio(now, initial)
            generations.append({"requested_token_budget": budget, "optimized_tokens": trainer.tokens_seen,
                                "samples": generate(model, tok)})
            ident, _, _ = binding(source_sha, s, init, c, trainer)
            checkpoint = cell_root / "checkpoints" / str(budget)
            ts = time.perf_counter()
            manifest = save_trainer_checkpoint(checkpoint, model=model, trainer=trainer, identity=ident)
            save_wall = time.perf_counter() - ts; ckpt_wall += save_wall
            steps = len(grads)
            observations.append({
                "parameters": s.parameter_count(), "requested_token_budget": budget,
                "optimized_tokens": trainer.tokens_seen, "optimizer_steps": trainer.optimizer_step,
                "compute_proxy": 6 * s.parameter_count() * trainer.tokens_seen,
                "train_loss": tr["loss"], "train_bpb": tr["bpb"],
                "validation_loss": va["loss"], "validation_bpb": va["bpb"],
                "last_interval_mean_update_loss": sum(losses) / len(losses),
                "gradient_norm": {"min": min(grads), "max": max(grads), "mean": sum(grads) / steps},
                "clip_frequency": clipped / steps, "clip_count": clipped, "clip_steps": steps,
                "parameter_update_ratio_interval": interval_ratio["update_to_weight_ratio"],
                "parameter_update_ratio_from_init": init_ratio["update_to_weight_ratio"],
                "changed_parameter_elements_interval": interval_ratio["changed_parameter_elements"],
                "optimizer_tensor_bytes": tensor_bytes(trainer.optimizer.state_dict()),
                "interval_training_wall_seconds": interval_wall,
                "interval_training_cpu_seconds": interval_cpu,
                "interval_tokens_per_second": (
                    sum(interval_sources.values()) * BATCH * (SEQ - 1) / interval_wall
                ),
                "cumulative_source_steps": dict(sorted(source_steps.items())),
                "checkpoint": {
                    "checkpoint_id": manifest["checkpoint_id"], "format": manifest["format"],
                    "format_version": manifest["format_version"],
                    "directory_bytes": directory_bytes(checkpoint), "save_wall_seconds": save_wall,
                    "artifact_bytes": {n: r["bytes"] for n, r in manifest["files"].items()},
                },
            })
            latest_budget = budget
            progress = {
                "schema": PROGRESS_SCHEMA, "control_identity_sha256": c["identity"],
                "model_spec_sha256": s.identity_sha256(), "observations": observations,
                "generations": generations, "resumes": resumes,
                "source_steps": dict(source_steps), "train_wall": train_wall, "train_cpu": train_cpu,
                "eval_wall": eval_wall, "ckpt_wall": ckpt_wall,
                "latest_budget": latest_budget, "latest_checkpoint": str(checkpoint),
                "last_process_id": pid, "complete": False,
            }
            atomic(progress_path, progress)
            prior = now; grads = []; losses = []; clipped = 0
            interval_wall = interval_cpu = 0.0; interval_sources = Counter(); next_idx += 1
            if stop_after is not None and budget >= stop_after:
                return {"schema": CELL_SCHEMA, "status": "INCOMPLETE_FORCED_RESUME",
                        "parameters": s.parameter_count(), "optimized_tokens": trainer.tokens_seen}

    if next_idx != len(BUDGETS): raise RuntimeError("cell did not reach final budget")
    report = {
        "schema": CELL_SCHEMA, "status": "COMPLETE", "source_sha": source_sha,
        "control_identity_sha256": c["identity"], "model_index": model_index,
        "model_spec": s.to_dict(), "model_spec_sha256": s.identity_sha256(),
        "parameters": s.parameter_count(), "observations": observations,
        "generations": generations, "resumes": resumes, "source_steps": dict(sorted(source_steps.items())),
        "cost": {
            "training_wall_seconds": train_wall, "training_cpu_seconds": train_cpu,
            "evaluation_wall_seconds": eval_wall, "checkpoint_wall_seconds": ckpt_wall,
            "optimized_tokens_per_training_second": trainer.tokens_seen / train_wall,
            "final_optimizer_tensor_bytes": observations[-1]["optimizer_tensor_bytes"],
        },
        "claims": {
            "foreign_pretrained_weights_used": False, "instruction_or_sft_used": False,
            "paid_compute_used": False, "representative_corpus": False,
            "quality_or_capability_claim": False,
        },
    }
    report["cell_report_sha256"] = hash_json(report); atomic(report_path, report)
    progress["complete"] = True; atomic(progress_path, progress)
    return report


def spawn(script: Path, root: Path, source: str, work: Path, index: int, threads: int,
          stop: int | None = None) -> None:
    cmd = [sys.executable, str(script), "cell", "--repo-root", str(root),
           "--source-sha", source, "--work-dir", str(work), "--model-index", str(index),
           "--threads", str(threads)]
    if stop is not None: cmd += ["--stop-after", str(stop)]
    subprocess.run(cmd, cwd=root, check=True)


def matrix(cells, field: str) -> list[list[float | int]]:
    rows = []
    for cell in cells:
        points = {p["requested_token_budget"]: p for p in cell["observations"] if p["requested_token_budget"]}
        rows.append([points[b][field] for b in BUDGETS])
    return rows


def efficiencies(cells) -> dict[str, Any]:
    maps = [{p["requested_token_budget"]: p for p in cell["observations"]} for cell in cells]
    parameter = []
    for i in range(4):
        row = []
        for b in BUDGETS:
            row.append(None if i == 0 else (
                (maps[i-1][b]["validation_loss"] - maps[i][b]["validation_loss"]) /
                (cells[i]["parameters"] - cells[i-1]["parameters"])
            ))
        parameter.append(row)
    token = []
    for points in maps:
        prior = points[0]; row = []
        for b in BUDGETS:
            cur = points[b]; dt = cur["optimized_tokens"] - prior["optimized_tokens"]
            row.append((prior["validation_loss"] - cur["validation_loss"]) / dt); prior = cur
        token.append(row)
    candidates = []
    for cell, points in zip(cells, maps, strict=True):
        final = points[BUDGETS[-1]]
        improvement = points[0]["validation_loss"] - final["validation_loss"]
        compute = final["compute_proxy"]
        candidates.append({
            "parameters": cell["parameters"], "requested_token_budget": BUDGETS[-1],
            "optimized_tokens": final["optimized_tokens"],
            "validation_loss_improvement_from_init": improvement, "compute_proxy": compute,
            "validation_improvement_per_compute_proxy": improvement / compute,
            "validation_improvement_per_1e12_compute_proxy": improvement * 1e12 / compute,
        })
    return {
        "parameter_efficiency_validation_loss_reduction_per_added_parameter": parameter,
        "token_efficiency_validation_loss_reduction_per_added_token": token,
        "best_validation_improvement_per_compute": {
            "comparison_basis": "largest_common_token_budget_vs_each_model_initial_validation_loss",
            "candidates": candidates,
            "winner": max(candidates, key=lambda x: x["validation_improvement_per_compute_proxy"]),
        },
    }


def run(root: Path, source: str, work: Path, output: Path, threads: int) -> dict[str, Any]:
    if git_head(root) != source: raise RuntimeError("exact-checkout mismatch")
    c = control(root); script = Path(__file__).resolve(); work.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    for i, s in enumerate(c["specs"]):
        _, progress, report = paths(work, s.parameter_count())
        if report.exists():
            existing = json.loads(report.read_text())
            if existing.get("control_identity_sha256") == c["identity"]: continue
        if i == 3 and not progress.exists():
            spawn(script, root, source, work, i, threads, stop=BUDGETS[2])
        spawn(script, root, source, work, i, threads)
    cells = []
    for s in c["specs"]:
        report = paths(work, s.parameter_count())[2]
        cell = json.loads(report.read_text())
        if cell["status"] != "COMPLETE" or cell["control_identity_sha256"] != c["identity"]:
            raise RuntimeError("incomplete/mismatched matrix cell")
        cells.append(cell)
    cells.sort(key=lambda x: x["parameters"])
    points = [p for cell in cells for p in cell["observations"] if p["requested_token_budget"]]
    eff = efficiencies(cells)
    largest = cells[-1]
    fresh = [r for r in largest["resumes"] if r["fresh_process"]]
    if not fresh: raise RuntimeError("fresh-process resume not exercised")
    report = {
        "schema": SCHEMA, "authority": AUTHORITY,
        "source": {
            "repository": REPO, "git_sha": source,
            "lineage": {
                "scaling_incumbent_pr": 162, "s2_mechanics_predecessor_pr": 144,
                "s2_mechanics_head_sha": S2_MECHANICS_SHA, "s1_s2_transition_successor_pr": 156,
                "data_tokenizer_incumbent_pr": 173, "data_tokenizer_head_sha": DATA10_SHA,
            },
        },
        "runtime": {"device": "cpu", "torch_threads": threads, "paid_compute": False,
                    "orchestrator_wall_seconds": time.perf_counter() - start},
        "controls": c["descriptor"], "control_identity_sha256": c["identity"], "data": c["data"],
        "models": [{"parameters": x["parameters"], "model_spec_sha256": x["model_spec_sha256"],
                    "model_spec": x["model_spec"]} for x in cells],
        "cells": cells, "cost_by_model": [{"parameters": x["parameters"], **x["cost"]} for x in cells],
        "matrices": {
            "parameters": list(COUNTS), "requested_token_budgets": list(BUDGETS),
            "optimized_tokens": matrix(cells, "optimized_tokens"),
            "train_loss": matrix(cells, "train_loss"),
            "validation_loss": matrix(cells, "validation_loss"),
            "train_bpb": matrix(cells, "train_bpb"),
            "validation_bpb": matrix(cells, "validation_bpb"),
            "compute_proxy": matrix(cells, "compute_proxy"),
        },
        "fit": _fit_log_plane(points), "efficiency": eff,
        "fresh_process_resume": {
            "required_model_parameters": largest["parameters"],
            "forced_split_requested_token_budget": BUDGETS[2],
            "verified_resumes": fresh, "passed": True,
        },
        "comparability": {
            "within_v2_matrix_only_model_geometry_changes": True,
            "previous_research41_v1_byte_s0_results_directly_poolable": False,
            "reason": "v2 changes tokenizer and corpus from v1; all four v2 cells share the new controls.",
        },
        "truth_boundary": {
            "base_pretraining_only": True, "random_init_only": True,
            "foreign_pretrained_weights_used": False, "instruction_or_sft_used": False,
            "paid_compute_used": False, "representative_corpus": False,
            "external_training_sources_approved": 0, "tokenizer_frozen": False,
            "quality_or_capability_claim": False, "stage_freeze_or_promotion": False,
            "fit_valid_only_inside_observed_box": True,
            "billions_or_trillions_extrapolation_authorized": False,
        },
    }
    report["report_sha256"] = hash_json(report); atomic(output, report); validate(report, source)
    return report


def validate(report: Mapping[str, Any], source: str | None = None) -> None:
    if report.get("schema") != SCHEMA or report.get("authority") != AUTHORITY:
        raise ValueError("schema/authority mismatch")
    if source is not None and report["source"]["git_sha"] != source:
        raise ValueError("source SHA mismatch")
    if report["matrices"]["parameters"] != list(COUNTS) or report["matrices"]["requested_token_budgets"] != list(BUDGETS):
        raise ValueError("matrix identity drift")
    for key in ("train_loss", "validation_loss", "train_bpb", "validation_bpb", "compute_proxy"):
        value = report["matrices"][key]
        if len(value) != 4 or any(len(row) != 5 for row in value): raise ValueError(f"{key} not 4x5")
    if report["controls"]["tokenizer"]["actual_vocab_size"] != TOK_VOCAB:
        raise ValueError("tokenizer drift")
    if report["data"]["representative_corpus"] is not False or report["data"]["train_heldout_exact_fingerprint_overlap"]:
        raise ValueError("data truth/isolation failure")
    if not report["fresh_process_resume"]["passed"] or not report["fresh_process_resume"]["verified_resumes"]:
        raise ValueError("fresh-process resume missing")
    for cell in report["cells"]:
        for point in cell["observations"][1:]:
            if point["compute_proxy"] != 6 * point["parameters"] * point["optimized_tokens"]:
                raise ValueError("compute proxy drift")
            if not all(math.isfinite(point[k]) and point[k] > 0 for k in ("train_loss", "validation_loss", "train_bpb", "validation_bpb")):
                raise ValueError("non-finite loss/BPB")
            if not 0 <= point["clip_frequency"] <= 1: raise ValueError("clip frequency")
            if point["parameter_update_ratio_interval"] < 0: raise ValueError("update ratio")
            if point["checkpoint"]["directory_bytes"] <= 0 or point["checkpoint"]["save_wall_seconds"] <= 0:
                raise ValueError("checkpoint evidence")
    truth = report["truth_boundary"]
    for key in ("foreign_pretrained_weights_used", "instruction_or_sft_used", "paid_compute_used",
                "representative_corpus", "tokenizer_frozen", "quality_or_capability_claim",
                "stage_freeze_or_promotion", "billions_or_trillions_extrapolation_authorized"):
        if truth[key] is not False: raise ValueError("truth boundary weakened")
    unsigned = dict(report); supplied = unsigned.pop("report_sha256", None)
    if supplied != hash_json(unsigned): raise ValueError("report self-hash mismatch")


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    runp = sub.add_parser("run")
    for p in (runp,):
        p.add_argument("--repo-root", type=Path, default=Path("."))
        p.add_argument("--source-sha", required=True); p.add_argument("--work-dir", type=Path, required=True)
        p.add_argument("--threads", type=int, default=2)
    runp.add_argument("--output", type=Path, required=True)
    cellp = sub.add_parser("cell")
    cellp.add_argument("--repo-root", type=Path, default=Path(".")); cellp.add_argument("--source-sha", required=True)
    cellp.add_argument("--work-dir", type=Path, required=True); cellp.add_argument("--model-index", type=int, required=True)
    cellp.add_argument("--threads", type=int, default=2); cellp.add_argument("--stop-after", type=int)
    valp = sub.add_parser("validate"); valp.add_argument("report", type=Path); valp.add_argument("--source-sha")
    args = parser.parse_args()
    if args.cmd == "run":
        report = run(args.repo_root.resolve(), args.source_sha, args.work_dir.resolve(), args.output, args.threads)
        print(json.dumps({
            "parameters": report["matrices"]["parameters"],
            "budgets": report["matrices"]["requested_token_budgets"],
            "validation_loss": report["matrices"]["validation_loss"],
            "validation_bpb": report["matrices"]["validation_bpb"],
            "winner": report["efficiency"]["best_validation_improvement_per_compute"]["winner"],
            "fresh_resume": report["fresh_process_resume"]["passed"],
            "report_sha256": report["report_sha256"],
        }, ensure_ascii=False, sort_keys=True)); return 0
    if args.cmd == "cell":
        print(json.dumps(run_cell(args.repo_root.resolve(), args.source_sha, args.work_dir.resolve(),
                                  args.model_index, args.threads, args.stop_after),
                         ensure_ascii=False, sort_keys=True)); return 0
    report = json.loads(args.report.read_text()); validate(report, args.source_sha)
    print(f"{SCHEMA}: PASS"); return 0


if __name__ == "__main__":
    raise SystemExit(main())
