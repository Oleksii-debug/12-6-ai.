"""TOK-37 ByteLevel BPE vocabulary sweep using the incumbent HF trainer only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from twelve_six.model import ModelSpec, TwelveSixDecoder
from twelve_six.scaling_experiment import controlled_specs
from twelve_six.tokenization.experiments import (
    CorpusFileIdentity,
    TokenizerProbe,
    TokenizerTrainingManifest,
    measure_probe,
    train_hf_tokenizer,
)

SCHEMA = "12-6.tok37-bpe-vocab-sweep.v1"
AUTHORITY = "LOCAL_FREE_TOKENIZER_SWEEP_NOT_TOKENIZER_FREEZE"
TOKENIZERS_VERSION = "0.23.1"
CORPUS_PATH = Path("data/synthetic/data10/uk-en-code-train.txt")
CORPUS_CONFIG_PATH = Path("configs/data/multilingual_uk_en_code_v1.experimental.json")
PURPOSE_PROFILE_PATH = Path(
    "requirements/profiles/linux-x86_64-tokenizer-experiment/profile.json"
)
PURPOSE_OVERLAY_PATH = Path(
    "requirements/profiles/linux-x86_64-tokenizer-experiment/overlay.lock.txt"
)
EXPECTED_OVERLAY_SHA256 = "11f27613ee7c15585796af39accde71b1e7c2791c24ff98d74c395262ee68544"
EXPECTED_PROFILE_SEMANTIC_SHA256 = (
    "e368fa4c9fb2fc924482de32d5057837959111e958649663813cb46dddf6b5e4"
)
REQUESTED_GRID = (256, 257, 320, 384, 512, 768, 1024)
SEED = 126
MODEL_PROBE_TARGET_PARAMETERS = 100_000
MODEL_PROBE_OPTIMIZED_TOKENS = 1_024
MODEL_PROBE_BATCH = 2
MODEL_PROBE_SEQUENCE = 17
MODEL_PROBE_STEPS = MODEL_PROBE_OPTIMIZED_TOKENS // (
    MODEL_PROBE_BATCH * (MODEL_PROBE_SEQUENCE - 1)
)

TRAIN_RECORDS: tuple[tuple[str, str], ...] = (
    (
        "uk-1",
        "Українська мова має відмінки, дієвідмінювання і словотвір. Ці дані "
        "потрібні для базового передтренування моделі.",
    ),
    (
        "uk-2",
        "Дослідники працюють із текстами різних жанрів, щоб модель бачила слова "
        "у називному, родовому, давальному, знахідному та орудному відмінках.",
    ),
    (
        "uk-3",
        "Київ, Львів і Ужгород мають різні мовні контексти; ґрунтовний корпус "
        "повинен містити літери ґ, ї, є, і та природні апострофи.",
    ),
    (
        "en-1",
        "The training corpus contains English prose with varied syntax and vocabulary "
        "so the base model learns next-token statistics rather than instructions.",
    ),
    (
        "en-2",
        "These records test deterministic data selection, source provenance, "
        "deduplication, and restart behavior for a universal language model.",
    ),
    (
        "en-3",
        "Data quality includes valid encoding, stable normalization, explicit source "
        "rights, and strict separation from held-out evaluation material.",
    ),
    (
        "code-1",
        "def stable_hash(value: str) -> str:\n"
        "    return hashlib.sha256(value.encode('utf-8')).hexdigest()\n",
    ),
    (
        "code-2",
        "class Counter:\n"
        "    def __init__(self):\n"
        "        self.value = 0\n"
        "    def increment(self):\n"
        "        self.value += 1\n"
        "        return self.value\n",
    ),
    (
        "code-3",
        "SELECT source_id, COUNT(*) FROM records\n"
        "WHERE split = 'train'\n"
        "GROUP BY source_id ORDER BY source_id;\n",
    ),
)

PROBES: tuple[TokenizerProbe, ...] = (
    TokenizerProbe(
        "uk-cases",
        "uk",
        "morphology-heldout",
        "книга книги книзі книгу книгою; учень учня учневі учнем",
    ),
    TokenizerProbe(
        "uk-verbs",
        "uk",
        "morphology-heldout",
        "працювати працюю працюєш працює працюємо працюють; прочитати прочитають",
    ),
    TokenizerProbe(
        "uk-orthography",
        "uk",
        "orthography-heldout",
        "п'ять, об'єкт, м'який, під'їзд, ґанок, їжак, Європа, Україна",
    ),
    TokenizerProbe(
        "en",
        "en",
        "language-heldout",
        "The multilingual base model compares token fertility on unseen English.",
    ),
    TokenizerProbe(
        "code",
        "code",
        "code-heldout",
        "for index, item in enumerate(records):\n    assert item.split == 'train'\n",
    ),
    TokenizerProbe(
        "unicode-mixed",
        "unicode",
        "unicode-heldout",
        "Україна — Kyiv — naïve café — λ = 3.14 — 😀",
    ),
    TokenizerProbe(
        "unicode-combining",
        "unicode",
        "unicode-edge-heldout",
        "e\u0301 ≠ é; ї\u0301; a\u0308; Z\u0351",
    ),
    TokenizerProbe(
        "unicode-zwj",
        "unicode",
        "unicode-edge-heldout",
        "👩‍💻 👨‍👩‍👧‍👦 🇺🇦 𐍈 数学 مرحبا",
    ),
)


class SweepError(RuntimeError):
    """Fail-closed TOK-37 evidence error."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _self_hash(payload: dict[str, Any]) -> str:
    return _sha256_bytes(_canonical_json(payload).encode("utf-8"))


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, encoding="utf-8"
    ).strip()


def _joined_training_text() -> str:
    return "\n".join(text for _, text in TRAIN_RECORDS) + "\n"


def _corpus_contract() -> dict[str, Any]:
    joined = _joined_training_text()
    raw = CORPUS_PATH.read_text(encoding="utf-8")
    if raw != joined:
        raise SweepError("DATA-10 tokenizer corpus bytes do not match fixed record segmentation")
    config = json.loads(CORPUS_CONFIG_PATH.read_text(encoding="utf-8"))
    local = config["local_mechanics_corpus"]
    corpus_sha = _sha256_bytes(raw.encode("utf-8"))
    if local["sha256"] != corpus_sha:
        raise SweepError("DATA-10 corpus SHA-256 drift")
    if bool(local["representative_corpus"]):
        raise SweepError("TOK-37 fixture contract unexpectedly claims representativeness")
    dataset_identity = _sha256_bytes(
        _canonical_json(
            {
                "schema": "12-6.tok37-dataset-identity.v1",
                "corpus_sha256": corpus_sha,
                "record_ids": [record_id for record_id, _ in TRAIN_RECORDS],
                "record_text_sha256": [
                    _sha256_bytes(text.encode("utf-8")) for _, text in TRAIN_RECORDS
                ],
            }
        ).encode("utf-8")
    )
    return {
        "dataset_id": "data10-project-authored-uk-en-code-v1",
        "dataset_identity_sha256": dataset_identity,
        "path": CORPUS_PATH.as_posix(),
        "sha256": corpus_sha,
        "bytes": len(raw.encode("utf-8")),
        "records": len(TRAIN_RECORDS),
        "record_ids": [record_id for record_id, _ in TRAIN_RECORDS],
        "representative_corpus": False,
        "authority": str(local["authority"]),
    }


def _purpose_environment_contract() -> dict[str, Any]:
    profile = json.loads(PURPOSE_PROFILE_PATH.read_text(encoding="utf-8"))
    if profile["profile_id"] != "linux-x86_64-tokenizer-experiment":
        raise SweepError("wrong purpose environment profile")
    if profile["python"]["version"] != "3.11.16":
        raise SweepError("purpose profile Python version drift")
    if profile["direct_requirements"] != ["tokenizers==0.23.1"]:
        raise SweepError("purpose profile tokenizer version drift")
    if profile["profile_sha256"] != EXPECTED_PROFILE_SEMANTIC_SHA256:
        raise SweepError("purpose profile semantic identity drift")
    overlay_sha = _sha256_file(PURPOSE_OVERLAY_PATH)
    if overlay_sha != EXPECTED_OVERLAY_SHA256:
        raise SweepError("purpose overlay lock SHA-256 drift")
    if profile["locks"]["overlay"]["sha256"] != overlay_sha:
        raise SweepError("purpose profile does not bind current overlay")
    base_path = Path(profile["base_profile"]["path"])
    if _sha256_file(base_path) != profile["base_profile"]["file_sha256"]:
        raise SweepError("base lock profile file identity drift")
    return {
        "profile_id": profile["profile_id"],
        "profile_semantic_sha256": profile["profile_sha256"],
        "profile_file_sha256": _sha256_file(PURPOSE_PROFILE_PATH),
        "overlay_sha256": overlay_sha,
        "python": profile["python"]["version"],
        "tokenizers": TOKENIZERS_VERSION,
    }


def _manifest(corpus: dict[str, Any], requested_vocab_size: int) -> TokenizerTrainingManifest:
    return TokenizerTrainingManifest(
        experiment_id=f"tok37-data10-bpe-r{requested_vocab_size}-v1",
        algorithm="bpe",
        tokenizers_version=TOKENIZERS_VERSION,
        dataset_id=str(corpus["dataset_id"]),
        dataset_manifest_sha256=str(corpus["dataset_identity_sha256"]),
        corpus_files=(
            CorpusFileIdentity(
                str(corpus["path"]),
                str(corpus["sha256"]),
                int(corpus["bytes"]),
            ),
        ),
        vocab_size=requested_vocab_size,
        min_frequency=2,
    )


def _artifact_dict(adapter: Any) -> dict[str, Any]:
    artifact = adapter.artifact_identity
    return {
        "algorithm": artifact.algorithm,
        "tokenizers_version": artifact.tokenizers_version,
        "training_manifest_sha256": artifact.training_manifest_sha256,
        "tokenizer_json_sha256": artifact.tokenizer_json_sha256,
        "vocab_sha256": artifact.vocab_sha256,
        "actual_vocab_size": artifact.vocab_size,
        "special_tokens": dict(artifact.special_tokens),
        "config_sha256": artifact.config_sha256,
    }


def _serialize_adapter(adapter: Any, path: Path) -> str:
    runtime = getattr(adapter, "_tokenizer", None)
    if runtime is None or not hasattr(runtime, "to_str"):
        raise SweepError("incumbent adapter does not expose serializable runtime")
    payload = runtime.to_str().encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    digest = _sha256_bytes(payload)
    if digest != adapter.artifact_identity.tokenizer_json_sha256:
        raise SweepError("serialized tokenizer file hash disagrees with artifact identity")
    return digest


def _probe_metrics(adapter: Any) -> dict[str, Any]:
    results = [measure_probe(adapter, probe, unknown_token_id=adapter.unk_id) for probe in PROBES]
    if not all(result.round_trip_exact for result in results):
        raise SweepError("held-out tokenizer round-trip failure")
    if any(result.unknown_tokens for result in results):
        raise SweepError("held-out tokenizer emitted unintended unknown token")
    by_group: dict[str, dict[str, Any]] = {}
    for language in ("uk", "en", "code", "unicode"):
        group = [result for result in results if result.language == language]
        codepoints = sum(result.codepoints for result in group)
        utf8_bytes = sum(result.utf8_bytes for result in group)
        tokens = sum(result.tokens for result in group)
        by_group[language] = {
            "probes": len(group),
            "codepoints": codepoints,
            "utf8_bytes": utf8_bytes,
            "tokens": tokens,
            "fertility_tokens_per_codepoint": tokens / codepoints if codepoints else 0.0,
            "tokens_per_utf8_byte": tokens / utf8_bytes if utf8_bytes else 0.0,
            "bpb_input_predicted_tokens": sum(max(result.tokens - 1, 0) for result in group),
            "bpb_input_utf8_bytes": utf8_bytes,
            "round_trip_exact": all(result.round_trip_exact for result in group),
            "unknown_tokens": sum(result.unknown_tokens for result in group),
        }
    return {
        "probes": [asdict(result) for result in results],
        "by_group": by_group,
        "tokens": sum(result.tokens for result in results),
        "utf8_bytes": sum(result.utf8_bytes for result in results),
        "round_trip_exact": True,
        "unknown_tokens": 0,
    }


def _training_unknowns(adapter: Any) -> int:
    return sum(
        token_id == adapter.unk_id
        for _, text in TRAIN_RECORDS
        for token_id in adapter.encode(text)
    )


def _parameter_tax(actual_vocab_size: int) -> dict[str, Any]:
    labels = ("100K", "250K", "500K", "1M")
    specs = controlled_specs()
    result: dict[str, Any] = {}
    for label, spec in zip(labels, specs, strict=True):
        target = spec.parameter_count()
        embedding = actual_vocab_size * spec.d_model
        byte_embedding = 256 * spec.d_model
        result[label] = {
            "controlled_model_parameters": target,
            "d_model": spec.d_model,
            "tied_lm_head": True,
            "embedding_parameters": embedding,
            "embedding_share": embedding / target,
            "incremental_parameters_vs_byte_vocab": embedding - byte_embedding,
            "incremental_share_vs_byte_vocab": (embedding - byte_embedding) / target,
        }
    return result


def _rank(values: list[tuple[int, float]]) -> dict[int, int]:
    ordered = sorted(values, key=lambda item: (item[1], item[0]))
    return {requested: rank + 1 for rank, (requested, _) in enumerate(ordered)}


def _prefilter_selection(results: list[dict[str, Any]], limit: int = 3) -> list[int]:
    eligible = [result for result in results if result["status"] == "PASS"]
    token_ranks = _rank(
        [(int(result["requested_vocab_size"]), float(result["held_out"]["tokens"])) for result in eligible]
    )
    vocab_ranks = _rank(
        [
            (int(result["requested_vocab_size"]), float(result["artifact"]["actual_vocab_size"]))
            for result in eligible
        ]
    )
    worst_ranks = _rank(
        [
            (
                int(result["requested_vocab_size"]),
                max(
                    float(result["held_out"]["by_group"][name]["fertility_tokens_per_codepoint"])
                    for name in ("uk", "en", "code")
                ),
            )
            for result in eligible
        ]
    )
    scored: list[tuple[int, int, int]] = []
    for result in eligible:
        requested = int(result["requested_vocab_size"])
        score = token_ranks[requested] + vocab_ranks[requested] + worst_ranks[requested]
        scored.append((score, int(result["artifact"]["actual_vocab_size"]), requested))
    return [requested for _, _, requested in sorted(scored)[:limit]]


def _pareto(
    results: Iterable[dict[str, Any]],
    *,
    include_model_bpb: bool,
) -> list[int]:
    eligible = [result for result in results if result["status"] == "PASS"]
    frontier: list[int] = []
    for candidate in eligible:
        cvocab = int(candidate["artifact"]["actual_vocab_size"])
        ctokens = int(candidate["held_out"]["tokens"])
        cbpb = (
            float(candidate["model_probe"]["aggregate_bpb"])
            if include_model_bpb and "model_probe" in candidate
            else None
        )
        dominated = False
        for other in eligible:
            if other is candidate:
                continue
            ovocab = int(other["artifact"]["actual_vocab_size"])
            otokens = int(other["held_out"]["tokens"])
            if include_model_bpb:
                if "model_probe" not in candidate or "model_probe" not in other:
                    continue
                obpb = float(other["model_probe"]["aggregate_bpb"])
                no_worse = ovocab <= cvocab and otokens <= ctokens and obpb <= float(cbpb)
                better = ovocab < cvocab or otokens < ctokens or obpb < float(cbpb)
            else:
                no_worse = ovocab <= cvocab and otokens <= ctokens
                better = ovocab < cvocab or otokens < ctokens
            if no_worse and better:
                dominated = True
                break
        if not dominated:
            frontier.append(int(candidate["requested_vocab_size"]))
    return sorted(frontier)


def _rebalance_100k(vocab_size: int) -> ModelSpec:
    base = controlled_specs()[0]
    target = MODEL_PROBE_TARGET_PARAMETERS
    candidate = replace(base, vocab_size=vocab_size)
    slope = candidate.n_layers * 3 * candidate.d_model
    constant = candidate.parameter_count() - slope * candidate.d_ff
    ideal = (target - constant) / slope
    if ideal <= 0:
        raise SweepError("vocabulary exhausts 100K probe model parameter budget")
    lower = max(8, int(ideal // 8) * 8)
    options = (lower, lower + 8)
    return min(
        (replace(candidate, d_ff=d_ff) for d_ff in options),
        key=lambda spec: (abs(spec.parameter_count() - target), spec.parameter_count() > target, -spec.d_ff),
    )


def _batch(stream: list[int], step: int) -> torch.Tensor:
    if not stream:
        raise SweepError("empty model-probe training stream")
    width = MODEL_PROBE_BATCH * MODEL_PROBE_SEQUENCE
    base = (step * width) % len(stream)
    rows = []
    for batch_index in range(MODEL_PROBE_BATCH):
        start = (base + batch_index * MODEL_PROBE_SEQUENCE) % len(stream)
        rows.append(
            [stream[(start + offset) % len(stream)] for offset in range(MODEL_PROBE_SEQUENCE)]
        )
    return torch.tensor(rows, dtype=torch.long)


@torch.no_grad()
def _eval_model(model: TwelveSixDecoder, adapter: Any) -> dict[str, Any]:
    model.eval()
    by_group: dict[str, dict[str, float | int]] = {}
    total_nll = 0.0
    total_bytes = 0
    total_targets = 0
    for language in ("uk", "en", "code", "unicode"):
        nll = 0.0
        utf8_bytes = 0
        targets = 0
        for probe in PROBES:
            if probe.language != language:
                continue
            ids = adapter.encode(probe.text)
            if len(ids) < 2:
                continue
            if len(ids) > model.spec.max_seq_len:
                raise SweepError(f"probe {probe.name} exceeds model probe context")
            tensor = torch.tensor([ids], dtype=torch.long)
            logits = model(tensor).logits[:, :-1, :]
            labels = tensor[:, 1:]
            nll += float(
                F.cross_entropy(
                    logits.reshape(-1, model.spec.vocab_size),
                    labels.reshape(-1),
                    reduction="sum",
                )
            )
            targets += labels.numel()
            utf8_bytes += len(probe.text.encode("utf-8"))
        bpb = nll / (utf8_bytes * math.log(2.0)) if utf8_bytes else 0.0
        by_group[language] = {
            "nll_nats": nll,
            "predicted_tokens": targets,
            "utf8_bytes": utf8_bytes,
            "bpb": bpb,
        }
        total_nll += nll
        total_targets += targets
        total_bytes += utf8_bytes
    return {
        "aggregate_nll_nats": total_nll,
        "aggregate_predicted_tokens": total_targets,
        "aggregate_utf8_bytes": total_bytes,
        "aggregate_bpb": total_nll / (total_bytes * math.log(2.0)),
        "by_group": by_group,
        "first_token_per_probe_unscored": True,
    }


def _model_probe(adapter: Any) -> dict[str, Any]:
    spec = _rebalance_100k(adapter.vocab_size)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    model = TwelveSixDecoder(spec)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
    )
    stream = adapter.encode(_joined_training_text())
    initial = _eval_model(model, adapter)
    model.train()
    last_loss = 0.0
    optimized = 0
    for step in range(MODEL_PROBE_STEPS):
        batch = _batch(stream, step)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch).logits[:, :-1, :]
        labels = batch[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, spec.vocab_size), labels.reshape(-1))
        if not torch.isfinite(loss):
            raise SweepError("non-finite model probe loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimized += labels.numel()
        last_loss = float(loss.detach())
    if optimized != MODEL_PROBE_OPTIMIZED_TOKENS:
        raise SweepError("model probe optimized-token ledger drift")
    final = _eval_model(model, adapter)
    return {
        "seed": SEED,
        "training_protocol": {
            "optimizer": "AdamW",
            "learning_rate": 3e-4,
            "betas": [0.9, 0.95],
            "weight_decay": 0.0,
            "gradient_clip_norm": 1.0,
            "batch_size": MODEL_PROBE_BATCH,
            "sequence_length": MODEL_PROBE_SEQUENCE,
            "steps": MODEL_PROBE_STEPS,
            "optimized_loss_tokens": optimized,
            "precision": "fp32",
            "paid_compute": False,
        },
        "model": spec.to_dict(),
        "model_identity_sha256": spec.identity_sha256(),
        "parameters": spec.parameter_count(),
        "target_parameters": MODEL_PROBE_TARGET_PARAMETERS,
        "parameter_delta": spec.parameter_count() - MODEL_PROBE_TARGET_PARAMETERS,
        "last_train_loss": last_loss,
        "initial": initial,
        **final,
    }


def run(*, source_sha: str, output_dir: Path) -> dict[str, Any]:
    if _git_head() != source_sha:
        raise SweepError("checkout HEAD does not match requested source SHA")
    if platform.python_version() != "3.11.16":
        raise SweepError("TOK-37 must run under purpose-profile CPython 3.11.16")
    if MODEL_PROBE_STEPS * MODEL_PROBE_BATCH * (MODEL_PROBE_SEQUENCE - 1) != MODEL_PROBE_OPTIMIZED_TOKENS:
        raise SweepError("model probe token budget is not exactly divisible")
    corpus = _corpus_contract()
    environment = _purpose_environment_contract()
    texts = tuple(text for _, text in TRAIN_RECORDS)
    results: list[dict[str, Any]] = []
    adapters: dict[int, Any] = {}

    for requested in REQUESTED_GRID:
        if requested < 257:
            results.append(
                {
                    "requested_vocab_size": requested,
                    "status": "INFEASIBLE_BY_INCUMBENT_CONTRACT",
                    "reason": "ByteLevel alphabet 256 plus mandatory <unk> requires requested_vocab_size >= 257",
                }
            )
            continue
        manifest = _manifest(corpus, requested)
        first = train_hf_tokenizer(manifest, texts)
        second = train_hf_tokenizer(manifest, texts)
        first_artifact = _artifact_dict(first)
        second_artifact = _artifact_dict(second)
        if first_artifact != second_artifact:
            raise SweepError(f"requested {requested}: repeated tokenizer artifact identity drift")
        if first.vocab_size != second.vocab_size:
            raise SweepError(f"requested {requested}: repeated actual vocabulary drift")
        train_unknowns = _training_unknowns(first) + _training_unknowns(second)
        if train_unknowns:
            raise SweepError(f"requested {requested}: unintended unknown token in training corpus")
        first_metrics = _probe_metrics(first)
        second_metrics = _probe_metrics(second)
        if first_metrics != second_metrics:
            raise SweepError(f"requested {requested}: repeated held-out encoding drift")
        artifact_a = output_dir / "tokenizers" / f"bpe-r{requested}-actual{first.vocab_size}-a.json"
        artifact_b = output_dir / "tokenizers" / f"bpe-r{requested}-actual{first.vocab_size}-b.json"
        hash_a = _serialize_adapter(first, artifact_a)
        hash_b = _serialize_adapter(second, artifact_b)
        if hash_a != hash_b or artifact_a.read_bytes() != artifact_b.read_bytes():
            raise SweepError(f"requested {requested}: repeated serialized tokenizer bytes drift")
        result = {
            "requested_vocab_size": requested,
            "status": "PASS",
            "training_manifest_sha256": manifest.sha256,
            "artifact": first_artifact,
            "repeat_artifact": second_artifact,
            "repeat_training_identity_exact": True,
            "serialized_artifacts": {
                "first": {"path": artifact_a.as_posix(), "sha256": hash_a},
                "repeat": {"path": artifact_b.as_posix(), "sha256": hash_b},
                "byte_identical": True,
            },
            "training_round_trip_exact": True,
            "training_unknown_tokens": 0,
            "held_out": first_metrics,
            "parameter_tax": _parameter_tax(first.vocab_size),
        }
        results.append(result)
        adapters[requested] = first

    selected = _prefilter_selection(results, limit=3)
    for result in results:
        requested = int(result["requested_vocab_size"])
        if requested in selected and result["status"] == "PASS":
            result["model_probe"] = _model_probe(adapters[requested])

    model_results = [
        result for result in results if result["status"] == "PASS" and "model_probe" in result
    ]
    if len(model_results) < 2:
        raise SweepError("fewer than two tokenizer candidates reached the model probe")
    for result in model_results:
        result["model_probe"]["selection_metrics"] = {
            "held_out_tokens": result["held_out"]["tokens"],
            "actual_vocab_size": result["artifact"]["actual_vocab_size"],
            "vocabulary_share_100k": result["parameter_tax"]["100K"]["embedding_share"],
        }

    provisional = min(
        model_results,
        key=lambda result: (
            float(result["model_probe"]["aggregate_bpb"]),
            max(
                float(result["model_probe"]["by_group"][name]["bpb"])
                for name in ("uk", "en", "code")
            ),
            float(result["parameter_tax"]["100K"]["embedding_share"]),
            int(result["held_out"]["tokens"]),
            int(result["artifact"]["actual_vocab_size"]),
        ),
    )
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "source_sha": source_sha,
        "corpus": corpus,
        "purpose_environment": environment,
        "grid": list(REQUESTED_GRID),
        "grid_note": "256 is retained as an explicit infeasible control; 257 is the legal ByteLevel+<unk> floor.",
        "fixed_probe_set": [asdict(probe) for probe in PROBES],
        "probe_used_for_tokenizer_training": False,
        "results": results,
        "prefilter_model_probe_requested_vocab_sizes": selected,
        "tokenizer_parameter_pareto_requested_vocab_sizes": _pareto(
            results, include_model_bpb=False
        ),
        "model_informed_pareto_requested_vocab_sizes": _pareto(
            model_results, include_model_bpb=True
        ),
        "provisional_candidate": {
            "requested_vocab_size": provisional["requested_vocab_size"],
            "actual_vocab_size": provisional["artifact"]["actual_vocab_size"],
            "artifact_config_sha256": provisional["artifact"]["config_sha256"],
            "vocab_sha256": provisional["artifact"]["vocab_sha256"],
            "aggregate_model_probe_bpb": provisional["model_probe"]["aggregate_bpb"],
            "status": "PROVISIONAL_DATA10_MECHANICS_ONLY_RETEST_ON_REPRESENTATIVE_CORPUS",
            "selection_rule": (
                "min aggregate model-probe BPB; then worst UK/EN/code BPB; then 100K "
                "embedding share; then held-out token count; then actual vocabulary size"
            ),
        },
        "truth_boundary": {
            "representative_corpus_available": False,
            "representative_corpus_claimed": False,
            "current_corpus_project_authored_synthetic": True,
            "tokenizer_frozen": False,
            "canonical_s0_unchanged": True,
            "no_benchmark_or_test_optimization": True,
            "no_paid_compute": True,
            "model_probe_is_fixture_scoped": True,
        },
    }
    payload["evidence_sha256"] = _self_hash(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    args = parser.parse_args()
    payload = run(source_sha=args.source_sha, output_dir=args.artifact_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "evidence_sha256": payload["evidence_sha256"],
                "provisional_candidate": payload["provisional_candidate"],
                "model_probe_candidates": payload[
                    "prefilter_model_probe_requested_vocab_sizes"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
