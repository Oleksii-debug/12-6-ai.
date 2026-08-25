"""EVAL-137 source-family generalization study on the bounded real corpus.

This experiment reuses DATA-21/22 rights-gated intake, the RESEARCH41 ~500K
geometry, the byte tokenizer, and the incumbent Trainer. Each source family has
one fixed evaluation partition that is never optimized in any arm, so mixed and
leave-family-out models are compared on identical records. Test-family results
never select or retune a training policy.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from twelve_six.checkpoint import hash_json, sha256_file
from twelve_six.data.source_intake import ELIGIBLE, load_candidate_registry, run_bounded_intake
from twelve_six.model import InitSpec, ModelSpec, TwelveSixDecoder
from twelve_six.packing import TextRecord, batch_examples, collate_rows, iter_packed_examples
from twelve_six.tokenization import ByteTokenizer

from .config import TrainerConfig
from .loss import causal_lm_loss
from .s0_evidence_contract import (
    TOKENIZER_CONFIG_SHA256,
    TOKENIZER_VOCAB_SHA256,
    validate_locked_environment_evidence,
)
from .s1_preflight import REPOSITORY
from .trainer import Trainer

SCHEMA_VERSION = "12-6.eval137-source-generalization.v1"
AUTHORITY = (
    "LOCAL_FREE_TWO_FAMILY_BOUNDED_REAL_SOURCE_GENERALIZATION_EVIDENCE_"
    "NOT_REPRESENTATIVE_CORPUS_OR_GENERAL_CAPABILITY_CLAIM"
)
REAL_SOURCE_REGISTRY = Path("configs/data/external_source_candidates_ua_en_v1.json")
PARAMETER_COUNT = 467_808
SEQUENCE_LENGTH = 64
BATCH_SIZE = 4
LOSS_TOKENS_PER_STEP = BATCH_SIZE * (SEQUENCE_LENGTH - 1)
OPTIMIZER_STEPS = 256
OPTIMIZED_TOKEN_BUDGET = OPTIMIZER_STEPS * LOSS_TOKENS_PER_STEP
CHUNK_TARGET_UTF8_BYTES = 4096
FAMILY_EVAL_MODULUS = 5
SOURCE_FAMILIES = (
    "ua.rada.open-data.laws-texts",
    "en.standardebooks.manual",
)
HOLDOUT_ARM_BY_FAMILY = {
    "ua.rada.open-data.laws-texts": "holdout_rada",
    "en.standardebooks.manual": "holdout_standardebooks",
}
DATA105_STATUS = "NOT_FOUND_IN_LIVE_REPOSITORY_AS_OF_2026-08-26"
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")

# Authored for EVAL-137 and evaluation-only. These controls mimic broad
# register/domain rather than source wording or factual content.
_PROJECT_CONTROLS: dict[str, tuple[str, ...]] = {
    "ua.rada.open-data.laws-texts": (
        "Навчальний нормативний приклад визначає порядок подання умовного документа та не описує реальний правовий акт.",
        "Умовна стаття встановлює, що внутрішній запис набирає чинності після перевірки позначених полів.",
        "Для цього проєктного прикладу термін застосовується лише в межах наведеного абзацу та не має юридичної сили.",
        "Якщо умовна вимога не виконана, тестовий запис повертається на технічну перевірку без правових наслідків.",
        "Цей синтетичний пункт містить нумеровану норму лише для вимірювання мовного моделювання українського формального стилю.",
        "Проєктний регламент описує послідовність дій абстрактного органу без посилань на установи, дати чи чинне законодавство.",
    ),
    "en.standardebooks.manual": (
        "In this project-authored style example, a heading is followed by a concise explanation of the local formatting rule.",
        "A technical note should use one term consistently when the surrounding passage assigns that term a narrow meaning.",
        "For this synthetic editorial control, punctuation is treated as part of the example rather than as a factual claim.",
        "The sample convention prefers a clear label before a short description so the reader can identify the relevant field.",
        "This project-owned prose describes an imaginary formatting practice solely for conditional language-model evaluation.",
        "When a fictional manuscript contains repeated markup, the example asks for consistent structure without citing any external guide.",
    ),
}


class SourceGeneralizationError(ValueError):
    """Raised when the fixed EVAL-137 contract cannot be satisfied."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceGeneralizationError(message)


def fixed_500k_model_spec() -> ModelSpec:
    """Exact 467,808-parameter member of the RESEARCH41 controlled family."""
    spec = ModelSpec(
        schema_version=1,
        vocab_size=256,
        max_seq_len=256,
        d_model=96,
        n_layers=4,
        n_heads=6,
        n_kv_heads=6,
        head_dim=16,
        d_ff=256,
        activation="swiglu",
        norm_kind="rmsnorm",
        norm_placement="pre",
        norm_eps=1e-5,
        position_embedding="rope",
        rope_theta=10_000.0,
        rope_rotary_dim=16,
        attention_bias=False,
        mlp_bias=False,
        attention_dropout=0.0,
        final_norm=True,
        tie_word_embeddings=True,
        lm_head_bias=False,
    )
    _require(spec.parameter_count() == PARAMETER_COUNT, "fixed ~500K geometry drift")
    return spec


def _trainer_config(*, seed: int) -> TrainerConfig:
    return TrainerConfig(
        learning_rate=3e-4,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        eps=1e-8,
        max_steps=OPTIMIZER_STEPS,
        warmup_steps=0,
        scheduler="constant",
        gradient_accumulation_steps=1,
        gradient_clip_norm=1.0,
        precision="fp32",
        seed=seed,
        deterministic_algorithms=True,
        deterministic_warn_only=False,
    )


def _split_oversize_line(line: str, target_bytes: int) -> list[str]:
    if len(line.encode("utf-8")) <= target_bytes:
        return [line]
    pieces: list[str] = []
    current: list[str] = []
    current_bytes = 0
    for char in line:
        char_bytes = len(char.encode("utf-8"))
        if current and current_bytes + char_bytes > target_bytes:
            pieces.append("".join(current))
            current = []
            current_bytes = 0
        current.append(char)
        current_bytes += char_bytes
    if current:
        pieces.append("".join(current))
    return pieces


def chunk_source_text(*, source_id: str, record_id: str, text: str) -> list[TextRecord]:
    """Create deterministic provenance-preserving chunks without family mixing."""
    _require(source_id in SOURCE_FAMILIES, f"unsupported source family: {source_id}")
    _require(bool(text.strip()), f"empty source text: {record_id}")
    logical_lines: list[str] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if stripped:
            logical_lines.extend(_split_oversize_line(stripped, CHUNK_TARGET_UTF8_BYTES))
    _require(bool(logical_lines), f"source text has no non-empty lines: {record_id}")

    chunks: list[str] = []
    current: list[str] = []
    current_bytes = 0
    for line in logical_lines:
        added = len(line.encode("utf-8")) + (1 if current else 0)
        if current and current_bytes + added > CHUNK_TARGET_UTF8_BYTES:
            chunks.append("\n".join(current))
            current = []
            current_bytes = 0
        current.append(line)
        current_bytes += len(line.encode("utf-8")) + (1 if len(current) > 1 else 0)
    if current:
        chunks.append("\n".join(current))

    return [
        TextRecord(
            record_id=f"{source_id}::{record_id}::chunk-{index:05d}",
            text=chunk,
            split="family_pool",
        )
        for index, chunk in enumerate(chunks)
        if len(chunk.encode("utf-8")) >= 2
    ]


def split_family_pool(records: Sequence[TextRecord]) -> tuple[list[TextRecord], list[TextRecord]]:
    """Fixed 80/20 train/evaluation partition declared before any result."""
    _require(len(records) >= 5, "source family yields fewer than five chunks")
    train: list[TextRecord] = []
    evaluation: list[TextRecord] = []
    for index, record in enumerate(records):
        if index % FAMILY_EVAL_MODULUS == 0:
            evaluation.append(
                TextRecord(record_id=record.record_id, text=record.text, split="evaluation")
            )
        else:
            train.append(TextRecord(record_id=record.record_id, text=record.text, split="train"))
    _require(bool(train) and bool(evaluation), "family split must have train and evaluation records")
    _require(
        {record.record_id for record in train}.isdisjoint(
            record.record_id for record in evaluation
        ),
        "family train/evaluation overlap",
    )
    return train, evaluation


# Backward-compatible name retained only for the focused contract test.
def split_seen_family(records: Sequence[TextRecord]) -> tuple[list[TextRecord], list[TextRecord]]:
    train, evaluation = split_family_pool(records)
    validation = [
        TextRecord(record_id=record.record_id, text=record.text, split="validation")
        for record in evaluation
    ]
    return train, validation


def _as_split(records: Sequence[TextRecord], split: str) -> list[TextRecord]:
    return [TextRecord(record_id=record.record_id, text=record.text, split=split) for record in records]


def _load_real_family_pools(
    root: Path, intake_output: Path
) -> tuple[dict[str, list[TextRecord]], dict[str, Any]]:
    registry_path = root / REAL_SOURCE_REGISTRY
    registry, sources = load_candidate_registry(registry_path)
    eligible = [source for source in sources if source.eligibility_status == ELIGIBLE]
    _require(
        tuple(source.source_id for source in eligible) == SOURCE_FAMILIES,
        "eligible source-family set drift",
    )
    _require(
        all(source.rights.allows_model_training is True for source in eligible),
        "training rights drift",
    )
    expected_objects = sum(len(source.acquisition_urls) for source in eligible)
    _require(expected_objects == 3, "EVAL-137 expects exactly three bounded acquisition objects")

    manifest = run_bounded_intake(registry, intake_output)
    counts = manifest.get("record_counts")
    _require(isinstance(counts, Mapping), "intake counts missing")
    _require(counts.get("attempted") == expected_objects, "intake attempt count drift")
    _require(counts.get("accepted") == expected_objects, "every eligible object must be accepted")
    _require(counts.get("rejected") == 0, "real-source intake rejected an object")

    accepted = [
        record
        for record in manifest.get("records", [])
        if isinstance(record, Mapping) and record.get("status") == "ACCEPTED"
    ]
    _require(len(accepted) == expected_objects, "accepted intake record count drift")
    families: dict[str, list[TextRecord]] = {family: [] for family in SOURCE_FAMILIES}
    provenance: list[dict[str, Any]] = []
    for record in accepted:
        source_id = str(record["source_id"])
        _require(source_id in families, f"unexpected accepted source family: {source_id}")
        _require(record.get("allows_model_training") is True, "accepted object lacks training permission")
        text_path = intake_output / str(record["text_path"])
        _require(text_path.is_file(), f"missing accepted text file: {record['id']}")
        text = text_path.read_text(encoding="utf-8").rstrip("\n")
        chunks = chunk_source_text(source_id=source_id, record_id=str(record["id"]), text=text)
        families[source_id].extend(chunks)
        provenance.append(
            {
                "record_id": record["id"],
                "source_id": source_id,
                "source_version": record["source_version"],
                "source_identity_sha256": record["source_identity_sha256"],
                "acquisition_url": record["acquisition_url"],
                "content_sha256": record["content_sha256"],
                "normalized_utf8_bytes": record["normalized_utf8_bytes"],
                "language": record["language"],
                "license_id": record["license_id"],
                "allows_model_training": record["allows_model_training"],
                "chunk_count": len(chunks),
            }
        )

    for family, records in families.items():
        _require(len(records) >= 5, f"{family} is too small for the predeclared family split")
    data_identity = {
        "candidate_registry_identity_sha256": registry["registry_identity_sha256"],
        "intake_manifest_sha256": manifest["manifest_sha256"],
        "family_order": list(SOURCE_FAMILIES),
        "chunk_target_utf8_bytes": CHUNK_TARGET_UTF8_BYTES,
        "family_eval_modulus": FAMILY_EVAL_MODULUS,
        "accepted_objects": provenance,
        "family_chunk_ids": {
            family: [record.record_id for record in records]
            for family, records in families.items()
        },
    }
    return families, {
        "registry_file_sha256": sha256_file(registry_path),
        "candidate_registry_identity_sha256": registry["registry_identity_sha256"],
        "intake_manifest_sha256": manifest["manifest_sha256"],
        "dataset_identity_sha256": hash_json(data_identity),
        "accepted_objects": provenance,
        "family_chunk_counts": {family: len(records) for family, records in families.items()},
        "family_utf8_bytes": {
            family: sum(len(record.text.encode("utf-8")) for record in records)
            for family, records in families.items()
        },
        "record_counts": dict(counts),
        "real_external_source_bytes": True,
        "representative_broad_pretraining_corpus": False,
        "limitation": (
            "Only two rights-approved real source families are currently fetch-eligible, represented by three bounded objects. "
            "The two families also differ in language and domain/register, so the experiment cannot separately identify source-style, language, and domain effects. "
            "This supports two informative leave-family-out arms, not a broad leave-one-source-out corpus study."
        ),
    }


def _tensor_batches(
    records: Sequence[TextRecord], *, split: str, tokenizer: ByteTokenizer, full_only: bool
) -> list[dict[str, torch.Tensor]]:
    examples = tuple(
        iter_packed_examples(
            records,
            tokenizer,
            expected_split=split,
            sequence_length=SEQUENCE_LENGTH,
        )
    )
    _require(bool(examples), f"{split} produced no packed examples")
    batches: list[dict[str, torch.Tensor]] = []
    for group in batch_examples(examples, batch_size=BATCH_SIZE, drop_last=full_only):
        rows = collate_rows(group, target_mode="labels")
        input_ids = torch.tensor(rows["input_ids"], dtype=torch.long)
        labels = torch.tensor(rows["labels"], dtype=torch.long)
        valid = int(labels[:, 1:].ne(-100).sum().item())
        if full_only and (input_ids.shape[0] != BATCH_SIZE or valid != LOSS_TOKENS_PER_STEP):
            continue
        batches.append({"input_ids": input_ids, "labels": labels})
    _require(bool(batches), f"{split} produced no usable batches")
    return batches


def training_trace(batch_count: int, *, seed: int) -> list[int]:
    """Seeded whole-pool permutation; cycle only when the budget exceeds the pool."""
    _require(batch_count > 0, "training batch count must be positive")
    order = list(range(batch_count))
    random.Random(seed).shuffle(order)
    return [order[index % batch_count] for index in range(OPTIMIZER_STEPS)]


@torch.no_grad()
def _evaluate_bpb(
    model: TwelveSixDecoder, batches: Sequence[Mapping[str, torch.Tensor]]
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    weighted_loss = 0.0
    tokens = 0
    try:
        for batch in batches:
            logits = model(batch["input_ids"]).logits
            labels = batch["labels"]
            count = int(labels[:, 1:].ne(-100).sum().item())
            loss = causal_lm_loss(logits, labels)
            _require(bool(torch.isfinite(loss).item()), "evaluation produced non-finite loss")
            weighted_loss += float(loss.item()) * count
            tokens += count
    finally:
        model.train(was_training)
    _require(tokens > 0, "evaluation has zero scoreable tokens")
    cross_entropy = weighted_loss / tokens
    return {
        "cross_entropy_nats": cross_entropy,
        "bpb": cross_entropy / math.log(2.0),
        "scored_byte_tokens": tokens,
    }


def _project_control_batches(
    family: str, tokenizer: ByteTokenizer
) -> list[dict[str, torch.Tensor]]:
    controls = [
        TextRecord(
            record_id=f"eval137-control::{family}::{index:02d}",
            text=text,
            split="control",
        )
        for index, text in enumerate(_PROJECT_CONTROLS[family])
    ]
    return _tensor_batches(controls, split="control", tokenizer=tokenizer, full_only=False)


def _run_arm(
    *,
    arm_id: str,
    train_records: Sequence[TextRecord],
    ordinary_validation_records: Sequence[TextRecord],
    family_evaluation_records: Mapping[str, Sequence[TextRecord]],
    family_seen_during_training: Mapping[str, bool],
    tokenizer: ByteTokenizer,
    spec: ModelSpec,
    init_spec: InitSpec,
    seed: int,
) -> dict[str, Any]:
    train_batches = _tensor_batches(
        train_records,
        split="train",
        tokenizer=tokenizer,
        full_only=True,
    )
    validation_batches = _tensor_batches(
        ordinary_validation_records,
        split="validation",
        tokenizer=tokenizer,
        full_only=False,
    )
    family_batches = {
        family: _tensor_batches(
            records,
            split="evaluation",
            tokenizer=tokenizer,
            full_only=False,
        )
        for family, records in family_evaluation_records.items()
    }
    control_batches = {
        family: _project_control_batches(family, tokenizer)
        for family in SOURCE_FAMILIES
    }

    random.seed(seed)
    torch.manual_seed(seed)
    model = TwelveSixDecoder(spec, init_spec)
    trainer = Trainer(model, _trainer_config(seed=seed), device="cpu")

    def evaluate_all() -> dict[str, Any]:
        return {
            "ordinary_validation": _evaluate_bpb(model, validation_batches),
            "family_evaluation": {
                family: {
                    "source_family_seen_in_training": bool(
                        family_seen_during_training[family]
                    ),
                    **_evaluate_bpb(model, batches),
                }
                for family, batches in family_batches.items()
            },
            "project_authored_same_domain_controls": {
                family: _evaluate_bpb(model, batches)
                for family, batches in control_batches.items()
            },
        }

    initial = evaluate_all()
    _require(
        trainer.tokens_seen == 0 and trainer.optimizer_step == 0,
        "initial evaluation mutated Trainer state",
    )

    trace = training_trace(len(train_batches), seed=seed)
    train_losses: list[float] = []
    for batch_index in trace:
        metrics = trainer.train_microbatch(train_batches[batch_index])
        train_losses.append(float(metrics.loss))
    trainer.assert_checkpoint_safe()
    _require(
        trainer.optimizer_step == OPTIMIZER_STEPS,
        f"{arm_id}: optimizer-step budget drift",
    )
    _require(
        trainer.tokens_seen == OPTIMIZED_TOKEN_BUDGET,
        f"{arm_id}: optimized-token budget drift",
    )

    before_eval_tokens = trainer.tokens_seen
    before_eval_steps = trainer.optimizer_step
    final = evaluate_all()
    _require(
        trainer.tokens_seen == before_eval_tokens,
        f"{arm_id}: evaluation mutated optimized tokens",
    )
    _require(
        trainer.optimizer_step == before_eval_steps,
        f"{arm_id}: evaluation mutated optimizer steps",
    )

    return {
        "arm_id": arm_id,
        "train_record_count": len(train_records),
        "ordinary_validation_record_count": len(ordinary_validation_records),
        "family_evaluation_record_counts": {
            family: len(records)
            for family, records in family_evaluation_records.items()
        },
        "family_seen_during_training": dict(family_seen_during_training),
        "train_batch_count": len(train_batches),
        "unique_train_batches_consumed": len(set(trace)),
        "training_batch_reuse_factor": OPTIMIZER_STEPS / len(train_batches),
        "training_trace_sha256": hash_json({"batch_indices": trace}),
        "optimized_tokens": trainer.tokens_seen,
        "optimizer_steps": trainer.optimizer_step,
        "mean_train_loss": sum(train_losses) / len(train_losses),
        "initial": initial,
        "final": final,
    }


def _comparisons(arms: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_id = {str(arm["arm_id"]): arm for arm in arms}
    mixed = by_id["mixed_real_baseline"]
    result: dict[str, Any] = {}
    for family in SOURCE_FAMILIES:
        heldout = by_id[HOLDOUT_ARM_BY_FAMILY[family]]
        mixed_metric = mixed["final"]["family_evaluation"][family]
        heldout_initial = heldout["initial"]["family_evaluation"][family]
        heldout_final = heldout["final"]["family_evaluation"][family]
        control = heldout["final"]["project_authored_same_domain_controls"][family]
        result[family] = {
            "mixed_seen_family_bpb": mixed_metric["bpb"],
            "leave_family_out_unseen_bpb": heldout_final["bpb"],
            "unseen_family_improvement_from_random_init_bpb": (
                heldout_initial["bpb"] - heldout_final["bpb"]
            ),
            "direct_family_exposure_advantage_bpb": (
                heldout_final["bpb"] - mixed_metric["bpb"]
            ),
            "leave_family_out_project_control_bpb": control["bpb"],
            "evaluation_record_count": heldout["family_evaluation_record_counts"][family],
            "same_evaluation_records_used_in_mixed_and_holdout": True,
        }
    return result


def _report_sha(report: Mapping[str, Any]) -> str:
    payload = dict(report)
    payload.pop("report_sha256", None)
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def validate_source_generalization_report(report: Mapping[str, Any]) -> None:
    _require(report.get("schema_version") == SCHEMA_VERSION, "report schema drift")
    _require(report.get("authority") == AUTHORITY, "report authority drift")
    identity = report.get("identity")
    _require(isinstance(identity, Mapping), "report identity missing")
    _require(identity.get("parameter_count") == PARAMETER_COUNT, "parameter count drift")
    _require(
        identity.get("optimized_token_budget_per_arm") == OPTIMIZED_TOKEN_BUDGET,
        "budget identity drift",
    )
    arms = report.get("arms")
    _require(
        isinstance(arms, list) and len(arms) == 3,
        "expected exactly three preregistered arms",
    )
    _require(
        [arm.get("arm_id") for arm in arms]
        == ["mixed_real_baseline", "holdout_rada", "holdout_standardebooks"],
        "arm order or identity drift",
    )
    expected_flags = {
        "mixed_real_baseline": {family: True for family in SOURCE_FAMILIES},
        "holdout_rada": {
            "ua.rada.open-data.laws-texts": False,
            "en.standardebooks.manual": True,
        },
        "holdout_standardebooks": {
            "ua.rada.open-data.laws-texts": True,
            "en.standardebooks.manual": False,
        },
    }
    family_counts: dict[str, int] | None = None
    for arm in arms:
        arm_id = str(arm.get("arm_id"))
        _require(
            arm.get("optimized_tokens") == OPTIMIZED_TOKEN_BUDGET,
            "matched token budget violated",
        )
        _require(
            arm.get("optimizer_steps") == OPTIMIZER_STEPS,
            "matched optimizer-step budget violated",
        )
        _require(
            arm.get("family_seen_during_training") == expected_flags[arm_id],
            "family exposure flag drift",
        )
        _require(
            0 < int(arm.get("unique_train_batches_consumed", 0)) <= OPTIMIZER_STEPS,
            "invalid unique training batch count",
        )
        counts = arm.get("family_evaluation_record_counts")
        _require(isinstance(counts, Mapping), "family evaluation counts missing")
        normalized_counts = {str(key): int(value) for key, value in counts.items()}
        if family_counts is None:
            family_counts = normalized_counts
        else:
            _require(
                normalized_counts == family_counts,
                "family evaluation records differ across arms",
            )
        final = arm.get("final")
        _require(isinstance(final, Mapping), "final metrics missing")
        ordinary = final.get("ordinary_validation")
        _require(isinstance(ordinary, Mapping), "ordinary validation missing")
        _require(math.isfinite(float(ordinary["bpb"])), "ordinary BPB non-finite")
        family_metrics = final.get("family_evaluation")
        _require(isinstance(family_metrics, Mapping), "family evaluation missing")
        for family in SOURCE_FAMILIES:
            metric = family_metrics.get(family)
            _require(isinstance(metric, Mapping), f"missing family metric: {family}")
            _require(
                math.isfinite(float(metric["bpb"])),
                f"family BPB non-finite: {family}",
            )
        controls = final.get("project_authored_same_domain_controls")
        _require(isinstance(controls, Mapping), "project controls missing")
        for metric in controls.values():
            _require(
                math.isfinite(float(metric["bpb"])),
                "project-control BPB non-finite",
            )
    _require(
        report.get("data105", {}).get("status") == DATA105_STATUS,
        "DATA-105 status drift",
    )
    comparisons = report.get("comparisons")
    _require(isinstance(comparisons, Mapping), "comparisons missing")
    _require(set(comparisons) == set(SOURCE_FAMILIES), "comparison family set drift")
    _require(
        report.get("report_sha256") == _report_sha(report),
        "report self-hash mismatch",
    )


def run_source_generalization(
    root: str | Path,
    *,
    source_sha: str,
    locked_environment_evidence: Mapping[str, Any],
    seed: int = 1337,
    torch_threads: int = 2,
) -> dict[str, Any]:
    """Execute the complete predeclared LOCAL_FREE EVAL-137 study."""
    _require(
        _GIT_SHA.fullmatch(source_sha) is not None,
        "source SHA must be full lowercase Git SHA",
    )
    _require(torch_threads > 0, "torch_threads must be positive")
    root = Path(root).resolve()
    environment = validate_locked_environment_evidence(
        locked_environment_evidence,
        source_sha=source_sha,
    )
    torch.set_num_threads(torch_threads)
    torch.use_deterministic_algorithms(True)
    tokenizer = ByteTokenizer()
    _require(tokenizer.vocab_size == 256, "byte tokenizer vocabulary drift")
    spec = fixed_500k_model_spec()
    init_spec = InitSpec()

    with tempfile.TemporaryDirectory(prefix="eval137-real-intake-") as temp_dir:
        family_pools, data = _load_real_family_pools(root, Path(temp_dir))
        partitions = {
            family: split_family_pool(family_pools[family])
            for family in SOURCE_FAMILIES
        }
        family_eval = {
            family: partitions[family][1]
            for family in SOURCE_FAMILIES
        }
        data["family_train_chunk_counts"] = {
            family: len(partitions[family][0])
            for family in SOURCE_FAMILIES
        }
        data["family_evaluation_chunk_counts"] = {
            family: len(partitions[family][1])
            for family in SOURCE_FAMILIES
        }
        data["family_evaluation_identity_sha256"] = hash_json(
            {
                family: [record.record_id for record in partitions[family][1]]
                for family in SOURCE_FAMILIES
            }
        )

        mixed_train = (
            partitions[SOURCE_FAMILIES[0]][0]
            + partitions[SOURCE_FAMILIES[1]][0]
        )
        mixed_validation = _as_split(
            partitions[SOURCE_FAMILIES[0]][1]
            + partitions[SOURCE_FAMILIES[1]][1],
            "validation",
        )
        rada_train = partitions["en.standardebooks.manual"][0]
        rada_validation = _as_split(
            partitions["en.standardebooks.manual"][1],
            "validation",
        )
        se_train = partitions["ua.rada.open-data.laws-texts"][0]
        se_validation = _as_split(
            partitions["ua.rada.open-data.laws-texts"][1],
            "validation",
        )

        arms = [
            _run_arm(
                arm_id="mixed_real_baseline",
                train_records=mixed_train,
                ordinary_validation_records=mixed_validation,
                family_evaluation_records=family_eval,
                family_seen_during_training={family: True for family in SOURCE_FAMILIES},
                tokenizer=tokenizer,
                spec=spec,
                init_spec=init_spec,
                seed=seed,
            ),
            _run_arm(
                arm_id="holdout_rada",
                train_records=rada_train,
                ordinary_validation_records=rada_validation,
                family_evaluation_records=family_eval,
                family_seen_during_training={
                    "ua.rada.open-data.laws-texts": False,
                    "en.standardebooks.manual": True,
                },
                tokenizer=tokenizer,
                spec=spec,
                init_spec=init_spec,
                seed=seed,
            ),
            _run_arm(
                arm_id="holdout_standardebooks",
                train_records=se_train,
                ordinary_validation_records=se_validation,
                family_evaluation_records=family_eval,
                family_seen_during_training={
                    "ua.rada.open-data.laws-texts": True,
                    "en.standardebooks.manual": False,
                },
                tokenizer=tokenizer,
                spec=spec,
                init_spec=init_spec,
                seed=seed,
            ),
        ]

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "authority": AUTHORITY,
        "identity": {
            "repository": REPOSITORY,
            "source_sha": source_sha,
            "model_identity_sha256": spec.identity_sha256(),
            "init_identity_sha256": init_spec.identity_sha256(),
            "parameter_count": spec.parameter_count(),
            "tokenizer_config_sha256": TOKENIZER_CONFIG_SHA256,
            "tokenizer_vocab_sha256": TOKENIZER_VOCAB_SHA256,
            "training_config": asdict(_trainer_config(seed=seed)),
            "sequence_length": SEQUENCE_LENGTH,
            "batch_size": BATCH_SIZE,
            "loss_tokens_per_step": LOSS_TOKENS_PER_STEP,
            "optimizer_steps_per_arm": OPTIMIZER_STEPS,
            "optimized_token_budget_per_arm": OPTIMIZED_TOKEN_BUDGET,
            "training_batch_order": (
                "seeded permutation of all packed batches; repeat only if the pool has fewer than 256 batches"
            ),
            "same_initialization_seed_for_all_arms": seed,
            "environment": dict(environment),
        },
        "data": data,
        "preregistration": {
            "source_families": list(SOURCE_FAMILIES),
            "family_definition": "rights-approved DATA-21/22 source_id/provider family",
            "chunk_target_utf8_bytes": CHUNK_TARGET_UTF8_BYTES,
            "family_evaluation_rule": "chunk_index_mod_5_equals_0",
            "family_evaluation_records_never_optimized_in_any_arm": True,
            "held_out_family_rule": "no record from held-out source_id enters optimization",
            "training_batch_order_rule": "seeded whole-pool permutation before applying the matched step budget",
            "selection_or_retuning_from_test_source": False,
            "project_control_source": (
                "EVAL-137 project-authored evaluation-only strings committed before execution"
            ),
            "matched_optimized_tokens": True,
        },
        "data105": {
            "status": DATA105_STATUS,
            "balancing_arm_executed": False,
            "reason": (
                "No live DATA-105 PR, branch, or candidate policy was found during pre-execution repository reconstruction; "
                "EVAL-137 therefore does not fabricate or post-hoc tune a balancing policy."
            ),
        },
        "arms": arms,
        "comparisons": _comparisons(arms),
        "interpretation_boundary": {
            "source_family_count": len(SOURCE_FAMILIES),
            "language_domain_source_confounding": True,
            "language_domain_source_confounding_note": (
                "Rada is Ukrainian legal text while Standard Ebooks is English technical/editorial text. With no third eligible real family, leave-family-out effects cannot be uniquely attributed to source style rather than language or domain."
            ),
            "broad_leave_one_source_out_claim": False,
            "representative_corpus_claim": False,
            "general_language_capability_claim": False,
            "metric_signs": {
                "unseen_family_improvement_from_random_init_bpb": (
                    "positive means training on the other family improved the held-out family BPB"
                ),
                "direct_family_exposure_advantage_bpb": (
                    "positive means mixed training with direct family exposure achieved lower BPB than leave-family-out"
                ),
            },
            "conclusion_rule": (
                "Cross-source transfer is evidenced only by improvement from random initialization on the fixed held-out family records "
                "when that entire family is absent from optimization. The gap to the mixed model quantifies additional benefit from direct "
                "family exposure and may reflect source/style, language, or domain-specific learning. Project-authored controls provide a "
                "register-level check but do not eliminate those confounds. No threshold is selected after observing test-family results."
            ),
        },
    }
    report["report_sha256"] = _report_sha(report)
    validate_source_generalization_report(report)
    return report
