"""Project-owned Ukrainian raw-Base conditional-likelihood diagnostics (EVAL-132)."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

DATASET_ID = "eval132-ua-raw-base-v1"
DATASET_SHA256 = "ca8d9c9d97c854127e0209871e8929f19e06ec91f3f19902bc8fda33481691ff"
SOURCE_ID = "project-authored:eval132:ua-raw-base-v1"
SOURCE_IDENTITY_SHA256 = "c55902af66a33b1695af60a4de05ab7973590427db7bddfd3422de64e1b29c13"
D06_REGISTRY_SHA256 = "9033007a7323c64ed9b27c3930de916a41cd5c7efd1dfb673871ca7886a47a31"
RESERVED_REGISTRY_IDENTITY_SHA256 = "f37e6d8a3d37f091287b03f8baf8dd1dcc66bef2a9b4b93b47538f3d65134685"
PHENOMENA = (
    "case_agreement",
    "gender_number_agreement",
    "verb_agreement",
    "common_word_order",
    "negation",
    "prepositions_case",
    "apostrophe_orthography",
    "morphological_continuations",
    "semantic_coherence",
)
CHANCE_PAIR_ACCURACY = 0.5
NEUTRAL_MARGIN_NATS_PER_SOURCE_BYTE = 0.0


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def source_path() -> Path:
    return _repo_root() / "data/evaluation/ua_raw_base_v1/source_rows.json"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_source(path: str | Path | None = None) -> dict[str, Any]:
    value = json.loads(Path(path or source_path()).read_text(encoding="utf-8"))
    if value.get("schema_version") != "12-6.ua-raw-base-source.v1":
        raise ValueError("unsupported EVAL-132 source schema")
    expected = value.get("source_identity_sha256")
    unhashed = dict(value)
    unhashed.pop("source_identity_sha256", None)
    actual = hashlib.sha256(_canonical_json(unhashed).encode("utf-8")).hexdigest()
    if expected != SOURCE_IDENTITY_SHA256 or actual != SOURCE_IDENTITY_SHA256:
        raise ValueError("EVAL-132 source identity mismatch")
    if value.get("source_id") != SOURCE_ID:
        raise ValueError("EVAL-132 source_id mismatch")
    if value.get("copyright_benchmark_copying") is not False:
        raise ValueError("EVAL-132 must remain project-authored")
    if value.get("external_factual_lookup") is not False:
        raise ValueError("EVAL-132 factual-free source invariant failed")
    return value


def generate_items(path: str | Path | None = None) -> list[dict[str, Any]]:
    rows = load_source(path)["rows"]
    if set(rows) != set(PHENOMENA):
        raise ValueError("EVAL-132 phenomenon inventory mismatch")
    items: list[dict[str, Any]] = []
    for phenomenon in PHENOMENA:
        phenomenon_rows = rows[phenomenon]
        if len(phenomenon_rows) != 24:
            raise ValueError(f"{phenomenon} must contain exactly 24 rows")
        for index, row in enumerate(phenomenon_rows, 1):
            if not isinstance(row, list) or len(row) != 4:
                raise ValueError("source rows must be [context, preferred, contrast, subtype]")
            context, preferred, contrast, subtype = row
            if not all(isinstance(value, str) and value for value in row):
                raise ValueError("EVAL-132 source strings must be non-empty")
            if preferred == contrast:
                raise ValueError("preferred and contrast continuations must differ")
            items.append(
                {
                    "item_id": f"ua-v1-{phenomenon}-{index:03d}",
                    "phenomenon": phenomenon,
                    "subtype": subtype,
                    "context": context,
                    "preferred": preferred,
                    "contrast": contrast,
                    "source": "PROJECT_AUTHORED_EVAL132",
                    "language": "uk",
                    "scoring": "conditional_mean_logprob_per_utf8_byte",
                }
            )
    if len(items) != 216 or len({item["item_id"] for item in items}) != 216:
        raise ValueError("EVAL-132 item identity invariant failed")
    actual = dataset_sha256(items)
    if actual != DATASET_SHA256:
        raise ValueError(f"EVAL-132 dataset identity mismatch: {actual}")
    return items


def canonical_jsonl(items: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(_canonical_json(dict(item)) for item in items) + "\n"


def dataset_sha256(items: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(canonical_jsonl(items).encode("utf-8")).hexdigest()


def rendered_variants(items: Sequence[Mapping[str, Any]] | None = None) -> tuple[str, ...]:
    suite = generate_items() if items is None else items
    values: list[str] = []
    for item in suite:
        values.append(str(item["context"]) + str(item["preferred"]))
        values.append(str(item["context"]) + str(item["contrast"]))
    return tuple(values)


def reserved_variant_hashes(items: Sequence[Mapping[str, Any]] | None = None) -> tuple[str, ...]:
    from twelve_six.data.corpus_v01 import norm

    hashes = {
        hashlib.sha256(norm(text, False).encode("utf-8")).hexdigest()
        for text in rendered_variants(items)
    }
    if len(hashes) != 432:
        raise ValueError("EVAL-132 reserved variant inventory must contain 432 unique hashes")
    return tuple(sorted(hashes))


def _model_logits(model: Any, input_ids: torch.Tensor) -> torch.Tensor:
    output = model(input_ids)
    logits = output.logits if hasattr(output, "logits") else output
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise TypeError("model must return [batch, sequence, vocab] logits")
    if logits.shape[-1] < 256:
        raise ValueError("EVAL-132 byte scorer requires vocabulary size >= 256")
    return logits


def _model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _state_sha256(model: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        value = tensor.detach().cpu().contiguous()
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def score_completion(model: Any, context: str, completion: str) -> dict[str, float | int]:
    context_bytes = context.encode("utf-8")
    completion_bytes = completion.encode("utf-8")
    if not context_bytes or not completion_bytes:
        raise ValueError("context and completion must be non-empty")
    combined = context_bytes + completion_bytes
    max_seq_len = int(model.spec.max_seq_len)
    if len(combined) > max_seq_len:
        raise ValueError(f"rendered byte sequence length {len(combined)} exceeds {max_seq_len}")
    input_ids = torch.tensor([list(combined)], dtype=torch.long, device=_model_device(model))
    log_probs = F.log_softmax(_model_logits(model, input_ids), dim=-1)
    first = len(context_bytes)
    values = [
        float(log_probs[0, first + offset - 1, target].item())
        for offset, target in enumerate(completion_bytes)
    ]
    total = math.fsum(values)
    byte_count = len(completion_bytes)
    return {
        "logprob_nats": total,
        "source_bytes": byte_count,
        "mean_logprob_nats_per_source_byte": total / byte_count,
        "conditional_bpb": -total / (math.log(2.0) * byte_count),
        "byte_tokens": byte_count,
        "tokens_per_source_byte": 1.0,
    }


def wilson95(correct: int, total: int) -> tuple[float, float]:
    if total <= 0 or not 0 <= correct <= total:
        raise ValueError("invalid binomial counts")
    z = 1.959963984540054
    p = correct / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denominator
    return center - half, center + half


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    correct = sum(bool(row["correct"]) for row in rows)
    margins = [float(row["margin_nats_per_source_byte"]) for row in rows]
    low, high = wilson95(correct, len(rows))
    return {
        "correct": correct,
        "n": len(rows),
        "accuracy": correct / len(rows),
        "accuracy_wilson95": [low, high],
        "mean_margin_nats_per_source_byte": statistics.fmean(margins),
        "median_margin_nats_per_source_byte": statistics.median(margins),
    }


@torch.no_grad()
def evaluate_model(
    model: Any,
    *,
    label: str,
    source: Mapping[str, Any],
    items: Sequence[Mapping[str, Any]] | None = None,
    optimized_tokens_getter: Callable[[], int] | None = None,
    include_item_rows: bool = True,
) -> dict[str, Any]:
    suite = generate_items() if items is None else list(items)
    before_hash = _state_sha256(model)
    before_tokens = optimized_tokens_getter() if optimized_tokens_getter is not None else None
    was_training = bool(model.training)
    model.eval()
    rows: list[dict[str, Any]] = []
    try:
        for item in suite:
            preferred = score_completion(model, str(item["context"]), str(item["preferred"]))
            contrast = score_completion(model, str(item["context"]), str(item["contrast"]))
            margin = float(preferred["mean_logprob_nats_per_source_byte"]) - float(
                contrast["mean_logprob_nats_per_source_byte"]
            )
            rows.append(
                {
                    "item_id": item["item_id"],
                    "phenomenon": item["phenomenon"],
                    "preferred": preferred,
                    "contrast": contrast,
                    "margin_nats_per_source_byte": margin,
                    "correct": margin > 0.0,
                    "tie": margin == 0.0,
                }
            )
    finally:
        model.train(was_training)
    after_hash = _state_sha256(model)
    after_tokens = optimized_tokens_getter() if optimized_tokens_getter is not None else None
    if before_hash != after_hash:
        raise RuntimeError("evaluation mutated model state")
    if before_tokens is not None and after_tokens != before_tokens:
        raise RuntimeError("evaluation mutated optimized-token counter")
    report: dict[str, Any] = {
        "label": label,
        "source": dict(source),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "model_state_sha256": before_hash,
        "state_unchanged": True,
        "optimized_tokens_delta": 0 if before_tokens is None else after_tokens - before_tokens,
        "overall": _aggregate(rows),
        "by_phenomenon": {
            phenomenon: _aggregate([row for row in rows if row["phenomenon"] == phenomenon])
            for phenomenon in PHENOMENA
        },
        "baseline": {
            "symmetric_pair_choice_chance_accuracy": CHANCE_PAIR_ACCURACY,
            "neutral_margin_nats_per_source_byte": NEUTRAL_MARGIN_NATS_PER_SOURCE_BYTE,
        },
        "scoring": {
            "kind": "raw_base_conditional_likelihood",
            "instruction_following": False,
            "primary": "mean conditional log-likelihood per UTF-8 source byte",
            "secondary": ["raw joint conditional log-likelihood", "conditional BPB"],
            "byte_tokenizer_native": True,
            "torch_no_grad": True,
        },
    }
    if include_item_rows:
        report["items"] = rows
    return report


def validate_reserved_registry(path: str | Path | None = None) -> dict[str, Any]:
    registry_path = Path(path or (_repo_root() / "data/external/reserved_fingerprints.json"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("registry_identity_sha256") != RESERVED_REGISTRY_IDENTITY_SHA256:
        raise ValueError("EVAL-132 reserved registry identity mismatch")
    matching = [entry for entry in registry.get("sets", []) if entry.get("benchmark_id") == "eval132-ua-raw-base"]
    if len(matching) != 1:
        raise ValueError("EVAL-132 reserved registry entry missing or duplicated")
    entry = matching[0]
    if tuple(entry.get("normalized_sha256", ())) != reserved_variant_hashes():
        raise ValueError("EVAL-132 reserved fingerprints differ from generated suite")
    if entry.get("variant_count") != 432 or entry.get("dataset_sha256") != DATASET_SHA256:
        raise ValueError("EVAL-132 reserved registry metadata mismatch")
    return registry
