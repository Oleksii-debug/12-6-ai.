from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from twelve_six.data.multilingual_pretraining import (
    MultilingualDataError,
    PretrainingRecord,
    admit_for_pretraining,
    assert_no_cross_split_overlap,
    build_token_budget_mixture,
    corpus_requirements,
    default_token_budget_strata,
    detect_language,
    replay_schedule,
    strict_normalize_utf8,
    tokenizer_cost,
)
from twelve_six.model import ModelSpec, TwelveSixDecoder, count_trainable_parameters
from twelve_six.packing import PACKING_CONFIG_HASH, TextRecord, iter_packed_examples
from twelve_six.tokenization import BYTE_TOKENIZER_HASH, BYTE_VOCAB_HASH, ByteTokenizer


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _record(
    *,
    record_id: str,
    text: str,
    language_hint: str | None,
    modality: str = "natural",
    split: str = "train",
    source_purpose: str = "pretraining",
) -> PretrainingRecord:
    return PretrainingRecord(
        record_id=record_id,
        source_id=f"project::{modality}",
        source_version="v1",
        source_manifest_sha256=_sha(f"manifest:{modality}"),
        split=split,
        source_purpose=source_purpose,
        modality=modality,  # type: ignore[arg-type]
        text=text,
        language_hint=language_hint,
        external=False,
        project_authored_synthetic=True,
    )


def test_language_evidence_distinguishes_uk_en_code_and_rejects_broken_unicode() -> None:
    ukrainian = (
        "Українська мова зберігає літери і, ї, є та ґ; ці дані потрібні для "
        "перевірки відмінків, закінчень і морфологічної різноманітності."
    )
    english = (
        "The training data for this model contains English sentences and these "
        "records are used only for base language modeling."
    )
    code = "def add(left: int, right: int) -> int:\n    return left + right\n"

    assert detect_language(ukrainian, language_hint="uk").label == "uk"
    assert detect_language(english, language_hint="en").label == "en"
    assert detect_language(code, modality="code").label == "code"

    with pytest.raises(MultilingualDataError, match="replacement"):
        strict_normalize_utf8("bad \ufffd text")
    with pytest.raises(MultilingualDataError, match="surrogate"):
        strict_normalize_utf8("bad \ud800 text")


def test_admission_fails_closed_on_heldout_rights_and_contamination() -> None:
    text = (
        "The model training data is project authored and this record exists "
        "only to test the multilingual pretraining admission firewall."
    )
    record = _record(record_id="en-1", text=text, language_hint="en")
    admitted = admit_for_pretraining(record)

    with pytest.raises(MultilingualDataError, match="cannot enter pretraining"):
        admit_for_pretraining(
            _record(
                record_id="en-val",
                text=text + " Validation copy.",
                language_hint="en",
                split="validation",
            )
        )

    with pytest.raises(MultilingualDataError, match="held out"):
        admit_for_pretraining(
            _record(
                record_id="en-benchmark",
                text=text + " Benchmark copy.",
                language_hint="en",
                source_purpose="benchmark",
            )
        )

    external = PretrainingRecord(
        record_id="external-1",
        source_id="external",
        source_version="v1",
        source_manifest_sha256=_sha("external-manifest"),
        split="train",
        source_purpose="pretraining",
        modality="natural",
        text=text,
        language_hint="en",
        external=True,
        rights_status="REVIEW_REQUIRED",
        allows_model_training=False,
    )
    with pytest.raises(MultilingualDataError, match="not explicitly approved"):
        admit_for_pretraining(external)

    with pytest.raises(MultilingualDataError, match="reserved"):
        admit_for_pretraining(
            record,
            reserved_normalized_sha256=frozenset({admitted.normalized_sha256}),
        )
    with pytest.raises(MultilingualDataError, match="held-out fingerprints"):
        assert_no_cross_split_overlap((admitted,), frozenset({admitted.normalized_sha256}))


def test_existing_mixture_plan_and_restart_are_deterministic() -> None:
    strata = default_token_budget_strata(
        {"uk": _sha("uk"), "en": _sha("en"), "code": _sha("code")}
    )
    plan = build_token_budget_mixture(
        strata,
        tokenizer_config_sha256=BYTE_TOKENIZER_HASH,
        tokenizer_vocab_sha256=BYTE_VOCAB_HASH,
        packing_config_sha256=PACKING_CONFIG_HASH,
        seed=126,
        num_shards=32,
    )

    full_counts, full_cursor = replay_schedule(plan, samples=1000)
    first_counts, cursor = replay_schedule(plan, samples=417)
    second_counts, resumed = replay_schedule(plan, samples=583, cursor=cursor)

    assert first_counts + second_counts == full_counts
    assert resumed == full_cursor
    assert resumed.next_sample_index == 1000
    assert set(full_counts) == {"uk", "en", "code"}
    assert 380 <= full_counts["uk"] <= 520
    assert 280 <= full_counts["en"] <= 420
    assert 140 <= full_counts["code"] <= 260


def test_token_cost_and_stage_corpus_requirements_are_explicit() -> None:
    bpe = tokenizer_cost(
        name="bpe-472",
        vocab_size=472,
        observed_tokens=286,
        byte_baseline_tokens=520,
        d_model=128,
    )
    unigram = tokenizer_cost(
        name="unigram-497",
        vocab_size=497,
        observed_tokens=284,
        byte_baseline_tokens=520,
        d_model=128,
    )

    assert bpe.token_reduction_vs_bytes == pytest.approx(0.45)
    assert unigram.token_reduction_vs_bytes == pytest.approx(0.45384615384615384)
    assert bpe.vocabulary_parameters_vs_byte == (472 - 256) * 128

    requirements = corpus_requirements()
    assert requirements["1M"]["total_train_tokens"] == 20_000_000
    assert requirements["10M"]["total_train_tokens"] == 200_000_000
    assert requirements["100M"]["total_train_tokens"] == 2_000_000_000
    assert requirements["100M"]["uk_train_tokens"] == 900_000_000


def test_real_s2_1m_forward_backward_on_multilingual_byte_packing() -> None:
    texts = [
        (
            "uk-1",
            "Українська мова має відмінки, дієвідмінювання і словотвір; ці дані "
            "перевіряють морфологічну різноманітність базового передтренування.",
            "uk",
            "natural",
        ),
        (
            "en-1",
            "The model training data contains English prose and these records test "
            "the deterministic base pretraining path without instruction tuning.",
            "en",
            "natural",
        ),
        (
            "code-1",
            "def stable_hash(value: str) -> str:\n"
            "    return hashlib.sha256(value.encode('utf-8')).hexdigest()\n",
            None,
            "code",
        ),
    ]
    admitted = tuple(
        admit_for_pretraining(
            _record(
                record_id=record_id,
                text=text,
                language_hint=hint,
                modality=modality,
            )
        )
        for record_id, text, hint, modality in texts
    )
    tokenizer = ByteTokenizer()
    packed = list(
        iter_packed_examples(
            (
                TextRecord(record.record_id, record.normalized_text, "train")
                for record in admitted
            ),
            tokenizer,
            expected_split="train",
        )
    )
    assert packed
    assert all(example.split == "train" for example in packed)

    config_path = Path("configs/stages/s2_1m.json")
    stage = json.loads(config_path.read_text(encoding="utf-8"))
    spec = ModelSpec.from_dict(stage["model"])
    torch.manual_seed(126)
    model = TwelveSixDecoder(spec)
    assert count_trainable_parameters(model) == 1_066_112

    example = packed[0]
    input_ids = torch.tensor([example.input_ids], dtype=torch.long)
    labels = torch.tensor([example.labels], dtype=torch.long)
    before = model.token_embedding.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    logits = model(input_ids).logits
    loss = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, spec.vocab_size),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    optimizer.step()
    assert not torch.equal(before, model.token_embedding.weight.detach())
