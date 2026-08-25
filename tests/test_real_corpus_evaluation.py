from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from twelve_six.data.real_corpus_holdout import (
    build_exclusion_proof,
    build_fixed_tokenizer_no_fit_proof,
    build_immutable_holdout,
    load_heldout_rows,
)
from twelve_six.real_corpus_evaluation import (
    CheckpointDescriptor,
    EvaluationBindings,
    RealCorpusEvaluationError,
    build_ladder_report,
    evaluate_checkpoint,
    verify_ladder_report,
)


@dataclass(frozen=True)
class Identity:
    version: str = "test-byte-v1"
    config_sha256: str = "a" * 64
    vocab_sha256: str = "b" * 64
    vocab_size: int = 256

    def to_dict(self):
        return {
            "version": self.version,
            "config_sha256": self.config_sha256,
            "vocab_sha256": self.vocab_sha256,
            "vocab_size": self.vocab_size,
        }


class ByteTokenizer:
    identity = Identity()
    vocab_size = 256

    def encode(self, text, *, add_bos=False, add_eos=False):
        assert not add_bos and not add_eos
        return list(text.encode("utf-8"))


class Spec:
    vocab_size = 256
    max_seq_len = 64
    attention_dropout = 0.0

    def __init__(self, identity):
        self._identity = identity

    def identity_sha256(self):
        return self._identity


class TransitionLM(torch.nn.Module):
    def __init__(self, spec_id: str):
        super().__init__()
        self.spec = Spec(spec_id)
        self.transition = torch.nn.Parameter(torch.zeros(256, 256))

    def forward(self, input_ids):
        assert not torch.is_grad_enabled()
        return type("Output", (), {"logits": self.transition[input_ids]})()


@dataclass
class TrainerState:
    marker: int
    tensor: torch.Tensor


class FakeTrainer:
    def __init__(self, model, *, tokens_seen):
        self.model = model
        self.micro_step = tokens_seen // 10
        self.optimizer_step = tokens_seen // 10
        self.tokens_seen = tokens_seen
        self._marker = torch.tensor([1.25])

    def state_dict(self):
        return TrainerState(self.optimizer_step, self._marker.clone())


def _records():
    common = {
        "source_kind": "EXTERNAL_REAL",
        "evaluation_use_authority_ref": "D03:rights",
        "provenance_ref": "source://test",
    }
    return [
        {
            **common,
            "record_id": "ua-1",
            "modality": "ua",
            "source_id": "ua-source",
            "source_family": "ua-family",
            "source_version": "1",
            "source_snapshot_sha256": "1" * 64,
            "text": "аааааааааа",
        },
        {
            **common,
            "record_id": "ua-2",
            "modality": "ua",
            "source_id": "ua-source",
            "source_family": "ua-family",
            "source_version": "1",
            "source_snapshot_sha256": "1" * 64,
            "text": "бббббббббб",
        },
        {
            **common,
            "record_id": "en-1",
            "modality": "en",
            "source_id": "en-source",
            "source_family": "en-family",
            "source_version": "1",
            "source_snapshot_sha256": "2" * 64,
            "text": "aaaaaaaaaa",
        },
        {
            **common,
            "record_id": "en-2",
            "modality": "en",
            "source_id": "en-source",
            "source_family": "en-family",
            "source_version": "1",
            "source_snapshot_sha256": "2" * 64,
            "text": "bbbbbbbbbb",
        },
        {
            **common,
            "record_id": "code-1",
            "modality": "code",
            "source_id": "code-source",
            "source_family": "code-family",
            "source_version": "1",
            "source_snapshot_sha256": "3" * 64,
            "text": "xxxxxxxxxx",
        },
        {
            **common,
            "record_id": "code-2",
            "modality": "code",
            "source_id": "code-source",
            "source_family": "code-family",
            "source_version": "1",
            "source_snapshot_sha256": "3" * 64,
            "text": "yyyyyyyyyy",
        },
    ]


def _suite(tmp_path: Path):
    manifest = build_immutable_holdout(
        _records(),
        tmp_path / "heldout",
        suite_name="eval131-test",
        evaluation_corpus_identity_sha256="4" * 64,
        benchmark_registry_sha256="5" * 64,
        decontamination_reference_bundle_sha256="6" * 64,
        decontamination_report_sha256="7" * 64,
    )
    manifest, rows = load_heldout_rows(tmp_path / "heldout")
    training = build_exclusion_proof(
        [{"record_id": "train-only", "text": "not held out"}],
        manifest,
        purpose="MODEL_TRAINING",
        candidate_identity_sha256="8" * 64,
    )
    tokenizer = build_fixed_tokenizer_no_fit_proof(
        manifest, tokenizer_identity_sha256="9" * 64
    )
    bindings = EvaluationBindings(
        training_corpus_identity_sha256="8" * 64,
        training_split_identity_sha256="c" * 64,
        decontamination_report_sha256="7" * 64,
        training_exclusion_proof=training,
        tokenizer_fit_exclusion_proof=tokenizer,
    )
    return manifest, rows, bindings


def _descriptor(kind, spec_id, *, tokens, checkpoint_char, label):
    return CheckpointDescriptor(
        label=label,
        kind=kind,
        checkpoint_identity_sha256=checkpoint_char * 64,
        model_spec_sha256=spec_id,
        initialization_identity_sha256="d" * 64,
        training_run_identity_sha256="e" * 64,
        parameter_count=256 * 256,
        optimized_tokens=tokens,
    )


def _teach_transitions(model, rows):
    with torch.no_grad():
        for row in rows:
            data = row["text"].encode("utf-8")
            for left, right in zip(data, data[1:]):
                model.transition[left, right] = 8.0


def test_no_grad_non_mutating_metrics_bootstrap_and_random_improvement(tmp_path: Path):
    manifest, rows, bindings = _suite(tmp_path)
    spec_id = "f" * 64
    tokenizer = ByteTokenizer()

    random_model = TransitionLM(spec_id)
    random_trainer = FakeTrainer(random_model, tokens_seen=0)
    random_report = evaluate_checkpoint(
        random_model,
        random_trainer,
        tokenizer,
        manifest,
        rows,
        _descriptor("RANDOM_INIT", spec_id, tokens=0, checkpoint_char="1", label="random"),
        bindings,
        context_tokens=32,
        bootstrap_replicates=32,
    )
    assert random_report["non_mutation"]["optimized_tokens_delta"] == 0
    random_ce = random_report["metrics"]["aggregate"]["point"][
        "cross_entropy_nats_per_token"
    ]
    assert random_ce > 5.0
    assert set(random_report["metrics"]["by_modality"]) == {"ua", "en", "code"}
    assert set(random_report["metrics"]["by_source_family"]) == {
        "ua-family",
        "en-family",
        "code-family",
    }

    learned_model = TransitionLM(spec_id)
    _teach_transitions(learned_model, rows)
    learned_trainer = FakeTrainer(learned_model, tokens_seen=1000)
    learned_report = evaluate_checkpoint(
        learned_model,
        learned_trainer,
        tokenizer,
        manifest,
        rows,
        _descriptor("LEARNED", spec_id, tokens=1000, checkpoint_char="2", label="learned"),
        bindings,
        context_tokens=32,
        bootstrap_replicates=32,
    )
    learned_bpb = learned_report["metrics"]["aggregate"]["point"]["bits_per_source_byte"]
    random_bpb = random_report["metrics"]["aggregate"]["point"]["bits_per_source_byte"]
    assert learned_bpb < random_bpb

    ladder = build_ladder_report([learned_report, random_report])
    assert verify_ladder_report(ladder) == ladder["report_sha256"]
    learned = next(
        value for value in ladder["checkpoint_reports"] if value["checkpoint"]["kind"] == "LEARNED"
    )
    improvement = learned["improvement_vs_same_model_random_init"]
    assert improvement["relative_bpb_improvement"] > 0.0
    assert ladder["dashboard_rows"]
    assert all("train_loss" not in key for row in ladder["dashboard_rows"] for key in row)

    repeated = evaluate_checkpoint(
        learned_model,
        learned_trainer,
        tokenizer,
        manifest,
        rows,
        _descriptor("LEARNED", spec_id, tokens=1000, checkpoint_char="2", label="learned"),
        bindings,
        context_tokens=32,
        bootstrap_replicates=32,
    )
    assert repeated["metrics"] == learned_report["metrics"]


def test_mutation_and_identity_drift_fail_closed(tmp_path: Path):
    manifest, rows, bindings = _suite(tmp_path)
    spec_id = "f" * 64
    tokenizer = ByteTokenizer()

    class MutatingLM(TransitionLM):
        def forward(self, input_ids):
            assert not torch.is_grad_enabled()
            with torch.no_grad():
                self.transition[0, 0].add_(1.0)
            return type("Output", (), {"logits": self.transition[input_ids]})()

    model = MutatingLM(spec_id)
    trainer = FakeTrainer(model, tokens_seen=0)
    with pytest.raises(RealCorpusEvaluationError, match="mutated model state"):
        evaluate_checkpoint(
            model,
            trainer,
            tokenizer,
            manifest,
            rows,
            _descriptor("RANDOM_INIT", spec_id, tokens=0, checkpoint_char="1", label="bad"),
            bindings,
            context_tokens=32,
            bootstrap_replicates=8,
        )

    clean = TransitionLM(spec_id)
    clean_trainer = FakeTrainer(clean, tokens_seen=0)
    wrong = _descriptor("RANDOM_INIT", "0" * 64, tokens=0, checkpoint_char="1", label="wrong")
    with pytest.raises(RealCorpusEvaluationError, match="ModelSpec identity mismatch"):
        evaluate_checkpoint(
            clean,
            clean_trainer,
            tokenizer,
            manifest,
            rows,
            wrong,
            bindings,
            context_tokens=32,
            bootstrap_replicates=8,
        )
