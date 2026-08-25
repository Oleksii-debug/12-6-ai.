from __future__ import annotations

import hashlib
import re
from pathlib import Path

import torch
from torch import nn

from tools.eval134_code_suite import training_documents, verify_reservation
from twelve_six.code_diagnostic import load_suite, score_probe, suite_file_sha256, summarize
from twelve_six.data.pipeline import normalize_text
from twelve_six.tokenization.byte import ByteTokenizer

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "eval/reserved/code_diag_v1/probes.jsonl"


class _Spec:
    max_seq_len = 256


class TinyLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.spec = _Spec()
        self.embedding = nn.Embedding(256, 16)
        self.head = nn.Linear(16, 256, bias=False)

    def forward(self, input_ids: torch.Tensor):
        class Output:
            pass

        output = Output()
        output.logits = self.head(self.embedding(input_ids))
        return output


def _state_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def test_suite_is_complete_and_project_synthetic() -> None:
    probes = load_suite(SUITE)
    assert len(probes) == 32
    assert {probe.category for probe in probes} == {
        "balanced_delimiters",
        "indentation_sensitive_continuation",
        "operator_type_syntax",
        "simple_function_call_structure",
        "variable_reuse",
        "string_comment_termination",
        "json_like_structure",
        "language_specific_syntax",
    }
    assert {
        probe.language for probe in probes if probe.category == "language_specific_syntax"
    } == {"python", "sql"}
    combined = "\n".join(
        probe.prefix + choice for probe in probes for choice in probe.choices
    )
    assert "qzv_" in combined
    assert all(token not in combined for token in ("stable_hash", "class Counter", "source_id"))
    assert suite_file_sha256(SUITE) == "87e8085ef7bd9bb6b9755e5e88b2db040226bfd0bffd6696a3ca5f2afb0fe865"


def test_reservation_and_training_overlap_gate() -> None:
    evidence = verify_reservation(ROOT)
    assert evidence["items"] == 32
    assert evidence["candidate_continuations"] == 64
    assert evidence["exact_registry_hashes_verified"] == 64
    assert evidence["normalized_registry_hashes_verified"] == 60
    assert evidence["training_overlap_count"] == 0
    assert evidence["synthetic_identifier_namespace_absent_from_training"] is True


def test_normalized_candidates_are_active_incumbent_forbidden_hashes() -> None:
    import json

    probes = load_suite(SUITE)
    registry = json.loads((ROOT / "data/s0/contamination_registry.json").read_text(encoding="utf-8"))
    forbidden = set(registry["forbidden_normalized_sha256"])
    candidate_hashes = {
        hashlib.sha256(normalize_text(probe.prefix + choice).encode("utf-8")).hexdigest()
        for probe in probes
        for choice in probe.choices
    }
    assert len(candidate_hashes) == 60
    assert candidate_hashes <= forbidden


def test_synthetic_literals_are_unseen_in_current_training_inputs() -> None:
    probes = load_suite(SUITE)
    candidate_text = "\n".join(
        probe.prefix + choice for probe in probes for choice in probe.choices
    )
    literals = set(re.findall(r"\b[78]\d{3}\b", candidate_text))
    assert literals
    training = "\n".join(text for _, text in training_documents(ROOT))
    assert all(literal not in training for literal in literals)


def test_scoring_is_nonmutating_and_byte_normalized() -> None:
    torch.manual_seed(17)
    model = TinyLM()
    model.train()
    before = _state_hash(model)
    probe = load_suite(SUITE)[0]
    score = score_probe(model, ByteTokenizer(), probe)
    assert _state_hash(model) == before
    assert model.training is True
    assert len(score.choices) == 2
    for choice in score.choices:
        assert choice.target_tokens == choice.target_bytes
        assert choice.byte_token_count == choice.target_bytes
        assert choice.tokenizer_tokens_per_source_byte == 1.0
        assert choice.bits_per_source_byte > 0.0


def test_summary_exposes_raw_and_normalized_metrics() -> None:
    torch.manual_seed(23)
    model = TinyLM()
    tokenizer = ByteTokenizer()
    scores = [score_probe(model, tokenizer, probe) for probe in load_suite(SUITE)[:4]]
    summary = summarize(scores)["overall"]
    assert 0.0 <= summary["raw_accuracy"] <= 1.0
    assert 0.0 <= summary["byte_normalized_accuracy"] <= 1.0
    assert summary["mean_correct_bits_per_source_byte"] > 0.0
