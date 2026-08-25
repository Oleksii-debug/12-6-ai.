from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from twelve_six.scale141_memorization import hashed_training_probe
from twelve_six.tokenization import ByteTokenizer


class IncrementModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.spec = SimpleNamespace(max_seq_len=256)

    def forward(self, input_ids: torch.Tensor):
        batch, length = input_ids.shape
        logits = torch.full((batch, length, 256), -20.0, device=input_ids.device)
        next_ids = (input_ids + 1) % 256
        logits.scatter_(2, next_ids.unsqueeze(-1), 20.0 + self.anchor)
        return SimpleNamespace(logits=logits)


def test_hash_only_probe_emits_no_training_text_and_is_non_mutating() -> None:
    rows = {
        modality: [
            {"record_id": f"{modality}-{index}", "text": "abcdefg"}
            for index in range(6)
        ]
        for modality in ("uk", "en", "code")
    }
    model = IncrementModel()
    before = model.anchor.detach().clone()
    result = hashed_training_probe(
        model,
        ByteTokenizer(),
        rows,
        seed=20260825,
        context_tokens=256,
    )
    assert result["text_emitted"] is False
    assert result["canary_injection"] is False
    assert result["privacy_leakage_claim"] == "NONE"
    assert result["model_non_mutation_passed"] is True
    assert result["sample_count"] == 18
    assert result["exact_short_continuation_rate"] == 1.0
    assert torch.equal(model.anchor.detach(), before)
    for item in result["items"]:
        assert "text" not in item
        assert len(item["content_sha256"]) == 64
