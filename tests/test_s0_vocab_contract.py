from __future__ import annotations

from pathlib import Path

import torch

from twelve_six import TwelveSixDecoder, load_stage_config

ROOT = Path(__file__).resolve().parents[1]


def test_s0_model_accepts_highest_raw_byte_token_id() -> None:
    stage = load_stage_config(ROOT / "configs" / "stages" / "s0_10k.json")
    assert stage.model.vocab_size == 256
    model = TwelveSixDecoder(stage.model, stage.init).eval()
    input_ids = torch.tensor([[0, 1, 254, 255]], dtype=torch.long)
    logits = model(input_ids).logits
    assert logits.shape == (1, 4, 256)
