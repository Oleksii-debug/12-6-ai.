from __future__ import annotations

from pathlib import Path

import pytest
import torch

from twelve_six.inference import FirstPartyInference, GenerationConfig, generate_token_ids
from twelve_six.inference.cli import build_parser, main as cli_main

ROOT = Path(__file__).resolve().parents[1]
S0_STAGE = ROOT / "configs/stages/s0_10k.json"


class StatelessOnlyBackend:
    eos_token_id = None
    max_context_tokens = 8

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, token_ids: tuple[int, ...] | list[int]) -> str:
        return bytes(token_ids).decode("utf-8", errors="replace")

    def next_token_logits(self, input_ids: tuple[int, ...]) -> list[float]:
        logits = [0.0] * 256
        logits[65] = 1.0
        return logits


def test_first_party_rejects_invalid_prompt_token_types_and_bounds() -> None:
    inference = FirstPartyInference.from_random_init_stage(S0_STAGE, seed=741)
    config = GenerationConfig(max_new_tokens=0)

    with pytest.raises(TypeError, match="must contain integers"):
        inference.generate_token_ids((True,), config, cache_mode="stateless")
    with pytest.raises(TypeError, match="must contain integers"):
        inference.generate_token_ids((65.0,), config, cache_mode="stateless")
    with pytest.raises(ValueError, match="outside vocabulary"):
        inference.generate_token_ids((-1,), config, cache_mode="stateless")
    with pytest.raises(ValueError, match="outside vocabulary"):
        inference.generate_token_ids((256,), config, cache_mode="stateless")
    with pytest.raises(ValueError, match="must be non-empty"):
        inference.generate_token_ids((), config, cache_mode="stateless")


def test_random_init_constructor_does_not_mutate_global_cpu_rng_state() -> None:
    torch.manual_seed(123456)
    before = torch.random.get_rng_state().clone()
    FirstPartyInference.from_random_init_stage(S0_STAGE, seed=741)
    after = torch.random.get_rng_state()
    assert torch.equal(before, after)


def test_invalid_cache_mode_fails_closed_before_generation() -> None:
    with pytest.raises(ValueError, match="cache_mode must be one of"):
        generate_token_ids(
            StatelessOnlyBackend(),
            (65,),
            GenerationConfig(max_new_tokens=1),
            cache_mode="bogus",  # type: ignore[arg-type]
        )


def test_cli_source_modes_are_mutually_exclusive() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(
            [
                "--checkpoint",
                "checkpoint",
                "--random-init-stage",
                str(S0_STAGE),
                "--prompt",
                "A",
            ]
        )
    assert excinfo.value.code == 2


def test_cli_rejects_custom_backend_loader_for_random_init() -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli_main(
            [
                "--random-init-stage",
                str(S0_STAGE),
                "--backend-loader",
                "example.module:factory",
                "--prompt",
                "A",
                "--max-new-tokens",
                "0",
            ]
        )
    assert excinfo.value.code == 2


def test_cli_rejects_empty_raw_token_id_prompt() -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli_main(
            [
                "--random-init-stage",
                str(S0_STAGE),
                "--token-ids",
                " ,  ",
                "--max-new-tokens",
                "0",
            ]
        )
    assert excinfo.value.code == 2
