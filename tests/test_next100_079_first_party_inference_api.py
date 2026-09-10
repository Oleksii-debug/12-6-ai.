from __future__ import annotations

from pathlib import Path

import pytest

from twelve_six.inference import FirstPartyInference, GenerationConfig, generate_token_ids
from twelve_six.inference.cli import main as cli_main
from twelve_six.inference.first_party import _checkpoint_spec
from twelve_six.model import ModelSpec, TwelveSixDecoder, load_stage_config

ROOT = Path(__file__).resolve().parents[1]
PRIMARY_STAGE = ROOT / "configs/candidates/model341_20m_candidate_a.json"
PRIMARY_MODEL_SHA256 = "fbff24d561a2818453554d58ca23fc6ace3303b078f1935a8576c4565bd92441"
LEARNED_10M_MODEL_SHA256 = "61caa5469123e23b9b72fc2024140bfca84c4c480dcb0a7e712ba800a4f22998"


class StatelessOnlyBackend:
    eos_token_id = None
    max_context_tokens = 8

    def __init__(self) -> None:
        self.seen: list[tuple[int, ...]] = []

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, token_ids: tuple[int, ...] | list[int]) -> str:
        return bytes(token_ids).decode("utf-8", errors="replace")

    def next_token_logits(self, input_ids: tuple[int, ...]) -> list[float]:
        self.seen.append(tuple(input_ids))
        logits = [0.0] * 256
        logits[65 + len(self.seen) - 1] = 10.0
        return logits


def test_token_id_generation_has_explicit_stateless_mode() -> None:
    backend = StatelessOnlyBackend()
    result = generate_token_ids(
        backend,
        (120,),
        GenerationConfig(max_new_tokens=2),
        cache_mode="stateless",
    )
    assert result.prompt_token_ids == (120,)
    assert result.generated_token_ids == (65, 66)
    assert backend.seen == [(120,), (120, 65)]


def test_static_mode_fails_clearly_when_backend_has_no_static_session() -> None:
    with pytest.raises(ValueError, match="does not support static-KV generation"):
        generate_token_ids(
            StatelessOnlyBackend(),
            (120,),
            GenerationConfig(max_new_tokens=1),
            cache_mode="static",
        )


def test_primary_model341_library_api_text_token_and_cache_parity() -> None:
    inference = FirstPartyInference.from_random_init_stage(PRIMARY_STAGE, seed=341)
    diagnostics = inference.diagnostics()

    assert type(inference.backend.model) is TwelveSixDecoder
    assert diagnostics["source_kind"] == "random_init"
    assert diagnostics["model_spec_sha256"] == PRIMARY_MODEL_SHA256
    assert diagnostics["parameter_count"] == 20_613_440
    assert diagnostics["max_context_tokens"] == 1024
    assert diagnostics["random_init_seed"] == 341

    model_identity = id(inference.backend.model)
    config = GenerationConfig(max_new_tokens=1)
    stateless = inference.generate_text("A", config, cache_mode="stateless")
    static = inference.generate_token_ids((65,), config, cache_mode="static")

    assert stateless.prompt_token_ids == (65,)
    assert static.prompt_token_ids == (65,)
    assert static.generated_token_ids == stateless.generated_token_ids
    assert static.text == stateless.text
    assert id(inference.backend.model) == model_identity


def test_primary_stage_is_exact_late_bound_model341_authority() -> None:
    stage = load_stage_config(PRIMARY_STAGE)
    assert stage.stage == "MODEL-341-20M-CANDIDATE-A"
    assert stage.expected_parameters == 20_613_440
    assert stage.model.identity_sha256() == PRIMARY_MODEL_SHA256
    assert stage.model.max_seq_len == 1024


def test_current_learned_10m_modelspec_checkpoint_parser_remains_compatible() -> None:
    spec = ModelSpec(
        schema_version=1,
        vocab_size=256,
        max_seq_len=1024,
        d_model=256,
        n_layers=12,
        n_heads=8,
        n_kv_heads=2,
        head_dim=32,
        d_ff=864,
        rope_rotary_dim=32,
    )
    assert spec.parameter_count() == 10_000_640
    assert spec.identity_sha256() == LEARNED_10M_MODEL_SHA256

    manifest = {
        "identity": {
            "model_spec": spec.to_dict(),
            "model_spec_hash": spec.identity_sha256(),
            "parameter_count": spec.parameter_count(),
            "training_config": {"training": {"context_length": 1024}},
        }
    }
    parsed = _checkpoint_spec(manifest)
    assert parsed == spec
    assert parsed.identity_sha256() == LEARNED_10M_MODEL_SHA256


def test_first_party_surface_has_no_chat_semantics() -> None:
    public = {name for name in dir(FirstPartyInference) if not name.startswith("_")}
    assert {"generate_text", "generate_token_ids", "from_checkpoint"} <= public
    assert {"chat", "messages", "roles", "system_prompt", "chat_template"}.isdisjoint(public)


def test_cli_accepts_token_ids_and_emits_plain_text_by_default(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli_main(
        [
            "--random-init-stage",
            str(ROOT / "configs/stages/s0_10k.json"),
            "--init-seed",
            "7",
            "--token-ids",
            "65,66",
            "--max-new-tokens",
            "1",
            "--cache-mode",
            "stateless",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "generation: input=token_ids cache=stateless" in captured.err
    assert not captured.out.lstrip().startswith("{")


def test_cli_rejects_out_of_vocab_token_id(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli_main(
        [
            "--random-init-stage",
            str(ROOT / "configs/stages/s0_10k.json"),
            "--token-ids",
            "256",
            "--max-new-tokens",
            "1",
        ]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert "outside vocabulary [0, 256)" in captured.err
