from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import torch

from twelve_six import TwelveSixDecoder, count_trainable_parameters, load_stage_config
from twelve_six.inference import GenerationConfig, generate
from twelve_six.inference.batching import BatchGenerationRequest, generate_batch_cached
from twelve_six.inference.sampling import greedy_token
from twelve_six.inference.static_kv import (
    StaticDecoderKVCache,
    allocate_static_kv_cache,
    decode_one_with_static_kv_cache,
    prefill_static_kv_cache,
)
from twelve_six.integration.s0_runtime import S0TorchInferenceBackend
from twelve_six.integration.torch_batching import S0TorchBatchedInferenceBackend
from twelve_six.model import DecoderKVCache
from twelve_six.tokenization import ByteTokenizer

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "candidates" / "model341_20m_candidate_a.json"
WORKER = "PERF-347-20M-STATIC-KV-CAPACITY"
PRIMARY_HEAD = "e4ff486fd90802fc123bebf60eed4e59196a98df"
PRIMARY_MODEL_SHA = "fbff24d561a2818453554d58ca23fc6ace3303b078f1935a8576c4565bd92441"
STATIC_KV_BLOB = "e4e8cf3746cbc7fc1e43f7c08b088b7df12e268b"
EXPECTED_PARAMETERS = 20_613_440
SEED = 347
ATOL = 1e-6
RTOL = 1e-6


class _StatelessBackend:
    def __init__(self, backend: S0TorchInferenceBackend) -> None:
        self.backend = backend
        self.max_context_tokens = backend.max_context_tokens
        self.eos_token_id = backend.eos_token_id

    def encode(self, text: str) -> list[int]:
        return self.backend.encode(text)

    def decode(self, token_ids: Sequence[int]) -> str:
        return self.backend.decode(token_ids)

    def next_token_logits(self, input_ids: Sequence[int]) -> Sequence[float]:
        return self.backend.next_token_logits(input_ids)


class _DynamicBackend(_StatelessBackend):
    def begin_generation(self, input_ids: Sequence[int]) -> object:
        return self.backend.begin_dynamic_generation(input_ids)


class _DynamicBatchedBackend:
    def __init__(self, backend: S0TorchBatchedInferenceBackend) -> None:
        self.backend = backend
        self.max_context_tokens = backend.max_context_tokens
        self.eos_token_id = backend.eos_token_id
        self.cache_row_filler_token_id = backend.cache_row_filler_token_id

    def encode(self, text: str) -> list[int]:
        return self.backend.encode(text)

    def decode(self, token_ids: Sequence[int]) -> str:
        return self.backend.decode(token_ids)

    def next_token_logits(self, input_ids: Sequence[int]) -> Sequence[float]:
        return self.backend.next_token_logits(input_ids)

    def next_token_logits_batch(
        self,
        input_ids: Sequence[Sequence[int]],
    ) -> Sequence[Sequence[float]]:
        return self.backend.next_token_logits_batch(input_ids)

    def begin_generation_batch(self, input_ids: Sequence[Sequence[int]]) -> object:
        return self.backend.begin_dynamic_generation_batch(input_ids)


def _cache_bytes(cache: DecoderKVCache) -> int:
    return sum(
        layer.key.numel() * layer.key.element_size()
        + layer.value.numel() * layer.value.element_size()
        for layer in cache.layers
    )


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach().float() - right.detach().float()).abs().max())


def _assert_close(left: torch.Tensor, right: torch.Tensor) -> float:
    torch.testing.assert_close(left, right, rtol=RTOL, atol=ATOL)
    return _max_abs(left, right)


def _expect_value_error(callable_: object) -> str:
    try:
        callable_()  # type: ignore[operator]
    except ValueError as error:
        return str(error)
    raise AssertionError("expected ValueError")


def _direct_parity_and_capacity(
    model: TwelveSixDecoder,
) -> dict[str, object]:
    prompt = torch.arange(64, dtype=torch.long).remainder(model.spec.vocab_size).unsqueeze(0)
    decode_tokens = [int(value) for value in range(80, 96)]

    static_cache = allocate_static_kv_cache(model, batch_size=1)
    static_storage = static_cache.storage_signature
    static_physical_before = static_cache.allocated_bytes
    static_prompt = prefill_static_kv_cache(model, prompt, static_cache).logits
    dynamic_prompt, dynamic_cache = model.prefill_kv_cache(prompt)
    stateless_prompt = model(prompt).logits

    max_static_dynamic = _assert_close(static_prompt, dynamic_prompt.logits)
    max_static_stateless = _assert_close(static_prompt, stateless_prompt)
    dynamic_before = _cache_bytes(dynamic_cache)
    static_logical_before = static_cache.logical_bytes
    running = prompt

    for token_id in decode_tokens:
        token = torch.tensor([[token_id]], dtype=torch.long)
        static_logits = decode_one_with_static_kv_cache(model, token, static_cache).logits
        dynamic_output, dynamic_cache = model.decode_one_with_kv_cache(token, dynamic_cache)
        running = torch.cat((running, token), dim=1)
        stateless_logits = model(running).logits[:, -1:, :]
        max_static_dynamic = max(
            max_static_dynamic,
            _assert_close(static_logits, dynamic_output.logits),
        )
        max_static_stateless = max(
            max_static_stateless,
            _assert_close(static_logits, stateless_logits),
        )

    dynamic_after = _cache_bytes(dynamic_cache)
    static_physical_after = static_cache.allocated_bytes
    static_logical_after = static_cache.logical_bytes
    storage_stable = static_cache.storage_signature == static_storage

    expected_static = (
        2
        * model.spec.n_layers
        * model.spec.n_kv_heads
        * model.spec.max_seq_len
        * model.spec.head_dim
        * 4
    )
    expected_dynamic_before = (
        2 * model.spec.n_layers * model.spec.n_kv_heads * 64 * model.spec.head_dim * 4
    )
    expected_dynamic_after = (
        2 * model.spec.n_layers * model.spec.n_kv_heads * 80 * model.spec.head_dim * 4
    )

    assert static_physical_before == expected_static == 8_388_608
    assert static_physical_after == static_physical_before
    assert static_logical_before == expected_dynamic_before == 524_288
    assert static_logical_after == expected_dynamic_after == 655_360
    assert dynamic_before == expected_dynamic_before
    assert dynamic_after == expected_dynamic_after
    assert dynamic_after - dynamic_before == 131_072
    assert storage_stable

    return {
        "prompt_tokens": 64,
        "decode_tokens": len(decode_tokens),
        "final_sequence_length": 80,
        "max_abs_static_vs_dynamic": max_static_dynamic,
        "max_abs_static_vs_stateless": max_static_stateless,
        "static_physical_bytes_before": static_physical_before,
        "static_physical_bytes_after": static_physical_after,
        "static_physical_growth_bytes": static_physical_after - static_physical_before,
        "static_logical_bytes_before": static_logical_before,
        "static_logical_bytes_after": static_logical_after,
        "dynamic_bytes_before": dynamic_before,
        "dynamic_bytes_after": dynamic_after,
        "dynamic_growth_bytes": dynamic_after - dynamic_before,
        "dynamic_growth_per_decode_token_bytes": (dynamic_after - dynamic_before)
        // len(decode_tokens),
        "storage_stable": storage_stable,
    }


def _generation_parity(model: TwelveSixDecoder) -> dict[str, object]:
    tokenizer = ByteTokenizer()
    backend = S0TorchInferenceBackend(model, tokenizer)
    stateless = _StatelessBackend(backend)
    dynamic = _DynamicBackend(backend)

    greedy_config = GenerationConfig(max_new_tokens=6, sample=False, seed=17)
    static_greedy = generate(backend, "cache", greedy_config)
    dynamic_greedy = generate(dynamic, "cache", greedy_config)
    stateless_greedy = generate(stateless, "cache", greedy_config)
    assert static_greedy == dynamic_greedy == stateless_greedy

    sample_config = GenerationConfig(
        max_new_tokens=6,
        sample=True,
        temperature=0.85,
        top_k=32,
        top_p=0.92,
        seed=SEED,
    )
    static_sample = generate(backend, "Base", sample_config)
    dynamic_sample = generate(dynamic, "Base", sample_config)
    stateless_sample = generate(stateless, "Base", sample_config)
    assert static_sample == dynamic_sample == stateless_sample

    stop_prompt = "stop"
    first_token = greedy_token(backend.next_token_logits(backend.encode(stop_prompt)))
    stop_config = GenerationConfig(max_new_tokens=5, stop_token_ids=(first_token,))
    static_stop = generate(backend, stop_prompt, stop_config)
    dynamic_stop = generate(dynamic, stop_prompt, stop_config)
    stateless_stop = generate(stateless, stop_prompt, stop_config)
    assert static_stop == dynamic_stop == stateless_stop
    assert static_stop.generated_token_ids == (first_token,)
    assert static_stop.stop_reason == "stop_token"

    backend.eos_token_id = first_token
    eos_stateless = _StatelessBackend(backend)
    eos_dynamic = _DynamicBackend(backend)
    eos_config = GenerationConfig(max_new_tokens=5)
    static_eos = generate(backend, stop_prompt, eos_config)
    dynamic_eos = generate(eos_dynamic, stop_prompt, eos_config)
    stateless_eos = generate(eos_stateless, stop_prompt, eos_config)
    assert static_eos == dynamic_eos == stateless_eos
    assert static_eos.generated_token_ids == (first_token,)
    assert static_eos.stop_reason == "eos"
    backend.eos_token_id = None

    return {
        "greedy_parity": True,
        "greedy_tokens": list(static_greedy.generated_token_ids),
        "sampling_parity": True,
        "sampling_seed": SEED,
        "sampled_tokens": list(static_sample.generated_token_ids),
        "stop_token_parity": True,
        "stop_token_id": first_token,
        "eos_parity": True,
        "eos_token_id": first_token,
    }


def _batching_parity(model: TwelveSixDecoder) -> dict[str, object]:
    tokenizer = ByteTokenizer()
    backend = S0TorchBatchedInferenceBackend(model, tokenizer)
    rows = (
        tuple(range(1, 9)),
        tuple(range(21, 29)),
    )

    with (
        backend.begin_generation_batch(rows) as static_session,
        backend.begin_dynamic_generation_batch(rows) as dynamic_session,
    ):
        static_logits = torch.tensor(static_session.next_token_logits_batch())
        dynamic_logits = torch.tensor(dynamic_session.next_token_logits_batch())
        max_abs = _assert_close(static_logits, dynamic_logits)
        storage = static_session.cache_storage_signature
        static_before = static_session.cache_bytes
        dynamic_before = dynamic_session.cache_bytes

        for pair in ((40, 41), (42, 43), (44, 45), (46, 47)):
            static_session.append_batch(pair)
            dynamic_session.append_batch(pair)
            static_logits = torch.tensor(static_session.next_token_logits_batch())
            dynamic_logits = torch.tensor(dynamic_session.next_token_logits_batch())
            max_abs = max(max_abs, _assert_close(static_logits, dynamic_logits))

        static_after = static_session.cache_bytes
        dynamic_after = dynamic_session.cache_bytes
        storage_stable = static_session.cache_storage_signature == storage

    assert static_before == 16_777_216
    assert static_after == static_before
    assert dynamic_before == 131_072
    assert dynamic_after == 196_608
    assert dynamic_after - dynamic_before == 65_536
    assert storage_stable

    requests = (
        BatchGenerationRequest("aa", GenerationConfig(max_new_tokens=5)),
        BatchGenerationRequest("bb", GenerationConfig(max_new_tokens=2)),
        BatchGenerationRequest(
            "cc",
            GenerationConfig(
                max_new_tokens=4,
                sample=True,
                temperature=0.9,
                top_k=16,
                top_p=0.95,
                seed=41,
            ),
        ),
    )
    static_result = generate_batch_cached(backend, requests, max_batch_size=3)
    dynamic_result = generate_batch_cached(
        _DynamicBatchedBackend(backend),
        requests,
        max_batch_size=3,
    )
    assert static_result.results == dynamic_result.results
    assert static_result.stats.model_batch_calls == dynamic_result.stats.model_batch_calls
    assert (
        static_result.stats.logical_cached_input_positions
        == dynamic_result.stats.logical_cached_input_positions
    )

    return {
        "batch_size": 2,
        "initial_sequence_length": 8,
        "decode_steps": 4,
        "max_abs_static_vs_dynamic": max_abs,
        "static_physical_bytes_before": static_before,
        "static_physical_bytes_after": static_after,
        "static_physical_growth_bytes": static_after - static_before,
        "dynamic_bytes_before": dynamic_before,
        "dynamic_bytes_after": dynamic_after,
        "dynamic_growth_bytes": dynamic_after - dynamic_before,
        "storage_stable": storage_stable,
        "partial_completion_generation_parity": True,
        "sampled_batch_row_parity": True,
    }


def _reset_reuse(model: TwelveSixDecoder) -> dict[str, object]:
    backend = S0TorchInferenceBackend(model, ByteTokenizer())
    with backend.begin_generation((1, 2, 3, 4)) as session:
        storage = session.cache_storage_signature
        physical = session.cache_bytes
        logical_initial = session.logical_cache_bytes
        session.append(5)
        logical_after_append = session.logical_cache_bytes
        session.reset((6, 7))
        logical_after_reset = session.logical_cache_bytes
        assert session.sequence_length == 2
        assert session.cache_storage_signature == storage
        assert session.cache_bytes == physical

    assert physical == 8_388_608
    assert logical_initial == 32_768
    assert logical_after_append == 40_960
    assert logical_after_reset == 16_384
    assert backend.active_generation_sessions == 0
    return {
        "storage_stable": True,
        "physical_bytes": physical,
        "physical_growth_bytes": 0,
        "logical_bytes_initial": logical_initial,
        "logical_bytes_after_append": logical_after_append,
        "logical_bytes_after_reset": logical_after_reset,
        "reset_sequence_length": 2,
        "session_closed_cleanly": True,
    }


def _maximum_context(model: TwelveSixDecoder) -> dict[str, object]:
    prompt = torch.arange(model.spec.max_seq_len - 1, dtype=torch.long)
    prompt = prompt.remainder(model.spec.vocab_size).unsqueeze(0)
    final_token = torch.tensor([[17]], dtype=torch.long)

    static_cache = allocate_static_kv_cache(model, batch_size=1)
    prefill_static_kv_cache(model, prompt, static_cache)
    storage = static_cache.storage_signature
    physical = static_cache.allocated_bytes
    _, dynamic_cache = model.prefill_kv_cache(prompt)

    static_final = decode_one_with_static_kv_cache(model, final_token, static_cache).logits
    dynamic_final, dynamic_cache = model.decode_one_with_kv_cache(final_token, dynamic_cache)
    full_input = torch.cat((prompt, final_token), dim=1)
    stateless_final = model(full_input).logits[:, -1:, :]
    max_static_dynamic = _assert_close(static_final, dynamic_final.logits)
    max_static_stateless = _assert_close(static_final, stateless_final)

    assert static_cache.valid_lengths == [model.spec.max_seq_len]
    assert dynamic_cache.sequence_length == model.spec.max_seq_len
    assert static_cache.storage_signature == storage
    assert static_cache.allocated_bytes == physical

    static_lengths_before = list(static_cache.valid_lengths)
    static_error = _expect_value_error(
        lambda: decode_one_with_static_kv_cache(model, final_token, static_cache)
    )
    dynamic_error = _expect_value_error(
        lambda: model.decode_one_with_kv_cache(final_token, dynamic_cache)
    )
    stateless_error = _expect_value_error(
        lambda: model(torch.cat((full_input, final_token), dim=1))
    )
    assert static_cache.valid_lengths == static_lengths_before
    assert static_cache.storage_signature == storage
    assert static_cache.allocated_bytes == physical

    return {
        "maximum_context_tokens": model.spec.max_seq_len,
        "final_legal_token_parity": True,
        "max_abs_static_vs_dynamic": max_static_dynamic,
        "max_abs_static_vs_stateless": max_static_stateless,
        "static_valid_length_after_final_legal_token": static_cache.valid_lengths[0],
        "static_physical_bytes": physical,
        "static_physical_growth_bytes": 0,
        "storage_stable": True,
        "overflow_rejected_before_static_length_mutation": True,
        "static_overflow_error": static_error,
        "dynamic_overflow_error": dynamic_error,
        "stateless_overflow_error": stateless_error,
    }


def qualify() -> dict[str, object]:
    torch.manual_seed(SEED)
    stage = load_stage_config(CONFIG)
    assert stage.stage == "MODEL-341-20M-CANDIDATE-A"
    assert stage.expected_parameters == EXPECTED_PARAMETERS
    assert stage.model.identity_sha256() == PRIMARY_MODEL_SHA
    assert stage.model.n_layers == 16
    assert stage.model.n_heads == 10
    assert stage.model.n_kv_heads == 2
    assert stage.model.head_dim == 32
    assert stage.model.max_seq_len == 1024

    model = TwelveSixDecoder(stage.model, stage.init)
    assert count_trainable_parameters(model) == EXPECTED_PARAMETERS
    assert next(model.parameters()).dtype == torch.float32
    assert next(model.parameters()).device.type == "cpu"
    model.eval()

    direct = _direct_parity_and_capacity(model)
    generation = _generation_parity(model)
    batching = _batching_parity(model)
    reset = _reset_reuse(model)
    maximum_context = _maximum_context(model)

    return {
        "schema_version": 1,
        "worker": WORKER,
        "qualification": "PASS",
        "pass": True,
        "execution": {
            "scope": "LOCAL_FREE CPU bounded inference qualification",
            "torch": torch.__version__,
            "device": str(next(model.parameters()).device),
            "cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "long_training_performed": False,
        },
        "authority": {
            "primary_model_worker": "MODEL-341-20M-CANDIDATE-A",
            "primary_head": PRIMARY_HEAD,
            "model_spec_sha256": stage.model.identity_sha256(),
            "static_kv_blob": STATIC_KV_BLOB,
        },
        "model": {
            "parameters": EXPECTED_PARAMETERS,
            "vocab_size": stage.model.vocab_size,
            "max_seq_len": stage.model.max_seq_len,
            "d_model": stage.model.d_model,
            "n_layers": stage.model.n_layers,
            "n_heads": stage.model.n_heads,
            "n_kv_heads": stage.model.n_kv_heads,
            "head_dim": stage.model.head_dim,
            "d_ff": stage.model.d_ff,
            "dtype": "float32",
        },
        "cache_accounting": {
            "formula": "2*n_layers*batch*n_kv_heads*length*head_dim*dtype_bytes",
            "dtype_bytes": 4,
            "batch1_full_static_bytes": 8_388_608,
            "batch2_full_static_bytes": 16_777_216,
            "batch1_dynamic_growth_per_token_bytes": 8_192,
            "batch2_dynamic_growth_per_token_bytes": 16_384,
        },
        "checks": {
            "stateless_dynamic_static_parity_and_capacity": direct,
            "generation_greedy_sampling_stop_eos": generation,
            "batching": batching,
            "reset_reuse": reset,
            "maximum_context": maximum_context,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    report = qualify()
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
