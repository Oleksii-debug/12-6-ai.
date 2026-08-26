from __future__ import annotations

import argparse
import json
import random
from collections.abc import Sequence
from pathlib import Path

import torch

from twelve_six import TwelveSixDecoder, count_trainable_parameters, load_stage_config
from twelve_six.inference import GenerationConfig
from twelve_six.inference.batching import (
    BatchGenerationRequest,
    generate_batch,
    generate_batch_cached,
)
from twelve_six.inference.sampling import sample_token
from twelve_six.inference.static_kv import (
    allocate_static_kv_cache,
    decode_one_with_static_kv_cache,
    prefill_static_kv_cache,
)
from twelve_six.integration.torch_batching import S0TorchBatchedInferenceBackend
from twelve_six.tokenization import ByteTokenizer

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "candidates" / "model341_20m_candidate_a.json"
WORKER = "NEXT100-077-20M-STATICKV-BATCH-STRESS"
MODEL341_HEAD = "e4ff486fd90802fc123bebf60eed4e59196a98df"
NEXT100_009_HEAD = "7e3fc17aa204f647e4493861ce0817a3e7a19e98"
MODEL_SPEC_SHA256 = "fbff24d561a2818453554d58ca23fc6ace3303b078f1935a8576c4565bd92441"
STATIC_KV_BLOB = "e4e8cf3746cbc7fc1e43f7c08b088b7df12e268b"
EXPECTED_PARAMETERS = 20_613_440
BATCH_SIZES = (1, 2, 4)
SEED = 77_341
ATOL = 1e-6
RTOL = 1e-6


class _DynamicBatchedBackend:
    """Expose the retained dynamic cache through the cached-batch protocol for parity."""

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


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach().float() - right.detach().float()).abs().max())


def _assert_close(left: torch.Tensor, right: torch.Tensor) -> float:
    torch.testing.assert_close(left, right, atol=ATOL, rtol=RTOL)
    return _max_abs(left, right)


def _expected_static_bytes(model: TwelveSixDecoder, batch_size: int) -> int:
    return (
        2
        * model.spec.n_layers
        * batch_size
        * model.spec.n_kv_heads
        * model.spec.max_seq_len
        * model.spec.head_dim
        * 4
    )


def _expected_dynamic_bytes(
    model: TwelveSixDecoder,
    *,
    batch_size: int,
    sequence_length: int,
) -> int:
    return (
        2
        * model.spec.n_layers
        * batch_size
        * model.spec.n_kv_heads
        * sequence_length
        * model.spec.head_dim
        * 4
    )


def _rows(batch_size: int, *, sequence_length: int, offset: int) -> tuple[tuple[int, ...], ...]:
    rows: list[tuple[int, ...]] = []
    for row_index in range(batch_size):
        base = offset + row_index * 17
        rows.append(
            tuple(
                (base + token_index) % 256
                for token_index in range(sequence_length)
            )
        )
    return tuple(rows)


def _parity_capacity(
    model: TwelveSixDecoder,
    backend: S0TorchBatchedInferenceBackend,
    batch_size: int,
) -> dict[str, object]:
    rows = _rows(batch_size, sequence_length=8, offset=11)
    running = [list(row) for row in rows]
    decode_steps = 3

    with (
        backend.begin_generation_batch(rows) as static_session,
        backend.begin_dynamic_generation_batch(rows) as dynamic_session,
    ):
        static_logits = torch.tensor(static_session.next_token_logits_batch())
        dynamic_logits = torch.tensor(dynamic_session.next_token_logits_batch())
        stateless_logits = torch.tensor(backend.next_token_logits_batch(rows))
        max_static_dynamic = _assert_close(static_logits, dynamic_logits)
        max_static_stateless = _assert_close(static_logits, stateless_logits)

        storage = static_session.cache_storage_signature
        static_before = static_session.cache_bytes
        dynamic_before = dynamic_session.cache_bytes
        expected_static = _expected_static_bytes(model, batch_size)
        assert static_before == expected_static
        assert dynamic_before == _expected_dynamic_bytes(
            model,
            batch_size=batch_size,
            sequence_length=8,
        )

        for step_index in range(decode_steps):
            token_ids = tuple(
                (80 + step_index * 13 + row_index) % model.spec.vocab_size
                for row_index in range(batch_size)
            )
            static_session.append_batch(token_ids)
            dynamic_session.append_batch(token_ids)
            for row, token_id in zip(running, token_ids, strict=True):
                row.append(token_id)

            static_logits = torch.tensor(static_session.next_token_logits_batch())
            dynamic_logits = torch.tensor(dynamic_session.next_token_logits_batch())
            stateless_logits = torch.tensor(backend.next_token_logits_batch(running))
            max_static_dynamic = max(
                max_static_dynamic,
                _assert_close(static_logits, dynamic_logits),
            )
            max_static_stateless = max(
                max_static_stateless,
                _assert_close(static_logits, stateless_logits),
            )
            assert static_session.cache_storage_signature == storage
            assert static_session.cache_bytes == static_before

        static_after = static_session.cache_bytes
        dynamic_after = dynamic_session.cache_bytes
        storage_stable = static_session.cache_storage_signature == storage

    expected_dynamic_after = _expected_dynamic_bytes(
        model,
        batch_size=batch_size,
        sequence_length=8 + decode_steps,
    )
    assert dynamic_after == expected_dynamic_after
    assert static_after == static_before
    assert storage_stable

    return {
        "stateless_parity": True,
        "dynamic_parity": True,
        "max_abs_static_vs_dynamic": max_static_dynamic,
        "max_abs_static_vs_stateless": max_static_stateless,
        "initial_sequence_length": 8,
        "decode_steps": decode_steps,
        "static_physical_bytes": static_before,
        "static_physical_bytes_after_decode": static_after,
        "static_backing_growth_bytes": static_after - static_before,
        "storage_identity_stable": storage_stable,
        "dynamic_bytes_before": dynamic_before,
        "dynamic_bytes_after": dynamic_after,
        "dynamic_growth_bytes": dynamic_after - dynamic_before,
        "dynamic_growth_per_decode_step_bytes": (
            (dynamic_after - dynamic_before) // decode_steps
        ),
    }


def _reset_reuse(
    model: TwelveSixDecoder,
    backend: S0TorchBatchedInferenceBackend,
    batch_size: int,
) -> dict[str, object]:
    initial_rows = _rows(batch_size, sequence_length=4, offset=3)
    second_rows = _rows(batch_size, sequence_length=5, offset=47)
    third_rows = _rows(batch_size, sequence_length=3, offset=101)

    with backend.begin_generation_batch(initial_rows) as static_session:
        storage = static_session.cache_storage_signature
        physical = static_session.cache_bytes
        assert physical == _expected_static_bytes(model, batch_size)

        max_static_dynamic = 0.0
        max_static_stateless = 0.0
        for rows in (second_rows, third_rows):
            static_session.reset_batch(rows)
            static_logits = torch.tensor(static_session.next_token_logits_batch())
            stateless_logits = torch.tensor(backend.next_token_logits_batch(rows))
            max_static_stateless = max(
                max_static_stateless,
                _assert_close(static_logits, stateless_logits),
            )
            with backend.begin_dynamic_generation_batch(rows) as dynamic_session:
                dynamic_logits = torch.tensor(dynamic_session.next_token_logits_batch())
                max_static_dynamic = max(
                    max_static_dynamic,
                    _assert_close(static_logits, dynamic_logits),
                )
            assert static_session.cache_storage_signature == storage
            assert static_session.cache_bytes == physical

        storage_stable = static_session.cache_storage_signature == storage

    assert storage_stable
    return {
        "reset": True,
        "reuse": True,
        "storage_identity_stable": storage_stable,
        "physical_bytes": physical,
        "physical_growth_bytes": 0,
        "max_abs_static_vs_dynamic_after_reuse": max_static_dynamic,
        "max_abs_static_vs_stateless_after_reuse": max_static_stateless,
    }


def _overflow_isolation(model: TwelveSixDecoder, batch_size: int) -> dict[str, object]:
    capacity = 4
    rows = _rows(batch_size, sequence_length=capacity, offset=7)
    tensor = torch.tensor(rows, dtype=torch.long)
    cache = allocate_static_kv_cache(model, batch_size=batch_size, capacity=capacity)
    prefill_static_kv_cache(model, tensor, cache)

    storage = cache.storage_signature
    physical = cache.allocated_bytes
    lengths_before = list(cache.valid_lengths)
    snapshots = tuple(
        (layer.key.clone(), layer.value.clone())
        for layer in cache.layers
    )
    next_ids = torch.tensor(
        [[(151 + row_index) % model.spec.vocab_size] for row_index in range(batch_size)],
        dtype=torch.long,
    )

    try:
        decode_one_with_static_kv_cache(model, next_ids, cache)
    except ValueError as error:
        message = str(error)
    else:
        raise AssertionError("expected static KV overflow rejection")

    assert cache.valid_lengths == lengths_before
    assert cache.storage_signature == storage
    assert cache.allocated_bytes == physical
    for layer, (key_before, value_before) in zip(cache.layers, snapshots, strict=True):
        assert torch.equal(layer.key, key_before)
        assert torch.equal(layer.value, value_before)

    return {
        "overflow_isolated": True,
        "error": message,
        "valid_lengths_unchanged": True,
        "storage_identity_stable": True,
        "backing_bytes_unchanged": True,
        "backing_contents_unchanged": True,
        "capacity": capacity,
        "physical_bytes": physical,
    }


def _sampling_requests(batch_size: int) -> tuple[BatchGenerationRequest, ...]:
    max_new_by_batch = {
        1: (3,),
        2: (1, 4),
        4: (1, 4, 2, 3),
    }
    return tuple(
        BatchGenerationRequest(
            "KV",
            GenerationConfig(
                max_new_tokens=max_new,
                sample=True,
                temperature=0.87,
                top_k=32,
                top_p=0.93,
                seed=77_000 + row_index,
            ),
        )
        for row_index, max_new in enumerate(max_new_by_batch[batch_size])
    )


def _partial_completion_and_sampling(
    backend: S0TorchBatchedInferenceBackend,
    batch_size: int,
    expected_static_bytes: int,
) -> dict[str, object]:
    requests = _sampling_requests(batch_size)
    static_first = generate_batch_cached(backend, requests, max_batch_size=batch_size)
    static_second = generate_batch_cached(backend, requests, max_batch_size=batch_size)
    dynamic = generate_batch_cached(
        _DynamicBatchedBackend(backend),
        requests,
        max_batch_size=batch_size,
    )
    stateless = generate_batch(backend, requests, max_batch_size=batch_size)

    assert static_first.results == static_second.results
    assert static_first.results == dynamic.results
    assert static_first.results == stateless.results
    assert static_first.stats.peak_cache_bytes == expected_static_bytes
    if batch_size == 1:
        assert static_first.stats.retired_row_decode_positions == 0
    else:
        assert static_first.stats.retired_row_decode_positions > 0

    return {
        "partial_row_completion": True,
        "seeded_sampling_deterministic": True,
        "static_repeat_parity": True,
        "static_dynamic_generation_parity": True,
        "static_stateless_generation_parity": True,
        "retired_row_decode_positions": static_first.stats.retired_row_decode_positions,
        "scheduled_decode_positions": static_first.stats.scheduled_decode_positions,
        "logical_decode_positions": static_first.stats.logical_decode_positions,
        "peak_static_cache_bytes": static_first.stats.peak_cache_bytes,
        "generated_token_ids": [
            list(result.generated_token_ids)
            for result in static_first.results
        ],
    }


def _find_eos_seed_set(
    logits: Sequence[float],
    batch_size: int,
) -> tuple[int, list[int]]:
    config = GenerationConfig(
        max_new_tokens=1,
        sample=True,
        temperature=0.91,
        top_k=32,
        top_p=0.94,
        seed=0,
    )

    def selected(seed: int) -> int:
        return sample_token(
            logits,
            rng=random.Random(seed),
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
        )

    first_seed = 11_000
    eos_token_id = selected(first_seed)
    seeds = [first_seed]
    candidate = first_seed + 1
    while len(seeds) < batch_size and candidate < first_seed + 20_000:
        if selected(candidate) != eos_token_id:
            seeds.append(candidate)
        candidate += 1
    if len(seeds) != batch_size:
        raise AssertionError("failed to find deterministic per-row EOS seed set")
    return eos_token_id, seeds


def _eos_per_row(
    backend: S0TorchBatchedInferenceBackend,
    batch_size: int,
) -> dict[str, object]:
    prompt = "E"
    logits = backend.next_token_logits(backend.encode(prompt))
    eos_token_id, seeds = _find_eos_seed_set(logits, batch_size)
    requests = tuple(
        BatchGenerationRequest(
            prompt,
            GenerationConfig(
                max_new_tokens=1,
                sample=True,
                temperature=0.91,
                top_k=32,
                top_p=0.94,
                seed=seed,
            ),
        )
        for seed in seeds
    )

    original_eos = backend.eos_token_id
    backend.eos_token_id = eos_token_id
    try:
        static = generate_batch_cached(backend, requests, max_batch_size=batch_size)
        dynamic = generate_batch_cached(
            _DynamicBatchedBackend(backend),
            requests,
            max_batch_size=batch_size,
        )
        stateless = generate_batch(backend, requests, max_batch_size=batch_size)
    finally:
        backend.eos_token_id = original_eos

    assert static.results == dynamic.results == stateless.results
    assert static.results[0].stop_reason == "eos"
    if batch_size > 1:
        assert any(result.stop_reason != "eos" for result in static.results[1:])

    return {
        "eos_per_row": True,
        "eos_token_id": eos_token_id,
        "row_seeds": seeds,
        "stop_reasons": [result.stop_reason for result in static.results],
        "static_dynamic_stateless_parity": True,
    }


def _stress_batch(
    model: TwelveSixDecoder,
    backend: S0TorchBatchedInferenceBackend,
    batch_size: int,
) -> dict[str, object]:
    expected_static_bytes = _expected_static_bytes(model, batch_size)
    parity = _parity_capacity(model, backend, batch_size)
    reset_reuse = _reset_reuse(model, backend, batch_size)
    overflow = _overflow_isolation(model, batch_size)
    sampling = _partial_completion_and_sampling(
        backend,
        batch_size,
        expected_static_bytes,
    )
    eos = _eos_per_row(backend, batch_size)

    return {
        "batch_size": batch_size,
        "exact_physical_static_bytes": expected_static_bytes,
        "parity_and_backing": parity,
        "reset_reuse": reset_reuse,
        "overflow_isolation": overflow,
        "partial_completion_and_sampling": sampling,
        "eos": eos,
    }


def qualify() -> dict[str, object]:
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    stage = load_stage_config(CONFIG)
    assert stage.stage == "MODEL-341-20M-CANDIDATE-A"
    assert stage.expected_parameters == EXPECTED_PARAMETERS
    assert stage.model.identity_sha256() == MODEL_SPEC_SHA256
    assert stage.model.n_layers == 16
    assert stage.model.n_heads == 10
    assert stage.model.n_kv_heads == 2
    assert stage.model.head_dim == 32
    assert stage.model.max_seq_len == 1024

    model = TwelveSixDecoder(stage.model, stage.init)
    assert count_trainable_parameters(model) == EXPECTED_PARAMETERS
    assert next(model.parameters()).device.type == "cpu"
    assert next(model.parameters()).dtype == torch.float32
    model.eval()

    backend = S0TorchBatchedInferenceBackend(model, ByteTokenizer())
    matrix = {
        str(batch_size): _stress_batch(model, backend, batch_size)
        for batch_size in BATCH_SIZES
    }
    physical_bytes = {
        str(batch_size): _expected_static_bytes(model, batch_size)
        for batch_size in BATCH_SIZES
    }
    assert physical_bytes == {
        "1": 8_388_608,
        "2": 16_777_216,
        "4": 33_554_432,
    }
    assert backend.active_generation_sessions == 0

    return {
        "schema_version": 1,
        "worker": WORKER,
        "qualification": "PASS",
        "pass": True,
        "execution": {
            "profile": "LOCAL_FREE",
            "device": "cpu",
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "torch_intraop_threads": torch.get_num_threads(),
            "training_performed": False,
            "paid_compute": False,
        },
        "authority": {
            "model341_head": MODEL341_HEAD,
            "next100_009_terminal_head": NEXT100_009_HEAD,
            "model_spec_sha256": MODEL_SPEC_SHA256,
            "static_kv_blob": STATIC_KV_BLOB,
            "model_parameters": EXPECTED_PARAMETERS,
        },
        "cache_accounting": {
            "formula": "2*n_layers*batch*n_kv_heads*capacity*head_dim*dtype_bytes",
            "dtype": "float32",
            "dtype_bytes": 4,
            "capacity_tokens": 1024,
            "physical_bytes_by_batch": physical_bytes,
            "dynamic_growth_per_decode_step_bytes_by_batch": {
                "1": 8_192,
                "2": 16_384,
                "4": 32_768,
            },
        },
        "requirements": {
            "batch_sizes_exercised": list(BATCH_SIZES),
            "stateless_parity": True,
            "dynamic_parity": True,
            "storage_identity_stability": True,
            "partial_row_completion": True,
            "eos_per_row": True,
            "reset": True,
            "reuse": True,
            "overflow_isolation": True,
            "seeded_sampling_determinism": True,
            "zero_decode_backing_growth": True,
        },
        "scheduler_boundary": {
            "continuous_batching_added": False,
            "production_scheduler_modified": False,
            "scope": "Existing fixed-row exact-equal-length cache batches only",
        },
        "claim_boundary": {
            "zero_decode_backing_growth_scope": "preallocated K/V tensors only",
            "zero_total_tensor_allocation_claimed": False,
            "learned_quality_claimed": False,
            "hardware_extrapolation_claimed": False,
        },
        "batches": matrix,
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
