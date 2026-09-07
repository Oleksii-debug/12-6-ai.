from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch

from twelve_six.inference.batching import (
    BatchGenerationRequest,
    generate_batch,
)
from twelve_six.inference.contracts import GenerationConfig
from twelve_six.inference.generation import generate
from twelve_six.integration.torch_batching import (
    S0TorchBatchedInferenceBackend,
    right_padded_next_token_logits,
)
from twelve_six.model import InitSpec, ModelSpec, TwelveSixDecoder
from twelve_six.tokenization import ByteTokenizer


def _tiny_model() -> TwelveSixDecoder:
    torch.manual_seed(20260825)
    spec = ModelSpec(
        schema_version=1,
        vocab_size=256,
        max_seq_len=24,
        d_model=16,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        head_dim=8,
        d_ff=32,
        rope_rotary_dim=8,
    )
    return TwelveSixDecoder(spec, InitSpec())


def test_right_padding_matches_independent_prefix_forwards() -> None:
    model = _tiny_model()
    rows = [[1, 2], [3, 4, 5, 6, 7], [8], [9, 10, 11]]

    actual, stats = right_padded_next_token_logits(model, rows, padding_token_id=0)
    expected = []
    model.eval()
    with torch.no_grad():
        for row in rows:
            tensor = torch.tensor([row], dtype=torch.long)
            expected.append(model(tensor).logits[0, -1].float().tolist())

    assert len(actual) == len(expected)
    for batch_row, independent_row in zip(actual, expected, strict=True):
        assert batch_row == pytest.approx(independent_row, abs=1e-6, rel=1e-6)
    assert stats.batch_size == 4
    assert stats.logical_input_positions == 11
    assert stats.padded_input_positions == 20
    assert stats.right_padding_positions == 9
    assert stats.input_tensor_bytes == 20 * torch.tensor([], dtype=torch.long).element_size()
    assert stats.output_logits_bytes == 4 * 5 * 256 * 4


def test_real_torch_batch_generation_matches_sequential_greedy() -> None:
    backend = S0TorchBatchedInferenceBackend(_tiny_model(), ByteTokenizer())
    requests = (
        BatchGenerationRequest("a", GenerationConfig(max_new_tokens=5)),
        BatchGenerationRequest("abcd", GenerationConfig(max_new_tokens=2)),
        BatchGenerationRequest("у", GenerationConfig(max_new_tokens=4)),
    )

    expected = tuple(generate(backend, request.prompt, request.config) for request in requests)
    actual = generate_batch(backend, requests, max_batch_size=3)

    assert actual.results == expected
    assert actual.stats.max_batch_observed == 3
    assert actual.stats.model_batch_calls < sum(
        len(result.generated_token_ids) for result in expected
    )
    assert actual.stats.right_padding_positions_scheduled > 0


class _FakeBatchBackend:
    eos_token_id: int | None = None
    max_context_tokens = 32

    def __init__(self) -> None:
        self.calls: list[tuple[int, ...]] = []

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, token_ids: Sequence[int]) -> str:
        return bytes(token_ids).decode("utf-8", errors="replace")

    def next_token_logits(self, input_ids: Sequence[int]) -> Sequence[float]:
        shift = (sum(input_ids) + len(input_ids) * 17) % 11
        return [float(((token_id * 7 + shift) % 13) - 6) for token_id in range(128)]

    def next_token_logits_batch(
        self,
        input_ids: Sequence[Sequence[int]],
    ) -> Sequence[Sequence[float]]:
        self.calls.append(tuple(len(row) for row in input_ids))
        return [self.next_token_logits(row) for row in input_ids]


class _StoppingBatchBackend(_FakeBatchBackend):
    eos_token_id = 67
    max_context_tokens = 8

    def next_token_logits(self, input_ids: Sequence[int]) -> Sequence[float]:
        first = input_ids[0]
        target = {ord("e"): 67, ord("t"): 66, ord("s"): 65}.get(first, 64)
        logits = [-100.0] * 128
        logits[target] = 100.0
        return logits


def test_sampling_seed_is_request_local_and_schedule_independent() -> None:
    sequential_backend = _FakeBatchBackend()
    batched_backend = _FakeBatchBackend()
    requests = (
        BatchGenerationRequest(
            "alpha",
            GenerationConfig(
                max_new_tokens=6,
                sample=True,
                seed=1,
                temperature=0.8,
                top_k=17,
                top_p=0.9,
            ),
        ),
        BatchGenerationRequest(
            "b",
            GenerationConfig(
                max_new_tokens=2,
                sample=True,
                seed=999,
                temperature=1.2,
                top_k=9,
                top_p=0.8,
            ),
        ),
        BatchGenerationRequest(
            "charlie",
            GenerationConfig(
                max_new_tokens=5,
                sample=True,
                seed=1,
                temperature=0.8,
                top_k=17,
                top_p=0.9,
            ),
        ),
    )

    expected = tuple(
        generate(sequential_backend, request.prompt, request.config) for request in requests
    )
    actual = generate_batch(
        batched_backend,
        requests,
        max_batch_size=2,
        max_padding_tokens=3,
    )

    assert actual.results == expected
    assert actual.stats.max_batch_observed <= 2
    assert batched_backend.calls


def test_eos_stops_and_context_finish_independently() -> None:
    sequential_backend = _StoppingBatchBackend()
    batched_backend = _StoppingBatchBackend()
    requests = (
        BatchGenerationRequest("e", GenerationConfig(max_new_tokens=4)),
        BatchGenerationRequest(
            "t",
            GenerationConfig(max_new_tokens=4, stop_token_ids=(66,)),
        ),
        BatchGenerationRequest(
            "s",
            GenerationConfig(max_new_tokens=4, stop_strings=("A",)),
        ),
        BatchGenerationRequest("12345678", GenerationConfig(max_new_tokens=4)),
    )

    expected = tuple(
        generate(sequential_backend, request.prompt, request.config) for request in requests
    )
    actual = generate_batch(batched_backend, requests, max_batch_size=4)

    assert actual.results == expected
    assert tuple(result.stop_reason for result in actual.results) == (
        "eos",
        "stop_token",
        "stop_string",
        "context_limit",
    )
    assert actual.results[2].text == ""
    assert actual.stats.sequences_evaluated == 3


def test_zero_padding_budget_buckets_only_equal_lengths() -> None:
    backend = _FakeBatchBackend()
    requests = (
        BatchGenerationRequest("a", GenerationConfig(max_new_tokens=1)),
        BatchGenerationRequest("b", GenerationConfig(max_new_tokens=1)),
        BatchGenerationRequest("cc", GenerationConfig(max_new_tokens=1)),
    )

    output = generate_batch(
        backend,
        requests,
        max_batch_size=8,
        max_padding_tokens=0,
    )

    assert sorted(backend.calls) == [(1, 1), (2,)]
    assert output.stats.right_padding_positions_scheduled == 0
    assert output.stats.model_batch_calls == 2


@pytest.mark.parametrize(
    ("max_batch_size", "max_padding_tokens"),
    [
        (0, None),
        (True, None),
        (1, -1),
        (1, True),
    ],
)
def test_invalid_batch_policy_fails_closed(
    max_batch_size: int,
    max_padding_tokens: int | None,
) -> None:
    with pytest.raises(ValueError):
        generate_batch(
            _FakeBatchBackend(),
            [BatchGenerationRequest("x")],
            max_batch_size=max_batch_size,
            max_padding_tokens=max_padding_tokens,
        )
