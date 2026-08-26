from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import resource
import threading
import time
from pathlib import Path

import torch

from twelve_six import TwelveSixDecoder, count_trainable_parameters, load_stage_config
from twelve_six.inference.static_kv import (
    allocate_static_kv_cache,
    decode_one_with_static_kv_cache,
    prefill_static_kv_cache,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "candidates" / "model341_20m_candidate_a.json"
SEED = 341
LENGTHS = (1, 2, 32, 128, 256, 512, 1023, 1024)
REJECT_LENGTH = 1025
RTOL = 1e-5
ATOL = 1e-5


def _current_rss_bytes() -> int:
    """Read current Linux RSS without adding a project dependency."""
    try:
        with Path("/proc/self/statm").open("r", encoding="ascii") as handle:
            rss_pages = int(handle.read().split()[1])
        return rss_pages * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, IndexError):
        # Fallback is a high-water mark rather than current RSS, but remains safe
        # for environments without procfs and does not under-report peak memory.
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


class RSSSampler:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self.peak_bytes = 0
        self.seconds = 0.0

    def _sample(self) -> None:
        while not self._stop.is_set():
            self.peak_bytes = max(self.peak_bytes, _current_rss_bytes())
            time.sleep(0.001)

    def __enter__(self) -> RSSSampler:
        self.peak_bytes = _current_rss_bytes()
        self.started = time.perf_counter()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.seconds = time.perf_counter() - self.started
        self._stop.set()
        self.thread.join()
        self.peak_bytes = max(self.peak_bytes, _current_rss_bytes())


def _rng_sha256() -> str:
    return hashlib.sha256(torch.get_rng_state().cpu().numpy().tobytes()).hexdigest()


def _static_cache_sha256(cache) -> str:
    digest = hashlib.sha256()
    for layer in cache.layers:
        digest.update(layer.key.detach().cpu().contiguous().numpy().tobytes())
        digest.update(layer.value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _dynamic_cache_sha256(cache) -> str:
    digest = hashlib.sha256()
    for layer in cache.layers:
        digest.update(layer.key.detach().cpu().contiguous().numpy().tobytes())
        digest.update(layer.value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _capture_rejection(callable_) -> dict[str, object]:
    try:
        callable_()
    except Exception as exc:  # boundary audit records the production exception verbatim
        return {"rejected": True, "type": type(exc).__name__, "error": str(exc)}
    return {"rejected": False, "type": None, "error": None}


def audit() -> dict[str, object]:
    torch.manual_seed(SEED)
    stage = load_stage_config(CONFIG)
    model = TwelveSixDecoder(stage.model, stage.init).eval()

    tokens = (
        (torch.arange(REJECT_LENGTH, dtype=torch.long) * 73 + 19) % stage.model.vocab_size
    ).view(1, -1)
    static_cache = allocate_static_kv_cache(model, batch_size=1)

    report: dict[str, object] = {
        "worker": "NEXT100-076-20M-CONTEXT-STRESS",
        "scope": "LOCAL_FREE CPU",
        "torch": torch.__version__,
        "seed": SEED,
        "model_spec_sha256": stage.model.identity_sha256(),
        "init_spec_sha256": stage.init.identity_sha256(),
        "parameter_count": count_trainable_parameters(model),
        "lengths": [*LENGTHS, REJECT_LENGTH],
    }

    rope = model.blocks[0].attn.rope
    original_cos_sin = rope.cos_sin
    rope_trace: list[tuple[int, int]] = []

    def traced_cos_sin(seq_len: int, **kwargs):
        rope_trace.append((int(seq_len), int(kwargs.get("position_offset", 0))))
        return original_cos_sin(seq_len, **kwargs)

    rope.cos_sin = traced_cos_sin  # type: ignore[method-assign]

    references: dict[int, torch.Tensor] = {}
    boundaries: list[dict[str, object]] = []
    for length in LENGTHS:
        input_ids = tokens[:, :length]
        gc.collect()
        rss_baseline = _current_rss_bytes()
        trace_start = len(rope_trace)
        with torch.inference_mode(), RSSSampler() as full_sample:
            full = model(input_ids).logits.detach().clone()
        full_trace = rope_trace[trace_start:]

        gc.collect()
        trace_start = len(rope_trace)
        with torch.inference_mode(), RSSSampler() as prefill_sample:
            prefill = prefill_static_kv_cache(model, input_ids, static_cache).logits.detach().clone()
        prefill_trace = rope_trace[trace_start:]

        delta = (full - prefill).abs()
        references[length] = full
        boundaries.append(
            {
                "tokens": length,
                "finite_logits": bool(torch.isfinite(full).all()),
                "finite_prefill_logits": bool(torch.isfinite(prefill).all()),
                "prefill_parity": bool(torch.allclose(full, prefill, rtol=RTOL, atol=ATOL)),
                "prefill_max_abs": float(delta.max()),
                "forward_latency_ms": full_sample.seconds * 1000.0,
                "prefill_latency_ms": prefill_sample.seconds * 1000.0,
                "rss_baseline_bytes": rss_baseline,
                "forward_peak_rss_bytes": full_sample.peak_bytes,
                "prefill_peak_rss_bytes": prefill_sample.peak_bytes,
                "kv_logical_bytes": static_cache.logical_bytes,
                "kv_allocated_bytes": static_cache.allocated_bytes,
                "kv_occupancy": length / stage.model.max_seq_len,
                "forward_rope_trace": full_trace,
                "prefill_rope_trace": prefill_trace,
            }
        )

    static_cache.reset()
    incremental_logits: list[torch.Tensor] = []
    incremental_marks: dict[int, float] = {}
    trace_start = len(rope_trace)
    cumulative_seconds = 0.0
    gc.collect()
    with torch.inference_mode(), RSSSampler() as incremental_sample:
        started = time.perf_counter()
        incremental_logits.append(
            prefill_static_kv_cache(model, tokens[:, :1], static_cache).logits.detach().clone()
        )
        cumulative_seconds += time.perf_counter() - started
        incremental_marks[1] = cumulative_seconds
        for position in range(1, stage.model.max_seq_len):
            started = time.perf_counter()
            incremental_logits.append(
                decode_one_with_static_kv_cache(
                    model, tokens[:, position : position + 1], static_cache
                ).logits.detach().clone()
            )
            cumulative_seconds += time.perf_counter() - started
            length = position + 1
            if length in LENGTHS:
                incremental_marks[length] = cumulative_seconds

    incremental = torch.cat(incremental_logits, dim=1)
    incremental_rope_trace = rope_trace[trace_start:]
    for boundary in boundaries:
        length = int(boundary["tokens"])
        delta = (references[length] - incremental[:, :length]).abs()
        boundary["incremental_decode_parity"] = bool(
            torch.allclose(references[length], incremental[:, :length], rtol=RTOL, atol=ATOL)
        )
        boundary["incremental_max_abs"] = float(delta.max())
        boundary["incremental_cumulative_latency_ms"] = incremental_marks[length] * 1000.0
        boundary["incremental_avg_ms_per_token"] = incremental_marks[length] * 1000.0 / length

    report["boundaries"] = boundaries
    report["incremental_peak_rss_bytes"] = incremental_sample.peak_bytes
    report["rope_offsets"] = {
        "trace_count": len(incremental_rope_trace),
        "first": incremental_rope_trace[:3],
        "last": incremental_rope_trace[-3:],
        "exact_0_through_1023": incremental_rope_trace
        == [(1, offset) for offset in range(stage.model.max_seq_len)],
        "max_offset": max(offset for _, offset in incremental_rope_trace),
    }

    # The static cache is now exactly full. 1025-equivalent operations must reject
    # before embedding, RoPE, valid-length, K/V storage, or RNG mutation.
    static_before = {
        "valid_lengths": tuple(static_cache.valid_lengths),
        "storage_signature": static_cache.storage_signature,
        "kv_sha256": _static_cache_sha256(static_cache),
        "rng_sha256": _rng_sha256(),
        "rope_calls": len(rope_trace),
    }
    embedding_calls = 0

    def embedding_hook(*_args) -> None:
        nonlocal embedding_calls
        embedding_calls += 1

    hook = model.token_embedding.register_forward_hook(embedding_hook)
    static_rejections = {
        "forward_1025": _capture_rejection(lambda: model(tokens)),
        "static_prefill_1025": _capture_rejection(
            lambda: prefill_static_kv_cache(model, tokens, static_cache)
        ),
        "static_decode_after_1024": _capture_rejection(
            lambda: decode_one_with_static_kv_cache(model, tokens[:, 1024:1025], static_cache)
        ),
    }
    hook.remove()
    static_embedding_calls = embedding_calls
    static_after = {
        "valid_lengths": tuple(static_cache.valid_lengths),
        "storage_signature": static_cache.storage_signature,
        "kv_sha256": _static_cache_sha256(static_cache),
        "rng_sha256": _rng_sha256(),
        "rope_calls": len(rope_trace),
    }
    static_rejections["state_unchanged"] = {
        "pass": static_before == static_after,
        "embedding_calls": static_embedding_calls,
    }

    # Independently verify the dynamic concatenating cache rejects a 1025th token
    # before mutation as well.
    with torch.inference_mode():
        _, dynamic_cache = model.prefill_kv_cache(tokens[:, :1024])
    dynamic_before = {
        "sequence_length": dynamic_cache.sequence_length,
        "kv_sha256": _dynamic_cache_sha256(dynamic_cache),
        "rng_sha256": _rng_sha256(),
        "rope_calls": len(rope_trace),
    }
    embedding_calls = 0
    hook = model.token_embedding.register_forward_hook(embedding_hook)
    dynamic_rejections = {
        "dynamic_prefill_1025": _capture_rejection(lambda: model.prefill_kv_cache(tokens)),
        "dynamic_decode_after_1024": _capture_rejection(
            lambda: model.decode_one_with_kv_cache(tokens[:, 1024:1025], dynamic_cache)
        ),
    }
    hook.remove()
    dynamic_embedding_calls = embedding_calls
    dynamic_after = {
        "sequence_length": dynamic_cache.sequence_length,
        "kv_sha256": _dynamic_cache_sha256(dynamic_cache),
        "rng_sha256": _rng_sha256(),
        "rope_calls": len(rope_trace),
    }
    dynamic_rejections["state_unchanged"] = {
        "pass": dynamic_before == dynamic_after,
        "embedding_calls": dynamic_embedding_calls,
    }

    report["fail_closed_1025"] = {
        "static": static_rejections,
        "dynamic": dynamic_rejections,
    }
    report["kv"] = {
        "dtype": str(next(model.parameters()).dtype).removeprefix("torch."),
        "shape_per_layer": [
            1,
            stage.model.n_kv_heads,
            stage.model.max_seq_len,
            stage.model.head_dim,
        ],
        "layers": stage.model.n_layers,
        "bytes_per_token": static_cache.allocated_bytes // stage.model.max_seq_len,
        "allocated_bytes": static_cache.allocated_bytes,
        "logical_bytes_at_1024": static_cache.logical_bytes,
    }

    boundary_ok = all(
        bool(item["finite_logits"])
        and bool(item["finite_prefill_logits"])
        and bool(item["prefill_parity"])
        and bool(item["incremental_decode_parity"])
        for item in boundaries
    )
    static_ok = (
        all(
            bool(static_rejections[name]["rejected"])
            for name in ("forward_1025", "static_prefill_1025", "static_decode_after_1024")
        )
        and bool(static_rejections["state_unchanged"]["pass"])
        and static_embedding_calls == 0
    )
    dynamic_ok = (
        all(
            bool(dynamic_rejections[name]["rejected"])
            for name in ("dynamic_prefill_1025", "dynamic_decode_after_1024")
        )
        and bool(dynamic_rejections["state_unchanged"]["pass"])
        and dynamic_embedding_calls == 0
    )

    report["pass"] = bool(
        boundary_ok
        and report["rope_offsets"]["exact_0_through_1023"]
        and static_ok
        and dynamic_ok
    )
    report["verdict"] = (
        "PASS_CONTEXT_BOUNDARY_1024_FAIL_CLOSED_1025" if report["pass"] else "FAIL"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "next100_076_model341_context_stress.json",
    )
    args = parser.parse_args()
    result = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": result["verdict"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
