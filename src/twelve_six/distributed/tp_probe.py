"""LOCAL_FREE multiprocess proof for the PyTorch-native 12-6 tensor-parallel seam."""

from __future__ import annotations

import multiprocessing as mp
import tempfile
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import Any

from .contracts import ParallelPlan
from .rank_layout import RankLayout
from .tensor_parallel import TensorParallelPlan, parallelize_decoder_tp


@dataclass(frozen=True, slots=True)
class CpuTensorParallelProbeResult:
    world_size: int
    tp_degree: int
    ranks_seen: tuple[int, ...]
    max_abs_forward_error: float
    tensor_parallel_plan_sha256: str
    checkpoint_layout_sha256: str
    parameter_partitioning_passed: bool
    state_dict_schema_stable: bool


def _tiny_tp_spec(tp_degree: int) -> Any:
    from twelve_six.model import ModelSpec

    return ModelSpec(
        schema_version=1,
        vocab_size=32,
        max_seq_len=8,
        d_model=tp_degree * 8,
        n_layers=2,
        n_heads=tp_degree * 2,
        n_kv_heads=tp_degree,
        head_dim=4,
        d_ff=tp_degree * 16,
        rope_rotary_dim=4,
    )


def _assert_local_partition(
    local_tensor: Any,
    full_tensor: Any,
    *,
    shard_dim: int,
    start: int,
    stop: int,
) -> None:
    import torch

    expected = full_tensor.narrow(shard_dim, start, stop - start)
    torch.testing.assert_close(local_tensor, expected, atol=0.0, rtol=0.0)


def _tp_worker(
    rank: int,
    tp_degree: int,
    init_file: str,
    output: Any,
) -> None:
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    from twelve_six.model import TwelveSixDecoder

    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=tp_degree,
    )
    try:
        torch.manual_seed(20260825)
        spec = _tiny_tp_spec(tp_degree)
        reference = TwelveSixDecoder(spec).eval()
        sharded = TwelveSixDecoder(spec).eval()
        sharded.load_state_dict(reference.state_dict())
        input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.long)
        reference_logits = reference(input_ids).logits

        state_before = {
            key: tuple(value.shape)
            for key, value in sharded.state_dict().items()
        }
        mesh = init_device_mesh("cpu", (tp_degree,), mesh_dim_names=("tp",))
        plan = TensorParallelPlan.from_model_spec(spec, tp_degree)
        layout = RankLayout(ParallelPlan(tensor_parallel=tp_degree))
        geometry = plan.rank_geometry_for_global_rank(layout, rank)
        parallelize_decoder_tp(sharded, mesh, plan=plan)

        block = sharded.blocks[0]
        full = reference.blocks[0]
        q_start = geometry.query_head_start * plan.head_dim
        q_stop = geometry.query_head_stop * plan.head_dim
        kv_start = geometry.kv_head_start * plan.head_dim
        kv_stop = geometry.kv_head_stop * plan.head_dim

        _assert_local_partition(
            block.attn.q_proj.weight.to_local(),
            full.attn.q_proj.weight,
            shard_dim=0,
            start=q_start,
            stop=q_stop,
        )
        for local_weight, full_weight in (
            (block.attn.k_proj.weight.to_local(), full.attn.k_proj.weight),
            (block.attn.v_proj.weight.to_local(), full.attn.v_proj.weight),
        ):
            _assert_local_partition(
                local_weight,
                full_weight,
                shard_dim=0,
                start=kv_start,
                stop=kv_stop,
            )
        _assert_local_partition(
            block.attn.out_proj.weight.to_local(),
            full.attn.out_proj.weight,
            shard_dim=1,
            start=q_start,
            stop=q_stop,
        )
        for local_weight, full_weight in (
            (block.mlp.gate_proj.weight.to_local(), full.mlp.gate_proj.weight),
            (block.mlp.up_proj.weight.to_local(), full.mlp.up_proj.weight),
        ):
            _assert_local_partition(
                local_weight,
                full_weight,
                shard_dim=0,
                start=geometry.ffn_start,
                stop=geometry.ffn_stop,
            )
        _assert_local_partition(
            block.mlp.down_proj.weight.to_local(),
            full.mlp.down_proj.weight,
            shard_dim=1,
            start=geometry.ffn_start,
            stop=geometry.ffn_stop,
        )

        state_after = {
            key: tuple(value.shape)
            for key, value in sharded.state_dict().items()
        }
        if state_after != state_before:
            raise RuntimeError("TP changed canonical state-dict FQNs or global tensor shapes")

        sharded_logits = sharded(input_ids).logits
        max_error = float((sharded_logits - reference_logits).abs().max().item())
        torch.testing.assert_close(sharded_logits, reference_logits, atol=2e-6, rtol=2e-6)
        output.put(
            (
                rank,
                max_error,
                plan.identity_sha256,
                plan.checkpoint_layout_sha256,
                True,
                True,
            )
        )
    finally:
        dist.destroy_process_group()


def run_cpu_tensor_parallel_probe(
    tp_degree: int = 2,
    *,
    timeout_seconds: float = 30.0,
) -> CpuTensorParallelProbeResult:
    """Run real DTensor/TP on local CPU/Gloo ranks; no GPU or cloud resource is used."""

    if not isinstance(tp_degree, int) or isinstance(tp_degree, bool):
        raise TypeError("tp_degree must be an integer")
    if tp_degree < 2:
        raise ValueError("tp_degree must be >= 2")
    if tp_degree > 4:
        raise ValueError("LOCAL_FREE TP probe is intentionally bounded to tp_degree <= 4")
    try:
        import torch.distributed as dist
        from torch.distributed.tensor.parallel import parallelize_module  # noqa: F401
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError("PyTorch DTensor/tensor parallel is unavailable") from exc
    if not dist.is_available() or not dist.is_gloo_available():
        raise RuntimeError("PyTorch Gloo backend is unavailable")

    context = mp.get_context("spawn")
    output = context.Queue()
    with tempfile.TemporaryDirectory(prefix="twelve-six-tp-") as directory:
        init_file = str(Path(directory) / "store")
        processes = [
            context.Process(
                target=_tp_worker,
                args=(rank, tp_degree, init_file, output),
            )
            for rank in range(tp_degree)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout_seconds)
        stuck = [process for process in processes if process.is_alive()]
        if stuck:
            for process in stuck:
                process.terminate()
            for process in stuck:
                process.join(5)
            raise RuntimeError("CPU tensor-parallel probe timed out")
        failures = [process.exitcode for process in processes if process.exitcode != 0]
        if failures:
            raise RuntimeError(f"CPU tensor-parallel probe child failures: {failures}")

        records = []
        for _ in range(tp_degree):
            try:
                records.append(output.get(timeout=5))
            except Empty as exc:
                raise RuntimeError("CPU tensor-parallel probe lost a child result") from exc

    ranks = tuple(sorted(record[0] for record in records))
    plan_hashes = {record[2] for record in records}
    checkpoint_hashes = {record[3] for record in records}
    if len(plan_hashes) != 1 or len(checkpoint_hashes) != 1:
        raise RuntimeError("TP plan/checkpoint layout identity differs across ranks")
    return CpuTensorParallelProbeResult(
        world_size=tp_degree,
        tp_degree=tp_degree,
        ranks_seen=ranks,
        max_abs_forward_error=max(record[1] for record in records),
        tensor_parallel_plan_sha256=plan_hashes.pop(),
        checkpoint_layout_sha256=checkpoint_hashes.pop(),
        parameter_partitioning_passed=all(record[4] for record in records),
        state_dict_schema_stable=all(record[5] for record in records),
    )
