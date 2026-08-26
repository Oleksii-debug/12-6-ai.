"""LOCAL_FREE Gloo smoke probe for process-group and logical-rank identity."""

from __future__ import annotations

import multiprocessing as mp
import tempfile
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import Any

from .contracts import ParallelPlan
from .rank_layout import RankLayout


@dataclass(frozen=True, slots=True)
class CpuProbeResult:
    world_size: int
    all_reduce_sum: int
    ranks_seen: tuple[int, ...]
    logical_layout_sha256: str


def _gloo_worker(
    rank: int,
    world_size: int,
    init_file: str,
    plan: ParallelPlan,
    output: Any,
) -> None:
    import torch
    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        layout = RankLayout(plan)
        coordinate = layout.coordinate(rank)
        value = torch.tensor(rank, dtype=torch.int64)
        dist.all_reduce(value)
        output.put((rank, int(value.item()), coordinate, layout.identity_sha256))
    finally:
        dist.destroy_process_group()


def run_cpu_gloo_probe(plan: ParallelPlan, *, timeout_seconds: float = 30.0) -> CpuProbeResult:
    """Spawn one local CPU/Gloo process per physical rank; never uses GPU/cloud resources."""

    plan.validate()
    if plan.world_size > 8:
        raise ValueError("CPU probe is intentionally bounded to world_size <= 8")
    try:
        import torch.distributed as dist
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError("PyTorch distributed is unavailable") from exc
    if not dist.is_available() or not dist.is_gloo_available():
        raise RuntimeError("PyTorch Gloo backend is unavailable")

    context = mp.get_context("spawn")
    output = context.Queue()
    with tempfile.TemporaryDirectory(prefix="twelve-six-gloo-") as directory:
        init_file = str(Path(directory) / "store")
        processes = [
            context.Process(
                target=_gloo_worker,
                args=(rank, plan.world_size, init_file, plan, output),
            )
            for rank in range(plan.world_size)
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
            raise RuntimeError("CPU/Gloo probe timed out")
        failures = [process.exitcode for process in processes if process.exitcode != 0]
        if failures:
            raise RuntimeError(f"CPU/Gloo probe child failures: {failures}")

        records = []
        for _ in range(plan.world_size):
            try:
                records.append(output.get(timeout=5))
            except Empty as exc:
                raise RuntimeError("CPU/Gloo probe lost a child result") from exc

    expected_sum = sum(range(plan.world_size))
    if any(record[1] != expected_sum for record in records):
        raise RuntimeError("CPU/Gloo all_reduce produced inconsistent rank sum")
    identities = {record[3] for record in records}
    if len(identities) != 1:
        raise RuntimeError("logical rank layout identity differs across CPU ranks")
    ranks = tuple(sorted(record[0] for record in records))
    return CpuProbeResult(
        world_size=plan.world_size,
        all_reduce_sum=expected_sum,
        ranks_seen=ranks,
        logical_layout_sha256=identities.pop(),
    )
