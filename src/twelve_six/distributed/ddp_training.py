"""Real LOCAL_FREE DDP execution for canonical 12-6 models.

This module deliberately stays separate from the single-process D02 Trainer. DDP
averages gradients across ranks, while D02's single-process token normalization
uses the local valid-token count. Unequal valid-token counts therefore require an
explicit global-token scale before backward; equal-sized toy batches can hide the
error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from queue import Empty
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel

from twelve_six.model import TwelveSixDecoder, load_stage_config
from twelve_six.training.config import TrainerConfig
from twelve_six.training.loss import causal_lm_loss
from twelve_six.training.trainer import build_optimizer

_SHA256_RE = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True, slots=True)
class CanonicalDDPProbeResult:
    stage: str
    parameter_count: int
    model_spec_sha256: str
    init_spec_sha256: str
    backend: str
    world_size: int
    local_valid_tokens: tuple[int, ...]
    global_valid_tokens: int
    global_loss: float
    max_cross_rank_parameter_diff: float
    max_reference_parameter_diff: float
    max_parameter_change: float
    reference_tolerance: float

    @property
    def passed(self) -> bool:
        return (
            self.world_size >= 2
            and len(set(self.local_valid_tokens)) > 1
            and self.global_valid_tokens == sum(self.local_valid_tokens)
            and self.max_cross_rank_parameter_diff == 0.0
            and self.max_reference_parameter_diff <= self.reference_tolerance
            and self.max_parameter_change > 0.0
        )


def ddp_token_weighted_loss_scale(
    local_valid_tokens: int,
    global_valid_tokens: int,
    world_size: int,
) -> float:
    """Scale a local mean loss so DDP's averaged gradient equals the global token mean."""
    for name, value in {
        "local_valid_tokens": local_valid_tokens,
        "global_valid_tokens": global_valid_tokens,
        "world_size": world_size,
    }.items():
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer")
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if local_valid_tokens > global_valid_tokens:
        raise ValueError("local_valid_tokens cannot exceed global_valid_tokens")
    return local_valid_tokens * world_size / global_valid_tokens


def _rank_batch(rank: int, *, vocab_size: int, sequence_length: int) -> tuple[Tensor, Tensor]:
    if sequence_length < 6:
        raise ValueError("sequence_length must be >= 6")
    tokens = (torch.arange(sequence_length, dtype=torch.long) + 17 * rank + 3) % vocab_size
    input_ids = tokens.unsqueeze(0)
    labels = input_ids.clone()
    masked_tail = rank * 2
    if masked_tail:
        labels[:, -masked_tail:] = -100
    return input_ids, labels


def _valid_target_tokens(labels: Tensor) -> int:
    return int(labels[:, 1:].ne(-100).sum().item())


def _gradient_norm(model: torch.nn.Module) -> float:
    total = torch.zeros((), dtype=torch.float64)
    for parameter in model.parameters():
        if parameter.grad is not None:
            grad = parameter.grad.detach().double()
            total += torch.sum(grad * grad)
    return float(torch.sqrt(total).item())


def _max_parameter_change(model: torch.nn.Module, before: list[Tensor]) -> float:
    maximum = 0.0
    for parameter, initial in zip(model.parameters(), before, strict=True):
        diff = float((parameter.detach() - initial).abs().max().item())
        maximum = max(maximum, diff)
    return maximum


def _cross_rank_parameter_diff(model: torch.nn.Module) -> float:
    maximum = torch.zeros((), dtype=torch.float64)
    for parameter in model.parameters():
        reference = parameter.detach().clone()
        dist.broadcast(reference, src=0)
        diff = (parameter.detach() - reference).abs().max().double()
        maximum = torch.maximum(maximum, diff)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    return float(maximum.item())


def _reference_parameter_diff(
    distributed_model: torch.nn.Module,
    *,
    stage_config_path: str,
    world_size: int,
    sequence_length: int,
    seed: int,
    trainer_config: TrainerConfig,
) -> float:
    stage = load_stage_config(stage_config_path)
    torch.manual_seed(seed)
    reference = TwelveSixDecoder(stage.model, stage.init)
    optimizer = build_optimizer(reference, trainer_config)
    inputs = []
    labels = []
    for rank in range(world_size):
        rank_inputs, rank_labels = _rank_batch(
            rank,
            vocab_size=stage.model.vocab_size,
            sequence_length=sequence_length,
        )
        inputs.append(rank_inputs)
        labels.append(rank_labels)
    global_inputs = torch.cat(inputs, dim=0)
    global_labels = torch.cat(labels, dim=0)
    loss = causal_lm_loss(reference(global_inputs).logits, global_labels)
    loss.backward()
    optimizer.step()

    maximum = 0.0
    for distributed_parameter, reference_parameter in zip(
        distributed_model.parameters(), reference.parameters(), strict=True
    ):
        diff = float(
            (distributed_parameter.detach() - reference_parameter.detach()).abs().max().item()
        )
        maximum = max(maximum, diff)
    return maximum


def _ddp_worker(
    rank: int,
    world_size: int,
    init_file: str,
    stage_config_path: str,
    sequence_length: int,
    seed: int,
    reference_tolerance: float,
    output: Any,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.set_num_threads(1)
        stage = load_stage_config(stage_config_path)
        if sequence_length > stage.model.max_seq_len:
            raise ValueError("probe sequence_length exceeds stage max_seq_len")
        torch.manual_seed(seed)
        model = TwelveSixDecoder(stage.model, stage.init)
        ddp_model = DistributedDataParallel(model)
        trainer_config = TrainerConfig(
            learning_rate=3e-4,
            weight_decay=0.0,
            max_steps=1,
            gradient_clip_norm=None,
            seed=seed,
        )
        optimizer = build_optimizer(ddp_model, trainer_config)
        before = [parameter.detach().clone() for parameter in model.parameters()]

        input_ids, labels = _rank_batch(
            rank,
            vocab_size=stage.model.vocab_size,
            sequence_length=sequence_length,
        )
        local_tokens = _valid_target_tokens(labels)
        local_loss = causal_lm_loss(ddp_model(input_ids).logits, labels)

        global_tokens_tensor = torch.tensor(local_tokens, dtype=torch.int64)
        dist.all_reduce(global_tokens_tensor, op=dist.ReduceOp.SUM)
        global_tokens = int(global_tokens_tensor.item())

        loss_sum_tensor = torch.tensor(
            float(local_loss.detach().item()) * local_tokens,
            dtype=torch.float64,
        )
        dist.all_reduce(loss_sum_tensor, op=dist.ReduceOp.SUM)
        global_loss = float(loss_sum_tensor.item() / global_tokens)

        scale = ddp_token_weighted_loss_scale(local_tokens, global_tokens, world_size)
        (local_loss * scale).backward()
        grad_norm = _gradient_norm(model)
        optimizer.step()

        cross_rank_diff = _cross_rank_parameter_diff(model)
        parameter_change = _max_parameter_change(model, before)
        reference_diff = 0.0
        if rank == 0:
            reference_diff = _reference_parameter_diff(
                model,
                stage_config_path=stage_config_path,
                world_size=world_size,
                sequence_length=sequence_length,
                seed=seed,
                trainer_config=trainer_config,
            )
        reference_diff_tensor = torch.tensor(reference_diff, dtype=torch.float64)
        dist.broadcast(reference_diff_tensor, src=0)
        reference_diff = float(reference_diff_tensor.item())

        if cross_rank_diff != 0.0:
            raise RuntimeError("DDP parameters diverged across ranks")
        if reference_diff > reference_tolerance:
            raise RuntimeError(
                "DDP update does not match single-process global-token reference: "
                f"max_abs_diff={reference_diff} tolerance={reference_tolerance}"
            )
        if parameter_change <= 0.0:
            raise RuntimeError("DDP optimizer step did not change model parameters")

        output.put(
            {
                "rank": rank,
                "local_tokens": local_tokens,
                "global_tokens": global_tokens,
                "global_loss": global_loss,
                "grad_norm": grad_norm,
                "cross_rank_diff": cross_rank_diff,
                "reference_diff": reference_diff,
                "parameter_change": parameter_change,
                "stage": stage.stage,
                "parameter_count": stage.model.parameter_count(),
                "model_spec_sha256": stage.model.identity_sha256(),
                "init_spec_sha256": stage.init.identity_sha256(),
            }
        )
    finally:
        dist.destroy_process_group()


def run_canonical_cpu_ddp_probe(
    stage_config_path: str | Path,
    *,
    world_size: int = 2,
    sequence_length: int = 16,
    seed: int = 20260825,
    reference_tolerance: float = 1e-5,
    timeout_seconds: float = 120.0,
) -> CanonicalDDPProbeResult:
    """Execute one real token-correct DDP update and compare it with a global reference."""
    if not isinstance(world_size, int) or isinstance(world_size, bool):
        raise TypeError("world_size must be an integer")
    if world_size < 2 or world_size > 4:
        raise ValueError("LOCAL_FREE CPU DDP probe requires 2 <= world_size <= 4")
    if not dist.is_available() or not dist.is_gloo_available():
        raise RuntimeError("PyTorch distributed Gloo backend is unavailable")
    if reference_tolerance <= 0:
        raise ValueError("reference_tolerance must be positive")

    stage_path = str(Path(stage_config_path).resolve())
    stage = load_stage_config(stage_path)
    if sequence_length > stage.model.max_seq_len:
        raise ValueError("probe sequence_length exceeds stage max_seq_len")

    context = mp.get_context("spawn")
    output = context.Queue()
    with tempfile.TemporaryDirectory(prefix="twelve-six-ddp-") as directory:
        init_file = str(Path(directory) / "store")
        processes = [
            context.Process(
                target=_ddp_worker,
                args=(
                    rank,
                    world_size,
                    init_file,
                    stage_path,
                    sequence_length,
                    seed,
                    reference_tolerance,
                    output,
                ),
            )
            for rank in range(world_size)
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
            raise RuntimeError("canonical CPU DDP probe timed out")
        failures = [process.exitcode for process in processes if process.exitcode != 0]
        if failures:
            raise RuntimeError(f"canonical CPU DDP probe child failures: {failures}")

        records = []
        for _ in range(world_size):
            try:
                records.append(output.get(timeout=5))
            except Empty as exc:
                raise RuntimeError("canonical CPU DDP probe lost a child result") from exc

    records.sort(key=lambda item: item["rank"])
    global_tokens = {item["global_tokens"] for item in records}
    global_losses = {item["global_loss"] for item in records}
    identities = {
        (item["stage"], item["parameter_count"], item["model_spec_sha256"], item["init_spec_sha256"])
        for item in records
    }
    if len(global_tokens) != 1 or len(global_losses) != 1 or len(identities) != 1:
        raise RuntimeError("DDP ranks disagree on global execution identity or statistics")
    local_tokens = tuple(int(item["local_tokens"]) for item in records)
    if len(set(local_tokens)) <= 1:
        raise RuntimeError("probe must exercise unequal valid-token counts across ranks")
    stage_name, parameter_count, model_sha, init_sha = identities.pop()
    result = CanonicalDDPProbeResult(
        stage=stage_name,
        parameter_count=parameter_count,
        model_spec_sha256=model_sha,
        init_spec_sha256=init_sha,
        backend="gloo",
        world_size=world_size,
        local_valid_tokens=local_tokens,
        global_valid_tokens=global_tokens.pop(),
        global_loss=global_losses.pop(),
        max_cross_rank_parameter_diff=max(item["cross_rank_diff"] for item in records),
        max_reference_parameter_diff=max(item["reference_diff"] for item in records),
        max_parameter_change=max(item["parameter_change"] for item in records),
        reference_tolerance=reference_tolerance,
    )
    if not result.passed:
        raise RuntimeError("canonical CPU DDP probe failed its acceptance contract")
    return result


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_evidence(result: CanonicalDDPProbeResult, *, source_sha: str) -> dict[str, Any]:
    if _SHA256_RE.fullmatch(source_sha) is None:
        raise ValueError("source_sha must be an exact lowercase 40-hex Git SHA")
    payload: dict[str, Any] = {
        "schema": "12-6.canonical-ddp-execution.v1",
        "authority": "LOCAL_FREE_DISTRIBUTED_MECHANICS_NOT_STAGE_OR_CAPABILITY_EVIDENCE",
        "source_sha": source_sha,
        "canonical_base": "random_init",
        "paid_compute": False,
        "promotion_authority": False,
        "behavioral_alignment": False,
        "execution": asdict(result) | {"passed": result.passed},
        "correctness": {
            "ddp_gradient_semantics": "AVERAGED_ACROSS_RANKS",
            "normalization": "GLOBAL_VALID_TOKEN_WEIGHTED",
            "unequal_rank_token_counts_required": True,
            "single_process_global_batch_reference": True,
        },
    }
    payload["report_sha256"] = _canonical_sha256(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-config", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()

    result = run_canonical_cpu_ddp_probe(
        args.stage_config,
        world_size=args.world_size,
        sequence_length=args.sequence_length,
        seed=args.seed,
    )
    evidence = build_evidence(result, source_sha=args.source_sha)
    output_path = Path(args.output)
    output_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(evidence, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
