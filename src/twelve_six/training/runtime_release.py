"""Verify the released training component and its read-only Plan 2 handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from contextlib import closing
from pathlib import Path
from typing import Any

import torch

from ..checkpoint import hash_json
from . import loss as training_loss

RELEASE_PATH = "configs/plan3/training_runtime_release_v1.json"
RELEASE_SHA256 = "12002302e78dd32593a682c06e9677ef2b45150b19dad5b9c64b737fc8e5234e"


class TrainingRuntimeReleaseError(ValueError):
    """A release artifact differs from the accepted component."""


def _read_lf(root: Path, relative: str) -> bytes:
    path = root / relative
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
        raise TrainingRuntimeReleaseError(f"release artifact is unavailable: {relative}")
    return path.read_bytes().replace(b"\r\n", b"\n")


def verify_runtime_release(root: Path) -> dict[str, Any]:
    """Verify the fixed release descriptor and its incumbent source files."""
    root = root.resolve(strict=True)
    raw = _read_lf(root, RELEASE_PATH)
    if hashlib.sha256(raw).hexdigest() != RELEASE_SHA256:
        raise TrainingRuntimeReleaseError("training runtime release descriptor changed")
    release = json.loads(raw)
    for relative, expected in release["runtime_source_sha256_lf"].items():
        if hashlib.sha256(_read_lf(root, relative)).hexdigest() != expected:
            raise TrainingRuntimeReleaseError(f"release source changed: {relative}")
    return release


def verify_plan2_handoff(root: Path, physical_release: Path) -> dict[str, Any]:
    """Reopen the accepted local data release and verify the aligned loss interface."""
    from tools import plan2_s18_physical_release_v1 as reader

    root = root.resolve(strict=True)
    handoff = verify_runtime_release(root)["plan2_handoff"]
    if (
        Path(__file__).resolve() != (root / "src/twelve_six/training/runtime_release.py").resolve()
        or Path(training_loss.__file__).resolve()
        != (root / "src/twelve_six/training/loss.py").resolve()
        or Path(reader.__file__).resolve()
        != (root / "tools/plan2_s18_physical_release_v1.py").resolve()
    ):
        raise TrainingRuntimeReleaseError("load the handoff runtime from the verified checkout")
    with closing(reader.iter_release_batches(root, physical_release, batch_size=2)) as batches:
        try:
            batch = next(batches)
        except StopIteration as exc:
            raise TrainingRuntimeReleaseError("accepted Plan 2 batch is missing") from exc
    if tuple(batch) != tuple(handoff["tensor_fields"]) or any(
        tensor.device.type != "cpu"
        or tensor.dtype != torch.long
        or list(tensor.shape) != handoff["first_batch_shape"]
        for tensor in batch.values()
    ):
        raise TrainingRuntimeReleaseError("Plan 2 batch does not match the aligned tensor contract")
    digests = {
        name: hashlib.sha256(tensor.numpy().astype("<i8", copy=False).tobytes()).hexdigest()
        for name, tensor in batch.items()
    }
    if digests != handoff["first_batch_tensor_sha256_i64le"]:
        raise TrainingRuntimeReleaseError("Plan 2 first-batch tensor bytes changed")
    labels = batch["target_ids"]
    active = batch["loss_mask"].bool() & labels.ne(-100)
    if active.sum().item() != handoff["first_batch_targets"]:
        raise TrainingRuntimeReleaseError("Plan 2 first-batch target count changed")
    vocab_size = handoff["vocab_size"]
    with torch.inference_mode():
        logits = torch.zeros((*labels.shape, vocab_size), dtype=torch.float64)
        logits.scatter_(2, labels.masked_fill(~active, 0).unsqueeze(-1), 2.0)
        loss = training_loss.causal_pair_loss(logits, labels, loss_mask=batch["loss_mask"]).item()
    expected = math.log1p((vocab_size - 1) * math.exp(-2.0))
    if not math.isclose(loss, expected, rel_tol=0, abs_tol=2e-12):
        raise TrainingRuntimeReleaseError("Plan 2 first batch failed the aligned loss check")
    result = {
        "runtime_release_sha256": RELEASE_SHA256,
        "plan2_handoff": handoff,
        "aligned_loss": loss,
        "optimizer_steps_executed": 0,
        "campaign_authorized": False,
    }
    return {**result, "receipt_sha256": hash_json(result)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--physical-release", type=Path)
    args = parser.parse_args()
    if args.physical_release is None:
        result = verify_runtime_release(args.root)
    else:
        result = verify_plan2_handoff(args.root, args.physical_release)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
