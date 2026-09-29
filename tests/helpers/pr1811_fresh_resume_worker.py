from __future__ import annotations

import hashlib
import json
import random
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from twelve_six.checkpoint import load_trainer_checkpoint
from twelve_six.scale141_recovery import resolve_recovery_generation
from twelve_six.training import Trainer, TrainerConfig
from twelve_six.trusted_parent_recovery_binding import (
    trusted_parent_recovery_binding_from_resolution,
    trusted_recovery_authority_token,
)

SOURCE_SHA = "2" * 40
RUN_HASH = "3" * 64
LEDGER_HASH = "8" * 64
MATERIALIZATION_HASH = "9" * 64
PACKING_IDENTITY_HASH = "a" * 64
EXPOSURE_PLAN_HASH = "b" * 64
ORDERED_NEXT_HASH = "c" * 64


class TinyLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        self.projection = nn.Linear(8, 16, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.projection(self.embedding(input_ids))


def _stack() -> tuple[TinyLM, Trainer]:
    cfg = TrainerConfig(max_steps=8, learning_rate=1e-3, seed=211)
    torch.manual_seed(cfg.seed)
    model = TinyLM()
    trainer = Trainer(model, cfg, device="cpu")
    return model, trainer


def _stable_digest(value: Any) -> str:
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"T")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(repr(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
            return
        if isinstance(item, Mapping):
            digest.update(b"D")
            for key in sorted(item, key=lambda candidate: repr(candidate)):
                visit(key)
                visit(item[key])
            return
        if isinstance(item, list):
            digest.update(b"L")
            for child in item:
                visit(child)
            return
        if isinstance(item, tuple):
            digest.update(b"U")
            for child in item:
                visit(child)
            return
        digest.update(type(item).__name__.encode("utf-8"))
        digest.update(b":")
        digest.update(repr(item).encode("utf-8"))

    visit(value)
    return digest.hexdigest()


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: pr1811_fresh_resume_worker.py ROOT REFERENCE_JSON")

    root = Path(sys.argv[1])
    reference = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    resolution = resolve_recovery_generation(
        root,
        expected_reference=reference,
        expected_source_sha=SOURCE_SHA,
        expected_run_manifest_hash=RUN_HASH,
        expected_step=reference["optimizer_step"],
        expected_tokens_seen=reference["tokens_seen"],
    )

    model, trainer = _stack()
    load_trainer_checkpoint(
        resolution.path,
        model=model,
        trainer=trainer,
        restore_rng=True,
        expected_checkpoint_id=reference["checkpoint_id"],
        expected_manifest_sha256=reference["manifest_sha256"],
        expected_git_sha=SOURCE_SHA,
        expected_run_manifest_hash=RUN_HASH,
        expected_step=reference["optimizer_step"],
        expected_tokens_seen=reference["tokens_seen"],
        expected_ledger_identity_sha256=LEDGER_HASH,
        expected_materialization_identity_sha256=MATERIALIZATION_HASH,
        expected_packing_identity_sha256=PACKING_IDENTITY_HASH,
        expected_exposure_plan_identity_sha256=EXPOSURE_PLAN_HASH,
        expected_ordered_next_exposure_identity_sha256=ORDERED_NEXT_HASH,
    )

    resume_state = resolution.resume_state
    if resume_state is None:
        raise RuntimeError("resolved recovery generation has no D04 resume state")

    placeholder_authority = {
        "repository": "Oleksii-debug/12-6-ai.",
        "git_sha": SOURCE_SHA,
        "evidence_sha256": "0" * 64,
        "workflow_run_id": 1811,
        "workflow_conclusion": "success",
        "terminal": True,
    }
    projected = trusted_parent_recovery_binding_from_resolution(
        resolution,
        provider_class="OTHER_FREE",
        provider_id="GITHUB_ACTIONS_STANDARD_LINUX_X64",
        provider_session_id="fresh-process-b",
        previous_provider_session_id="publisher-process-a",
        terminal_recovery_authority=placeholder_authority,
    )
    authority = dict(placeholder_authority)
    authority["evidence_sha256"] = projected["binding_sha256"]
    trusted = trusted_parent_recovery_binding_from_resolution(
        resolution,
        provider_class="OTHER_FREE",
        provider_id="GITHUB_ACTIONS_STANDARD_LINUX_X64",
        provider_session_id="fresh-process-b",
        previous_provider_session_id="publisher-process-a",
        terminal_recovery_authority=authority,
    )
    if trusted["binding_sha256"] != authority["evidence_sha256"]:
        raise RuntimeError("trusted parent binding did not remain stable after authority binding")

    result = {
        "model_state_sha256": _stable_digest(model.state_dict()),
        "trainer_state_sha256": _stable_digest(trainer.state_dict()),
        "micro_step": trainer.micro_step,
        "optimizer_step": trainer.optimizer_step,
        "tokens_seen": trainer.tokens_seen,
        "checkpoint_id": reference["checkpoint_id"],
        "manifest_sha256": reference["manifest_sha256"],
        "d04_state_identity_sha256": resume_state["state_identity_sha256"],
        "ordered_next_exposure_identity_sha256": resume_state[
            "ordered_next_exposure_identity_sha256"
        ],
        "trusted_parent_binding_sha256": trusted["binding_sha256"],
        "trusted_parent_checkpoint_id": trusted["checkpoint_id"],
        "trusted_parent_manifest_sha256": trusted["checkpoint_manifest_sha256"],
        "trusted_recovery_authority_token": trusted_recovery_authority_token(authority),
        "rng_probe": {
            "python": random.random(),
            "numpy": float(np.random.random()),
            "torch": float(torch.rand(1).item()),
        },
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
