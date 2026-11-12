"""Commit verified ordered exposure accounting with the trainer checkpoint."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from copy import deepcopy
from typing import Any

from torch import Tensor

from ..checkpoint import CheckpointIdentity, hash_json
from ..data.deterministic_exposure_order import (
    authorize_ordered_live_batch,
    ordered_next_exposure_identity,
    validate_exposure_plan_preflight,
)
from ..data.identity_safe_exposure_guard import IdentitySafeExposureReplayGuard
from ..data.ordered_exposure_resume import load_ordered_resume_state, ordered_resume_state_dict
from ..data.unique_loss_ledger_v2 import LedgerError
from .config import TrainingRecipe
from .run_control import RunBoundary, RunController, RunPhase, RunReceipt
from .trainer import Batch, Trainer

SCHEMA = "12-6.ordered-exposure-accounting.v1"


class OrderedExposureRuntime:
    """Attach a sealed one-pass exposure plan to the normal single-writer runner."""

    def __init__(
        self,
        *,
        ledger: Mapping[str, Any],
        plan: Mapping[str, Any],
        content_manifest: Mapping[str, Any],
        expected_ledger_sha256: str,
        expected_plan_sha256: str,
        expected_content_sha256: str,
        unique_target_budget: int,
    ) -> None:
        self._ledger = deepcopy(dict(ledger))
        self._plan = deepcopy(dict(plan))
        self._content_manifest = deepcopy(dict(content_manifest))
        self._ledger_sha = expected_ledger_sha256
        self._plan_sha = expected_plan_sha256
        self._content_sha = expected_content_sha256
        self._budget = unique_target_budget
        proof = validate_exposure_plan_preflight(
            self._plan,
            self._ledger,
            expected_ledger_identity_sha256=self._ledger_sha,
            expected_unique_budget=self._budget,
        )
        if proof["plan_identity_sha256"] != self._plan_sha:
            raise LedgerError("exposure plan differs from its expected identity")
        content_refs = {
            "ledger_identity_sha256": self._ledger_sha,
            "plan_identity_sha256": self._plan_sha,
            "materialization_identity_sha256": self._ledger["materialization_identity_sha256"],
            "packing_identity_sha256": self._ledger["packing_identity_sha256"],
            "loss_bearing_target_count": self._budget,
        }
        observed_refs = {key: self._content_manifest.get(key) for key in content_refs}
        if hash_json(observed_refs) != hash_json(content_refs):
            raise LedgerError(
                "loss-bearing content does not match exposure ledger, plan and budget"
            )
        self._batch_count = len(self._plan["batches"])

    def _progress(
        self,
        identity: CheckpointIdentity,
        *,
        step: int,
        micro_step: int,
        targets: int,
    ) -> dict[str, Any]:
        return {
            "run_manifest_sha256": identity.run_manifest_hash,
            "dataset_manifest_sha256": identity.dataset_manifest_hash,
            "ledger_sha256": self._ledger_sha,
            "plan_sha256": self._plan_sha,
            "content_manifest_sha256": self._content_sha,
            "unique_target_budget": self._budget,
            "optimizer_step": step,
            "micro_step": micro_step,
            "unique_targets_seen": targets,
        }

    @staticmethod
    def _binding(progress: dict[str, Any]) -> dict[str, Any]:
        # The D04 binding hashes this embedded progress record. Using the
        # enclosing checkpoint digest here would make the cursor self-referential.
        return {
            "checkpoint_generation": f"step-{progress['optimizer_step']:08d}",
            "checkpoint_manifest_sha256": hash_json(progress),
            "optimizer_step": progress["optimizer_step"],
            "trainer_nonignored_target_count": progress["unique_targets_seen"],
        }

    def _guard(self, identity: CheckpointIdentity) -> IdentitySafeExposureReplayGuard:
        if self._content_manifest.get("tokenizer_identity_sha256") != identity.tokenizer_hash:
            raise LedgerError("loss-bearing content tokenizer differs from the run tokenizer")
        initial = self._progress(identity, step=0, micro_step=0, targets=0)
        return IdentitySafeExposureReplayGuard(
            self._ledger,
            expected_ledger_identity_sha256=self._ledger_sha,
            authorized_budget=self._budget,
            trainer_state_binding=self._binding(initial),
            loss_bearing_content_manifest=self._content_manifest,
            expected_loss_bearing_manifest_identity_sha256=self._content_sha,
        )

    def _validate_config(self, trainer: Trainer, identity: CheckpointIdentity) -> None:
        microbatches = trainer.config.max_steps * trainer.config.gradient_accumulation_steps
        if microbatches != self._batch_count:
            raise LedgerError("ordered exposure batch count differs from the trainer recipe")
        training = identity.training_config.get("training")
        if isinstance(training, Mapping) and not {"recipe", "recipe_sha256"}.isdisjoint(training):
            try:
                recipe = TrainingRecipe.from_dict(training.get("recipe"))
            except (TypeError, ValueError) as exc:
                raise LedgerError("ordered exposure run has an invalid training recipe") from exc
            if (
                recipe.target_unique_loss_positions != self._budget
                or recipe.maximum_total_exposures < self._budget
            ):
                raise LedgerError("ordered exposure target budget differs from the trainer recipe")
        data = identity.training_config.get("data")
        if isinstance(data, Mapping):
            pins = {
                "ledger_identity_sha256": self._ledger_sha,
                "exposure_plan_identity_sha256": self._plan_sha,
                "loss_bearing_manifest_identity_sha256": self._content_sha,
                "materialization_identity_sha256": self._ledger["materialization_identity_sha256"],
                "packing_identity_sha256": self._ledger["packing_identity_sha256"],
            }
            for key, expected in pins.items():
                if key in data and data[key] != expected:
                    raise LedgerError(f"ordered exposure {key} differs from the bound run")

    def _cursor(
        self,
        guard: IdentitySafeExposureReplayGuard,
        progress: dict[str, Any],
    ) -> dict[str, Any]:
        if (
            guard.claim_sequence != progress["micro_step"]
            or guard.consumed_loss_positions != progress["unique_targets_seen"]
        ):
            raise LedgerError("trainer progress differs from authorized unique exposures")
        guard.bind_checkpoint_state(self._binding(progress))
        return {
            "schema": SCHEMA,
            "progress": progress,
            "ordered_state": ordered_resume_state_dict(
                guard,
                self._plan,
                expected_plan_identity_sha256=self._plan_sha,
            ),
        }

    def prepare(
        self,
        controller: RunController,
        trainer: Trainer,
        identity: CheckpointIdentity,
    ) -> RunReceipt:
        """Verify authorities and record the initial unconsumed exposure cursor."""
        self._validate_config(trainer, identity)
        initial = self._progress(identity, step=0, micro_step=0, targets=0)
        cursor = self._cursor(self._guard(identity), initial)
        return controller.prepare(trainer, identity, exposure_cursor=cursor)

    def _authorized_batches(
        self,
        guard: IdentitySafeExposureReplayGuard,
        source: Iterable[Batch],
    ) -> Iterator[Batch]:
        for batch in source:
            if "input_ids" not in batch or "target_ids" not in batch:
                raise LedgerError("ordered training requires input_ids and aligned target_ids")
            if "labels" in batch:
                raise LedgerError("ordered training cannot mix labels and target_ids")
            verified = {}
            for key in ("input_ids", "target_ids", "loss_mask"):
                if key in batch:
                    value = batch[key]
                    if not isinstance(value, Tensor):
                        raise LedgerError(f"ordered training {key} must be a tensor")
                    verified[key] = value.detach().clone()
            index = guard.claim_sequence
            expected = ordered_next_exposure_identity(
                guard,
                self._plan,
                batch_index=index,
                expected_plan_identity_sha256=self._plan_sha,
            )
            authorize_ordered_live_batch(
                guard,
                self._plan,
                batch_index=index,
                expected_plan_identity_sha256=self._plan_sha,
                expected_ordered_next_exposure_identity_sha256=expected,
                input_ids=verified["input_ids"],
                target_ids=verified["target_ids"],
                loss_mask=verified.get("loss_mask"),
            )
            yield verified

    def run(
        self,
        controller: RunController,
        trainer: Trainer,
        identity: CheckpointIdentity,
        source: Callable[[int], Iterable[Batch]],
        *,
        expected_receipt_sha256: str,
        recover: bool = False,
        control: Callable[[RunBoundary], RunPhase | None] | None = None,
        admit_progress: Callable[[RunReceipt], None] | None = None,
    ) -> RunReceipt:
        """Open the source at the exact next authorized batch after verified restore."""
        self._validate_config(trainer, identity)
        guard = self._guard(identity)
        restored: Mapping[str, Any] | None = None

        def remember_cursor(cursor: Mapping[str, Any]) -> None:
            nonlocal restored
            restored = cursor

        def batches(receipt: RunReceipt) -> Iterable[Batch]:
            progress = self._progress(
                identity,
                step=receipt.committed_step,
                micro_step=receipt.committed_step * trainer.config.gradient_accumulation_steps,
                targets=receipt.tokens_seen,
            )
            if (
                restored is None
                or set(restored) != {"schema", "progress", "ordered_state"}
                or restored["schema"] != SCHEMA
                or hash_json(restored["progress"]) != hash_json(progress)
                or not isinstance(restored["ordered_state"], Mapping)
            ):
                raise LedgerError("verified exposure cursor differs from committed run progress")
            state = restored["ordered_state"]
            load_ordered_resume_state(
                guard,
                self._plan,
                state,
                expected_ordered_resume_identity_sha256=state.get("ordered_resume_identity_sha256"),
                expected_plan_identity_sha256=self._plan_sha,
                expected_trainer_state_binding=self._binding(progress),
            )
            if guard.claim_sequence != progress["micro_step"]:
                raise LedgerError("restored exposure cursor differs from the trainer microbatch")
            return self._authorized_batches(guard, source(guard.claim_sequence))

        def cursor(current: Trainer) -> dict[str, Any]:
            return self._cursor(
                guard,
                self._progress(
                    identity,
                    step=current.optimizer_step,
                    micro_step=current.micro_step,
                    targets=current.tokens_seen,
                ),
            )

        return controller.run(
            trainer,
            identity,
            batches,
            expected_receipt_sha256=expected_receipt_sha256,
            recover=recover,
            control=control,
            read_exposure_cursor=cursor,
            restore_exposure_cursor=remember_cursor,
            admit_progress=admit_progress,
        )
