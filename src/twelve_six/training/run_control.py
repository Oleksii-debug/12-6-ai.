"""Durable single-writer control for checkpointed Trainer runs."""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Callable, Iterable, Mapping
from contextlib import closing
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path

from ..checkpoint import (
    CheckpointIdentity,
    hash_json,
    load_trainer_checkpoint,
    save_trainer_checkpoint,
    verify_checkpoint,
)
from ..checkpoint.durability import fsync_parent_directory
from ..checkpoint.recovery_lock import exclusive_recovery_lock
from .config import TrainingRecipe
from .trainer import Batch, Trainer

_SHA256 = re.compile(r"[0-9a-f]{64}")


class RunControlError(RuntimeError):
    """A run transition cannot be proven safe."""


class RunPhase(str, Enum):
    PREPARED = "PREPARED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"
    FAILED = "FAILED"
    COMPLETED = "COMPLETED"


@dataclass(frozen=True, slots=True)
class RunReceipt:
    run_sha256: str
    identity_sha256: str
    generation: int
    control_epoch: int
    phase: RunPhase
    control_enabled: bool
    control_pending: bool
    committed_step: int
    tokens_seen: int
    pending_step: int | None
    pending_tokens: int | None
    checkpoint_epoch: int
    checkpoint_id: str
    parent_sha256: str | None
    receipt_sha256: str


@dataclass(frozen=True, slots=True)
class RunBoundary:
    run_sha256: str
    committed_step: int
    tokens_seen: int
    checkpoint_id: str


def _need(ok: bool, message: str) -> None:
    if not ok:
        raise RunControlError(message)


class RunController:
    """Serialize a run identity within one checkpoint-controlled namespace.

    The kernel lock is held across optimizer calls. An intent is committed to
    the journal before each update, and a verified checkpoint precedes the
    commit receipt. Recovery can only start from the last committed state.
    """

    def __init__(self, root: str | Path, run_sha256: str) -> None:
        _need(
            isinstance(run_sha256, str) and _SHA256.fullmatch(run_sha256) is not None,
            "run identity must be an exact lowercase SHA-256",
        )
        self.run_sha256 = run_sha256
        self.namespace = Path(root).absolute() / run_sha256

    @staticmethod
    def _connect(root: Path, *, create: bool = False) -> sqlite3.Connection:
        path = root / "control.sqlite3"
        _need(not path.is_symlink() and (create or path.is_file()), "run journal is unavailable")
        connection = sqlite3.connect(path, timeout=10)
        try:
            connection.execute("PRAGMA synchronous=FULL")
            if create:
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS events "
                    "(generation INTEGER PRIMARY KEY, payload TEXT NOT NULL, digest TEXT NOT NULL)"
                )
        except BaseException:
            connection.close()
            raise
        return connection

    def _read(self, connection: sqlite3.Connection) -> RunReceipt | None:
        records = connection.execute(
            "SELECT generation, payload, digest FROM events ORDER BY generation DESC LIMIT 2"
        ).fetchall()
        if not records:
            return None
        generation, raw, digest = records[0]
        try:
            doc = json.loads(raw)
            _need(
                digest == hash_json(doc)
                and doc["generation"] == generation
                and doc["run_sha256"] == self.run_sha256,
                "run receipt identity changed",
            )
            _need(
                (
                    len(records) == 2
                    and records[1][0] == generation - 1
                    and doc["parent_sha256"] == records[1][2]
                )
                or (len(records) == 1 and generation == 0 and doc["parent_sha256"] is None),
                "run receipt parent changed",
            )
            return RunReceipt(**{**doc, "phase": RunPhase(doc["phase"]), "receipt_sha256": digest})
        except (KeyError, TypeError, ValueError) as exc:
            raise RunControlError("malformed run receipt") from exc

    def _append(
        self,
        connection: sqlite3.Connection,
        previous: RunReceipt | None,
        **changes,
    ) -> RunReceipt:
        state = {
            "run_sha256": self.run_sha256,
            "control_epoch": 0,
            "phase": RunPhase.PREPARED,
            "control_enabled": False,
            "control_pending": False,
            "committed_step": 0,
            "tokens_seen": 0,
            "pending_step": None,
            "pending_tokens": None,
            "checkpoint_epoch": 0,
        }
        if previous is not None:
            state.update(asdict(previous))
            state.pop("receipt_sha256")
        state.update(
            changes,
            generation=0 if previous is None else previous.generation + 1,
            parent_sha256=None if previous is None else previous.receipt_sha256,
        )
        digest = hash_json(state)
        with connection:
            connection.execute(
                "INSERT INTO events (generation, payload, digest) VALUES (?, ?, ?)",
                (state["generation"], json.dumps(state, sort_keys=True, allow_nan=False), digest),
            )
        return RunReceipt(**state, receipt_sha256=digest)

    def prepare(self, trainer: Trainer, identity: CheckpointIdentity) -> RunReceipt:
        """Create the initial receipt once for a bound run manifest."""
        identity_sha256 = self._initial_identity(identity)
        self._require_fresh(trainer)
        self._recipe(trainer, identity)
        self._prepare_base()
        with (
            exclusive_recovery_lock(self.namespace) as root,
            closing(self._connect(root, create=True)) as connection,
        ):
            _need(self._read(connection) is None, "run has already been prepared")
            path = self._checkpoint_path(root, 0, 0)
            path.parent.mkdir(exist_ok=True)
            fsync_parent_directory(self.namespace)
            fsync_parent_directory(path.parent)
            if path.exists() or path.is_symlink():
                checkpoint_id = self._recover_checkpoint(path, identity)
                self._restore(path, trainer, checkpoint_id, step=0, tokens=0)
            else:
                checkpoint_id = self._save_checkpoint(path, trainer, identity)
            return self._append(
                connection,
                None,
                identity_sha256=identity_sha256,
                checkpoint_id=checkpoint_id,
            )

    def _prepare_base(self) -> None:
        base = self.namespace.parent
        missing = []
        parent = base
        while not parent.exists():
            _need(parent != parent.parent, "run storage volume is unavailable")
            missing.append(parent)
            parent = parent.parent
        if parent != parent.parent:
            fsync_parent_directory(parent)
        for directory in reversed(missing):
            directory.mkdir(exist_ok=True)
            fsync_parent_directory(directory)

    def status(self) -> RunReceipt:
        """Read the current durable receipt."""
        with closing(self._connect(self.namespace)) as connection:
            receipt = self._read(connection)
            _need(receipt is not None, "run has not been prepared")
            assert receipt is not None
            return receipt

    def _initial_identity(self, identity: CheckpointIdentity) -> str:
        identity.validate()
        _need(
            identity.run_manifest_hash == self.run_sha256
            and identity.step == 0
            and identity.tokens_seen == 0,
            "initial checkpoint identity does not bind this run",
        )
        return hash_json(asdict(identity))

    @staticmethod
    def _require_fresh(trainer: Trainer) -> None:
        trainer.assert_checkpoint_safe()
        _need(
            trainer.optimizer_step == 0 and trainer.tokens_seen == 0,
            "a writer must start with a fresh trainer",
        )

    @staticmethod
    def _recipe(trainer: Trainer, identity: CheckpointIdentity) -> TrainingRecipe | None:
        training = identity.training_config.get("training")
        if not isinstance(training, Mapping) or {"recipe", "recipe_sha256"}.isdisjoint(training):
            return None
        try:
            recipe = TrainingRecipe.from_dict(training.get("recipe"))
        except (TypeError, ValueError) as exc:
            raise RunControlError("invalid bound training recipe") from exc
        _need(
            training.get("recipe_sha256") == recipe.identity_sha256()
            and trainer.config == recipe.trainer,
            "trainer differs from bound training recipe",
        )
        return recipe

    @staticmethod
    def _checkpoint_path(root: Path, epoch: int, step: int) -> Path:
        return root / "checkpoints" / f"{epoch:08d}-{step:08d}"

    @staticmethod
    def _save_checkpoint(path: Path, trainer: Trainer, identity: CheckpointIdentity) -> str:
        trainer.assert_checkpoint_safe()
        manifest = save_trainer_checkpoint(
            path, model=trainer.model, trainer=trainer, identity=identity
        )
        return manifest["checkpoint_id"]

    @staticmethod
    def _recover_checkpoint(path: Path, expected: CheckpointIdentity) -> str:
        manifest = verify_checkpoint(path)
        identity = {key: manifest["identity"][key] for key in asdict(expected)}
        _need(
            hash_json(identity) == hash_json(asdict(expected)),
            "checkpoint is not bound to the reserved run position",
        )
        # The previous writer may have exited after the rename but before fsync.
        fsync_parent_directory(path)
        return manifest["checkpoint_id"]

    def _pending_checkpoint(
        self,
        root: Path,
        current: RunReceipt,
        identity: CheckpointIdentity,
    ) -> tuple[CheckpointIdentity, str] | None:
        if current.pending_step is None or current.pending_tokens is None:
            return None
        path = self._checkpoint_path(root, current.control_epoch, current.pending_step)
        if not path.exists() and not path.is_symlink():
            return None
        expected = replace(identity, step=current.pending_step, tokens_seen=current.pending_tokens)
        return expected, self._recover_checkpoint(path, expected)

    def _restore(
        self,
        path: Path,
        trainer: Trainer,
        checkpoint_id: str,
        *,
        step: int,
        tokens: int,
    ) -> None:
        load_trainer_checkpoint(
            path,
            model=trainer.model,
            trainer=trainer,
            expected_checkpoint_id=checkpoint_id,
            expected_run_manifest_hash=self.run_sha256,
            expected_step=step,
            expected_tokens_seen=tokens,
        )

    def _settle_boundary(
        self,
        connection: sqlite3.Connection,
        receipt: RunReceipt,
        control: Callable[[RunBoundary], RunPhase | None] | None,
    ) -> RunReceipt:
        if not receipt.control_pending:
            return receipt
        _need(control is not None, "recovery requires the interrupted control policy")
        assert control is not None
        command = control(
            RunBoundary(
                receipt.run_sha256,
                receipt.committed_step,
                receipt.tokens_seen,
                receipt.checkpoint_id,
            )
        )
        _need(
            command is None or command is RunPhase.PAUSED or command is RunPhase.STOPPED,
            "unsupported boundary control command",
        )
        return self._append(
            connection,
            receipt,
            control_pending=False,
            phase=RunPhase.RUNNING if command is None else command,
        )

    def run(
        self,
        trainer: Trainer,
        identity: CheckpointIdentity,
        batches: Callable[[RunReceipt], Iterable[Batch]],
        *,
        expected_receipt_sha256: str,
        recover: bool = False,
        control: Callable[[RunBoundary], RunPhase | None] | None = None,
    ) -> RunReceipt:
        """Advance one writer from the durable position supplied to the batch factory.

        Control must return the same decision when recovery replays a boundary.
        The factory must resume at the committed step and token position it receives.
        """
        with (
            exclusive_recovery_lock(self.namespace) as root,
            closing(self._connect(root)) as connection,
        ):
            current = self._read(connection)
            _need(
                current is not None and current.receipt_sha256 == expected_receipt_sha256,
                "stale run receipt",
            )
            assert current is not None
            phase = current.phase
            _need(
                (recover and phase in {RunPhase.FAILED, RunPhase.RUNNING})
                or (not recover and phase in {RunPhase.PREPARED, RunPhase.PAUSED}),
                "run phase does not permit this writer",
            )
            _need(
                not recover or not current.control_enabled or control is not None,
                "recovery requires the interrupted control policy",
            )
            self._require_fresh(trainer)
            _need(
                current.identity_sha256 == self._initial_identity(identity),
                "checkpoint identity differs from prepared run",
            )
            recipe = self._recipe(trainer, identity)
            pending = self._pending_checkpoint(root, current, identity) if recover else None
            if pending is not None:
                expected, checkpoint_id = pending
                self._restore(
                    self._checkpoint_path(root, current.control_epoch, expected.step),
                    trainer,
                    checkpoint_id,
                    step=expected.step,
                    tokens=expected.tokens_seen,
                )
                current = self._append(
                    connection,
                    current,
                    pending_step=None,
                    pending_tokens=None,
                    committed_step=expected.step,
                    tokens_seen=expected.tokens_seen,
                    checkpoint_epoch=current.control_epoch,
                    checkpoint_id=checkpoint_id,
                    control_pending=current.control_enabled,
                )
            else:
                self._restore(
                    self._checkpoint_path(root, current.checkpoint_epoch, current.committed_step),
                    trainer,
                    current.checkpoint_id,
                    step=current.committed_step,
                    tokens=current.tokens_seen,
                )
            try:
                if recover and current.control_pending:
                    current = self._settle_boundary(connection, current, control)
                    if current.phase in {RunPhase.PAUSED, RunPhase.STOPPED}:
                        return current
                current = self._append(
                    connection,
                    current,
                    control_epoch=current.control_epoch + 1,
                    phase=RunPhase.RUNNING,
                    pending_step=None,
                    pending_tokens=None,
                    control_enabled=control is not None,
                )
                return self._advance(
                    root,
                    connection,
                    current,
                    trainer,
                    identity,
                    recipe,
                    batches,
                    control,
                )
            except BaseException:
                latest = self._read(connection)
                if latest is not None and latest.phase is RunPhase.RUNNING:
                    self._append(connection, latest, phase=RunPhase.FAILED)
                raise

    def _advance(
        self,
        root: Path,
        connection: sqlite3.Connection,
        current: RunReceipt,
        trainer: Trainer,
        identity: CheckpointIdentity,
        recipe: TrainingRecipe | None,
        batches: Callable[[RunReceipt], Iterable[Batch]],
        control: Callable[[RunBoundary], RunPhase | None] | None,
    ) -> RunReceipt:
        if trainer.optimizer_step == trainer.config.max_steps:
            return self._append(connection, current, phase=RunPhase.COMPLETED)
        iterator = iter(batches(current))
        while trainer.optimizer_step < trainer.config.max_steps:
            step = current.committed_step + 1
            current = self._append(connection, current, pending_step=step)
            metrics = None
            for _ in range(trainer.config.gradient_accumulation_steps):
                try:
                    batch = next(iterator)
                except StopIteration as exc:
                    raise RunControlError("input ended before the reserved optimizer step") from exc
                metrics = trainer.train_microbatch(batch)
            _need(
                metrics is not None
                and metrics.optimizer_stepped
                and metrics.optimizer_step == step,
                "trainer did not commit the reserved optimizer step",
            )
            _need(
                recipe is None or trainer.tokens_seen <= recipe.maximum_total_exposures,
                "optimizer step exceeds the bound training exposure budget",
            )
            current = self._append(connection, current, pending_tokens=trainer.tokens_seen)
            checkpoint_id = self._save_checkpoint(
                self._checkpoint_path(root, current.control_epoch, step),
                trainer,
                replace(identity, step=step, tokens_seen=trainer.tokens_seen),
            )
            current = self._append(
                connection,
                current,
                pending_step=None,
                pending_tokens=None,
                committed_step=step,
                tokens_seen=trainer.tokens_seen,
                checkpoint_epoch=current.control_epoch,
                checkpoint_id=checkpoint_id,
                control_pending=current.control_enabled,
            )
            current = self._settle_boundary(connection, current, control)
            if current.phase in {RunPhase.PAUSED, RunPhase.STOPPED}:
                return current
        return self._append(connection, current, phase=RunPhase.COMPLETED)
