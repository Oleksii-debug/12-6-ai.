from __future__ import annotations

import json
from pathlib import Path

import pytest

from twelve_six.post_base.contract import (
    CanonicalBasePolicy,
    PostBaseConsumptionContract,
    PostBaseStage,
    TokenizerCompatibility,
    snapshot_directory,
)
from twelve_six.post_base.sft_recovery import (
    POSTBASE352_TOKENIZER_ID,
    TERMINAL_SCHEMA,
    _verify_active_pointer,
    terminalize_sft_mechanics,
)
from twelve_six.post_base.sft_runner import (
    MAX_MECHANICS_STEPS,
    FixtureProvenance,
    FixtureSourceKind,
    MechanicsSplit,
    SFTCheckpointStore,
    SFTMechanicsDataset,
    SFTMechanicsExample,
    SFTMechanicsPlan,
)
from twelve_six.posttraining.contracts import CheckpointRef, LineageKind


class RecoveryToyBackend:
    backend_id = "local-recovery-toy-sft-v1"

    @staticmethod
    def _state(state: object) -> dict[str, float]:
        if not isinstance(state, dict) or "weight" not in state:
            raise TypeError("invalid toy state")
        return state

    def load_input_checkpoint(self, checkpoint_root: Path) -> object:
        payload = json.loads((checkpoint_root / "weights.json").read_text(encoding="utf-8"))
        return {"weight": float(payload["weight"])}

    def train_step(
        self,
        state: object,
        example: SFTMechanicsExample,
        *,
        step: int,
        seed: int,
    ) -> dict[str, float]:
        del step, seed
        toy = self._state(state)
        target = float(example.assistant_text)
        error = target - toy["weight"]
        loss = error * error
        toy["weight"] += 0.5 * error
        return {"loss": loss}

    def evaluate(
        self,
        state: object,
        examples: tuple[SFTMechanicsExample, ...],
    ) -> dict[str, float]:
        toy = self._state(state)
        losses = [
            (float(example.assistant_text) - toy["weight"]) ** 2
            for example in examples
        ]
        return {"mse": sum(losses) / len(losses)}

    def save_checkpoint(self, state: object, checkpoint_root: Path) -> None:
        toy = self._state(state)
        (checkpoint_root / "weights.json").write_text(
            json.dumps({"weight": toy["weight"]}, sort_keys=True),
            encoding="utf-8",
        )

    def load_checkpoint(self, checkpoint_root: Path) -> object:
        return self.load_input_checkpoint(checkpoint_root)


def _provenance(source_id: str) -> FixtureProvenance:
    return FixtureProvenance(
        source_id=source_id,
        source_kind=FixtureSourceKind.SYNTHETIC_LOCAL,
        source_revision="recovery-fixture-v1",
        generator_id="local:postbase353-recovery-fixture",
    )


def _dataset(*, target: str = "1.0") -> SFTMechanicsDataset:
    return SFTMechanicsDataset(
        dataset_id="postbase353-recovery-fixture-v1",
        train=(
            SFTMechanicsExample(
                record_id="recovery-train-1",
                split=MechanicsSplit.TRAIN,
                user_text="Return the project-owned scalar target.",
                assistant_text=target,
                provenance=_provenance("project:recovery/train-1"),
            ),
            SFTMechanicsExample(
                record_id="recovery-train-2",
                split=MechanicsSplit.TRAIN,
                user_text="Return the same project-owned scalar target.",
                assistant_text=target,
                provenance=_provenance("project:recovery/train-2"),
            ),
        ),
        evaluation=(
            SFTMechanicsExample(
                record_id="recovery-eval-1",
                split=MechanicsSplit.EVALUATION,
                user_text="Evaluate the project-owned scalar target.",
                assistant_text=target,
                provenance=FixtureProvenance(
                    source_id="project:recovery/eval-1",
                    source_kind=FixtureSourceKind.PROJECT_OWNED,
                    source_revision="recovery-fixture-v1",
                ),
            ),
        ),
    )


def _tokenizer(*, tokenizer_id: str = POSTBASE352_TOKENIZER_ID) -> TokenizerCompatibility:
    return TokenizerCompatibility(
        tokenizer_id=tokenizer_id,
        config_sha256="3" * 64,
        vocab_sha256="4" * 64,
        vocab_size=256,
    )


def _base_root(tmp_path: Path) -> Path:
    root = tmp_path / "canonical_base"
    root.mkdir()
    (root / "weights.json").write_text('{"weight": 0.0}', encoding="utf-8")
    return root


def _base_checkpoint(base_root: Path) -> CheckpointRef:
    return CheckpointRef(
        checkpoint_id="fixture/base/postbase353-recovery",
        sha256=snapshot_directory(base_root).identity_sha256,
        git_sha="abcdef0",
        stage="learned_fixture",
        lineage=LineageKind.BASE,
    )


def _contract(
    *,
    dataset: SFTMechanicsDataset,
    base_root: Path,
    tokenizer: TokenizerCompatibility,
) -> PostBaseConsumptionContract:
    return PostBaseConsumptionContract(
        contract_id="postbase353-recovery-contract",
        base_checkpoint=_base_checkpoint(base_root),
        base_policy=CanonicalBasePolicy(),
        tokenizer=tokenizer,
        dataset=dataset.to_contract_provenance(),
        stage=PostBaseStage.COMMUNICATION_SUPERVISION,
    )


def _plan(
    *,
    dataset: SFTMechanicsDataset,
    base_root: Path,
    tokenizer: TokenizerCompatibility | None = None,
    max_steps: int = 2,
) -> SFTMechanicsPlan:
    bound_tokenizer = tokenizer or _tokenizer()
    return SFTMechanicsPlan(
        run_id="postbase353-recovery-exact-head",
        backend_id=RecoveryToyBackend.backend_id,
        contract=_contract(
            dataset=dataset,
            base_root=base_root,
            tokenizer=bound_tokenizer,
        ),
        tokenizer=bound_tokenizer,
        max_steps=max_steps,
        seed=20260826,
    )


def test_terminal_gate_produces_scoped_pass_and_preserves_base(tmp_path: Path) -> None:
    base_root = _base_root(tmp_path)
    base_before = snapshot_directory(base_root)
    dataset = _dataset()
    plan = _plan(dataset=dataset, base_root=base_root)
    experiment_root = tmp_path / "postbase353_recovery"

    terminal = terminalize_sft_mechanics(
        plan=plan,
        dataset=dataset,
        canonical_base_root=base_root,
        experiment_root=experiment_root,
        backend=RecoveryToyBackend(),
    )

    payload = json.loads(terminal.authority_path.read_text(encoding="utf-8"))
    assert payload["schema"] == TERMINAL_SCHEMA
    assert payload["status"] == "PASS"
    assert payload["scope"] == "fixture_only_sft_mechanics"
    assert payload["execution_profile"] == "LOCAL_FREE"
    assert payload["real_communication_campaign_authorized"] is False
    assert payload["executed_steps"] == 2
    assert payload["tokenizer"]["tokenizer_id"] == POSTBASE352_TOKENIZER_ID
    assert payload["dataset"]["manifest_sha256"] == dataset.manifest_sha256
    assert snapshot_directory(base_root) == base_before
    assert terminal.mechanics.canonical_snapshot_before == (
        terminal.mechanics.canonical_snapshot_after
    )

    store = SFTCheckpointStore(experiment_root, plan.run_id, create=False)
    assert store.active_generation() == terminal.mechanics.final_checkpoint.generation
    assert tuple(path.name for path in sorted(store.root.glob("generation_*"))) == (
        "generation_000000",
        "generation_000001",
        "generation_000002",
    )


def test_terminal_gate_rejects_postbase352_logical_tokenizer_before_workspace(
    tmp_path: Path,
) -> None:
    base_root = _base_root(tmp_path)
    dataset = _dataset()
    wrong_tokenizer = _tokenizer(tokenizer_id="fixture-byte-v1")
    plan = _plan(dataset=dataset, base_root=base_root, tokenizer=wrong_tokenizer)
    experiment_root = tmp_path / "must_not_exist"

    with pytest.raises(ValueError, match="s0-byte-v1"):
        terminalize_sft_mechanics(
            plan=plan,
            dataset=dataset,
            canonical_base_root=base_root,
            experiment_root=experiment_root,
            backend=RecoveryToyBackend(),
        )

    assert not experiment_root.exists()


def test_exact_tokenizer_hash_drift_is_rejected_by_consumption_contract(tmp_path: Path) -> None:
    base_root = _base_root(tmp_path)
    dataset = _dataset()
    expected = _tokenizer()
    contract = _contract(dataset=dataset, base_root=base_root, tokenizer=expected)
    drifted = TokenizerCompatibility(
        tokenizer_id=expected.tokenizer_id,
        config_sha256="5" * 64,
        vocab_sha256=expected.vocab_sha256,
        vocab_size=expected.vocab_size,
    )

    with pytest.raises(ValueError, match="exactly match"):
        SFTMechanicsPlan(
            run_id="tokenizer-drift",
            backend_id=RecoveryToyBackend.backend_id,
            contract=contract,
            tokenizer=drifted,
            max_steps=1,
            seed=0,
        )


def test_dataset_identity_drift_is_rejected_before_workspace(tmp_path: Path) -> None:
    base_root = _base_root(tmp_path)
    contracted = _dataset(target="1.0")
    supplied = _dataset(target="2.0")
    plan = _plan(dataset=contracted, base_root=base_root)
    experiment_root = tmp_path / "must_not_exist"

    with pytest.raises(ValueError, match="do not match"):
        terminalize_sft_mechanics(
            plan=plan,
            dataset=supplied,
            canonical_base_root=base_root,
            experiment_root=experiment_root,
            backend=RecoveryToyBackend(),
        )

    assert not experiment_root.exists()


def test_mechanics_step_bound_rejects_overrun(tmp_path: Path) -> None:
    base_root = _base_root(tmp_path)
    dataset = _dataset()
    tokenizer = _tokenizer()
    contract = _contract(dataset=dataset, base_root=base_root, tokenizer=tokenizer)

    with pytest.raises(ValueError, match="max_steps"):
        SFTMechanicsPlan(
            run_id="unbounded-mechanics",
            backend_id=RecoveryToyBackend.backend_id,
            contract=contract,
            tokenizer=tokenizer,
            max_steps=MAX_MECHANICS_STEPS + 1,
            seed=0,
        )


def test_foreign_model_output_fixture_is_rejected() -> None:
    with pytest.raises(ValueError, match="foreign-model"):
        FixtureProvenance(
            source_id="foreign:generator",
            source_kind=FixtureSourceKind.SYNTHETIC_LOCAL,
            source_revision="v1",
            generator_id="foreign:model",
            foreign_model_output=True,
        )


def test_checkpoint_backend_tamper_is_rejected(tmp_path: Path) -> None:
    base_root = _base_root(tmp_path)
    dataset = _dataset()
    plan = _plan(dataset=dataset, base_root=base_root, max_steps=1)
    experiment_root = tmp_path / "postbase353_recovery"
    backend = RecoveryToyBackend()

    terminalize_sft_mechanics(
        plan=plan,
        dataset=dataset,
        canonical_base_root=base_root,
        experiment_root=experiment_root,
        backend=backend,
    )
    store = SFTCheckpointStore(experiment_root, plan.run_id, create=False)
    tampered = store.generation_path(1) / "backend" / "weights.json"
    tampered.write_text('{"weight": 999.0}', encoding="utf-8")

    with pytest.raises(RuntimeError, match="immutable manifest"):
        store.load_state(backend, 1)


def test_active_rollback_pointer_manifest_binding_detects_tamper(tmp_path: Path) -> None:
    base_root = _base_root(tmp_path)
    dataset = _dataset()
    plan = _plan(dataset=dataset, base_root=base_root, max_steps=1)
    experiment_root = tmp_path / "postbase353_recovery"

    terminal = terminalize_sft_mechanics(
        plan=plan,
        dataset=dataset,
        canonical_base_root=base_root,
        experiment_root=experiment_root,
        backend=RecoveryToyBackend(),
    )
    store = SFTCheckpointStore(experiment_root, plan.run_id, create=False)
    pointer = json.loads(store.active_pointer.read_text(encoding="utf-8"))
    pointer["manifest_sha256"] = "0" * 64
    store.active_pointer.write_text(json.dumps(pointer), encoding="utf-8")

    with pytest.raises(RuntimeError, match="does not bind"):
        _verify_active_pointer(
            store,
            expected_generation=terminal.mechanics.final_checkpoint.generation,
        )
