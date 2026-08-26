from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from safetensors.numpy import load as load_safetensors_bytes
from safetensors.numpy import save as save_safetensors_bytes

from twelve_six.checkpoint.core import (
    CheckpointCompatibilityError,
    CheckpointIdentity,
    CheckpointIntegrityError,
    hash_json,
    load_checkpoint,
    save_checkpoint,
)
from twelve_six.checkpoint.state_tree import pack_state_tree, unpack_state_tree
from twelve_six.model import TwelveSixDecoder, load_stage_config

ROOT = Path(__file__).resolve().parents[1]
MODEL341_CONFIG = ROOT / "configs/candidates/model341_20m_candidate_a.json"
MODEL341_BASE_SHA = "e4ff486fd90802fc123bebf60eed4e59196a98df"
MODEL341_MODEL_SPEC_SHA256 = "fbff24d561a2818453554d58ca23fc6ace3303b078f1935a8576c4565bd92441"
MODEL341_PARAMETER_COUNT = 20_613_440
TOKENIZER_SHA256 = "b" * 64
TOKENIZER_VOCAB_SHA256 = "c" * 64
DATASET_SHA256 = "d" * 64
RUN_SHA256 = "e" * 64
ENVIRONMENT_LOCK_SHA256 = "f" * 64


def _stage():
    stage = load_stage_config(MODEL341_CONFIG)
    assert stage.stage == "MODEL-341-20M-CANDIDATE-A"
    assert stage.expected_parameters == MODEL341_PARAMETER_COUNT
    assert stage.model.parameter_count() == MODEL341_PARAMETER_COUNT
    assert stage.model.identity_sha256() == MODEL341_MODEL_SPEC_SHA256
    return stage


def _new_model(seed: int) -> TwelveSixDecoder:
    stage = _stage()
    torch.manual_seed(seed)
    model = TwelveSixDecoder(stage.model, stage.init)
    assert sum(parameter.numel() for parameter in model.parameters()) == MODEL341_PARAMETER_COUNT
    return model


def _identity(*, optimizer: dict[str, Any] | None = None) -> CheckpointIdentity:
    stage = _stage()
    return CheckpointIdentity(
        git_sha=MODEL341_BASE_SHA,
        model_spec=stage.model.to_dict(),
        parameter_count=MODEL341_PARAMETER_COUNT,
        tokenizer_hash=TOKENIZER_SHA256,
        tokenizer_vocab_hash=TOKENIZER_VOCAB_SHA256,
        dataset_manifest_hash=DATASET_SHA256,
        run_manifest_hash=RUN_SHA256,
        training_config={"purpose": "next100075-production-corruption-matrix"},
        seed=341075,
        precision="float32",
        step=1,
        tokens_seen=16,
        optimizer=optimizer or {"name": "none"},
        scheduler=None,
        environment_lock_hash=ENVIRONMENT_LOCK_SHA256,
    )


def _rebind_manifest(checkpoint: Path, changed_payloads: tuple[str, ...] = ()) -> None:
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in changed_payloads:
        payload = (checkpoint / name).read_bytes()
        manifest["files"][name] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
    manifest["checkpoint_id"] = hash_json(
        {"identity": manifest["identity"], "files": manifest["files"]}
    )
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (checkpoint / "MANIFEST.sha256").write_text(
        f"{manifest_sha}  manifest.json\n", encoding="ascii"
    )


def _rewrite_state(checkpoint: Path, combined_state: Any) -> None:
    packed = pack_state_tree(combined_state)
    (checkpoint / "state.safetensors").write_bytes(save_safetensors_bytes(packed.tensors))
    (checkpoint / "state.json").write_text(
        json.dumps(packed.tree, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    _rebind_manifest(checkpoint, ("state.safetensors", "state.json"))


def _copy_checkpoint(source: Path, destination: Path) -> Path:
    shutil.copytree(source, destination)
    return destination


def _sentinel(model: TwelveSixDecoder) -> tuple[str, torch.Tensor]:
    name, value = next(iter(model.state_dict().items()))
    return name, value.detach().clone()


def _assert_unmutated(model: TwelveSixDecoder, sentinel: tuple[str, torch.Tensor]) -> None:
    name, before = sentinel
    torch.testing.assert_close(model.state_dict()[name], before, rtol=0.0, atol=0.0)


@pytest.fixture(scope="session")
def model341_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    checkpoint = tmp_path_factory.mktemp("next100075") / "model341-valid"
    model = _new_model(341075)
    save_checkpoint(checkpoint, model=model, identity=_identity())
    return checkpoint


@pytest.fixture(scope="session")
def model341_sgd_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    checkpoint = tmp_path_factory.mktemp("next100075-sgd") / "model341-sgd"
    model = _new_model(341076)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    first_parameter = next(model.parameters())
    first_parameter.grad = torch.ones_like(first_parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    save_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        identity=_identity(optimizer={"name": "SGD", "lr": 0.05, "momentum": 0.9}),
    )
    return checkpoint


@pytest.fixture(scope="session")
def model341_target() -> TwelveSixDecoder:
    return _new_model(941075)


def test_control_exact_model341_checkpoint_loads(model341_checkpoint: Path) -> None:
    target = _new_model(941076)
    result = load_checkpoint(
        model341_checkpoint,
        model=target,
        restore_rng=False,
        expected_git_sha=MODEL341_BASE_SHA,
        expected_model_spec_hash=MODEL341_MODEL_SPEC_SHA256,
        expected_tokenizer_hash=TOKENIZER_SHA256,
    )
    assert result.manifest["identity"]["parameter_count"] == MODEL341_PARAMETER_COUNT
    assert result.manifest["identity"]["model_spec_hash"] == MODEL341_MODEL_SPEC_SHA256


def test_case_01_missing_tensor_rejected_before_mutation(
    tmp_path: Path, model341_checkpoint: Path, model341_target: TwelveSixDecoder
) -> None:
    checkpoint = _copy_checkpoint(model341_checkpoint, tmp_path / "missing-tensor")
    arrays = load_safetensors_bytes((checkpoint / "weights.safetensors").read_bytes())
    arrays.pop("token_embedding.weight")
    (checkpoint / "weights.safetensors").write_bytes(save_safetensors_bytes(arrays))
    _rebind_manifest(checkpoint, ("weights.safetensors",))
    sentinel = _sentinel(model341_target)

    with pytest.raises(CheckpointCompatibilityError, match="state_dict keys differ"):
        load_checkpoint(checkpoint, model=model341_target, restore_rng=False)
    _assert_unmutated(model341_target, sentinel)


def test_case_02_extra_tensor_rejected_before_mutation(
    tmp_path: Path, model341_checkpoint: Path, model341_target: TwelveSixDecoder
) -> None:
    checkpoint = _copy_checkpoint(model341_checkpoint, tmp_path / "extra-tensor")
    arrays = load_safetensors_bytes((checkpoint / "weights.safetensors").read_bytes())
    arrays["__next100075_extra__"] = np.zeros((1,), dtype=np.float32)
    (checkpoint / "weights.safetensors").write_bytes(save_safetensors_bytes(arrays))
    _rebind_manifest(checkpoint, ("weights.safetensors",))
    sentinel = _sentinel(model341_target)

    with pytest.raises(CheckpointCompatibilityError, match="state_dict keys differ"):
        load_checkpoint(checkpoint, model=model341_target, restore_rng=False)
    _assert_unmutated(model341_target, sentinel)


def test_case_03_shape_mismatch_rejected_before_mutation(
    tmp_path: Path, model341_checkpoint: Path, model341_target: TwelveSixDecoder
) -> None:
    checkpoint = _copy_checkpoint(model341_checkpoint, tmp_path / "shape-mismatch")
    arrays = load_safetensors_bytes((checkpoint / "weights.safetensors").read_bytes())
    arrays["token_embedding.weight"] = arrays["token_embedding.weight"][:1].copy()
    (checkpoint / "weights.safetensors").write_bytes(save_safetensors_bytes(arrays))
    _rebind_manifest(checkpoint, ("weights.safetensors",))
    sentinel = _sentinel(model341_target)

    with pytest.raises(CheckpointCompatibilityError, match="shape mismatch"):
        load_checkpoint(checkpoint, model=model341_target, restore_rng=False)
    _assert_unmutated(model341_target, sentinel)


def test_case_04_dtype_mismatch_rejected_before_mutation(
    tmp_path: Path, model341_checkpoint: Path, model341_target: TwelveSixDecoder
) -> None:
    checkpoint = _copy_checkpoint(model341_checkpoint, tmp_path / "dtype-mismatch")
    arrays = load_safetensors_bytes((checkpoint / "weights.safetensors").read_bytes())
    arrays["token_embedding.weight"] = arrays["token_embedding.weight"].astype(np.float64)
    (checkpoint / "weights.safetensors").write_bytes(save_safetensors_bytes(arrays))
    _rebind_manifest(checkpoint, ("weights.safetensors",))
    sentinel = _sentinel(model341_target)

    with pytest.raises(CheckpointCompatibilityError, match="dtype mismatch"):
        load_checkpoint(checkpoint, model=model341_target, restore_rng=False)
    _assert_unmutated(model341_target, sentinel)


def test_case_05_manifest_hash_mismatch_rejected_before_mutation(
    tmp_path: Path, model341_checkpoint: Path, model341_target: TwelveSixDecoder
) -> None:
    checkpoint = _copy_checkpoint(model341_checkpoint, tmp_path / "manifest-mismatch")
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["identity"]["step"] = 2
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    sentinel = _sentinel(model341_target)

    with pytest.raises(CheckpointIntegrityError):
        load_checkpoint(checkpoint, model=model341_target, restore_rng=False)
    _assert_unmutated(model341_target, sentinel)


def test_case_06_optimizer_state_corruption_rejected_before_mutation(
    tmp_path: Path, model341_sgd_checkpoint: Path
) -> None:
    checkpoint = _copy_checkpoint(model341_sgd_checkpoint, tmp_path / "optimizer-corruption")
    tree = json.loads((checkpoint / "state.json").read_text(encoding="utf-8"))
    arrays = load_safetensors_bytes((checkpoint / "state.safetensors").read_bytes())
    combined = copy.deepcopy(unpack_state_tree(tree, arrays))
    optimizer_state = combined["optimizer"]["state"]
    first_parameter_state = next(iter(optimizer_state.values()))
    momentum = first_parameter_state["momentum_buffer"]
    first_parameter_state["momentum_buffer"] = momentum.reshape(-1)[:1].clone()
    _rewrite_state(checkpoint, combined)

    target = _new_model(941077)
    target_optimizer = torch.optim.SGD(target.parameters(), lr=0.05, momentum=0.9)
    sentinel = _sentinel(target)

    with pytest.raises(CheckpointCompatibilityError, match="optimizer state tensor shape mismatch"):
        load_checkpoint(
            checkpoint,
            model=target,
            optimizer=target_optimizer,
            restore_rng=False,
        )
    _assert_unmutated(target, sentinel)
    assert target_optimizer.state == {}


def test_case_07_rng_corruption_rejected_before_mutation(
    tmp_path: Path, model341_checkpoint: Path, model341_target: TwelveSixDecoder
) -> None:
    checkpoint = _copy_checkpoint(model341_checkpoint, tmp_path / "rng-corruption")
    tree = json.loads((checkpoint / "state.json").read_text(encoding="utf-8"))
    arrays = load_safetensors_bytes((checkpoint / "state.safetensors").read_bytes())
    combined = copy.deepcopy(unpack_state_tree(tree, arrays))
    combined["rng"]["torch"]["cpu"] = "not-a-torch-rng-state"
    _rewrite_state(checkpoint, combined)
    sentinel = _sentinel(model341_target)

    with pytest.raises(CheckpointCompatibilityError, match="torch CPU RNG state is invalid"):
        load_checkpoint(checkpoint, model=model341_target, restore_rng=True)
    _assert_unmutated(model341_target, sentinel)


def test_case_08_counter_corruption_rejected_before_mutation(
    tmp_path: Path, model341_checkpoint: Path, model341_target: TwelveSixDecoder
) -> None:
    checkpoint = _copy_checkpoint(model341_checkpoint, tmp_path / "counter-corruption")
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["identity"]["step"] = -1
    manifest["identity"]["tokens_seen"] = -32
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    _rebind_manifest(checkpoint)
    sentinel = _sentinel(model341_target)

    with pytest.raises(CheckpointIntegrityError, match="step and tokens_seen"):
        load_checkpoint(checkpoint, model=model341_target, restore_rng=False)
    _assert_unmutated(model341_target, sentinel)


def test_case_09_modelspec_mismatch_rejected_before_mutation(
    model341_checkpoint: Path, model341_target: TwelveSixDecoder
) -> None:
    sentinel = _sentinel(model341_target)
    with pytest.raises(CheckpointCompatibilityError, match="identity mismatch"):
        load_checkpoint(
            model341_checkpoint,
            model=model341_target,
            restore_rng=False,
            expected_model_spec_hash="0" * 64,
        )
    _assert_unmutated(model341_target, sentinel)


def test_case_10_tokenizer_mismatch_rejected_before_mutation(
    model341_checkpoint: Path, model341_target: TwelveSixDecoder
) -> None:
    sentinel = _sentinel(model341_target)
    with pytest.raises(CheckpointCompatibilityError, match="identity mismatch"):
        load_checkpoint(
            model341_checkpoint,
            model=model341_target,
            restore_rng=False,
            expected_tokenizer_hash="0" * 64,
        )
    _assert_unmutated(model341_target, sentinel)


def test_case_11_partial_write_rejected_before_mutation(
    tmp_path: Path, model341_checkpoint: Path, model341_target: TwelveSixDecoder
) -> None:
    checkpoint = _copy_checkpoint(model341_checkpoint, tmp_path / "partial-write")
    weights_path = checkpoint / "weights.safetensors"
    payload = weights_path.read_bytes()
    weights_path.write_bytes(payload[: max(1, len(payload) // 2)])
    sentinel = _sentinel(model341_target)

    with pytest.raises(CheckpointIntegrityError):
        load_checkpoint(checkpoint, model=model341_target, restore_rng=False)
    _assert_unmutated(model341_target, sentinel)
