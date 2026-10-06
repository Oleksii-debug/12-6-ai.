"""D05 must not publish checksum-valid checkpoints containing non-finite weights.

Tiny synthetic CPU serialization cases; no real training or scale acceptance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from twelve_six.checkpoint import CheckpointIdentity, core
from twelve_six.checkpoint.core import CheckpointIntegrityError


def _identity() -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha="a" * 40,
        model_spec={"kind": "nonfinite-model-publication", "width": 3},
        parameter_count=3,
        tokenizer_hash="b" * 64,
        tokenizer_vocab_hash="c" * 64,
        dataset_manifest_hash="d" * 64,
        run_manifest_hash="e" * 64,
        training_config={"steps": 1},
        seed=703,
        precision="fp32",
        step=0,
        tokens_seen=0,
        optimizer={"name": "AdamW"},
        scheduler=None,
        environment_lock_hash="f" * 64,
    )


class _DetachedTensorModel(torch.nn.Module):
    def __init__(self, *, bad_field: str, dtype: torch.dtype) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.1, 0.2, 0.3]))
        self.register_buffer("statistic", torch.ones(3, dtype=dtype))
        self.bad_field = bad_field

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        # Return an invalid detached tensor, leaving the live module finite.
        detached = state[self.bad_field].clone()
        detached[0] = float("nan")
        state[self.bad_field] = detached
        return state


class _NumpyStateModel:
    def __init__(self, *, dtype: str, value: complex) -> None:
        self.array = np.asarray([value, 1], dtype=np.dtype(dtype))

    def state_dict(self):
        return {"array": self.array}


@pytest.mark.parametrize(
    ("field", "dtype"),
    [
        ("weight", torch.float32),
        ("statistic", torch.float32),
        ("statistic", torch.bfloat16),
    ],
    ids=["detached-weight", "float-buffer", "bfloat16-buffer"],
)
def test_detached_nonfinite_model_tensor_rejected_before_any_checkpoint_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    field: str, dtype: torch.dtype,
) -> None:
    model = _DetachedTensorModel(bad_field=field, dtype=dtype)
    original = {k: v.detach().clone() for k, v in model.named_parameters()}
    attempted_writes: list[bool] = []

    def forbidden_write(*args, **kwargs) -> None:
        attempted_writes.append(True)
        raise AssertionError("model finiteness must precede SafeTensors publication")

    monkeypatch.setattr(core, "save_safetensors", forbidden_write)
    target = tmp_path / f"not-published-{field}-дані"
    with pytest.raises(CheckpointIntegrityError, match="non-finite model tensor"):
        core.save_checkpoint(target, model=model, identity=_identity())
    assert attempted_writes == []
    assert not target.exists()
    assert not list(tmp_path.glob(f".{target.name}.tmp-*"))
    for name, old in original.items():
        torch.testing.assert_close(dict(model.named_parameters())[name], old, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("dtype", "value"),
    [
        ("float32", float("nan")),
        ("float64", float("inf")),
        ("complex64", complex(float("nan"), 0.0)),
    ],
)
def test_nonfinite_numpy_model_state_rejected(
    tmp_path: Path, dtype: str, value: complex,
) -> None:
    model = _NumpyStateModel(dtype=dtype, value=value)
    target = tmp_path / "nonfinite-array"
    with pytest.raises(CheckpointIntegrityError, match="non-finite model tensor"):
        core.save_checkpoint(target, model=model, identity=_identity())
    assert not target.exists()
    assert not list(tmp_path.glob(".nonfinite-array.tmp-*"))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_finite_model_state_still_publishes_immutable_verified_checkpoint(
    tmp_path: Path, dtype: torch.dtype,
) -> None:
    model = torch.nn.Linear(3, 1).to(dtype=dtype)
    directory = tmp_path / "finite-модель із пробілами"
    manifest = core.save_checkpoint(directory, model=model, identity=_identity())
    verified = core.verify_checkpoint(directory)
    assert verified["checkpoint_id"] == manifest["checkpoint_id"]
    assert directory.is_dir()
    assert not list(tmp_path.glob(f".{directory.name}.tmp-*"))


@pytest.mark.parametrize(
    ("dtype", "bad"),
    [
        (torch.float32, np.array([float("nan"), 1], dtype=np.float32)),
        (torch.bfloat16, np.array([0x7FC1, 0x3F80], dtype=np.uint16)),
        (torch.complex64, np.array([complex(float("nan"), 1), 1], dtype=np.complex64)),
    ],
    ids=["float-nan", "bfloat16-bits-nan", "complex-nan"],
)
def test_nonfinite_verified_tensor_refused_before_application(
    dtype: torch.dtype, bad: np.ndarray,
) -> None:
    destination = torch.nn.Parameter(torch.zeros(2, dtype=dtype))
    original = destination.detach().clone()
    with pytest.raises(
        core.CheckpointCompatibilityError,
        match="checkpoint model tensor contains non-finite values",
    ):
        core._materialize_for_target(bad, destination)
    torch.testing.assert_close(destination.detach(), original, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("dtype", "bad"),
    [
        ("float32", np.array([float("nan"), 1], dtype=np.float32)),
        ("float64", np.array([float("inf"), 1], dtype=np.float64)),
        ("complex64", np.array([complex(float("nan"), 1), 1], dtype=np.complex64)),
    ],
)
def test_nonfinite_verified_numpy_target_refused_before_application(
    dtype: str, bad: np.ndarray,
) -> None:
    destination = np.zeros(2, dtype=dtype)
    with pytest.raises(
        core.CheckpointCompatibilityError,
        match="checkpoint model tensor contains non-finite values",
    ):
        core._materialize_for_target(bad, destination)
    np.testing.assert_array_equal(destination, np.zeros_like(destination))


def test_nonfinite_model_payload_cannot_partially_apply_other_valid_weights() -> None:
    model = torch.nn.Linear(2, 2)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    payload = {
        "weight": np.full((2, 2), float("nan"), dtype=np.float32),
        "bias": np.array([1.0, 2.0], dtype=np.float32),
    }
    with pytest.raises(
        core.CheckpointCompatibilityError,
        match="checkpoint model tensor contains non-finite values",
    ):
        core._prepare_model_weights(model, payload, strict=True)
    for name, saved in before.items():
        torch.testing.assert_close(model.state_dict()[name], saved, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_clean_model_payload_remains_materializable(
    dtype: torch.dtype,
) -> None:
    model = torch.nn.Linear(3, 1).to(dtype=dtype)
    payload = core._model_state_to_numpy(model)
    restored = core._prepare_model_weights(model, payload, strict=True)
    for name, expected in model.state_dict().items():
        torch.testing.assert_close(restored[name], expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_checksum_valid_nonfinite_legacy_checkpoint_refused_before_model_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype,
) -> None:
    """A resealed malformed file passes SHA checks but cannot enter live weights."""
    import json

    from safetensors.numpy import load_file, save_file

    source = torch.nn.Linear(3, 1).to(dtype=dtype)
    directory = tmp_path / "resealed-nonfinite-дані"
    core.save_checkpoint(directory, model=source, identity=_identity())
    weights_path = directory / core.WEIGHTS_NAME
    arrays = load_file(str(weights_path))
    if dtype == torch.bfloat16:
        assert arrays["weight"].dtype == np.uint16
        arrays["weight"].reshape(-1)[0] = np.uint16(0x7FC1)
    else:
        arrays["weight"].reshape(-1)[0] = np.float32(float("nan"))
    save_file(arrays, str(weights_path))

    manifest_path = directory / core.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][core.WEIGHTS_NAME] = {
        "sha256": core.sha256_file(weights_path),
        "bytes": weights_path.stat().st_size,
    }
    manifest["checkpoint_id"] = core.hash_json({
        "identity": manifest["identity"], "files": manifest["files"],
    })
    core._write_json(manifest_path, manifest)
    manifest_digest = core.sha256_file(manifest_path)
    (directory / core.MANIFEST_CHECKSUM_NAME).write_text(
        f"{manifest_digest}  {core.MANIFEST_NAME}\n", encoding="ascii",
    )
    verified = core.verify_checkpoint(directory)
    assert verified["checkpoint_id"] == manifest["checkpoint_id"]

    target = torch.nn.Linear(3, 1).to(dtype=dtype)
    before = {name: tensor.detach().clone() for name, tensor in target.state_dict().items()}
    applied: list[bool] = []

    def forbidden_apply(*args, **kwargs) -> None:
        applied.append(True)
        raise AssertionError("non-finite verified model may never be applied")

    monkeypatch.setattr(core, "_apply_model_weights", forbidden_apply)
    with pytest.raises(
        core.CheckpointCompatibilityError,
        match="checkpoint model tensor contains non-finite values",
    ):
        core.load_checkpoint(
            directory, model=target, strict_model=True, restore_rng=False,
        )
    assert applied == []
    for name, original in before.items():
        torch.testing.assert_close(target.state_dict()[name], original, rtol=0, atol=0)
