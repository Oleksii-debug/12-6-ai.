"""Plan 3 / Section 1: frozen architecture and scratch-init contract."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from twelve_six.model import (
    BaseInitManifest,
    InitSpec,
    ModelSpec,
    build_canonical_base,
    canonical_json_sha256,
    count_trainable_parameters,
)


def tiny_spec() -> ModelSpec:
    return ModelSpec(
        schema_version=1,
        vocab_size=32,
        max_seq_len=8,
        d_model=16,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=4,
        d_ff=32,
        rope_rotary_dim=4,
    )


def manifest(spec: ModelSpec, init: InitSpec, seed: int = 1234) -> BaseInitManifest:
    return BaseInitManifest(
        schema_version=1,
        model_spec_sha256=spec.identity_sha256(),
        init_spec_sha256=init.identity_sha256(),
        seed=seed,
    )


def test_model_spec_roundtrip_legacy_hash_and_parameter_geometry() -> None:
    spec = tiny_spec()
    assert ModelSpec.from_dict(spec.to_dict()) == spec
    assert spec.identity_sha256() == canonical_json_sha256(spec.to_dict())
    assert spec.q_dim == 16
    assert spec.kv_dim == 8
    assert spec.parameter_count() == spec.parameter_breakdown()["total"]
    assert spec.parameter_breakdown()["attention_weights_per_layer"] == 2 * 16 * (16 + 8)
    assert spec.parameter_breakdown()["mlp_weights_per_layer"] == 3 * 16 * 32


def test_rng_isolated_deterministic_scratch_base_and_tying() -> None:
    spec, init = tiny_spec(), InitSpec()
    init_manifest = manifest(spec, init)
    before = torch.get_rng_state().clone()

    torch.manual_seed(31)
    first = build_canonical_base(spec, init, init_manifest)
    torch.manual_seed(99999)
    second = build_canonical_base(spec, init, init_manifest)
    # Reproducible regardless of caller RNG/launcher state.
    assert all(
        torch.equal(p1, p2) for p1, p2 in zip(first.parameters(), second.parameters())
    )
    assert first.lm_head.weight is first.token_embedding.weight
    assert count_trainable_parameters(first) == spec.parameter_count()
    assert first(torch.tensor([[1, 2, 3]], dtype=torch.long)).logits.shape == (1, 3, 32)

    different = build_canonical_base(spec, init, manifest(spec, init, seed=1235))
    assert not torch.equal(first.token_embedding.weight, different.token_embedding.weight)

    torch.set_rng_state(before)
    state = torch.get_rng_state().clone()
    build_canonical_base(spec, init, init_manifest)
    assert torch.equal(torch.get_rng_state(), state)


def test_scratch_base_seed_never_reseeds_launcher_accelerators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # torch.manual_seed seeds accelerator RNGs even within fork_rng(devices=[]).
    # Detect this side effect on a CPU-only runner; no real GPU is required.
    accelerator_seed_calls: list[int] = []
    monkeypatch.setattr(
        torch.cuda,
        "manual_seed_all",
        lambda seed: accelerator_seed_calls.append(seed),
    )
    spec, init = tiny_spec(), InitSpec()
    cpu_state = torch.get_rng_state().clone()
    build_canonical_base(spec, init, manifest(spec, init))
    assert accelerator_seed_calls == []
    assert torch.equal(torch.get_rng_state(), cpu_state)


def test_manifest_schema_serialization_and_integrity() -> None:
    spec, init = tiny_spec(), InitSpec()
    m = manifest(spec, init)
    assert InitSpec.from_dict(init.to_dict()) == init
    assert BaseInitManifest.from_dict(m.to_dict()) == m
    assert m.identity_sha256() == canonical_json_sha256(m.to_dict())
    assert len(m.identity_sha256()) == 64
    with pytest.raises(ValueError, match="fields must match"):
        BaseInitManifest.from_dict({**m.to_dict(), "unsafe": "extra"})
    with pytest.raises(ValueError, match="fields must match"):
        BaseInitManifest.from_dict({"seed": 7})


@pytest.mark.parametrize(
    "bad",
    [
        {"schema_version": 2},
        {"seed": -1},
        {"seed": True},
        {"seed": 2**63},
        {"weights_origin": "instruct"},
        {"weights_origin": "pretrained"},
        {"ancestor_checkpoint_sha256": "f" * 64},
        {"model_spec_sha256": "not-a-digest"},
        {"init_spec_sha256": "F" * 64},
    ],
)
def test_manifest_rejects_foreign_base_weights_and_malformed_identity(bad: dict) -> None:
    m = manifest(tiny_spec(), InitSpec())
    with pytest.raises(ValueError):
        BaseInitManifest.from_dict({**m.to_dict(), **bad})


def test_manifest_mutation_cannot_bypass_scratch_ancestry_boundary() -> None:
    spec, init = tiny_spec(), InitSpec()
    for field, value in (
        ("weights_origin", "pretrained"),
        ("ancestor_checkpoint_sha256", "a" * 64),
        ("seed", -1),
    ):
        forged = manifest(spec, init)
        object.__setattr__(forged, field, value)
        with pytest.raises(ValueError):
            build_canonical_base(spec, init, forged)


def test_manifest_fails_closed_on_swapped_model_or_init_spec() -> None:
    spec, init = tiny_spec(), InitSpec()
    m = manifest(spec, init)
    with pytest.raises(ValueError, match="ModelSpec identity mismatch"):
        build_canonical_base(replace(spec, vocab_size=33), init, m)
    with pytest.raises(ValueError, match="InitSpec identity mismatch"):
        build_canonical_base(spec, replace(init, std=0.015), m)


@pytest.mark.parametrize(
    "field",
    [
        "attention_bias",
        "mlp_bias",
        "final_norm",
        "tie_word_embeddings",
        "lm_head_bias",
    ],
)
@pytest.mark.parametrize("bad", ["false", 0, 1, None])
def test_model_spec_refuses_ambiguous_boolean_switches(field: str, bad: object) -> None:
    with pytest.raises(TypeError, match="must be a boolean"):
        replace(tiny_spec(), **{field: bad})


@pytest.mark.parametrize(
    "field,bad",
    [
        ("norm_eps", float("nan")),
        ("rope_theta", float("inf")),
        ("attention_dropout", float("nan")),
        ("attention_dropout", 1.0),
        ("rope_rotary_dim", 3),
        ("n_kv_heads", 3),
        ("head_dim", 3),
    ],
)
def test_model_spec_invalid_geometries_and_precision_fail_closed(field: str, bad) -> None:
    with pytest.raises(ValueError):
        replace(tiny_spec(), **{field: bad})


@pytest.mark.parametrize("bad", [True, 1.0, "1", 0, 2])
def test_init_spec_schema_version_is_strict_v1_integer(bad: object) -> None:
    # Python equality permits True == 1 and 1.0 == 1; the wire contract cannot.
    with pytest.raises(ValueError, match="unsupported InitSpec schema_version"):
        InitSpec(schema_version=bad)
    with pytest.raises(ValueError, match="unsupported InitSpec schema_version"):
        InitSpec.from_dict({**InitSpec().to_dict(), "schema_version": bad})


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 0.0, True])
def test_init_spec_rejects_pathological_std(bad) -> None:
    with pytest.raises((ValueError, TypeError)):
        InitSpec(std=bad)


@pytest.mark.parametrize(
    "field,bad",
    [
        ("rope_theta", float("nan")),
        ("norm_eps", float("inf")),
        ("n_heads", 3),
        ("vocab_size", -1),
    ],
)
def test_rehashed_mutated_model_spec_fails_at_weight_construction(field: str, bad) -> None:
    spec, init = tiny_spec(), InitSpec()
    object.__setattr__(spec, field, bad)
    # Even an attacker who creates a new manifest bound to the forged bytes
    # must not enter the Torch constructor with an unvalidated architecture.
    forged_manifest = manifest(spec, init)
    with pytest.raises((ValueError, TypeError)):
        build_canonical_base(spec, init, forged_manifest)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.01, True])
def test_rehashed_mutated_init_spec_fails_at_weight_construction(bad) -> None:
    spec, init = tiny_spec(), InitSpec()
    object.__setattr__(init, "std", bad)
    forged_manifest = manifest(spec, init)
    with pytest.raises((ValueError, TypeError)):
        build_canonical_base(spec, init, forged_manifest)
