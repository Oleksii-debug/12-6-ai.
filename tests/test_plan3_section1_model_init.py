"""Plan 3 / Section 1: frozen architecture and scratch-init contract."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from twelve_six.model import (
    BaseInitManifest,
    InitSpec,
    ModelSpec,
    build_canonical_base,
    canonical_json_sha256,
    count_trainable_parameters,
    load_stage_config,
)
from twelve_six.training.loss import causal_lm_loss


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

_SCALE_ROWS = (
    ("SCALE-500M", 1280, 24, 494_204_160),
    ("SCALE-750M", 1536, 26, 755_842_560),
    ("SCALE-1B", 1792, 26, 1_018_983_168),
    ("SCALE-2B", 2048, 40, 1_996_654_592),
    ("SCALE-3B", 2304, 48, 3_005_966_592),
    ("SCALE-4B", 2560, 52, 4_003_207_680),
    ("SCALE-5B", 2816, 54, 5_017_022_208),
)


def _scale_path(name: str) -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "stages" / (
        name.lower().replace("-", "_") + ".json"
    )


def test_seven_scale_configs_have_distinct_scratch_identities_and_exact_counts() -> None:
    names = tuple(row[0] for row in _SCALE_ROWS)
    assert names == (
        "SCALE-500M", "SCALE-750M", "SCALE-1B", "SCALE-2B",
        "SCALE-3B", "SCALE-4B", "SCALE-5B",
    )
    spec_hashes: set[str] = set()
    manifest_hashes: set[str] = set()
    for name, width, depth, expected in _SCALE_ROWS:
        path = _scale_path(name)
        raw = json.loads(path.read_text(encoding="utf-8"))
        loaded = load_stage_config(path)
        manifest = BaseInitManifest.from_dict(raw["base_init_manifest"])
        model = loaded.model
        assert (model.d_model, model.n_layers) == (width, depth)
        assert model.vocab_size == 32_768 and model.max_seq_len == 2048
        assert model.n_kv_heads * 4 == model.n_heads
        assert model.d_ff == 3 * width and model.head_dim == 64
        # Independent matrix-shape sum, including tied embeddings and RMSNorms.
        independent = model.vocab_size * width + depth * (
            width * model.n_heads * 64 + 2 * width * model.n_kv_heads * 64
            + model.n_heads * 64 * width + 3 * width * model.d_ff + 2 * width
        ) + width
        assert independent == expected == loaded.expected_parameters
        assert manifest.model_spec_sha256 == model.identity_sha256()
        assert manifest.init_spec_sha256 == loaded.init.identity_sha256()
        assert manifest.weights_origin == "random_init"
        assert manifest.ancestor_checkpoint_sha256 is None
        assert raw["profile_id"] == raw["stage"] == name
        assert raw["profile_schema_version"] == 1
        assert raw["qualification"] == "CONFIG_DEFINED"
        assert raw["paid_training_authorized"] is False
        assert raw["resource_evidence_class"] == "ARITHMETIC_ONLY_NOT_MEASURED"
        assert raw["tokenizer_contract"] == "32K_VERSIONED_MANIFEST_REQUIRED_AT_RUN_BINDING"
        spec_hashes.add(model.identity_sha256())
        manifest_hashes.add(manifest.identity_sha256())
    assert len(spec_hashes) == len(manifest_hashes) == 7


def test_scale_config_rejects_swapped_and_tampered_authorities(tmp_path: Path) -> None:
    first = json.loads(_scale_path("SCALE-500M").read_text(encoding="utf-8"))
    second = json.loads(_scale_path("SCALE-750M").read_text(encoding="utf-8"))
    assert first["base_init_manifest"]["seed"] != second["base_init_manifest"]["seed"]
    for key, value in (
        ("expected_parameters", second["expected_parameters"]),
        ("expected_model_identity_sha256", second["expected_model_identity_sha256"]),
        ("canonical_base", "pretrained"),
    ):
        changed = dict(first, **{key: value})
        path = tmp_path / "tampered.json"
        path.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(ValueError):
            load_stage_config(path)
    swapped = dict(first, base_init_manifest=second["base_init_manifest"])
    manifest = BaseInitManifest.from_dict(swapped["base_init_manifest"])
    loaded = load_stage_config(_scale_path("SCALE-500M"))
    assert manifest.model_spec_sha256 != loaded.model.identity_sha256()
    with pytest.raises(ValueError, match="foreign"):
        BaseInitManifest.from_dict(dict(
            first["base_init_manifest"], weights_origin="pretrained",
        ))

@pytest.mark.parametrize("name", [row[0] for row in _SCALE_ROWS])
def test_seven_scale_seeds_use_one_tiny_scratch_decoder_proxy(name: str) -> None:
    """Proxy math/gradients only; does not instantiate the large profile."""
    raw = json.loads(_scale_path(name).read_text(encoding="utf-8"))
    seed = BaseInitManifest.from_dict(raw["base_init_manifest"]).seed
    spec, init = tiny_spec(), InitSpec.from_dict(raw["init"])
    first = build_canonical_base(spec, init, manifest(spec, init, seed=seed))
    second = build_canonical_base(spec, init, manifest(spec, init, seed=seed))
    assert all(torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters()))
    tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    logits = first(tokens).logits
    assert logits.shape == (1, 4, spec.vocab_size)
    loss = causal_lm_loss(logits, tokens)
    assert torch.isfinite(loss)
    loss.backward()
    assert first.token_embedding.weight.grad is not None
    assert torch.isfinite(first.token_embedding.weight.grad).all()
