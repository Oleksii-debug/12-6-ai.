import pytest

from twelve_six.product_architecture import (
    REQUIRED_BOUNDARIES,
    BaseModelIdentity,
    BoundaryContract,
    ComponentBinding,
    ProductArchitecture,
    ProductLayer,
)


def _digest(char: str) -> str:
    return char * 64


def _model(
    checkpoint_char: str,
    *,
    model_char: str = "a",
    init_char: str = "b",
    tokenizer_char: str = "c",
) -> BaseModelIdentity:
    return BaseModelIdentity(
        schema_version=1,
        model_spec_sha256=_digest(model_char),
        init_spec_sha256=_digest(init_char),
        tokenizer_sha256=_digest(tokenizer_char),
        checkpoint_sha256=_digest(checkpoint_char),
    )


def _components() -> tuple[ComponentBinding, ...]:
    return tuple(
        ComponentBinding(
            layer=layer,
            component_id=f"twelve_six.{layer.value}",
            contract_version=1,
            implementation_version="0.1.0",
        )
        for layer in ProductLayer
    )


def _architecture(
    *,
    components: tuple[ComponentBinding, ...] | None = None,
    boundaries: tuple[BoundaryContract, ...] | None = None,
    model: BaseModelIdentity | None = None,
) -> ProductArchitecture:
    return ProductArchitecture(
        schema_version=1,
        components=_components() if components is None else components,
        boundaries=REQUIRED_BOUNDARIES if boundaries is None else boundaries,
        active_model=_model("d") if model is None else model,
    )


def test_architecture_requires_every_canonical_product_layer() -> None:
    missing_tools = tuple(
        binding for binding in _components() if binding.layer is not ProductLayer.TOOLS
    )

    with pytest.raises(ValueError, match="layer mismatch"):
        _architecture(components=missing_tools)


def test_architecture_rejects_duplicate_layer_binding() -> None:
    components = _components() + (_components()[0],)

    with pytest.raises(ValueError, match="exactly one component"):
        _architecture(components=components)


def test_component_binding_rejects_non_layer_endpoint() -> None:
    with pytest.raises(ValueError, match="layer must be a ProductLayer"):
        ComponentBinding(  # type: ignore[arg-type]
            layer="base_model",
            component_id="twelve_six.base_model",
            contract_version=1,
            implementation_version="0.1.0",
        )


def test_architecture_requires_all_stable_typed_boundaries() -> None:
    without_memory = tuple(
        boundary for boundary in REQUIRED_BOUNDARIES if boundary.contract != "memory"
    )

    with pytest.raises(ValueError, match="missing or incompatible required boundary"):
        _architecture(boundaries=without_memory)


def test_architecture_rejects_boundary_version_drift() -> None:
    drifted = tuple(
        BoundaryContract(
            boundary.producer,
            boundary.consumer,
            boundary.contract,
            2 if boundary.contract == "tools" else boundary.version,
        )
        for boundary in REQUIRED_BOUNDARIES
    )

    with pytest.raises(ValueError, match="missing or incompatible required boundary"):
        _architecture(boundaries=drifted)


@pytest.mark.parametrize(
    "forbidden_layer",
    [
        ProductLayer.COGNITION_MEMORY,
        ProductLayer.TOOLS,
        ProductLayer.LIVE_AGENT,
        ProductLayer.EVOLUTION,
    ],
)
def test_base_model_rejects_direct_runtime_coupling(forbidden_layer: ProductLayer) -> None:
    boundaries = REQUIRED_BOUNDARIES + (
        BoundaryContract(ProductLayer.BASE_MODEL, forbidden_layer, "forbidden", 1),
    )

    with pytest.raises(ValueError, match="must remain isolated behind ModelGateway"):
        _architecture(boundaries=boundaries)


def test_swapping_scale_and_checkpoint_keeps_runtime_graph_unchanged() -> None:
    original = _architecture()
    successor_model = _model(
        "9",
        model_char="e",
        init_char="f",
        tokenizer_char="1",
    )
    successor = original.with_active_model(successor_model)

    assert successor.active_model != original.active_model
    assert successor.active_model.model_spec_sha256 != original.active_model.model_spec_sha256
    assert successor.components == original.components
    assert successor.boundaries == original.boundaries
    assert successor.non_model_identity_sha256() == original.non_model_identity_sha256()
    assert successor.identity_sha256() != original.identity_sha256()


def test_manifest_exposes_complete_machine_readable_graph() -> None:
    architecture = _architecture()
    manifest = architecture.to_dict()

    assert {item["layer"] for item in manifest["components"]} == {
        layer.value for layer in ProductLayer
    }
    assert len(manifest["boundaries"]) == len(REQUIRED_BOUNDARIES)
    assert manifest["non_model_identity_sha256"] == architecture.non_model_identity_sha256()


def test_non_model_graph_identity_is_order_independent() -> None:
    original = _architecture()
    reordered = _architecture(
        components=tuple(reversed(original.components)),
        boundaries=tuple(reversed(original.boundaries)),
    )

    assert reordered.non_model_identity_sha256() == original.non_model_identity_sha256()
    assert reordered.identity_sha256() == original.identity_sha256()


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("model_spec_sha256", "ABC"),
        ("checkpoint_sha256", "ABC"),
        ("tokenizer_sha256", None),
    ],
)
def test_base_model_identity_fails_closed_on_noncanonical_digest(
    field_name: str,
    bad_value: object,
) -> None:
    kwargs: dict[str, object] = {
        "schema_version": 1,
        "model_spec_sha256": _digest("a"),
        "init_spec_sha256": _digest("b"),
        "tokenizer_sha256": _digest("c"),
        "checkpoint_sha256": _digest("d"),
    }
    kwargs[field_name] = bad_value

    with pytest.raises(ValueError, match="lowercase hexadecimal SHA-256"):
        BaseModelIdentity(**kwargs)  # type: ignore[arg-type]
