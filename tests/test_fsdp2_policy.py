from __future__ import annotations

import pytest

import twelve_six.distributed.fsdp2_policy as policy_module
from twelve_six.distributed.fsdp2_policy import (
    FSDP2ReshardPolicy,
    apply_fsdp2_reshard_policy,
    fsdp2_reshard_policy_spec,
)


@pytest.mark.parametrize(
    ("policy", "non_root", "root"),
    [
        (FSDP2ReshardPolicy.FULL_SHARD, True, True),
        (FSDP2ReshardPolicy.ROOT_KEEP_UNSHARDED, True, False),
        (FSDP2ReshardPolicy.SHARD_GRAD_OP, False, False),
    ],
)
def test_reshard_policy_maps_to_maintained_fsdp2_flags(policy, non_root, root) -> None:
    spec = fsdp2_reshard_policy_spec(policy)
    assert spec.non_root_reshard_after_forward is non_root
    assert spec.root_reshard_after_forward is root


def test_root_keep_reuses_canonical_apply_and_changes_root_only(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []

    class DummyModel:
        def set_reshard_after_forward(self, value, recurse=True):
            calls.append(("root_setter", (value, recurse)))

    model = DummyModel()

    def fake_apply(candidate, mesh, *, reshard_after_forward=True):
        assert candidate is model
        assert mesh == "mesh"
        calls.append(("apply", reshard_after_forward))
        return candidate

    monkeypatch.setattr(policy_module, "apply_fsdp2", fake_apply)
    result = apply_fsdp2_reshard_policy(
        model,  # type: ignore[arg-type]
        "mesh",
        policy=FSDP2ReshardPolicy.ROOT_KEEP_UNSHARDED,
    )
    assert result is model
    assert calls == [("apply", True), ("root_setter", (False, False))]


def test_full_and_shard_grad_do_not_need_root_mutation(monkeypatch) -> None:
    class DummyModel:
        def set_reshard_after_forward(self, value, recurse=True):
            raise AssertionError("uniform policies must not mutate the root separately")

    model = DummyModel()
    observed: list[bool] = []

    def fake_apply(candidate, mesh, *, reshard_after_forward=True):
        observed.append(reshard_after_forward)
        return candidate

    monkeypatch.setattr(policy_module, "apply_fsdp2", fake_apply)
    apply_fsdp2_reshard_policy(
        model,  # type: ignore[arg-type]
        object(),
        policy=FSDP2ReshardPolicy.FULL_SHARD,
    )
    apply_fsdp2_reshard_policy(
        model,  # type: ignore[arg-type]
        object(),
        policy=FSDP2ReshardPolicy.SHARD_GRAD_OP,
    )
    assert observed == [True, False]
