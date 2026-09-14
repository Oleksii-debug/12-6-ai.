from __future__ import annotations

import copy
from typing import Any

import pytest

from twelve_six.data import expanded_global_dedup_v9 as v9


def test_copy_deepcopy_dispatch_mutation_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatch = getattr(copy, "_deepcopy_dispatch")

    def poisoned_dict_copier(value: object, memo: dict[int, object]) -> dict[str, object]:
        del value, memo
        return {"poisoned": True}

    monkeypatch.setitem(dispatch, dict, poisoned_dict_copier)
    with pytest.raises(
        v9.ExpandedDedupError,
        match="copy deepcopy dispatch handler replaced: dict",
    ):
        v9._verify_stdlib_runtime_semantic_closure()


@pytest.mark.parametrize(
    "delegate_name",
    ["_LEGACY_VALIDATE_RADA_ROWS", "_LEGACY_RUN_EXPANDED_DEDUP"],
)
def test_legacy_delegate_substitution_fails_closed(
    monkeypatch: pytest.MonkeyPatch, delegate_name: str
) -> None:
    def replacement(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(v9, delegate_name, replacement)
    with pytest.raises(
        v9.ExpandedDedupError,
        match="V9 legacy .* delegate replaced",
    ):
        v9._verify_private_runtime_semantic_closure()


def test_private_helper_substitution_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    def replacement(value: Any, _label: str) -> Any:
        return value

    monkeypatch.setattr(v9._impl, "_mapping", replacement)
    with pytest.raises(
        v9.ExpandedDedupError,
        match=r"V9 private runtime function (globals|code) replaced: _mapping",
    ):
        v9._verify_private_runtime_semantic_closure()


def test_private_runtime_closure_accepts_unmodified_incumbent() -> None:
    v9._verify_private_runtime_semantic_closure()
