from dataclasses import replace

import pytest

from twelve_six.agent_runtime.context import ContextEntry, ContextError, build_context, canonical_bytes


def e(name, recency, priority=10, kind="retrieval", critical=False, content="hello"):
    return ContextEntry(name, f"source:{name}", kind, recency, priority, content, critical)


def test_long_horizon_compaction_preserves_owner_system_invariants():
    pinned = [
        e("system", 0, kind="system", critical=True, content="DO NOT EXPOSE SECRETS"),
        e("owner", 1, kind="owner", critical=True, content="NO PAID COMPUTE"),
    ]
    history = [e(f"h{i}", i + 2, content="irrelevant" * 50) for i in range(400)]
    result = build_context(pinned + history, max_bytes=1400, max_entries=5)
    assert len(canonical_bytes(result)) <= 1400
    assert result["omitted_count"] >= 397
    survivors = {v["entry_id"]: v for v in result["entries"]}
    for item in pinned:
        assert survivors[item.entry_id]["content"] == item.content
        assert survivors[item.entry_id]["source_id"] == item.source_id


def test_reproducible_independent_of_input_order_and_model_provider():
    items = [e(f"x{i}", i, priority=i % 10) for i in range(30)]
    a = build_context(items, max_bytes=850, max_entries=5)
    b = build_context(list(reversed(items)), max_bytes=850, max_entries=5)
    assert canonical_bytes(a) == canonical_bytes(b)
    assert a["schema"] == "12-6.agent-context.v1"
    assert "model_id" not in a and "tokenizer_id" not in a


def test_priority_and_recency_order():
    result = build_context(
        [e("low", 100, 1), e("old_high", 1, 100), e("new_high", 50, 100)],
        max_bytes=10000,
        max_entries=1,
    )
    assert [v["entry_id"] for v in result["entries"]] == ["new_high"]


def test_protected_bytes_never_silently_truncated():
    pinned = e("p", 0, kind="owner", critical=True, content="O" * 1000)
    with pytest.raises(ContextError, match="protected invariants"):
        build_context([pinned], max_bytes=100, max_entries=1)
    with pytest.raises(ContextError, match="protected invariants"):
        build_context([pinned, e("q", 2, kind="system", critical=True)], max_bytes=9000, max_entries=1)


def test_adversarial_provenance_identity_and_types_fail_closed():
    good = e("good", 1)
    for values in (
        [good, good],
        [replace(good, critical=True)],
        [replace(good, priority=True)],
        [replace(good, source_id="")],
        [replace(good, recency=-1)],
        [replace(good, source_kind="outside")],
    ):
        with pytest.raises(ContextError):
            build_context(values, max_bytes=2000, max_entries=2)
    with pytest.raises(ContextError):
        build_context([good], max_bytes=True, max_entries=2)


def test_empty_and_large_drop_proofs_fit_budget():
    empty = build_context([], max_bytes=400, max_entries=1)
    assert empty["entries"] == [] and empty["omitted_count"] == 0
    result = build_context([e("big", 1, content="x" * 10000)], max_bytes=400, max_entries=1)
    assert result["omitted_count"] == 1
    assert len(canonical_bytes(result)) <= 400
