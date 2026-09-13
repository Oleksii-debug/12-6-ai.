from __future__ import annotations

import copy

import numpy as np
import pytest

from twelve_six.checkpoint.state_tree import (
    StateTreeError,
    pack_state_tree,
    unpack_state_tree,
)


def test_state_tree_round_trip_remains_exact() -> None:
    original = {
        "bytes": b"abc",
        "tuple": (1, 2),
        "list": [True, None, "x"],
        "scalar": np.int64(7),
        "array": np.arange(4, dtype=np.float32).reshape(2, 2),
    }
    packed = pack_state_tree(original)
    restored = unpack_state_tree(packed.tree, packed.tensors)

    assert restored["bytes"] == b"abc"
    assert restored["tuple"] == (1, 2)
    assert restored["list"] == [True, None, "x"]
    assert restored["scalar"] == np.int64(7)
    np.testing.assert_array_equal(restored["array"], original["array"])


@pytest.mark.parametrize(
    ("tree", "match"),
    [
        ({"unexpected": "raw-mapping"}, "without __kind__"),
        (
            {"__kind__": "list", "items": [], "extra": 1},
            "noncanonical list node fields",
        ),
        (
            {"__kind__": "list", "items": {"not": "a-list"}},
            "items must be a list",
        ),
        ({"__kind__": "bytes", "base64": "%%%"}, "base64 payload is invalid"),
        (
            {"__kind__": "numpy_scalar", "dtype": "object", "value": "x"},
            "object numpy scalar",
        ),
    ],
)
def test_noncanonical_state_tree_nodes_fail_closed(tree: object, match: str) -> None:
    with pytest.raises(StateTreeError, match=match):
        unpack_state_tree(tree, {})


def test_duplicate_mapping_keys_are_rejected() -> None:
    tree = {
        "__kind__": "mapping",
        "items": [["same", 1], ["same", 2]],
    }
    with pytest.raises(StateTreeError, match="duplicate mapping key"):
        unpack_state_tree(tree, {})


def test_unreferenced_tensor_payload_is_rejected() -> None:
    with pytest.raises(StateTreeError, match="unreferenced tensor payloads"):
        unpack_state_tree(
            1,
            {"tensor_00000000": np.asarray([1], dtype=np.int64)},
        )


def test_tensor_payload_cannot_be_referenced_twice() -> None:
    tensor_node = {
        "__kind__": "tensor",
        "key": "tensor_00000000",
        "backend": "numpy",
        "device": None,
    }
    tree = {"__kind__": "list", "items": [tensor_node, copy.deepcopy(tensor_node)]}
    tensors = {"tensor_00000000": np.asarray([1], dtype=np.int64)}

    with pytest.raises(StateTreeError, match="referenced more than once"):
        unpack_state_tree(tree, tensors)


def test_unknown_tensor_backend_is_rejected() -> None:
    tree = {
        "__kind__": "tensor",
        "key": "tensor_00000000",
        "backend": "pickle",
        "device": None,
    }
    tensors = {"tensor_00000000": np.asarray([1], dtype=np.int64)}

    with pytest.raises(StateTreeError, match="unknown tensor backend"):
        unpack_state_tree(tree, tensors)


def test_mutating_packed_tree_with_unknown_field_is_detected() -> None:
    packed = pack_state_tree({"value": np.asarray([1, 2], dtype=np.float32)})
    tampered = copy.deepcopy(packed.tree)
    tampered["extra"] = "checksum-consistent rewrite"

    with pytest.raises(StateTreeError, match="noncanonical mapping node fields"):
        unpack_state_tree(tampered, packed.tensors)
