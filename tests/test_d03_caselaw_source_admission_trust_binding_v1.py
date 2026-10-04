from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

MODULE = Path(__file__).parents[1] / "tools" / "run_d03_caselaw_source_admission_replay_v1.py"
spec = importlib.util.spec_from_file_location("caselaw_source_admission_replay_trust", MODULE)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def _product_cfg() -> dict:
    return {
        "source": {
            "dataset": "common-pile/caselaw_access_project",
            "revision": "31e65135501af50f8285b52489bc3b39fd0fc5d5",
            "ordering_policy": "PRESERVE_ORIGINAL_CAP_00044_THEN_APPEND_CAP_00043",
            "objects": [
                {
                    "file": "cap_00044.jsonl.gz",
                    "url": "https://example.invalid/cap_00044.jsonl.gz",
                    "sha256": (
                        "e04bdff817b34d5fd2ab0b4aff272adfe6e863aacef7d4cd24db47e4b7c8db33"
                    ),
                    "bytes": 9_441_419,
                },
                {
                    "file": "cap_00043.jsonl.gz",
                    "url": "https://example.invalid/cap_00043.jsonl.gz",
                    "sha256": (
                        "f299c45effc957e1e02d3930e68c5d6dc643eb9d585093ff7445f36095c8531f"
                    ),
                    "bytes": 10_476_512,
                },
            ],
            "family": "en.common-pile.caselaw",
            "allowed_source_fields": [
                "Caselaw Access Project",
                "Court Listener",
                "CourtListener",
            ],
        }
    }


def _admission_module() -> SimpleNamespace:
    def validate_candidate_identity(identity: dict, policy: dict) -> None:
        if dict(identity) != policy["candidate_binding"]:
            raise RuntimeError("candidate identity substitution")

    return SimpleNamespace(validate_candidate_identity=validate_candidate_identity)


def test_observed_product_binding_is_derived_from_authenticated_product_source() -> None:
    observed = mod.build_observed_product_binding(_product_cfg())
    assert observed["product_pr"] == 904
    assert observed["product_semantic_commit_sha"] == mod.PRODUCT_SEMANTIC_COMMIT
    assert observed["dataset"] == "common-pile/caselaw_access_project"
    assert observed["revision"] == "31e65135501af50f8285b52489bc3b39fd0fc5d5"
    assert observed["family"] == "en.common-pile.caselaw"
    assert observed["objects"][0] == {
        "file": "cap_00044.jsonl.gz",
        "bytes": 9_441_419,
        "sha256": "e04bdff817b34d5fd2ab0b4aff272adfe6e863aacef7d4cd24db47e4b7c8db33",
    }
    assert "url" not in observed["objects"][0]


def test_product_side_coherent_substitution_fails_against_unchanged_policy() -> None:
    cfg = _product_cfg()
    expected = mod.build_observed_product_binding(cfg)
    admission = _admission_module()
    policy = {"candidate_binding": expected}
    assert mod.bind_authenticated_product_identity(cfg, admission, policy) == expected

    substituted = deepcopy(cfg)
    substituted["source"]["revision"] = "coherent-product-side-substitution"
    with pytest.raises(
        mod.ReplayError,
        match="authenticated Product identity does not match admission authority",
    ):
        mod.bind_authenticated_product_identity(substituted, admission, policy)


def test_product_object_substitution_fails_against_unchanged_policy() -> None:
    cfg = _product_cfg()
    expected = mod.build_observed_product_binding(cfg)
    admission = _admission_module()
    policy = {"candidate_binding": expected}

    substituted = deepcopy(cfg)
    substituted["source"]["objects"][0]["sha256"] = "0" * 64
    with pytest.raises(mod.ReplayError, match="does not match admission authority"):
        mod.bind_authenticated_product_identity(substituted, admission, policy)


def test_claimed_authority_commits_map_exact_paths_to_executed_blobs() -> None:
    mod.verify_authority_provenance()


def test_wrong_commit_path_binding_fails_closed() -> None:
    with pytest.raises(mod.ReplayError, match="provenance"):
        mod._verify_commit_path_blob(
            mod.REPO_ROOT,
            mod.PRODUCT_SEMANTIC_COMMIT,
            mod.ADMISSION_POLICY,
            mod.ADMISSION_POLICY_BLOB_SHA1,
            "wrong source-admission provenance",
        )
