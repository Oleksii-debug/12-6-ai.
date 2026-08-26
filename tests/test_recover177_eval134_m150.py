from __future__ import annotations

import hashlib
import json
from pathlib import Path

from twelve_six.code_diagnostic import (
    canonical_json_sha256,
    load_suite,
    suite_file_sha256,
)
from twelve_six.data.pipeline import normalize_text
from twelve_six.milestone150_learned_base_ladder import SCALE_ORDER, model_spec

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "eval/reserved/code_diag_v1/probes.jsonl"
MANIFEST = ROOT / "eval/reserved/code_diag_v1/manifest.json"
PROVENANCE = ROOT / "eval/reserved/code_diag_v1/reservation_provenance.json"


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _root(values: set[str]) -> str:
    return _sha_text("\n".join(sorted(values)) + "\n")


def test_eval134_reserved_suite_identity_and_registry_roots() -> None:
    probes = load_suite(SUITE)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    unsigned = dict(manifest)
    identity = unsigned.pop("suite_identity_sha256")
    assert len(probes) == 32
    assert suite_file_sha256(SUITE) == manifest["data_sha256"]
    assert canonical_json_sha256(unsigned) == identity
    assert identity == provenance["suite_identity_sha256"]
    candidates = tuple(
        probe.prefix + choice for probe in probes for choice in probe.choices
    )
    exact = {_sha_text(value) for value in candidates}
    normalized = {_sha_text(normalize_text(value)) for value in candidates}
    assert len(candidates) == 64
    assert len(exact) == 64
    assert len(normalized) == 60
    assert _root(exact) == provenance["candidate_exact_sha256_root"]
    assert _root(normalized) == provenance["candidate_normalized_sha256_root"]
    assert provenance["origin_pr"] == 287
    assert provenance["origin_head_sha"] == "74fee51945c83ebdf39e171a894741964ba51b6d"


def test_m150_ladder_is_byte_vocab_only_for_eval134_model_scoring() -> None:
    assert {model_spec(scale).vocab_size for scale in SCALE_ORDER} == {256}
    assert {model_spec(scale).parameter_count() for scale in SCALE_ORDER} == {
        95_568,
        467_808,
        1_037_696,
    }
