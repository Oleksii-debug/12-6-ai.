"""Plan 2 S9: mixture determinism, admission, caps, versioning and restart."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools import plan2_corpus_mixture_v1 as mixture
from tools import plan2_exact_dedup_v1 as exact
from tools import plan2_normalization_evidence_v1 as norm
from tools import plan2_privacy_gate_v1 as privacy
from tools import plan2_reserved_eval_firewall_v1 as firewall

ROOT = Path(__file__).resolve().parents[1]
POLICY_RAW = (ROOT / mixture.POLICY_PATH).read_bytes()
RESERVED = (ROOT / firewall.RESERVE_PATH).read_bytes()
SOURCE = "uk.kubernetes.docs.what-is-kubernetes"
BASE = (
    "Практичний довідник пояснює інженерам налаштування перевіреної "
    "відкритої платформи для надійної роботи з технічними документами."
)
OTHER = (
    "Культурний архів міста зберігає рідкісні рукописи та історичні "
    "свідчення для незалежних дослідників сучасної гуманітаристики."
)


def _fixture(source: str, *lines: str):
    payload = "".join(line + "\n" for line in lines).encode()
    offsets = []
    pos = 0
    for index, chunk in enumerate(payload.splitlines(keepends=True)):
        offsets.append({
            "index": index, "start_byte": pos,
            "end_byte": pos + len(chunk), "sha256": exact._sha(chunk),
        })
        pos += len(chunk)
    core = {
        "schema_version": norm.MANIFEST_SCHEMA,
        "source_id": source,
        "record_count": len(offsets),
        "records": offsets,
        "normalized_bytes": len(payload),
        "normalized_sha256": exact._sha(payload),
        "classification": {"language": "uk", "modality": "text"},
        "training_corpus_authorized": False,
        "tokenizer_fit_authorized": False,
        "evaluation_authorized": False,
    }
    normalized = {**core, "manifest_sha256": exact._sha(exact._canonical(core))}
    receipt = privacy.inspect_normalized(
        normalized, payload, policy_sha256="f" * 64)
    return normalized, payload, receipt


def _policy():
    return mixture._parse_policy(POLICY_RAW)


def test_exact_pinned_policy_and_current_source_metadata():
    policy = _policy()
    assert mixture._git_blob(POLICY_RAW) == mixture.POLICY_GIT_BLOB
    assert policy["sources"][0]["source_id"] == SOURCE
    assert policy["sources"][0]["domain"] == "technical_documentation"
    assert policy["sources"][0]["sampling_weight"] == 1


def test_survivor_selection_is_order_independent_and_stays_candidate_only():
    cohorts = (_fixture(SOURCE, BASE), _fixture("foreign.source", OTHER))
    first = mixture.inspect_mixture(cohorts, RESERVED, POLICY_RAW)
    second = mixture.inspect_mixture(tuple(reversed(cohorts)), RESERVED, POLICY_RAW)
    assert first == second
    assert first["selected_record_ids"] == [SOURCE + ":r00000000"]
    assert first["excluded_reason_counts"]["UNMAPPED_SOURCE"] == 1
    assert first["training_corpus_authorized"] is False
    assert first["tokenizer_fit_authorized"] is False
    assert first["real_final_test_material_accessed"] is False
    assert first["actual_tokenizer_token_count"] is None
    assert first["token_measure"].endswith("NOT_MODEL_TOKENS")
    assert first["contributions"]["language"]["uk"]["records"] == 1
    assert first["contributions"]["domain"]["technical_documentation"]["records"] == 1
    assert first["contributions"]["modality"]["text"]["token_proxy_units"] > 0
    assert BASE not in json.dumps(first, ensure_ascii=False)


def test_caps_across_source_family_domain_language_modality():
    policy = _policy()
    policy["max_records_total"] = 10
    policy["caps"] = {key: 1 for key in mixture.BUCKETS}
    policy["sources"] = [
        {"source_id": "source.a", "source_family": "same.family",
         "language": "uk", "domain": "domain.a", "modality": "text",
         "sampling_weight": 1},
        {"source_id": "source.b", "source_family": "same.family",
         "language": "uk", "domain": "domain.b", "modality": "text",
         "sampling_weight": 2},
    ]
    ids = ["source.a:r00000000", "source.a:r00000001", "source.b:r00000000"]
    rows = [
        {"record_id": rid, "source_id": rid.split(":")[0],
         "modality": "uk", "text": BASE}
        for rid in ids
    ]
    receipt = {
        "decontaminated_record_ids": ids, "excluded_record_ids": [],
        "manifest_sha256": "a" * 64,
    }
    result = mixture.compose_mixture(receipt, rows, policy)
    assert result["selected_record_count"] == 1
    assert len(result["excluded"]) == 2
    assert sum(v["records"] for v in result["contributions"]["family"].values()) == 1
    assert all(reason.endswith("_CAP") for reason in result["excluded_reason_counts"])


def test_weighted_rank_and_policy_version_change_new_dataset_identity():
    policy = _policy()
    cohorts = (_fixture(SOURCE, BASE, OTHER),)
    s8 = firewall.inspect_firewall(cohorts, RESERVED)
    _s7, rows = firewall._retained_rows(cohorts)
    original = mixture.compose_mixture(s8, rows, policy)
    changed = copy.deepcopy(policy)
    changed["revision"] = "plan2-mixture-fixture-2026-10-08-v2"
    changed["sources"][0]["sampling_weight"] = 9
    revised = mixture.compose_mixture(s8, rows, changed)
    assert original["policy_sha256"] != revised["policy_sha256"]
    assert original["dataset_candidate_sha256"] != revised["dataset_candidate_sha256"]
    assert original["selected_record_ids"] == revised["selected_record_ids"]


def test_s8_eval_overlap_cannot_reenter_mixture_even_with_policy_source():
    _manifest, holdout, _authorities = firewall._reserve(RESERVED)
    cohorts = (_fixture(SOURCE, BASE),
               _fixture("foreign.contaminated", holdout[0]["text"]))
    result = mixture.inspect_mixture(cohorts, RESERVED, POLICY_RAW)
    assert result["selected_record_count"] == 1
    assert len(result["selected_record_ids"]) + len(result["excluded"]) == 2
    assert result["excluded_reason_counts"]["S8_DECONTAMINATED"] >= 1


@pytest.mark.parametrize("mutation", [
    "revision", "seed", "caps", "weight", "extra_source", "truncation",
])
def test_policy_drift_is_rejected_before_mixture(mutation):
    changed = _policy()
    if mutation == "revision":
        changed["revision"] += "-forged"
    elif mutation == "seed":
        changed["seed"] = "0" * 64
    elif mutation == "caps":
        changed["caps"]["family"] = 0
    elif mutation == "weight":
        changed["sources"][0]["sampling_weight"] = 999
    elif mutation == "extra_source":
        changed["sources"].append(copy.deepcopy(changed["sources"][0]))
    else:
        raw = POLICY_RAW[:23]
        with pytest.raises(mixture.Plan2MixtureError):
            mixture.inspect_mixture((_fixture(SOURCE, BASE),), RESERVED, raw)
        return
    raw = mixture._canonical(changed)
    assert raw != POLICY_RAW
    with pytest.raises(mixture.Plan2MixtureError, match="unapproved mixture"):
        mixture.inspect_mixture((_fixture(SOURCE, BASE),), RESERVED, raw)


@pytest.mark.parametrize("mutation", ["bool-cap", "duplicate", "bad-source", "zero-weight"])
def test_invalid_new_policy_version_fails_structural_validation(mutation):
    candidate = _policy()
    if mutation == "bool-cap":
        candidate["caps"]["family"] = True
    elif mutation == "duplicate":
        candidate["sources"].append(copy.deepcopy(candidate["sources"][0]))
    elif mutation == "bad-source":
        candidate["sources"][0]["source_id"] = "../escape"
    else:
        candidate["sources"][0]["sampling_weight"] = 0
    with pytest.raises(mixture.Plan2MixtureError):
        mixture._parse_policy(mixture._canonical(candidate))


def test_forged_upstream_privacy_or_heldout_fixture_denied():
    normalized, payload, receipt = _fixture(SOURCE, BASE)
    altered = copy.deepcopy(receipt)
    altered["manifest_sha256"] = "0" * 64
    with pytest.raises(mixture.Plan2MixtureError):
        mixture.inspect_mixture(((normalized, payload, altered),), RESERVED, POLICY_RAW)
    bad_reserved = RESERVED.replace(b"LOCAL_FREE_SYNTHETIC", b"PRODUCTION_RELEASE")
    with pytest.raises(mixture.Plan2MixtureError):
        mixture.inspect_mixture((_fixture(SOURCE, BASE),), bad_reserved, POLICY_RAW)


def test_missing_or_duplicate_s8_records_cannot_be_silently_sampled():
    policy = _policy()
    row = {
        "record_id": SOURCE + ":r00000000",
        "source_id": SOURCE, "modality": "uk", "text": BASE,
    }
    receipt = {"decontaminated_record_ids": [row["record_id"]],
               "excluded_record_ids": [], "manifest_sha256": "a" * 64}
    with pytest.raises(mixture.Plan2MixtureError, match="missing S8"):
        mixture.compose_mixture(receipt, [], policy)
    with pytest.raises(mixture.Plan2MixtureError, match="duplicate"):
        mixture.compose_mixture(receipt, [row, row], policy)


def test_physical_restart_rebuild_and_corruption_fail_closed(tmp_path):
    first = mixture.stage_mixture(ROOT, tmp_path / "first")
    restarted = mixture.stage_mixture(ROOT, tmp_path / "first")
    fresh = mixture.stage_mixture(ROOT, tmp_path / "fresh")
    assert first == restarted == fresh
    assert first["input_s8_survivor_count"] >= first["selected_record_count"] > 0
    assert first["dataset_candidate_sha256"]
    assert first["training_corpus_authorized"] is False
    manifest = tmp_path / "first" / "corpus-mixture-manifest.json"
    assert manifest.read_bytes() == (
        tmp_path / "fresh" / "corpus-mixture-manifest.json").read_bytes()
    manifest.write_bytes(b'{"forged":true}')
    with pytest.raises(mixture.Plan2MixtureError, match="immutable mixture"):
        mixture.stage_mixture(ROOT, tmp_path / "first")


def test_symlink_staging_refused(tmp_path):
    (tmp_path / "real").mkdir()
    (tmp_path / "linked").symlink_to(tmp_path / "real", target_is_directory=True)
    with pytest.raises(mixture.Plan2MixtureError, match="symlink mixture"):
        mixture.stage_mixture(ROOT, tmp_path / "linked" / "mix")
