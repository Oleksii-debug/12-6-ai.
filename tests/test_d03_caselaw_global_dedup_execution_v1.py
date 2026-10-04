from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).parents[1]
MODULE = ROOT / "tools" / "run_d03_caselaw_global_dedup_execution_v1.py"


def _run_isolated(body: str) -> None:
    prelude = f"""
from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path

MODULE = Path({str(MODULE)!r})
spec = importlib.util.spec_from_file_location("caselaw_global_dedup_execution", MODULE)
assert spec is not None and spec.loader is not None
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def row(source_id: str, *, family: str = "base", size: int = 5) -> dict:
    return {{
        "source_id": source_id,
        "source_family": family,
        "declared_capacity_bytes": size,
    }}
"""
    proc = subprocess.run(
        [sys.executable, "-c", dedent(prelude) + "\n" + dedent(body)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_compose_graph_is_add_only_and_preserves_inputs() -> None:
    _run_isolated(
        """
base_inventory = {
    "schema_version": "fixture",
    "sources": [row("base:a")],
    "lineage_edges": [{"left_source_id": "base:a", "right_source_id": "base:a"}],
    "final_refresh_required": True,
}
base_payloads = {"base:a": b"alpha"}
extension_sources = [row("caselaw:a", family=mod.caselaw.SOURCE_FAMILY, size=4)]
extension_payloads = {"caselaw:a": b"beta"}
inventory_before = deepcopy(base_inventory)

inventory, payloads = mod._compose_graph(
    base_inventory,
    base_payloads,
    extension_sources,
    extension_payloads,
)

assert base_inventory == inventory_before
assert inventory["sources"] == [base_inventory["sources"][0], extension_sources[0]]
assert inventory["lineage_edges"] == base_inventory["lineage_edges"]
assert inventory["final_refresh_required"] is False
assert set(payloads) == {"base:a", "caselaw:a"}
assert payloads["base:a"] == b"alpha"
assert payloads["caselaw:a"] == b"beta"
"""
    )


def test_compose_graph_rejects_source_id_collision() -> None:
    _run_isolated(
        """
try:
    mod._compose_graph(
        {"sources": [row("same")]},
        {"same": b"a"},
        [row("same", family=mod.caselaw.SOURCE_FAMILY)],
        {"same": b"b"},
    )
except mod.CaselawGlobalDedupError as exc:
    assert "collision" in str(exc)
else:
    raise AssertionError("collision was accepted")
"""
    )


def test_compose_graph_requires_exact_payload_coverage() -> None:
    _run_isolated(
        """
for base_payloads, extension_payloads, expected in (
    ({"other": b"a"}, {"caselaw:a": b"b"}, "base inventory/payload"),
    ({"base:a": b"a"}, {"other": b"b"}, "extension inventory/payload"),
):
    try:
        mod._compose_graph(
            {"sources": [row("base:a")]},
            base_payloads,
            [row("caselaw:a", family=mod.caselaw.SOURCE_FAMILY)],
            extension_payloads,
        )
    except mod.CaselawGlobalDedupError as exc:
        assert expected in str(exc)
    else:
        raise AssertionError(f"{expected} mismatch was accepted")
"""
    )


def test_survivor_wrapper_accounts_caselaw_after_cross_family_selection() -> None:
    _run_isolated(
        """
dedup = {
    "report_sha256": "1" * 64,
    "sources": [
        row("base:a", size=10),
        row("caselaw:a", family=mod.caselaw.SOURCE_FAMILY, size=7),
        row("caselaw:b", family=mod.caselaw.SOURCE_FAMILY, size=11),
    ],
}
projection = {
    "schema_version": "fixture-survivors",
    "survivor_authority_sha256": "2" * 64,
    "pre_dedup_source_object_count": 3,
    "post_dedup_survivor_source_object_count": 2,
    "pre_dedup_declared_capacity_bytes": 28,
    "post_dedup_declared_capacity_bytes": 21,
    "duplicate_discount_bytes": 7,
    "duplicate_cluster_count": 1,
    "duplicate_clusters": [
        {
            "member_source_ids": ["base:a", "caselaw:a"],
            "selected_source_id": "base:a",
            "selected_declared_capacity_bytes": 10,
        }
    ],
    "survivor_source_ids": ["base:a", "caselaw:b"],
}

authority = mod._outer_survivor_authority(dedup, projection)

assert authority["matcher_report_sha256"] == "1" * 64
assert authority["caselaw"]["post_dedup_survivor_source_object_count"] == 1
assert authority["caselaw"]["post_dedup_survivor_declared_capacity_bytes"] == 11
assert authority["truth_boundary"]["canonical_capacity_credited"] == 0
assert authority["truth_boundary"]["authorized_optimized_target_exposure"] == 0
assert authority["truth_boundary"]["tokenizer_fit_authorized"] is False
assert len(authority["survivor_authority_sha256"]) == 64
"""
    )


def test_survivor_wrapper_rejects_unknown_selected_source() -> None:
    _run_isolated(
        """
try:
    mod._outer_survivor_authority(
        {"report_sha256": "1" * 64, "sources": [row("base:a")]},
        {
            "schema_version": "fixture",
            "survivor_authority_sha256": "2" * 64,
            "survivor_source_ids": ["missing"],
        },
    )
except mod.CaselawGlobalDedupError as exc:
    assert "unknown survivor" in str(exc)
else:
    raise AssertionError("unknown survivor was accepted")
"""
    )


def test_production_arithmetic_is_exact() -> None:
    _run_isolated(
        """
assert mod.EXPECTED_BASE_OBJECTS == 263
assert mod.EXPECTED_BASE_BYTES == 6_093_965
assert mod.EXPECTED_COMBINED_OBJECTS == 5_921
assert mod.EXPECTED_COMBINED_BYTES == 12_056_112
assert mod.PAYLOAD_BYTES_SEMANTICS == "DECLARED_CAPACITY_BYTES"
assert mod.EXPECTED_MAIN == "bd2d445dfd8fbd7ec6759c1398bb913f4e0c0093"
assert mod.CASELAW_FINAL_HEAD == "deaf0730fe04a12e9abb8f3cecb14d6ad2cc7a4d"
"""
    )


def test_authority_surface_includes_matcher_adapter_and_materializer() -> None:
    _run_isolated(
        """
expected = {
    "src/twelve_six/data/caselaw_source_admitted_dedup_intake.py",
    "src/twelve_six/data/incumbent_dedup_indexed_execution.py",
    "src/twelve_six/data/_incumbent_dedup_indexed_execution_core.py",
    "tools/run_d03_expanded_global_dedup_v9.py",
    "tools/run_d03_nomis_free_clean_successor_v1.py",
    "src/twelve_six/data/external_llm_provenance_quarantine_v1.py",
    "configs/data/d03_external_llm_provenance_quarantine_v1.json",
    "tools/run_next100_065f_global_dedup_v8.py",
    "tools/materialize_data_bulk_code1_permissive_python_bundle.py",
}
assert expected <= set(mod.AUTHORITY_PATHS)
"""
    )


def test_declared_capacity_is_separate_from_comparison_payload_bytes() -> None:
    _run_isolated(
        """
inventory = {
    "sources": [
        row("base:a", size=5),
        row("base:b", size=4),
    ]
}
payloads = {
    "base:a": b"alpha\\n\\n",
    "base:b": b"beta",
}
assert sum(len(raw) for raw in payloads.values()) == 11
assert mod._declared_capacity_bytes(inventory, payloads, label="fixture") == 9
"""
    )


def test_declared_capacity_rejects_aliases_negative_and_coverage_drift() -> None:
    _run_isolated(
        """
for bad in (True, 5.0, "5", -1):
    try:
        mod._declared_capacity_bytes(
            {"sources": [row("base:a", size=bad)]},
            {"base:a": b"payload"},
            label="fixture",
        )
    except mod.CaselawGlobalDedupError as exc:
        assert "declared capacity must be exact nonnegative int" in str(exc)
    else:
        raise AssertionError(f"invalid capacity accepted: {bad!r}")

try:
    mod._declared_capacity_bytes(
        {"sources": [row("base:a", size=5)]},
        {"other": b"payload"},
        label="fixture",
    )
except mod.CaselawGlobalDedupError as exc:
    assert "inventory/payload coverage mismatch" in str(exc)
else:
    raise AssertionError("coverage drift accepted")
"""
    )


def test_nomis_removal_proof_is_exact_and_capacity_delta_matches() -> None:
    _run_isolated(
        """
proof = deepcopy(mod.NOMIS_REMOVAL_PROOF)
mod._verify_removal_proof(proof)
assert proof["blocked_payload_bytes"] == 1_659
assert proof["pre_source_object_count"] == 35
assert proof["post_source_object_count"] == 34
assert proof["pre_source_capacity_bytes"] - proof["post_source_capacity_bytes"] == 1_659
assert proof["removed_before_new_global_dedup"] is True

proof["post_source_capacity_bytes"] += 1
try:
    mod._verify_removal_proof(proof)
except mod.CaselawGlobalDedupError as exc:
    assert "Nomis deauthorization proof drift" in str(exc)
else:
    raise AssertionError("mutated removal proof accepted")
"""
    )


def test_evidence_declares_capacity_semantics_explicitly() -> None:
    raw = MODULE.read_text(encoding="utf-8")
    assert raw.count('"payload_bytes_semantics": PAYLOAD_BYTES_SEMANTICS') == 3
    assert '"declared_capacity_bytes": EXPECTED_BASE_BYTES' in raw
    assert '"comparison_payload_bytes": base_comparison_payload_bytes' in raw
    assert '"comparison_payload_bytes": caselaw_comparison_payload_bytes' in raw
    assert '"comparison_payload_bytes": combined_comparison_payload_bytes' in raw
    assert '"nomis1864_deauthorization": removal' in raw


def test_historical_reconstruction_closure_includes_v5_pipeline() -> None:
    _run_isolated(
        """
assert "twelve_six.data.pipeline" in mod._HISTORICAL_MATCHER_MODULES
"""
    )


def test_incumbent_runtime_attestation_precedes_reference_report_execution() -> None:
    _run_isolated(
        """
import inspect

source = inspect.getsource(mod.execute)
warmup = source.index("_preflight_attested_lineage_warmup(matcher)")
sample = source.index("_preflight_attested_reference_sample(matcher, inventory, payloads)")
reference = source.index("reference = matcher.audit_payloads(inventory, payloads)")
assert warmup < sample < reference

# The verifier was moved into both actual preflight helpers. Do not merely
# assert the names of helpers: require unchanged incumbent attestation before
# and after lineage warmup and after the physical reference sample.
warmup_source = inspect.getsource(mod._preflight_attested_lineage_warmup)
attestation = "indexed.attest_incumbent_runtime(matcher)"
assert warmup_source.count(attestation) == 2
first = warmup_source.index(attestation)
work = warmup_source.index("matches = lineage(fingerprints, ())")
last = warmup_source.rindex(attestation)
assert first < work < last

sample_source = inspect.getsource(mod._preflight_attested_reference_sample)
physical = sample_source.index(
    "sample_report = matcher.audit_payloads(sample_inventory, sample_payloads)"
)
attested = sample_source.index(attestation)
assert physical < attested
"""
    )


def test_lineage_preflight_attests_before_and_after_synthetic_warmup() -> None:
    _run_isolated(
        """
from types import SimpleNamespace
observed = []

def lineage(fingerprints, edges):
    assert len(fingerprints) == 16 and edges == ()
    assert all(item["row"]["source_family"] == "local-preflight-only"
               for item in fingerprints)
    observed.append("warmup")
    return [
        {"match_type": "lineage_same_origin_alias", "capacity_collapsing": True}
        for _ in range(8)
    ]

def attest(matcher):
    assert matcher._lineage_matches is lineage
    observed.append("attest")

mod.indexed.attest_incumbent_runtime = attest
mod._preflight_attested_lineage_warmup(SimpleNamespace(_lineage_matches=lineage))
assert observed == ["attest", *["warmup"] * 10, "attest"]
"""
    )


def test_lineage_preflight_rejects_changed_semantics_without_second_attest() -> None:
    _run_isolated(
        """
from types import SimpleNamespace
observed = []

def attest(matcher):
    observed.append("attest")

mod.indexed.attest_incumbent_runtime = attest
matcher = SimpleNamespace(_lineage_matches=lambda items, edges: [])
try:
    mod._preflight_attested_lineage_warmup(matcher)
except mod.CaselawGlobalDedupError as exc:
    assert "warmup semantics drift" in str(exc)
else:
    raise AssertionError("invalid lineage semantics accepted")
assert observed == ["attest"]
"""
    )


def test_lineage_preflight_propagates_second_attestation_failure() -> None:
    _run_isolated(
        """
from types import SimpleNamespace
calls = []

def attest(matcher):
    calls.append("attest")
    if len(calls) == 2:
        raise mod.indexed.IndexedExecutionError("V3 callable code drift: _lineage_matches")

def lineage(fingerprints, edges):
    return [
        {"match_type": "lineage_same_origin_alias", "capacity_collapsing": True}
        for _ in range(8)
    ]

mod.indexed.attest_incumbent_runtime = attest
try:
    mod._preflight_attested_lineage_warmup(SimpleNamespace(_lineage_matches=lineage))
except mod.indexed.IndexedExecutionError as exc:
    assert "V3 callable code drift: _lineage_matches" in str(exc)
else:
    raise AssertionError("post-execution attestation drift was bypassed")
assert calls == ["attest", "attest"]
"""
    )


def test_small_reference_preflight_samples_real_graph_edges_and_attests() -> None:
    _run_isolated(
        """
from types import SimpleNamespace
calls = []
rows = [{"source_id": f"id:{i}"} for i in range(20)]
edges = [
    {"left_source_id": "id:0", "right_source_id": "id:1"},
    {"left_source_id": "id:8", "right_source_id": "id:9"},
    {"left_source_id": "id:18", "right_source_id": "id:19"},
]
inventory = {"sources": rows, "lineage_edges": edges, "local_free_only": True}
payloads = {row["source_id"]: b"original-input" for row in rows}
def audit(sample, raw):
    calls.append("reference")
    expected_ids = {f"id:{i}" for i in [*range(8), *range(12, 20)]}
    assert [row["source_id"] for row in sample["sources"]] == [
        *[f"id:{i}" for i in range(8)], *[f"id:{i}" for i in range(12, 20)]
    ]
    assert set(raw) == expected_ids
    assert list(raw) == [row["source_id"] for row in sample["sources"]]
    assert len(sample["lineage_edges"]) == 2
    assert sample["lineage_edges"] == [edges[0], edges[2]]
    assert sample["local_free_only"] is True
    return {"report_sha256": "f" * 64}
def verify(report):
    calls.append("verify")
    assert report == {"report_sha256": "f" * 64}
def attest(matcher):
    calls.append("attest")
mod.indexed.attest_incumbent_runtime = attest
matcher = SimpleNamespace(audit_payloads=audit, verify_report=verify)
mod._preflight_attested_reference_sample(matcher, inventory, payloads)
assert calls == ["reference", "verify", "attest"]
"""
    )


def test_small_reference_preflight_fails_closed_on_post_reference_drift() -> None:
    _run_isolated(
        """
from types import SimpleNamespace
calls = []
def audit(*args):
    calls.append("reference")
    return {}
def verify(*args):
    calls.append("verify")
def attest(*args):
    calls.append("attest")
    raise mod.indexed.IndexedExecutionError("V3 callable code drift: _lineage_matches")
mod.indexed.attest_incumbent_runtime = attest
matcher = SimpleNamespace(audit_payloads=audit, verify_report=verify)
rows = [{"source_id": f"id:{i}"} for i in range(16)]
try:
    mod._preflight_attested_reference_sample(
        matcher, {"sources": rows, "lineage_edges": []},
        {row["source_id"]: b"x" for row in rows},
    )
except mod.indexed.IndexedExecutionError as exc:
    assert "_lineage_matches" in str(exc)
else:
    raise AssertionError("post-reference drift was ignored")
assert calls == ["reference", "verify", "attest"]
"""
    )


def test_small_reference_preflight_rejects_insufficient_or_duplicate_inputs() -> None:
    _run_isolated(
        """
from types import SimpleNamespace
matcher = SimpleNamespace(audit_payloads=lambda *args: None)
for count in (0, 15):
    rows = [{"source_id": f"id:{i}"} for i in range(count)]
    try:
        mod._preflight_attested_reference_sample(
            matcher, {"sources": rows, "lineage_edges": []},
            {row["source_id"]: b"x" for row in rows},
        )
    except mod.CaselawGlobalDedupError as exc:
        assert "at least 16" in str(exc)
    else:
        raise AssertionError("undersized preflight accepted")
rows = [{"source_id": f"id:{i % 15}"} for i in range(16)]
try:
    mod._preflight_attested_reference_sample(
        matcher, {"sources": rows, "lineage_edges": []},
        {row["source_id"]: b"x" for row in rows},
    )
except mod.CaselawGlobalDedupError as exc:
    assert "not unique" in str(exc)
else:
    raise AssertionError("repeated source ID accepted")
"""
    )


def test_lineage_preflight_precedes_expensive_reference() -> None:
    _run_isolated(
        """
import inspect
source = inspect.getsource(mod.execute)
preflight = source.index("_preflight_attested_lineage_warmup(matcher)")
sample = source.index("_preflight_attested_reference_sample(matcher, inventory, payloads)")
assert preflight < sample
reference = source.index("reference = matcher.audit_payloads(inventory, payloads)")
assert preflight < reference
"""
    )


def test_survivor_wrapper_rejects_nonexact_declared_capacity() -> None:
    _run_isolated(
        """
for bad in (True, 11.0, "11", -1):
    try:
        mod._outer_survivor_authority(
            {
                "report_sha256": "1" * 64,
                "sources": [
                    row(
                        "caselaw:a",
                        family=mod.caselaw.SOURCE_FAMILY,
                        size=bad,
                    )
                ],
            },
            {
                "schema_version": "fixture",
                "survivor_authority_sha256": "2" * 64,
                "survivor_source_ids": ["caselaw:a"],
            },
        )
    except mod.CaselawGlobalDedupError as exc:
        assert "exact nonnegative int" in str(exc)
    else:
        raise AssertionError(f"nonexact declared capacity accepted: {bad!r}")
"""
    )


def test_provenance_truth_uses_scoped_nonclaim() -> None:
    _run_isolated(
        """
import inspect

authority = mod._outer_survivor_authority(
    {
        "report_sha256": "1" * 64,
        "sources": [
            row(
                "caselaw:a",
                family=mod.caselaw.SOURCE_FAMILY,
                size=11,
            )
        ],
    },
    {
        "schema_version": "fixture",
        "survivor_authority_sha256": "2" * 64,
        "survivor_source_ids": ["caselaw:a"],
    },
)
truth = authority["truth_boundary"]
assert truth["whole_corpus_external_llm_cleanliness_claimed"] is False
assert "external_llm_or_api_used_for_data_or_intelligence" not in truth

execute_source = inspect.getsource(mod.execute)
assert '"whole_corpus_external_llm_cleanliness_claimed": False' in execute_source
assert '"external_llm_or_api_used_for_data_or_intelligence": False' not in execute_source
"""
    )


def test_survivor_wrapper_rejects_malformed_or_duplicate_source_ids() -> None:
    _run_isolated(
        """
cases = (
    (
        {"report_sha256": "1" * 64, "sources": [{"source_family": "base"}]},
        ["base:a"],
        "source row invalid",
    ),
    (
        {
            "report_sha256": "1" * 64,
            "sources": [row("base:a"), row("base:a")],
        },
        ["base:a"],
        "duplicate dedup source id",
    ),
    (
        {"report_sha256": "1" * 64, "sources": [row("base:a")]},
        ["base:a", "base:a"],
        "duplicate survivor source id",
    ),
)
for dedup, survivor_ids, expected in cases:
    try:
        mod._outer_survivor_authority(
            dedup,
            {
                "schema_version": "fixture",
                "survivor_authority_sha256": "2" * 64,
                "survivor_source_ids": survivor_ids,
            },
        )
    except mod.CaselawGlobalDedupError as exc:
        assert expected in str(exc)
    else:
        raise AssertionError(f"{expected} was accepted")
"""
    )


def test_two_clean_workflow_uses_exact_numeric_and_stable_identity_checks() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "def require_exact_int(value: object, expected: int, field: str) -> None:" in workflow
    for field in (
        '"schema_version"',
        '"execution_profile"',
        '"execution_claim_issue"',
        '"execution_pr"',
        '"execution_head_sha"',
    ):
        assert field in workflow


def test_source_admission_provenance_scope_is_explicit_and_fail_closed() -> None:
    _run_isolated(
        """
valid = {
    "truth_boundary": {
        "upstream_source_evidence_external_llm_or_api_used": False,
        "current_corpus_external_llm_free_claimed_by_this_adapter": False,
    }
}
scope = mod._source_admission_provenance_scope(valid)
assert scope == {
    "upstream_source_evidence_external_llm_or_api_used": False,
    "current_corpus_external_llm_free_claimed_by_this_adapter": False,
    "historical_source_admission_report_promoted_as_corpus_global_truth": False,
}

for mutated in (
    {},
    {"truth_boundary": {}},
    {
        "truth_boundary": {
            "upstream_source_evidence_external_llm_or_api_used": True,
            "current_corpus_external_llm_free_claimed_by_this_adapter": False,
        }
    },
    {
        "truth_boundary": {
            "upstream_source_evidence_external_llm_or_api_used": False,
            "current_corpus_external_llm_free_claimed_by_this_adapter": True,
        }
    },
):
    try:
        mod._source_admission_provenance_scope(mutated)
    except mod.CaselawGlobalDedupError:
        pass
    else:
        raise AssertionError(f"invalid provenance scope accepted: {mutated!r}")
"""
    )


def test_caselaw_physical_workflow_pins_exact_poppler_pdftotext() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert 'POPPLER_VERSION: "25.06.0"' in workflow
    assert (
        'POPPLER_TARBALL_SHA256: '
        '"8199532d38984fab46dbd0020ec9c40f20e928e33e9b4cc6043572603a821d83"'
        in workflow
    )
    install = workflow.index("- name: Install local-free pinned Poppler build prerequisites")
    build = workflow.index("- name: Build exact Poppler pdftotext from pinned upstream source")
    prepare = workflow.index("- name: Prepare exact historical authority worktrees")
    execute = workflow.index("- name: Execute incumbent reference plus indexed global dedup")
    assert install < build < prepare < execute
    for marker in (
        "sha256sum --check --strict",
        "command -v pdftotext",
        'grep -F "pdftotext version ${POPPLER_VERSION}"',
        "-DENABLE_UTILS=ON",
        "-DENABLE_GLIB=OFF",
        "-DENABLE_QT6=OFF",
    ):
        assert marker in workflow


def test_two_clean_report_hash_uses_real_lf_byte() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    slash = chr(92)
    correct = f'return raw + (b"{slash}n" if newline else b"")'
    incorrect = f'return raw + (b"{slash}{slash}n" if newline else b"")'
    assert correct in workflow
    assert incorrect not in workflow


def test_publication_success_is_create_only_and_cleans_controls() -> None:
    _run_isolated(
        """
import json
from tempfile import TemporaryDirectory

with TemporaryDirectory() as raw:
    root = Path(raw)
    values = (
        (root / "report.json", {"kind": "report", "value": 1}),
        (root / "survivors.json", {"kind": "survivors", "value": 2}),
        (root / "evidence.json", {"kind": "evidence", "value": 3}),
    )
    mod._publish_json_outputs(values)

    prepared = tuple(
        (path, mod._canonical(dict(value)) + b"\\n") for path, value in values
    )
    marker, manifest, stages, _ = mod._publication_control_paths(prepared)
    assert not marker.exists()
    assert not manifest.exists()
    assert not any(stage.exists() for stage in stages)
    for path, value in values:
        assert json.loads(path.read_text(encoding="utf-8")) == value

    try:
        mod._publish_json_outputs(values)
    except mod.CaselawGlobalDedupError as exc:
        assert "refusing to overwrite" in str(exc)
    else:
        raise AssertionError("terminal outputs were overwritten")
"""
    )


def test_publication_recovers_partial_hardlink_transaction() -> None:
    _run_isolated(
        """
import json
from tempfile import TemporaryDirectory

with TemporaryDirectory() as raw:
    root = Path(raw)
    values = (
        (root / "report.json", {"kind": "report"}),
        (root / "survivors.json", {"kind": "survivors"}),
        (root / "evidence.json", {"kind": "evidence"}),
    )
    prepared = tuple(
        (path, mod._canonical(dict(value)) + b"\\n") for path, value in values
    )
    marker, manifest, stages, pathset_id = mod._publication_control_paths(prepared)
    _, manifest_payload = mod._publication_manifest(prepared, stages, pathset_id)

    mod._write_create_only_durable(marker, b"")
    mod._write_create_only_durable(manifest, manifest_payload)
    for (_, payload), stage in zip(prepared, stages, strict=True):
        mod._write_create_only_durable(stage, payload)
    mod._link_staged_output(stages[0], prepared[0][0])

    # A fresh invocation must prove ownership, roll the incomplete transaction
    # back, and then publish one coherent terminal set.
    mod._publish_json_outputs(values)

    assert not marker.exists()
    assert not manifest.exists()
    assert not any(stage.exists() for stage in stages)
    for path, value in values:
        assert json.loads(path.read_text(encoding="utf-8")) == value
"""
    )


def test_publication_tampered_linked_stage_fails_closed() -> None:
    _run_isolated(
        """
from tempfile import TemporaryDirectory

with TemporaryDirectory() as raw:
    root = Path(raw)
    values = (
        (root / "report.json", {"kind": "report"}),
        (root / "survivors.json", {"kind": "survivors"}),
        (root / "evidence.json", {"kind": "evidence"}),
    )
    prepared = tuple(
        (path, mod._canonical(dict(value)) + b"\\n") for path, value in values
    )
    marker, manifest, stages, pathset_id = mod._publication_control_paths(prepared)
    _, manifest_payload = mod._publication_manifest(prepared, stages, pathset_id)

    mod._write_create_only_durable(marker, b"")
    mod._write_create_only_durable(manifest, manifest_payload)
    for (_, payload), stage in zip(prepared, stages, strict=True):
        mod._write_create_only_durable(stage, payload)
    mod._link_staged_output(stages[0], prepared[0][0])
    stages[0].write_bytes(b"tampered")

    try:
        mod._publish_json_outputs(values)
    except mod.CaselawGlobalDedupError as exc:
        assert "digest mismatch" in str(exc)
    else:
        raise AssertionError("tampered incomplete publication was accepted")

    # Fail closed: recovery evidence remains for diagnosis; no second terminal
    # set may be produced around the tampered hard link.
    assert marker.exists()
    assert manifest.exists()
    assert prepared[0][0].exists()
    assert not prepared[1][0].exists()
    assert not prepared[2][0].exists()
"""
    )


def test_publication_link_failure_rolls_back_every_created_path() -> None:
    _run_isolated(
        """
from tempfile import TemporaryDirectory

with TemporaryDirectory() as raw:
    root = Path(raw)
    values = (
        (root / "report.json", {"kind": "report"}),
        (root / "survivors.json", {"kind": "survivors"}),
        (root / "evidence.json", {"kind": "evidence"}),
    )
    prepared = tuple(
        (path, mod._canonical(dict(value)) + b"\\n") for path, value in values
    )
    marker, manifest, stages, _ = mod._publication_control_paths(prepared)
    real_link = mod.os.link

    def fail_link(_source, _target):
        raise OSError("injected hard-link failure")

    mod.os.link = fail_link
    try:
        try:
            mod._publish_json_outputs(values)
        except mod.CaselawGlobalDedupError as exc:
            assert "cannot atomically publish output" in str(exc)
        else:
            raise AssertionError("injected publication failure was ignored")
    finally:
        mod.os.link = real_link

    assert not marker.exists()
    assert not manifest.exists()
    assert not any(stage.exists() for stage in stages)
    assert not any(path.exists() for path, _ in values)
"""
    )


def test_publication_rejects_duplicate_output_path_before_writing() -> None:
    _run_isolated(
        """
from tempfile import TemporaryDirectory

with TemporaryDirectory() as raw:
    root = Path(raw)
    path = root / "same.json"
    try:
        mod._publish_json_outputs(((path, {"a": 1}), (path, {"b": 2})))
    except mod.CaselawGlobalDedupError as exc:
        assert "duplicate output path" in str(exc)
    else:
        raise AssertionError("duplicate output path was accepted")
    assert not path.exists()
"""
    )


def test_publication_manifest_json_is_bounded_and_strict() -> None:
    _run_isolated(
        """
from tempfile import TemporaryDirectory

with TemporaryDirectory() as raw:
    root = Path(raw)
    path = root / "manifest.json"
    bad_inputs = (
        b'{"a":1,"a":1}\\n',
        b'{"a":NaN}\\n',
        b'{"a":1.25}\\n',
        b'{"a":' + b'[' * 3000 + b'0' + b']' * 3000 + b'}\\n',
        b"x" * (mod.PUBLICATION_MANIFEST_MAX_BYTES + 1),
    )
    for payload in bad_inputs:
        path.write_bytes(payload)
        try:
            mod._load_publication_manifest(path)
        except mod.CaselawGlobalDedupError as exc:
            assert "unreadable" in str(exc)
        else:
            raise AssertionError(f"unsafe manifest JSON accepted: {payload[:20]!r}")
"""
    )


def test_publication_manifest_rejects_extra_root_and_target_keys() -> None:
    _run_isolated(
        """
from tempfile import TemporaryDirectory

with TemporaryDirectory() as raw:
    root = Path(raw)
    values = ((root / "out.json", {"kind": "out"}),)
    prepared = tuple(
        (path, mod._canonical(dict(value)) + b"\\n") for path, value in values
    )
    marker, manifest_path, stages, pathset_id = mod._publication_control_paths(prepared)
    manifest, _ = mod._publication_manifest(prepared, stages, pathset_id)

    root_extra = deepcopy(manifest)
    root_extra["extra"] = "forbidden"
    core = {
        key: value
        for key, value in root_extra.items()
        if key != "manifest_identity_sha256"
    }
    root_extra["manifest_identity_sha256"] = mod._sha256(mod._canonical(core))
    manifest_path.write_bytes(mod._canonical(root_extra) + b"\\n")
    try:
        mod._load_publication_manifest(manifest_path)
    except mod.CaselawGlobalDedupError as exc:
        assert "keys invalid" in str(exc)
    else:
        raise AssertionError("extra root key was accepted")
    manifest_path.unlink()

    target_extra = deepcopy(manifest)
    target_extra["targets"][0]["extra"] = "forbidden"
    core = {
        key: value
        for key, value in target_extra.items()
        if key != "manifest_identity_sha256"
    }
    target_extra["manifest_identity_sha256"] = mod._sha256(mod._canonical(core))
    mod._write_create_only_durable(marker, b"")
    mod._write_create_only_durable(
        manifest_path, mod._canonical(target_extra) + b"\\n"
    )
    try:
        mod._recover_incomplete_publication(
            marker, manifest_path, prepared, stages, pathset_id
        )
    except mod.CaselawGlobalDedupError as exc:
        assert "target keys invalid" in str(exc)
    else:
        raise AssertionError("extra target key was accepted")
    assert marker.exists()
    assert manifest_path.exists()
"""
    )

def test_publication_rejects_resolved_output_aliases_before_writing() -> None:
    _run_isolated(
        """
from tempfile import TemporaryDirectory

with TemporaryDirectory() as raw:
    root = Path(raw)
    (root / "nested").mkdir()
    direct = root / "same.json"
    alias = root / "nested" / ".." / "same.json"
    assert direct != alias
    assert direct.resolve() == alias.resolve()
    try:
        mod._publish_json_outputs(
            ((direct, {"a": 1}), (alias, {"b": 2}))
        )
    except mod.CaselawGlobalDedupError as exc:
        assert "duplicate output path" in str(exc)
    else:
        raise AssertionError("resolved output alias was accepted")
    assert not direct.exists()
"""
    )

