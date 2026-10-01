from __future__ import annotations

import importlib.util
import inspect
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "tools" / "run_d03_franko1901_global_dedup_execution_v1.py"


def _load():
    spec = importlib.util.spec_from_file_location("franko1901_global_dedup_execution", MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(source_id: str, *, family: str = "base", size: int = 5) -> dict[str, object]:
    return {
        "source_id": source_id,
        "source_family": family,
        "declared_capacity_bytes": size,
    }


def test_compose_graph_is_add_only_and_preserves_inputs() -> None:
    mod = _load()
    base_inventory = {
        "schema_version": "fixture",
        "sources": [_row("base:a")],
        "lineage_edges": [{"left_source_id": "base:a", "right_source_id": "base:a"}],
        "final_refresh_required": True,
    }
    base_payloads = {"base:a": b"alpha"}
    extension_sources = [_row("franko:a", family=mod.franko1901.SOURCE_FAMILY, size=4)]
    extension_payloads = {"franko:a": b"beta"}
    before = deepcopy(base_inventory)

    inventory, payloads = mod._compose_graph(
        base_inventory, base_payloads, extension_sources, extension_payloads
    )

    assert base_inventory == before
    assert inventory["sources"] == [base_inventory["sources"][0], extension_sources[0]]
    assert inventory["lineage_edges"] == base_inventory["lineage_edges"]
    assert inventory["final_refresh_required"] is False
    assert payloads == {"base:a": b"alpha", "franko:a": b"beta"}


@pytest.mark.parametrize(
    ("base_payloads", "extension_payloads", "message"),
    [
        ({"other": b"a"}, {"franko:a": b"b"}, "base inventory/payload"),
        ({"base:a": b"a"}, {"other": b"b"}, "extension inventory/payload"),
    ],
)
def test_compose_graph_rejects_incomplete_payload_coverage(
    base_payloads: dict[str, bytes],
    extension_payloads: dict[str, bytes],
    message: str,
) -> None:
    mod = _load()
    with pytest.raises(mod.Franko1901GlobalDedupError, match=message):
        mod._compose_graph(
            {"sources": [_row("base:a")]},
            base_payloads,
            [_row("franko:a", family=mod.franko1901.SOURCE_FAMILY)],
            extension_payloads,
        )


def test_compose_graph_rejects_source_id_collision() -> None:
    mod = _load()
    with pytest.raises(mod.Franko1901GlobalDedupError, match="collision"):
        mod._compose_graph(
            {"sources": [_row("same")]},
            {"same": b"a"},
            [_row("same", family=mod.franko1901.SOURCE_FAMILY)],
            {"same": b"b"},
        )


def test_survivor_authority_accounts_only_franko_family() -> None:
    mod = _load()
    dedup = {
        "report_sha256": "1" * 64,
        "sources": [
            _row("base:a", size=10),
            _row("franko:a", family=mod.franko1901.SOURCE_FAMILY, size=7),
            _row("franko:b", family=mod.franko1901.SOURCE_FAMILY, size=11),
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
        "duplicate_clusters": [],
        "survivor_source_ids": ["base:a", "franko:b"],
    }

    authority = mod._outer_survivor_authority(dedup, projection)

    assert authority["franko1901"]["post_dedup_survivor_source_object_count"] == 1
    assert authority["franko1901"]["post_dedup_survivor_declared_capacity_bytes"] == 11
    assert authority["truth_boundary"]["canonical_capacity_credited"] == 0
    assert authority["truth_boundary"]["authorized_optimized_target_exposure"] == 0
    assert authority["truth_boundary"]["whole_corpus_external_llm_cleanliness_claimed"] is False
    assert len(authority["survivor_authority_sha256"]) == 64


@pytest.mark.parametrize("bad", [True, 11.0, "11", -1])
def test_survivor_authority_rejects_nonexact_capacity(bad: object) -> None:
    mod = _load()
    with pytest.raises(mod.Franko1901GlobalDedupError, match="exact nonnegative int"):
        mod._outer_survivor_authority(
            {
                "report_sha256": "1" * 64,
                "sources": [
                    _row(
                        "franko:a",
                        family=mod.franko1901.SOURCE_FAMILY,
                        size=bad,  # type: ignore[arg-type]
                    )
                ],
            },
            {
                "schema_version": "fixture",
                "survivor_authority_sha256": "2" * 64,
                "survivor_source_ids": ["franko:a"],
            },
        )


def test_production_arithmetic_binds_exact_franko_candidate() -> None:
    mod = _load()
    assert mod.EXPECTED_BASE_OBJECTS == 264
    assert mod.EXPECTED_BASE_BYTES == 6_095_624
    assert mod.EXPECTED_FRANKO1901_OBJECTS == 30_660
    assert mod.EXPECTED_FRANKO1901_BYTES == 1_762_005
    assert mod.EXPECTED_COMBINED_OBJECTS == 30_924
    assert mod.EXPECTED_COMBINED_BYTES == 7_857_629
    assert mod.franko1901.UPSTREAM_PRODUCT_PR == 1025
    assert mod.FRANKO1901_FINAL_HEAD == "5816f0ff4ca2f53053123063cd2471442a07d974"
    assert mod.EXPECTED_MAIN == "ba9e49cedba4a110e1c4f7d83702e8fcf8a42461"


def test_composed_inventory_names_exact_franko_product_authority() -> None:
    mod = _load()
    inventory, _ = mod._compose_graph(
        {"sources": [_row("base:a")]},
        {"base:a": b"alpha"},
        [_row("franko:a", family=mod.franko1901.SOURCE_FAMILY, size=4)],
        {"franko:a": b"beta"},
    )
    rule = inventory["terminal_refresh_rule"]
    assert "PR #1025 source-admitted Franko1901" in rule
    assert "PR #1347" not in rule


def test_authority_surface_binds_franko_and_incumbent_execution() -> None:
    mod = _load()
    expected = {
        "src/twelve_six/data/franko1901_dedup_intake.py",
        "configs/data/d03_franko1901_fresh_execution_authority_v1.json",
        "src/twelve_six/data/incumbent_dedup_indexed_execution.py",
        "src/twelve_six/data/_incumbent_dedup_indexed_execution_core.py",
        "tools/run_d03_expanded_global_dedup_v9.py",
        "tools/run_next100_065f_global_dedup_v8.py",
    }
    assert expected <= set(mod.AUTHORITY_PATHS)


def test_execution_head_requires_explicit_exact_selected_head(monkeypatch) -> None:
    mod = _load()
    selected = "a" * 40
    monkeypatch.setattr(
        mod,
        "_git",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout=selected + "\n", stderr=""
        ),
    )
    assert mod._bind_execution_head(selected) == selected


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "a" * 39,
        "A" * 40,
        "g" * 40,
    ],
)
def test_execution_head_rejects_malformed_selection(bad: str) -> None:
    mod = _load()
    with pytest.raises(
        mod.Franko1901GlobalDedupError,
        match="exact lowercase 40-hex",
    ):
        mod._bind_execution_head(bad)


def test_execution_head_rejects_synthetic_or_stale_checkout(monkeypatch) -> None:
    mod = _load()
    selected = "a" * 40
    observed = "b" * 40
    monkeypatch.setattr(
        mod,
        "_git",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout=observed + "\n", stderr=""
        ),
    )
    with pytest.raises(mod.Franko1901GlobalDedupError, match="execution HEAD drift"):
        mod._bind_execution_head(selected)


def test_execute_binds_selected_head_before_authority_and_evidence() -> None:
    mod = _load()
    source = inspect.getsource(mod.execute)
    bind = source.index("execution_head = _bind_execution_head(expected_execution_head)")
    carrier = source.index("execution_carrier_blob = verify_execution_carrier()")
    authority = source.index("authority_blobs = verify_repository_authority()")
    evidence = source.index("evidence_core = {")
    assert bind < carrier < authority < evidence


def test_cli_requires_expected_execution_head() -> None:
    raw = MODULE.read_text(encoding="utf-8")
    assert '"--expected-execution-head"' in raw
    assert "synthetic PR merge commits are rejected" in raw


def test_execution_carrier_binds_worktree_bytes_to_head() -> None:
    mod = _load()
    observed = mod.verify_execution_carrier()
    assert len(observed) == 40


def test_repository_authority_rejects_dirty_authority_worktree(monkeypatch) -> None:
    mod = _load()
    original_git = mod._git

    def fake_git(*args, **kwargs):
        if args[:3] == ("diff", "--quiet", "HEAD"):
            return SimpleNamespace(returncode=1, stdout="", stderr="")
        return original_git(*args, **kwargs)

    monkeypatch.setattr(mod, "_git", fake_git)
    with pytest.raises(mod.Franko1901GlobalDedupError, match="worktree drift"):
        mod.verify_repository_authority()


def test_execute_records_exact_carrier_blob_before_evidence() -> None:
    mod = _load()
    source = inspect.getsource(mod.execute)
    carrier = source.index("execution_carrier_blob = verify_execution_carrier()")
    evidence = source.index("evidence_core = {")
    assert carrier < evidence
    assert '"execution_carrier_git_blob_sha1": execution_carrier_blob' in source


def test_repository_authority_is_current_main_ancestry_bound() -> None:
    mod = _load()
    blobs = mod.verify_repository_authority()
    assert set(blobs) == set(mod.AUTHORITY_PATHS)
    assert all(len(value) == 40 for value in blobs.values())


def test_bulk_workspace_gate_creates_and_requires_empty_directory(tmp_path: Path) -> None:
    mod = _load()
    workspace = tmp_path / "bulk"
    resolved = mod._prepare_empty_bulk_workspace(workspace)
    assert resolved == workspace.resolve(strict=True)
    assert list(workspace.iterdir()) == []

    (workspace / "stale.txt").write_text("stale", encoding="utf-8")
    with pytest.raises(mod.Franko1901GlobalDedupError, match="bulk workspace must be empty"):
        mod._prepare_empty_bulk_workspace(workspace)


def test_bulk_workspace_gate_rejects_symlink(tmp_path: Path) -> None:
    mod = _load()
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "bulk-link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink not available on this platform")
    with pytest.raises(mod.Franko1901GlobalDedupError, match="must not be a symlink"):
        mod._prepare_empty_bulk_workspace(link)


def test_v7_worktree_requires_exact_expected_head(monkeypatch, tmp_path: Path) -> None:
    mod = _load()

    def fake_git(root: Path, *args: str, check: bool = True):
        del root, check
        if args == ("rev-parse", "HEAD"):
            return SimpleNamespace(returncode=0, stdout="0" * 40 + "\n", stderr="")
        raise AssertionError(args)

    monkeypatch.setattr(mod, "_git_in", fake_git)
    with pytest.raises(mod.Franko1901GlobalDedupError, match="V7 worktree HEAD drift"):
        mod._verify_v7_worktree(tmp_path)


def test_v7_worktree_rejects_tracked_or_noncache_untracked_drift(
    monkeypatch, tmp_path: Path
) -> None:
    mod = _load()
    statuses = [
        " M src/twelve_six/data/cross_source_capacity_audit_v7.py\n",
        "?? injected.py\n",
    ]
    for status in statuses:
        def fake_git(root: Path, *args: str, check: bool = True, status=status):
            del root, check
            if args == ("rev-parse", "HEAD"):
                return SimpleNamespace(
                    returncode=0,
                    stdout=mod.v8.EXPECTED_V7_HEAD + "\n",
                    stderr="",
                )
            if args == ("status", "--porcelain=v1", "--untracked-files=all"):
                return SimpleNamespace(returncode=0, stdout=status, stderr="")
            raise AssertionError(args)

        monkeypatch.setattr(mod, "_git_in", fake_git)
        with pytest.raises(mod.Franko1901GlobalDedupError, match="V7 worktree is not clean"):
            mod._verify_v7_worktree(tmp_path)


def test_v7_worktree_rejects_untracked_python_bytecode(monkeypatch, tmp_path: Path) -> None:
    mod = _load()

    def fake_git(root: Path, *args: str, check: bool = True):
        del root, check
        if args == ("rev-parse", "HEAD"):
            return SimpleNamespace(
                returncode=0,
                stdout=mod.v8.EXPECTED_V7_HEAD + "\n",
                stderr="",
            )
        if args == ("status", "--porcelain=v1", "--untracked-files=all"):
            return SimpleNamespace(
                returncode=0,
                stdout="?? src/twelve_six/data/__pycache__/matcher.cpython-312.pyc\n",
                stderr="",
            )
        raise AssertionError(args)

    monkeypatch.setattr(mod, "_git_in", fake_git)
    with pytest.raises(mod.Franko1901GlobalDedupError, match="V7 worktree is not clean"):
        mod._verify_v7_worktree(tmp_path)


def test_historical_reconstruction_disables_bytecode_cache_writes() -> None:
    mod = _load()
    source = inspect.getsource(mod._reconstruct_v8_with_historical_namespace)
    enable = source.index("sys.dont_write_bytecode = True")
    reconstruct = source.index("v9_runner.reconstruct_v8_source_inputs(")
    restore = source.index("sys.dont_write_bytecode = previous_dont_write_bytecode")
    assert enable < reconstruct < restore


def test_execute_binds_v7_worktree_before_reconstruction() -> None:
    mod = _load()
    source = inspect.getsource(mod.execute)
    workspace = source.index("bulk_workspace = _prepare_empty_bulk_workspace(bulk_workspace)")
    bind = source.index("verified_v7_head = _verify_v7_worktree(v7_root)")
    reconstruct = source.index("_reconstruct_v8_with_historical_namespace(")
    reverify = source.index(
        "_verify_v7_worktree(v7_root) == verified_v7_head",
        reconstruct,
    )
    evidence = source.index('"v7_head_sha": verified_v7_head')
    assert workspace < bind < reconstruct < reverify < evidence


def test_historical_matcher_namespace_includes_pipeline() -> None:
    mod = _load()
    assert "twelve_six.data.pipeline" in mod._HISTORICAL_MATCHER_MODULES


def test_runtime_attestation_precedes_indexed_matcher_path() -> None:
    mod = _load()
    source = inspect.getsource(mod.execute)
    attest = source.index("indexed.attest_incumbent_runtime(matcher)")
    indexed = source.index("indexed_report = indexed.audit_payloads_indexed(")
    assert attest < indexed


def test_production_execution_does_not_reintroduce_all_pairs_reference() -> None:
    mod = _load()
    source = inspect.getsource(mod.execute)
    assert "matcher.audit_payloads(inventory, payloads)" not in source
    assert '"indexed_executor_performance_equivalence_authority": "MERGED_PR_1459"' in source


def test_execution_does_not_rebuild_index_only_for_telemetry() -> None:
    mod = _load()
    source = inspect.getsource(mod.execute)
    assert "candidate_pair_indices_with_stats" not in source
    assert "execution_stats(" not in source


def test_execute_uses_exact_three_part_franko_authority() -> None:
    mod = _load()
    source = inspect.getsource(mod.execute)
    assert "franko1901.validate_and_project_franko(" in source
    assert "candidate_jsonl," in source
    assert "historical_terminal_evidence_json," in source
    assert "fresh_execution_authority_json," in source
    assert "validate_and_project_franko1901" not in source


def test_terminal_summary_requires_exact_arithmetic() -> None:
    mod = _load()
    good = {
        "source_count": mod.EXPECTED_COMBINED_OBJECTS,
        "terminal_candidates": {
            "declared_capacity_bytes_before": mod.EXPECTED_COMBINED_BYTES,
            "conservative_unique_capacity_bytes_after": mod.EXPECTED_COMBINED_BYTES - 17,
            "duplicate_discount_bytes": 17,
            "duplicate_cluster_count": 3,
        },
    }
    assert mod._validated_terminal_summary(good) == good["terminal_candidates"]

    for field, bad in (
        ("conservative_unique_capacity_bytes_after", mod.EXPECTED_COMBINED_BYTES + 1),
        ("duplicate_discount_bytes", 18),
        ("duplicate_cluster_count", -1),
    ):
        broken = {
            "source_count": good["source_count"],
            "terminal_candidates": dict(good["terminal_candidates"]),
        }
        broken["terminal_candidates"][field] = bad
        with pytest.raises(mod.Franko1901GlobalDedupError):
            mod._validated_terminal_summary(broken)


def test_terminal_summary_rejects_bool_aliases() -> None:
    mod = _load()
    report = {
        "source_count": mod.EXPECTED_COMBINED_OBJECTS,
        "terminal_candidates": {
            "declared_capacity_bytes_before": mod.EXPECTED_COMBINED_BYTES,
            "conservative_unique_capacity_bytes_after": True,
            "duplicate_discount_bytes": mod.EXPECTED_COMBINED_BYTES - 1,
            "duplicate_cluster_count": 0,
        },
    }
    with pytest.raises(mod.Franko1901GlobalDedupError):
        mod._validated_terminal_summary(report)


def _good_survivor_projection(mod):
    after = mod.EXPECTED_COMBINED_BYTES - 17
    report = {
        "report_sha256": "1" * 64,
        "source_count": mod.EXPECTED_COMBINED_OBJECTS,
        "terminal_candidates": {
            "declared_capacity_bytes_before": mod.EXPECTED_COMBINED_BYTES,
            "conservative_unique_capacity_bytes_after": after,
            "duplicate_discount_bytes": 17,
            "duplicate_cluster_count": 1,
        },
    }
    projection = {
        "schema_version": mod.v9_semantics.SURVIVOR_SCHEMA,
        "matcher_report_sha256": report["report_sha256"],
        "pre_dedup_source_object_count": mod.EXPECTED_COMBINED_OBJECTS,
        "post_dedup_survivor_source_object_count": 2,
        "pre_dedup_declared_capacity_bytes": mod.EXPECTED_COMBINED_BYTES,
        "post_dedup_declared_capacity_bytes": after,
        "duplicate_discount_bytes": 17,
        "duplicate_cluster_count": 1,
        "duplicate_clusters": [
            {
                "member_source_ids": ["base:a", "franko:a"],
                "selected_source_id": "base:a",
                "selected_declared_capacity_bytes": after,
            }
        ],
        "survivor_source_ids": ["base:a", "franko:b"],
    }
    projection["survivor_authority_sha256"] = mod._sha256(mod._canonical(projection))
    return report, projection


def _reseal_survivor_projection(mod, projection: dict[str, object]) -> None:
    core = dict(projection)
    core.pop("survivor_authority_sha256", None)
    projection["survivor_authority_sha256"] = mod._sha256(mod._canonical(core))


def test_survivor_projection_is_cross_bound_to_terminal_report() -> None:
    mod = _load()
    report, projection = _good_survivor_projection(mod)
    mod._validate_survivor_projection(report, projection)


def test_survivor_projection_rejects_forged_self_identity() -> None:
    mod = _load()
    report, projection = _good_survivor_projection(mod)
    projection["survivor_authority_sha256"] = "0" * 64
    with pytest.raises(
        mod.Franko1901GlobalDedupError,
        match="survivor projection identity drift",
    ):
        mod._validate_survivor_projection(report, projection)


@pytest.mark.parametrize(
    "bad",
    [
        "A" * 64,
        "g" * 64,
        "0" * 63,
        True,
    ],
)
def test_survivor_projection_rejects_noncanonical_identity_format(bad: object) -> None:
    mod = _load()
    report, projection = _good_survivor_projection(mod)
    projection["survivor_authority_sha256"] = bad
    with pytest.raises(
        mod.Franko1901GlobalDedupError,
        match="survivor projection identity format invalid",
    ):
        mod._validate_survivor_projection(report, projection)


@pytest.mark.parametrize(
    ("field", "bad", "message"),
    [
        ("matcher_report_sha256", "2" * 64, "matcher identity"),
        ("pre_dedup_source_object_count", 1, "pre-dedup count"),
        ("pre_dedup_declared_capacity_bytes", 1, "pre-dedup bytes"),
        ("post_dedup_declared_capacity_bytes", 1, "post-dedup bytes"),
        ("duplicate_discount_bytes", 16, "duplicate discount"),
        ("duplicate_cluster_count", 0, "cluster count"),
    ],
)
def test_survivor_projection_rejects_terminal_drift(
    field: str, bad: object, message: str
) -> None:
    mod = _load()
    report, projection = _good_survivor_projection(mod)
    projection[field] = bad
    _reseal_survivor_projection(mod, projection)
    with pytest.raises(mod.Franko1901GlobalDedupError, match=message):
        mod._validate_survivor_projection(report, projection)


def test_survivor_projection_rejects_id_cardinality_drift() -> None:
    mod = _load()
    report, projection = _good_survivor_projection(mod)
    projection["survivor_source_ids"] = ["base:a", "base:a"]
    _reseal_survivor_projection(mod, projection)
    with pytest.raises(mod.Franko1901GlobalDedupError, match="ids invalid"):
        mod._validate_survivor_projection(report, projection)


def test_execute_validates_survivor_projection_before_outer_publication() -> None:
    mod = _load()
    source = inspect.getsource(mod.execute)
    derived = source.index("selection_projection = v9_semantics._derive_survivors(indexed_report)")
    validated = source.index("_validate_survivor_projection(indexed_report, selection_projection)")
    wrapped = source.index("survivors = _outer_survivor_authority(indexed_report, selection_projection)")
    assert derived < validated < wrapped


def test_source_provenance_negative_is_explicitly_source_local() -> None:
    mod = _load()
    scope = mod._source_admission_provenance_scope(
        {"truth_boundary": {"external_llm_or_api_used_for_data_or_intelligence": False}}
    )
    assert scope == {
        "upstream_source_package_external_llm_or_api_used": False,
        "current_corpus_external_llm_free_claimed_by_this_carrier": False,
        "source_local_negative_promoted_as_corpus_global_truth": False,
    }
    with pytest.raises(mod.Franko1901GlobalDedupError):
        mod._source_admission_provenance_scope(
            {"truth_boundary": {"external_llm_or_api_used_for_data_or_intelligence": True}}
        )


def test_truth_boundary_never_promotes_source_local_external_llm_negative() -> None:
    mod = _load()
    authority = mod._outer_survivor_authority(
        {
            "report_sha256": "1" * 64,
            "sources": [_row("franko:a", family=mod.franko1901.SOURCE_FAMILY)],
        },
        {
            "schema_version": "fixture",
            "survivor_authority_sha256": "2" * 64,
            "survivor_source_ids": ["franko:a"],
        },
    )
    truth = authority["truth_boundary"]
    assert truth["whole_corpus_external_llm_cleanliness_claimed"] is False
    assert "external_llm_or_api_used_for_data_or_intelligence" not in truth


def test_runtime_environment_is_provider_neutral_for_local_execution(monkeypatch) -> None:
    mod = _load()
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("RUNNER_ENVIRONMENT", raising=False)
    monkeypatch.delenv("RUNNER_OS", raising=False)
    monkeypatch.delenv("RUNNER_ARCH", raising=False)
    observed = mod._runtime_environment()
    assert observed == {
        "python_platform": mod.sys.platform,
        "github_actions": False,
        "runner_environment": "local",
    }


def test_runtime_environment_records_github_runner_without_claiming_provider_policy(
    monkeypatch,
) -> None:
    mod = _load()
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("RUNNER_ENVIRONMENT", "github-hosted")
    monkeypatch.setenv("RUNNER_OS", "Linux")
    monkeypatch.setenv("RUNNER_ARCH", "X64")
    assert mod._runtime_environment() == {
        "python_platform": mod.sys.platform,
        "github_actions": True,
        "runner_environment": "github-hosted",
        "runner_os": "Linux",
        "runner_arch": "X64",
    }


def test_runtime_environment_rejects_ambiguous_github_runner(monkeypatch) -> None:
    mod = _load()
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.delenv("RUNNER_ENVIRONMENT", raising=False)
    monkeypatch.setenv("RUNNER_OS", "Linux")
    monkeypatch.setenv("RUNNER_ARCH", "X64")
    with pytest.raises(
        mod.Franko1901GlobalDedupError,
        match="runner environment identity missing",
    ):
        mod._runtime_environment()


def test_evidence_profile_is_local_free_not_hardcoded_github_hosted() -> None:
    raw = MODULE.read_text(encoding="utf-8")
    assert '"execution_profile": "LOCAL_FREE"' in raw
    assert '"runtime_environment": _runtime_environment()' in raw
    assert "GITHUB_HOSTED_FREE_LOCAL_FREE" not in raw


def test_linux_max_rss_preserves_kib_even_above_ten_million(monkeypatch) -> None:
    mod = _load()
    monkeypatch.setattr(mod.sys, "platform", "linux")
    monkeypatch.setattr(
        mod,
        "resource",
        SimpleNamespace(
            RUSAGE_SELF=0,
            getrusage=lambda _: SimpleNamespace(ru_maxrss=12_000_000),
        ),
    )
    assert mod._max_rss_kib() == 12_000_000


def test_darwin_max_rss_converts_bytes_to_kib(monkeypatch) -> None:
    mod = _load()
    monkeypatch.setattr(mod.sys, "platform", "darwin")
    monkeypatch.setattr(
        mod,
        "resource",
        SimpleNamespace(
            RUSAGE_SELF=0,
            getrusage=lambda _: SimpleNamespace(ru_maxrss=4097),
        ),
    )
    assert mod._max_rss_kib() == 5


def test_windows_max_rss_uses_windows_backend(monkeypatch) -> None:
    mod = _load()
    monkeypatch.setattr(mod.sys, "platform", "win32")
    monkeypatch.setattr(mod, "_windows_peak_working_set_kib", lambda: 777)
    assert mod._max_rss_kib() == 777


def test_unknown_platform_max_rss_fails_closed(monkeypatch) -> None:
    mod = _load()
    monkeypatch.setattr(mod.sys, "platform", "freebsd")
    monkeypatch.setattr(
        mod,
        "resource",
        SimpleNamespace(
            RUSAGE_SELF=0,
            getrusage=lambda _: SimpleNamespace(ru_maxrss=500),
        ),
    )
    assert mod._max_rss_kib() is None


def test_execute_requires_positive_measured_rss_before_evidence() -> None:
    mod = _load()
    source = inspect.getsource(mod.execute)
    rss_measure = source.index("process_max_rss_kib = _max_rss_kib()")
    rss_gate = source.index('"process max RSS unavailable; refusing terminal execution evidence"')
    evidence = source.index("evidence_core = {")
    assert rss_measure < rss_gate < evidence
    assert '"process_max_rss_kib": process_max_rss_kib' in source


def test_cli_requires_both_franko_authority_documents() -> None:
    raw = MODULE.read_text(encoding="utf-8")
    assert (
        'parser.add_argument("--historical-terminal-evidence-json", type=Path, required=True)'
        in raw
    )
    assert (
        'parser.add_argument("--fresh-execution-authority-json", type=Path, required=True)'
        in raw
    )


def test_write_json_is_true_create_only(tmp_path: Path) -> None:
    mod = _load()
    output = tmp_path / "evidence.json"
    mod._write_json(output, {"b": 2, "a": 1})
    assert output.read_bytes() == b'{"a":1,"b":2}\n'

    with pytest.raises(mod.Franko1901GlobalDedupError, match="refusing to overwrite"):
        mod._write_json(output, {"changed": True})
    assert output.read_bytes() == b'{"a":1,"b":2}\n'


def test_write_json_nonfinite_fails_before_publication(tmp_path: Path) -> None:
    mod = _load()
    output = tmp_path / "evidence.json"
    with pytest.raises(ValueError):
        mod._write_json(output, {"rss": float("nan")})
    assert not output.exists()


def test_output_set_rolls_back_when_later_path_exists(tmp_path: Path) -> None:
    mod = _load()
    report = tmp_path / "report.json"
    survivors = tmp_path / "survivors.json"
    evidence = tmp_path / "evidence.json"
    evidence.write_bytes(b"existing-evidence\n")

    with pytest.raises(mod.Franko1901GlobalDedupError, match="refusing to overwrite"):
        mod._publish_json_outputs(
            (
                (report, {"kind": "report"}),
                (survivors, {"kind": "survivors"}),
                (evidence, {"kind": "evidence"}),
            )
        )

    assert not report.exists()
    assert not survivors.exists()
    assert evidence.read_bytes() == b"existing-evidence\n"


def test_output_set_serializes_all_values_before_publication(tmp_path: Path) -> None:
    mod = _load()
    report = tmp_path / "report.json"
    evidence = tmp_path / "evidence.json"

    with pytest.raises(ValueError):
        mod._publish_json_outputs(
            (
                (report, {"kind": "report"}),
                (evidence, {"rss": float("nan")}),
            )
        )

    assert not report.exists()
    assert not evidence.exists()


def _publication_outputs(tmp_path: Path):
    return (
        (tmp_path / "report.json", {"kind": "report"}),
        (tmp_path / "survivors.json", {"kind": "survivors"}),
        (tmp_path / "evidence.json", {"kind": "evidence"}),
    )


@pytest.mark.parametrize("interrupt_after", [1, 2])
def test_output_set_recovers_process_interruption_after_final_link(
    tmp_path: Path, monkeypatch, interrupt_after: int
) -> None:
    mod = _load()
    outputs = _publication_outputs(tmp_path)
    original_link = mod._link_staged_output
    calls = 0

    def interrupting_link(stage_path: Path, final_path: Path) -> None:
        nonlocal calls
        original_link(stage_path, final_path)
        calls += 1
        if calls == interrupt_after:
            raise KeyboardInterrupt("simulated process interruption")

    monkeypatch.setattr(mod, "_link_staged_output", interrupting_link)
    with pytest.raises(KeyboardInterrupt, match="simulated process interruption"):
        mod._publish_json_outputs(outputs)

    prepared = tuple(
        (path, mod._canonical(dict(value)) + b"\n") for path, value in outputs
    )
    marker = mod._publication_marker_path(prepared)
    assert marker.exists()
    assert sum(path.exists() for path, _ in outputs) == interrupt_after

    monkeypatch.setattr(mod, "_link_staged_output", original_link)
    mod._publish_json_outputs(outputs)

    assert not marker.exists()
    assert [path.read_bytes() for path, _ in outputs] == [
        b'{"kind":"report"}\n',
        b'{"kind":"survivors"}\n',
        b'{"kind":"evidence"}\n',
    ]
    assert not list(tmp_path.glob(".*.stage-*"))


def test_invalid_manifest_without_payload_is_recovered(tmp_path: Path) -> None:
    mod = _load()
    outputs = _publication_outputs(tmp_path)
    prepared = tuple(
        (path, mod._canonical(dict(value)) + b"\n") for path, value in outputs
    )
    marker, manifest, stages, _ = mod._publication_control_paths(prepared)
    mod._write_create_only_durable(marker, b"")
    mod._write_create_only_durable(manifest, b"{broken\n")

    assert not any(path.exists() for path, _ in outputs)
    assert not any(stage.exists() for stage in stages)

    mod._publish_json_outputs(outputs)

    assert not marker.exists()
    assert not manifest.exists()
    assert [path.read_bytes() for path, _ in outputs] == [
        b'{"kind":"report"}\n',
        b'{"kind":"survivors"}\n',
        b'{"kind":"evidence"}\n',
    ]


def test_invalid_manifest_with_stage_residue_fails_closed(tmp_path: Path) -> None:
    mod = _load()
    outputs = _publication_outputs(tmp_path)
    prepared = tuple(
        (path, mod._canonical(dict(value)) + b"\n") for path, value in outputs
    )
    marker, manifest, stages, _ = mod._publication_control_paths(prepared)
    mod._write_create_only_durable(marker, b"")
    mod._write_create_only_durable(manifest, b"{broken\n")
    mod._write_create_only_durable(stages[0], b"partial")

    with pytest.raises(
        mod.Franko1901GlobalDedupError,
        match="invalid incomplete manifest coexists with payload paths",
    ):
        mod._publish_json_outputs(outputs)

    assert marker.exists()
    assert manifest.exists()
    assert stages[0].read_bytes() == b"partial"
    assert not any(path.exists() for path, _ in outputs)


def test_incomplete_marker_never_deletes_digest_mismatched_final(
    tmp_path: Path, monkeypatch
) -> None:
    mod = _load()
    outputs = _publication_outputs(tmp_path)
    original_link = mod._link_staged_output

    def interrupt_first(stage_path: Path, final_path: Path) -> None:
        original_link(stage_path, final_path)
        raise KeyboardInterrupt("simulated process interruption")

    monkeypatch.setattr(mod, "_link_staged_output", interrupt_first)
    with pytest.raises(KeyboardInterrupt):
        mod._publish_json_outputs(outputs)

    report = outputs[0][0]
    report.write_bytes(b"tampered\n")
    monkeypatch.setattr(mod, "_link_staged_output", original_link)

    with pytest.raises(
        mod.Franko1901GlobalDedupError,
        match="incomplete publication output digest mismatch",
    ):
        mod._publish_json_outputs(outputs)

    assert report.read_bytes() == b"tampered\n"


def test_preexisting_authority_fails_before_incomplete_marker(tmp_path: Path) -> None:
    mod = _load()
    outputs = _publication_outputs(tmp_path)
    evidence = outputs[-1][0]
    evidence.write_bytes(b"preexisting\n")
    prepared = tuple(
        (path, mod._canonical(dict(value)) + b"\n") for path, value in outputs
    )
    marker = mod._publication_marker_path(prepared)

    with pytest.raises(mod.Franko1901GlobalDedupError, match="refusing to overwrite"):
        mod._publish_json_outputs(outputs)

    assert evidence.read_bytes() == b"preexisting\n"
    assert not marker.exists()
    assert not outputs[0][0].exists()
    assert not outputs[1][0].exists()


def test_no_training_or_capacity_promotion_in_execution_evidence() -> None:
    raw = MODULE.read_text(encoding="utf-8")
    assert '"canonical_capacity_credited": 0' in raw
    assert '"authorized_optimized_target_exposure": 0' in raw
    assert '"tokenizer_fit_authorized": False' in raw
    assert '"training_executed": False' in raw
    assert '"learned_weights_created": False' in raw
    assert '"final_test_outcomes_read": False' in raw
    assert '"paid_compute_used": False' in raw
