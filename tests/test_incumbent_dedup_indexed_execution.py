import ast
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


class _LazyIndexed:
    _module: ModuleType | None = None

    def __getattr__(self, name: str) -> Any:
        module = self._module
        if module is None:
            module = importlib.import_module(
                "twelve_six.data.incumbent_dedup_indexed_execution"
            )
            self._module = module
        return getattr(module, name)


indexed = _LazyIndexed()


class FakeV1:
    @staticmethod
    def normalize_for_contamination(text: str, modality: str) -> str:
        del modality
        return " ".join(text.casefold().split())


def _fp(
    source_id: str,
    *,
    modality: str = "uk",
    origin: str | None = None,
    raw: str | None = None,
    normalized: str | None = None,
    shingles: frozenset[str] = frozenset(),
    skeleton: frozenset[str] = frozenset(),
    text: str = "unrelated body",
):
    return {
        "row": {
            "source_id": source_id,
            "modality": modality,
            "origin_key": origin or source_id,
        },
        "raw_sha256": raw or f"raw-{source_id}",
        "normalized_sha256": normalized or f"norm-{source_id}",
        "shingles": shingles,
        "skeleton_shingles": skeleton,
        "text": text,
    }


def test_terminal_science_constants_are_exact_and_non_overridable():
    assert indexed.EXPECTED_DATA232_GIT_BLOB_SHA1 == "dab5da98dfc43133aa8f3c2e3c78c809252b741b"
    assert indexed.EXPECTED_THRESHOLDS == {
        "natural_shingle_tokens": 3,
        "natural_near_jaccard": 0.80,
        "natural_fragment_containment": 0.88,
        "natural_fragment_min_tokens": 18,
        "code_shingle_tokens": 7,
        "code_near_jaccard": 0.86,
        "code_fragment_containment": 0.90,
        "code_fragment_min_tokens": 16,
        "code_copy_jaccard": 0.82,
        "code_copy_min_tokens": 16,
    }
    assert tuple(indexed.inspect.signature(indexed.attest_incumbent_runtime).parameters) == ("v3",)
    assert "expected_v3_blob" not in indexed.inspect.signature(indexed.audit_payloads_indexed).parameters
    assert "expected_v1_blob" not in indexed.inspect.signature(indexed.audit_payloads_indexed).parameters


def test_executable_attestation_rejects_in_memory_callable_substitution(tmp_path):
    path = tmp_path / "authority.py"
    path.write_text("VALUE = 7\ndef semantic(value):\n    return value + VALUE\n", encoding="utf-8")
    spec = importlib.util.spec_from_file_location("_indexed_attestation_fixture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    indexed._attest_executable_module(module, "FIXTURE")
    module.semantic = lambda value: value
    with pytest.raises(indexed.IndexedExecutionError, match="callable"):
        indexed._attest_executable_module(module, "FIXTURE")


def test_candidate_index_contains_each_incumbent_necessary_condition():
    shared_edge = "This shared publisher footer has comfortably more than thirty two characters."
    rows = [
        _fp("a", origin="same-origin"),
        _fp("b", origin="same-origin"),
        _fp("c", raw="same-raw"),
        _fp("d", raw="same-raw"),
        _fp("e", normalized="same-normalized"),
        _fp("f", normalized="same-normalized"),
        _fp("g", shingles=frozenset({"natural-shingle"})),
        _fp("h", shingles=frozenset({"natural-shingle"})),
        _fp("i", modality="code", skeleton=frozenset({"skeleton-shingle"})),
        _fp("j", modality="code", skeleton=frozenset({"skeleton-shingle"})),
        _fp("k", text=f"header\n{shared_edge}\nbody"),
        _fp("l", text=f"other\n{shared_edge}\ntail"),
        _fp("m"),
    ]
    pairs = set(indexed.candidate_pair_indices(FakeV1, rows))
    assert (0, 1) in pairs
    assert (2, 3) in pairs
    assert (4, 5) in pairs
    assert (6, 7) in pairs
    assert (8, 9) in pairs
    assert (10, 11) in pairs
    assert all(12 not in pair for pair in pairs)


def test_content_shingles_do_not_cross_code_natural_boundary():
    rows = [
        _fp("natural", shingles=frozenset({"same"})),
        _fp("code", modality="code", shingles=frozenset({"same"})),
    ]
    assert indexed.candidate_pair_indices(FakeV1, rows) == []


def test_candidate_budget_fails_closed():
    rows = [_fp(str(index), origin="same") for index in range(5)]
    with pytest.raises(indexed.IndexedExecutionError, match="candidate pair budget exceeded"):
        indexed.candidate_pair_indices(FakeV1, rows, max_candidate_pairs=2)


def test_index_posting_budget_fails_before_unbounded_growth():
    rows = [_fp("a", shingles=frozenset({"s1", "s2", "s3"}))]
    with pytest.raises(indexed.IndexedExecutionError, match="index posting work budget exceeded"):
        indexed.candidate_pair_indices(FakeV1, rows, max_index_postings=5)


def test_repeated_key_amplification_collapses_identical_bucket_signature():
    shared = frozenset(f"shared-{index}" for index in range(1_000))
    rows = [_fp(str(index), shingles=shared) for index in range(10)]
    pairs, stats = indexed.candidate_pair_indices_with_stats(
        FakeV1,
        rows,
        max_pair_expansions=100,
    )
    assert len(pairs) == 45
    assert stats["unique_candidate_pairs"] == 45
    assert stats["pair_expansion_attempts"] == 45
    assert stats["unique_bucket_signatures"] == 1


def test_pair_expansion_budget_is_independent_of_unique_candidate_budget():
    rows = [
        _fp("0", shingles=frozenset({"a", "b"})),
        _fp("1", shingles=frozenset({"a"})),
        _fp("2", shingles=frozenset({"b"})),
    ]
    with pytest.raises(indexed.IndexedExecutionError, match="pair expansion work budget exceeded"):
        indexed.candidate_pair_indices_with_stats(
            FakeV1,
            rows,
            max_candidate_pairs=100,
            max_pair_expansions=1,
        )


def test_execution_stats_exact_rada_scale_and_work_telemetry():
    rada = indexed.execution_stats(101_559, 0, index_postings=123, pair_expansion_attempts=45)
    assert rada["incumbent_all_pair_dispatches"] == 5_157_064_461
    assert rada["index_postings"] == 123
    assert rada["pair_expansion_attempts"] == 45
    combined = indexed.execution_stats(101_821, 0)
    assert combined["incumbent_all_pair_dispatches"] == 5_183_707_110


def test_execution_stats_rejects_impossible_candidate_count():
    with pytest.raises(indexed.IndexedExecutionError, match="exceeds all-pairs"):
        indexed.execution_stats(2, 2)


def test_dedicated_indexed_harness_never_imports_authority_during_collection():
    target = "twelve_six.data.incumbent_dedup_indexed_execution"
    test_dir = Path(__file__).parent
    dedicated = (
        "test_incumbent_dedup_direct_import_behavior.py",
        "test_incumbent_dedup_imported_member_closure.py",
        "test_incumbent_dedup_indexed_execution.py",
        "test_incumbent_dedup_json_re_transitive_closure.py",
        "test_incumbent_dedup_runtime_closure.py",
    )

    violations: list[str] = []
    for filename in dedicated:
        tree = ast.parse((test_dir / filename).read_text(encoding="utf-8"), filename=filename)
        for node in tree.body:
            if isinstance(node, ast.Import):
                if any(alias.name == target for alias in node.names):
                    violations.append(f"{filename}:{node.lineno}: import {target}")
            elif isinstance(node, ast.ImportFrom):
                if node.module == target:
                    violations.append(f"{filename}:{node.lineno}: from {target} import ...")
                elif node.module == "twelve_six.data" and any(
                    alias.name == "incumbent_dedup_indexed_execution" for alias in node.names
                ):
                    violations.append(
                        f"{filename}:{node.lineno}: from twelve_six.data import authority"
                    )

    assert violations == []


    blocker_script = r"""
import sys

target = "twelve_six.data.incumbent_dedup_indexed_execution"


class BlockAuthorityImport:
    def find_spec(self, fullname, path=None, target=None):
        del path, target
        if fullname == "twelve_six.data.incumbent_dedup_indexed_execution":
            raise RuntimeError(f"collection imported forbidden authority: {fullname}")
        return None


sys.meta_path.insert(0, BlockAuthorityImport())
import pytest

raise SystemExit(pytest.main(["--collect-only", "-q", *sys.argv[1:]]))
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            blocker_script,
            *(str(test_dir / filename) for filename in dedicated),
        ],
        cwd=test_dir.parent,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

