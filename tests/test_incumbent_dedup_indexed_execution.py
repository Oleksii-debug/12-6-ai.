import ast
import functools
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

_ISOLATED_INDEXED_TEST_NODE = "TWELVE_SIX_ISOLATED_INDEXED_TEST_NODE"


def _isolated_indexed_test(test):
    @functools.wraps(test)
    def wrapper(*args, **kwargs):
        current = os.environ.get("PYTEST_CURRENT_TEST", "").rsplit(" (", 1)[0]
        if os.environ.get(_ISOLATED_INDEXED_TEST_NODE) == current and current:
            return test(*args, **kwargs)
        if not current:
            return test(*args, **kwargs)

        env = os.environ.copy()
        env[_ISOLATED_INDEXED_TEST_NODE] = current
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", current],
            cwd=Path(__file__).resolve().parent.parent,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=180,
        )
        assert completed.returncode == 0, (
            f"isolated indexed test failed: {current}\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )

    return wrapper


class _LazyIndexed:
    _module: ModuleType | None = None

    def _load_module(self) -> ModuleType:
        module = object.__getattribute__(self, "_module")
        if module is None:
            module = importlib.import_module(
                "twelve_six.data.incumbent_dedup_indexed_execution"
            )
            object.__setattr__(self, "_module", module)
        return module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._load_module(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_module":
            object.__setattr__(self, name, value)
            return
        setattr(self._load_module(), name, value)

    def __delattr__(self, name: str) -> None:
        if name == "_module":
            object.__delattr__(self, name)
            return
        delattr(self._load_module(), name)


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


@_isolated_indexed_test
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


@_isolated_indexed_test
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


@_isolated_indexed_test
def test_candidate_index_contains_each_incumbent_necessary_condition():
    shared_edge = (
        "This shared publisher footer is deliberately longer than eighty normalized "
        "characters so it can satisfy the incumbent boilerplate predicate."
    )
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


@_isolated_indexed_test
def test_content_shingles_do_not_cross_code_natural_boundary():
    rows = [
        _fp("natural", shingles=frozenset({"same"})),
        _fp("code", modality="code", shingles=frozenset({"same"})),
    ]
    assert indexed.candidate_pair_indices(FakeV1, rows) == []


@_isolated_indexed_test
def test_candidate_budget_fails_closed():
    rows = [_fp(str(index), origin="same") for index in range(5)]
    with pytest.raises(indexed.IndexedExecutionError, match="candidate pair budget exceeded"):
        indexed.candidate_pair_indices(FakeV1, rows, max_candidate_pairs=2)


@_isolated_indexed_test
def test_index_posting_budget_fails_before_unbounded_growth():
    rows = [_fp("a", shingles=frozenset({"s1", "s2", "s3"}))]
    with pytest.raises(indexed.IndexedExecutionError, match="index posting work budget exceeded"):
        indexed.candidate_pair_indices(FakeV1, rows, max_index_postings=5)


@_isolated_indexed_test
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


@_isolated_indexed_test
def test_high_frequency_single_shingle_noise_is_pruned_before_quadratic_pairs():
    rows = []
    for index in range(100):
        shingles = frozenset(
            {"shared-noise", *(f"unique-{index}-{offset}" for offset in range(20))}
        )
        rows.append(_fp(str(index), shingles=shingles))
    pairs, stats = indexed.candidate_pair_indices_with_stats(FakeV1, rows)
    assert pairs == []
    assert stats["unique_candidate_pairs"] == 0


@_isolated_indexed_test
def test_threshold_boundary_content_and_skeleton_pairs_are_retained():
    natural_large = frozenset(f"natural-{index}" for index in range(5))
    natural_subset = frozenset(f"natural-{index}" for index in range(4))
    code_large = frozenset(f"code-{index}" for index in range(100))
    code_subset = frozenset(f"code-{index}" for index in range(86))
    skeleton_large = frozenset(f"skeleton-{index}" for index in range(50))
    skeleton_subset = frozenset(f"skeleton-{index}" for index in range(41))
    rows = [
        _fp("natural-a", shingles=natural_large),
        _fp("natural-b", shingles=natural_subset),
        _fp("code-a", modality="code", shingles=code_large, skeleton=skeleton_large),
        _fp("code-b", modality="code", shingles=code_subset, skeleton=skeleton_subset),
    ]
    pairs = set(indexed.candidate_pair_indices(FakeV1, rows))
    assert (0, 1) in pairs
    assert (2, 3) in pairs


@_isolated_indexed_test
def test_fragment_containment_candidate_survives_low_jaccard():
    large = frozenset(f"fragment-{index}" for index in range(100))
    contained = frozenset(f"fragment-{index}" for index in range(20))
    pairs = set(
        indexed.candidate_pair_indices(
            FakeV1,
            [_fp("large", shingles=large), _fp("contained", shingles=contained)],
        )
    )
    assert (0, 1) in pairs


@_isolated_indexed_test
def test_small_set_exhaustion_retains_every_natural_containment_candidate():
    universe = tuple(f"token-{index}" for index in range(5))
    subsets = [
        frozenset(
            universe[index]
            for index in range(len(universe))
            if mask & (1 << index)
        )
        for mask in range(1, 1 << len(universe))
    ]
    rows = [_fp(str(index), shingles=subset) for index, subset in enumerate(subsets)]
    candidates = set(indexed.candidate_pair_indices(FakeV1, rows))

    for left_index, left in enumerate(subsets):
        for right_index in range(left_index + 1, len(subsets)):
            right = subsets[right_index]
            containment = len(left & right) / min(len(left), len(right))
            if containment >= indexed.EXPECTED_THRESHOLDS["natural_near_jaccard"]:
                assert (left_index, right_index) in candidates


@_isolated_indexed_test
def test_boilerplate_weight_prefix_retains_multi_line_eighty_character_overlap():
    shared_a = "A" * 40
    shared_b = "B" * 45
    rows = [
        _fp("left", text=f"head\n{shared_a}\n{shared_b}\ntail"),
        _fp("right", text=f"other\n{shared_a}\n{shared_b}\nend"),
    ]
    pairs = set(indexed.candidate_pair_indices(FakeV1, rows))
    assert (0, 1) in pairs


@_isolated_indexed_test
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


@_isolated_indexed_test
def test_execution_stats_exact_rada_scale_and_work_telemetry():
    rada = indexed.execution_stats(101_559, 0, index_postings=123, pair_expansion_attempts=45)
    assert rada["incumbent_all_pair_dispatches"] == 5_157_064_461
    assert rada["index_postings"] == 123
    assert rada["pair_expansion_attempts"] == 45
    combined = indexed.execution_stats(101_821, 0)
    assert combined["incumbent_all_pair_dispatches"] == 5_183_707_110


@_isolated_indexed_test
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

