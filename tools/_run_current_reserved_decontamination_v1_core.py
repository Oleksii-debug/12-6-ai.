"""Execute current reserved-evaluation decontamination with exact code binding.

Payload-bearing inputs remain ephemeral. Durable publication is one atomic directory
containing only the canonical hash-only report/evidence plus a code- and input-bound
receipt. This runner never grants corpus, tokenizer, exposure, or training authority.
"""
from __future__ import annotations

import argparse
import ast
import ctypes
import errno
import hashlib
import json
import marshal
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import types
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import twelve_six.data._data232_decontamination_matching as matching_impl
import twelve_six.data.current_reserved_decontamination_v1 as current_impl
import twelve_six.data.decontamination_authority_v2 as authority_impl
from twelve_six.data.current_reserved_decontamination_v1 import (
    execute_reserved_decontamination,
    verify_execution_evidence,
)

RECEIPT_SCHEMA = "12-6.current-reserved-decontamination-run-receipt.v1"
REPORT_NAME = "decontamination_report.json"
EVIDENCE_NAME = "execution_evidence.json"
RECEIPT_NAME = "execution_receipt.json"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SAFE_AUTHORITY_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$")
_EXPECTED_DATA232_SCHEMA = "12-6.data232-decontamination-report.v2"
_EXPECTED_EXECUTION_SCHEMA = "12-6.current-reserved-decontamination-execution.v1"
_EXPECTED_ALGORITHM = "data232-deterministic-overlap-cluster-v2"
_EXPECTED_NORMALIZATION = "data232-contamination-normalization-v2"
_EXPECTED_CODE_SKELETON = "data232-code-skeleton-v1"
_EXPECTED_THRESHOLDS = {
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
_EXPECTED_OUTCOME_PARTS = (
    "accuracy",
    "bpb",
    "loss",
    "margin",
    "metric",
    "outcome",
    "perplexity",
    "result",
    "score",
)
_EXPECTED_INVISIBLE = dict.fromkeys(
    map(ord, "\ufeff\u00ad\u200b\u200c\u200d\u2060"),
    None,
)
_EXPECTED_KEYWORDS = {
    "and",
    "as",
    "async",
    "await",
    "break",
    "case",
    "catch",
    "class",
    "const",
    "continue",
    "def",
    "delete",
    "do",
    "else",
    "elif",
    "except",
    "export",
    "false",
    "finally",
    "for",
    "from",
    "function",
    "if",
    "import",
    "in",
    "instanceof",
    "interface",
    "lambda",
    "let",
    "match",
    "new",
    "none",
    "not",
    "null",
    "or",
    "package",
    "pass",
    "raise",
    "return",
    "select",
    "static",
    "struct",
    "switch",
    "throw",
    "true",
    "try",
    "type",
    "var",
    "when",
    "where",
    "while",
    "with",
    "yield",
}
_EXPECTED_REGEX_BINDINGS = {
    "TOKEN_RE": (r"\w+|[^\w\s]", re.UNICODE),
    "CODE_TOKEN_RE": (
        r"(?:[A-Za-z_][A-Za-z0-9_]*)|(?:0[xX][0-9A-Fa-f]+|\d+(?:\.\d+)?)|"
        r"(?:==|!=|<=|>=|->|=>|::|\+\+|--|&&|\|\||<<|>>|\*\*)|(?:[^\s])",
        0,
    ),
    "LINE_COMMENT": (r"(?m)(?://|#).*$", 0),
    "BLOCK_COMMENT": (r"(?s)/\*.*?\*/", 0),
    "STRING": (
        r"(?s)(?:'''(?:\\.|[^\\])*?'''|\"\"\"(?:\\.|[^\\])*?\"\"\"|"
        r"'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\")",
        0,
    ),
}
_RECEIPT_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "decontamination_implementation_git_sha",
        "input_files_sha256",
        "output_files_sha256",
        "status",
        "training_corpus_identity_sha256",
        "input_survivor_authority_sha256",
        "training_handoff_identity_sha256",
        "reserved_payload_binding_identity_sha256",
        "selection_validation_identity_sha256",
        "final_test_identity_sha256",
        "decontamination_report_sha256",
        "execution_identity_sha256",
        "final_test_payload_accessed_for_decontamination",
        "final_test_outcomes_read",
        "durable_bundle_hash_only",
        "raw_text_persisted_in_bundle",
        "record_ids_persisted_in_bundle",
        "authorized_training_exposure",
        "tokenizer_fit_authorized",
        "training_executed",
        "paid_compute_used",
        "receipt_identity_sha256",
    }
)
_REPORT_TOP_LEVEL_KEYS = frozenset(
    {
        "schema",
        "worker_id",
        "status",
        "training_corpus_identity",
        "selection_validation_identity",
        "final_test_identity",
        "all_reserved_authorities_identity",
        "matching",
        "counts",
        "quarantined_source_family_sha256",
        "excluded_records",
        "contaminated_clusters",
        "match_evidence",
        "evaluation_authorities",
        "hash_only_evidence",
        "final_test_outcomes_read",
        "model_architecture_or_hyperparameters_selected",
        "training_executed",
        "local_free_only",
        "report_sha256",
    }
)
_EXECUTION_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "training_corpus_identity_sha256",
        "input_survivor_authority_sha256",
        "training_handoff_identity_sha256",
        "reserved_payload_binding_identity_sha256",
        "selection_validation_identity_sha256",
        "final_test_identity_sha256",
        "decontamination_report_sha256",
        "counts",
        "final_test_payload_accessed_for_decontamination",
        "final_test_outcomes_read",
        "durable_evidence_hash_only",
        "model_architecture_or_hyperparameters_selected",
        "authorized_training_exposure",
        "tokenizer_fit_authorized",
        "training_executed",
        "paid_compute_used",
        "execution_identity_sha256",
    }
)
_MATCHING_KEYS = frozenset(
    {
        "algorithm",
        "normalization",
        "code_skeleton",
        "thresholds",
        "cluster_policy",
        "family_policy",
    }
)
_COUNT_KEYS = frozenset(
    {
        "training_records",
        "evaluation_records",
        "excluded_training_records",
        "quarantined_source_families",
        "match_evidence_records",
    }
)
_EXCLUDED_RECORD_KEYS = frozenset(
    {
        "record_id_sha256",
        "source_id_sha256",
        "source_family_sha256",
        "lineage_family_sha256",
        "modality",
        "raw_sha256",
        "normalized_sha256",
        "reason",
    }
)
_CLUSTER_KEYS = frozenset(
    {
        "cluster_id_sha256",
        "training_record_id_sha256",
        "evaluation_record_id_sha256",
    }
)
_EVAL_MATCH_KEYS = frozenset(
    {
        "match_type",
        "score",
        "train_raw_sha256",
        "eval_raw_sha256",
        "train_normalized_sha256",
        "eval_normalized_sha256",
        "cross_source_family",
        "train_record_id_sha256",
        "eval_record_id_sha256",
    }
)
_PEER_MATCH_KEYS = frozenset(
    {
        "match_type",
        "score",
        "train_raw_sha256",
        "peer_raw_sha256",
        "train_normalized_sha256",
        "peer_normalized_sha256",
        "cross_source_family",
        "match_scope",
        "train_record_id_sha256",
        "peer_record_id_sha256",
    }
)
_AUTHORITY_KEYS = frozenset(
    {"authority_id", "identity_sha256", "role", "source_sha"}
)
_FORBIDDEN_DURABLE_KEYS = frozenset(
    {
        "text",
        "raw_text",
        "source_text",
        "content",
        "prefix",
        "continuation",
        "canary_text",
        "record_id",
        "training_record_id",
        "evaluation_record_id",
        "eval_record_id",
        "peer_record_id",
    }
)
_IMPLEMENTATION_RELATIVE_PATHS = (
    "tools/run_current_reserved_decontamination_v1.py",
    "src/twelve_six/data/current_reserved_decontamination_v1.py",
    "src/twelve_six/data/decontamination_authority_v2.py",
    "src/twelve_six/data/_data232_decontamination_matching.py",
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _file_bytes(value: object) -> bytes:
    return _canonical_bytes(value) + b"\n"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_json_with_sha(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    digest = _sha256_bytes(raw)
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value, digest


def _load_jsonl_with_sha(path: Path) -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            digest.update(raw)
            if not raw.strip():
                raise ValueError(
                    f"blank JSONL line {line_number} is forbidden: {path}"
                )
            value = json.loads(raw.decode("utf-8"))
            if not isinstance(value, dict):
                raise TypeError(
                    f"JSONL line {line_number} must be an object: {path}"
                )
            rows.append(value)
    if not rows:
        raise ValueError(f"JSONL input must be non-empty: {path}")
    return rows, digest.hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be lowercase 64-hex SHA-256")
    return value


def _require_git_sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _GIT_SHA_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be lowercase 40-hex Git SHA")
    return value


def _require_exact_keys(
    value: object,
    expected: frozenset[str],
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{name} key set drift")
    return value


def _require_nonnegative_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _require_sha256_list(value: object, name: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return [
        _require_sha256(item, f"{name}[{index}]")
        for index, item in enumerate(value)
    ]


def _reject_forbidden_durable_keys(value: object, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if key_text.lower() in _FORBIDDEN_DURABLE_KEYS:
                raise ValueError(
                    f"durable bundle contains forbidden raw/unhashed field: "
                    f"{path}.{key_text}"
                )
            _reject_forbidden_durable_keys(child, f"{path}.{key_text}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden_durable_keys(child, f"{path}[{index}]")


def _verify_report_durable_schema(report: Mapping[str, Any]) -> None:
    authority_impl.verify_report(report)
    _reject_forbidden_durable_keys(report, "$.report")
    _require_exact_keys(report, _REPORT_TOP_LEVEL_KEYS, "durable report")
    if report.get("schema") != _EXPECTED_DATA232_SCHEMA:
        raise ValueError("durable report schema drift")
    if report.get("worker_id") != "DATA-232-DECONTAMINATION-AUTHORITY-V2":
        raise ValueError("durable report worker identity drift")
    if report.get("status") not in {"PASS_CLEAN", "PASS_WITH_EXCLUSIONS"}:
        raise ValueError("durable report status is not a terminal pass")
    for key in (
        "training_corpus_identity",
        "selection_validation_identity",
        "final_test_identity",
        "all_reserved_authorities_identity",
        "report_sha256",
    ):
        _require_sha256(report.get(key), f"report.{key}")

    matching = _require_exact_keys(
        report.get("matching"),
        _MATCHING_KEYS,
        "durable report matching",
    )
    if matching.get("algorithm") != _EXPECTED_ALGORITHM:
        raise ValueError("durable report matching algorithm drift")
    if matching.get("normalization") != _EXPECTED_NORMALIZATION:
        raise ValueError("durable report normalization drift")
    if matching.get("code_skeleton") != _EXPECTED_CODE_SKELETON:
        raise ValueError("durable report code-skeleton drift")
    thresholds = _require_exact_keys(
        matching.get("thresholds"),
        frozenset(_EXPECTED_THRESHOLDS),
        "durable report thresholds",
    )
    if dict(thresholds) != _EXPECTED_THRESHOLDS:
        raise ValueError("durable report threshold values drift")
    if (
        matching.get("cluster_policy")
        != "exclude every training node in an evaluation-connected component"
    ):
        raise ValueError("durable report cluster policy drift")
    if (
        matching.get("family_policy")
        != "quarantine contaminated cross-source training family"
    ):
        raise ValueError("durable report family policy drift")

    counts = _require_exact_keys(
        report.get("counts"),
        _COUNT_KEYS,
        "durable report counts",
    )
    for key in _COUNT_KEYS:
        _require_nonnegative_int(counts.get(key), f"report.counts.{key}")

    _require_sha256_list(
        report.get("quarantined_source_family_sha256"),
        "report.quarantined_source_family_sha256",
    )
    excluded = report.get("excluded_records")
    if not isinstance(excluded, list):
        raise ValueError("report.excluded_records must be a list")
    for index, row in enumerate(excluded):
        item = _require_exact_keys(
            row,
            _EXCLUDED_RECORD_KEYS,
            f"report.excluded_records[{index}]",
        )
        for key in (
            "record_id_sha256",
            "source_id_sha256",
            "source_family_sha256",
            "raw_sha256",
            "normalized_sha256",
        ):
            _require_sha256(
                item.get(key),
                f"report.excluded_records[{index}].{key}",
            )
        lineage = item.get("lineage_family_sha256")
        if lineage is not None:
            _require_sha256(
                lineage,
                f"report.excluded_records[{index}].lineage_family_sha256",
            )
        if item.get("modality") not in {"uk", "ua", "en", "code", "text"}:
            raise ValueError("durable excluded-record modality drift")
        if item.get("reason") not in {
            "source_family_quarantine",
            "evaluation_connected_component",
        }:
            raise ValueError("durable excluded-record reason drift")

    clusters = report.get("contaminated_clusters")
    if not isinstance(clusters, list):
        raise ValueError("report.contaminated_clusters must be a list")
    for index, row in enumerate(clusters):
        item = _require_exact_keys(
            row,
            _CLUSTER_KEYS,
            f"report.contaminated_clusters[{index}]",
        )
        _require_sha256(
            item.get("cluster_id_sha256"),
            f"report.contaminated_clusters[{index}].cluster_id_sha256",
        )
        _require_sha256_list(
            item.get("training_record_id_sha256"),
            f"report.contaminated_clusters[{index}].training_record_id_sha256",
        )
        _require_sha256_list(
            item.get("evaluation_record_id_sha256"),
            f"report.contaminated_clusters[{index}].evaluation_record_id_sha256",
        )

    match_evidence = report.get("match_evidence")
    if not isinstance(match_evidence, list):
        raise ValueError("report.match_evidence must be a list")
    for index, row in enumerate(match_evidence):
        if not isinstance(row, Mapping):
            raise ValueError(f"report.match_evidence[{index}] must be an object")
        if "eval_record_id_sha256" in row:
            item = _require_exact_keys(
                row,
                _EVAL_MATCH_KEYS,
                f"report.match_evidence[{index}]",
            )
            hash_keys = (
                "train_raw_sha256",
                "eval_raw_sha256",
                "train_normalized_sha256",
                "eval_normalized_sha256",
                "train_record_id_sha256",
                "eval_record_id_sha256",
            )
        elif "peer_record_id_sha256" in row:
            item = _require_exact_keys(
                row,
                _PEER_MATCH_KEYS,
                f"report.match_evidence[{index}]",
            )
            hash_keys = (
                "train_raw_sha256",
                "peer_raw_sha256",
                "train_normalized_sha256",
                "peer_normalized_sha256",
                "train_record_id_sha256",
                "peer_record_id_sha256",
            )
            if item.get("match_scope") not in {
                "training_cross_source_mirror",
                "training_within_family",
            }:
                raise ValueError("durable match-evidence scope drift")
        else:
            raise ValueError("durable match evidence lacks hashed counterpart identity")
        if item.get("match_type") not in {
            "raw_exact",
            "normalized_exact",
            "near_match",
            "document_fragment",
            "code_fork_copy",
        }:
            raise ValueError("durable match-evidence type drift")
        score = item.get("score")
        if (
            not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not math.isfinite(float(score))
            or not 0.0 <= float(score) <= 1.0
        ):
            raise ValueError("durable match-evidence score drift")
        if not isinstance(item.get("cross_source_family"), bool):
            raise ValueError("durable match-evidence family flag drift")
        for key in hash_keys:
            _require_sha256(item.get(key), f"report.match_evidence[{index}].{key}")

    authorities = report.get("evaluation_authorities")
    if not isinstance(authorities, list) or not authorities:
        raise ValueError("report.evaluation_authorities must be a non-empty list")
    for index, row in enumerate(authorities):
        item = _require_exact_keys(
            row,
            _AUTHORITY_KEYS,
            f"report.evaluation_authorities[{index}]",
        )
        authority_id = item.get("authority_id")
        if (
            not isinstance(authority_id, str)
            or _SAFE_AUTHORITY_ID_RE.fullmatch(authority_id) is None
        ):
            raise ValueError("durable evaluation authority_id is not a safe identifier")
        _require_sha256(
            item.get("identity_sha256"),
            f"report.evaluation_authorities[{index}].identity_sha256",
        )
        if item.get("role") not in {
            "selection_validation",
            "final_test",
            "auxiliary_reserved",
        }:
            raise ValueError("durable evaluation authority role drift")
        _require_git_sha(
            item.get("source_sha"),
            f"report.evaluation_authorities[{index}].source_sha",
        )

    if report.get("hash_only_evidence") is not True:
        raise ValueError("durable report is not hash-only")
    for key in (
        "final_test_outcomes_read",
        "model_architecture_or_hyperparameters_selected",
        "training_executed",
    ):
        if report.get(key) is not False:
            raise ValueError(f"durable report truth boundary widened: {key}")
    if report.get("local_free_only") is not True:
        raise ValueError("durable report left LOCAL_FREE boundary")


def _verify_execution_durable_schema(
    evidence: Mapping[str, Any],
    report: Mapping[str, Any],
) -> None:
    _reject_forbidden_durable_keys(evidence, "$.evidence")
    _require_exact_keys(
        evidence,
        _EXECUTION_TOP_LEVEL_KEYS,
        "durable execution evidence",
    )
    if evidence.get("schema_version") != _EXPECTED_EXECUTION_SCHEMA:
        raise ValueError("durable execution schema drift")
    if evidence.get("status") != report.get("status"):
        raise ValueError("durable execution/report status mismatch")
    for key in (
        "training_corpus_identity_sha256",
        "input_survivor_authority_sha256",
        "training_handoff_identity_sha256",
        "reserved_payload_binding_identity_sha256",
        "selection_validation_identity_sha256",
        "final_test_identity_sha256",
        "decontamination_report_sha256",
        "execution_identity_sha256",
    ):
        _require_sha256(evidence.get(key), f"evidence.{key}")
    if (
        evidence.get("decontamination_report_sha256")
        != report.get("report_sha256")
    ):
        raise ValueError("durable execution/report identity mismatch")

    counts = _require_exact_keys(
        evidence.get("counts"),
        _COUNT_KEYS,
        "durable execution counts",
    )
    if dict(counts) != dict(report.get("counts", {})):
        raise ValueError("durable execution/report counts mismatch")
    for key in _COUNT_KEYS:
        _require_nonnegative_int(counts.get(key), f"evidence.counts.{key}")

    if evidence.get("final_test_payload_accessed_for_decontamination") is not True:
        raise ValueError("final-test decontamination payload-access truth was erased")
    if evidence.get("durable_evidence_hash_only") is not True:
        raise ValueError("durable execution evidence is not hash-only")
    for key in (
        "final_test_outcomes_read",
        "model_architecture_or_hyperparameters_selected",
        "tokenizer_fit_authorized",
        "training_executed",
        "paid_compute_used",
    ):
        if evidence.get(key) is not False:
            raise ValueError(f"durable execution truth boundary widened: {key}")
    if evidence.get("authorized_training_exposure") != 0 or isinstance(
        evidence.get("authorized_training_exposure"), bool
    ):
        raise ValueError("durable execution training exposure is not exact zero")


def _verify_durable_bundle_contract(
    report: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> None:
    _verify_report_durable_schema(report)
    verify_execution_evidence(evidence, report)
    _verify_execution_durable_schema(evidence, report)


def _git_head(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _tracked_tree_is_clean(repo_root: Path) -> bool:
    unstaged = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--"],
        cwd=repo_root,
        check=False,
    )
    staged = subprocess.run(
        ["git", "diff", "--cached", "--quiet", "HEAD", "--"],
        cwd=repo_root,
        check=False,
    )
    if unstaged.returncode not in (0, 1) or staged.returncode not in (0, 1):
        raise RuntimeError("unable to verify tracked working-tree cleanliness")
    return unstaged.returncode == 0 and staged.returncode == 0


def _normalized_code(code: types.CodeType) -> types.CodeType:
    constants = tuple(
        _normalized_code(item) if isinstance(item, types.CodeType) else item
        for item in code.co_consts
    )
    return code.replace(
        co_filename="<bound-source>",
        co_consts=constants,
    )


def _normalized_code_bytes(code: types.CodeType) -> bytes:
    return marshal.dumps(_normalized_code(code))


def _require_module_function_code(
    module: object,
    source_path: Path,
    module_name: str,
) -> None:
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    function_names = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    compiled = compile(
        source,
        str(source_path),
        "exec",
        dont_inherit=True,
        optimize=sys.flags.optimize,
    )
    expected_code = {
        item.co_name: item
        for item in compiled.co_consts
        if isinstance(item, types.CodeType)
    }
    for name in sorted(function_names):
        expected = expected_code.get(name)
        actual = getattr(module, name, None)
        if expected is None or not isinstance(actual, types.FunctionType):
            raise RuntimeError(
                f"loaded implementation callable set drift: {module_name}.{name}"
            )
        if actual.__module__ != module_name:
            raise RuntimeError(
                f"loaded implementation callable origin drift: {module_name}.{name}"
            )
        if _normalized_code_bytes(actual.__code__) != _normalized_code_bytes(expected):
            raise RuntimeError(
                f"loaded implementation callable code drift: {module_name}.{name}"
            )


def _require_matcher_behavior_globals() -> None:
    if type(matching_impl.DEFAULT_THRESHOLDS) is not dict:
        raise RuntimeError("DATA-232 behavior-global type drift: DEFAULT_THRESHOLDS")
    if dict(matching_impl.DEFAULT_THRESHOLDS) != _EXPECTED_THRESHOLDS:
        raise RuntimeError("DATA-232 behavior-global drift: DEFAULT_THRESHOLDS")
    if type(matching_impl.OUTCOME_PARTS) is not tuple:
        raise RuntimeError("DATA-232 behavior-global type drift: OUTCOME_PARTS")
    if matching_impl.OUTCOME_PARTS != _EXPECTED_OUTCOME_PARTS:
        raise RuntimeError("DATA-232 behavior-global drift: OUTCOME_PARTS")
    if type(matching_impl.INVISIBLE) is not dict:
        raise RuntimeError("DATA-232 behavior-global type drift: INVISIBLE")
    if matching_impl.INVISIBLE != _EXPECTED_INVISIBLE:
        raise RuntimeError("DATA-232 behavior-global drift: INVISIBLE")
    if type(matching_impl.KEYWORDS) is not set:
        raise RuntimeError("DATA-232 behavior-global type drift: KEYWORDS")
    if matching_impl.KEYWORDS != _EXPECTED_KEYWORDS:
        raise RuntimeError("DATA-232 behavior-global drift: KEYWORDS")
    pattern_type = type(re.compile(""))
    for name, (pattern, flags) in _EXPECTED_REGEX_BINDINGS.items():
        actual = getattr(matching_impl, name, None)
        expected = re.compile(pattern, flags)
        if type(actual) is not pattern_type:
            raise RuntimeError(f"DATA-232 behavior-global type drift: {name}")
        if actual.pattern != expected.pattern or actual.flags != expected.flags:
            raise RuntimeError(f"DATA-232 behavior-global drift: {name}")


def _require_behavior_closure(repo_root: Path) -> None:
    runner_module = sys.modules[__name__]
    modules = {
        "tools/run_current_reserved_decontamination_v1.py": (
            runner_module,
            __name__,
        ),
        "src/twelve_six/data/current_reserved_decontamination_v1.py": (
            current_impl,
            current_impl.__name__,
        ),
        "src/twelve_six/data/decontamination_authority_v2.py": (
            authority_impl,
            authority_impl.__name__,
        ),
        "src/twelve_six/data/_data232_decontamination_matching.py": (
            matching_impl,
            matching_impl.__name__,
        ),
    }
    for relative, (module, module_name) in modules.items():
        _require_module_function_code(
            module,
            (repo_root / relative).resolve(),
            module_name,
        )

    _require_matcher_behavior_globals()
    if (
        execute_reserved_decontamination
        is not current_impl.execute_reserved_decontamination
    ):
        raise RuntimeError("runner decontamination callable binding drift")
    if verify_execution_evidence is not current_impl.verify_execution_evidence:
        raise RuntimeError("runner evidence-verifier callable binding drift")
    if current_impl.build_report is not authority_impl.build_report:
        raise RuntimeError("current adapter report-builder binding drift")
    if current_impl.verify_report is not authority_impl.verify_report:
        raise RuntimeError("current adapter report-verifier binding drift")
    for name in (
        "_blocked_pairs",
        "_fingerprint",
        "_pair",
        "_thresholds",
        "_train_pairs",
        "authority_composite_identity",
        "sha256_bytes",
        "stable_identity",
        "validate_authority_metadata",
    ):
        if getattr(authority_impl, name) is not getattr(matching_impl, name):
            raise RuntimeError(f"DATA-232 callable binding drift: {name}")

    expected_constants = {
        "SCHEMA": _EXPECTED_DATA232_SCHEMA,
        "ALGORITHM": _EXPECTED_ALGORITHM,
        "NORMALIZATION": _EXPECTED_NORMALIZATION,
        "CODE_SKELETON": _EXPECTED_CODE_SKELETON,
    }
    for name, expected in expected_constants.items():
        if getattr(matching_impl, name) != expected:
            raise RuntimeError(f"DATA-232 constant drift: {name}")
        if getattr(authority_impl, name) != expected:
            raise RuntimeError(f"DATA-232 imported constant drift: {name}")
    if dict(matching_impl.DEFAULT_THRESHOLDS) != _EXPECTED_THRESHOLDS:
        raise RuntimeError("DATA-232 default-threshold drift")
    if dict(authority_impl.DEFAULT_THRESHOLDS) != _EXPECTED_THRESHOLDS:
        raise RuntimeError("DATA-232 imported default-threshold drift")
    if current_impl.EXECUTION_SCHEMA != _EXPECTED_EXECUTION_SCHEMA:
        raise RuntimeError("current decontamination execution-schema drift")


def _loaded_implementation_sources() -> dict[str, Path]:
    module_paths = {
        "src/twelve_six/data/current_reserved_decontamination_v1.py": (
            current_impl.__file__
        ),
        "src/twelve_six/data/decontamination_authority_v2.py": authority_impl.__file__,
        "src/twelve_six/data/_data232_decontamination_matching.py": (
            matching_impl.__file__
        ),
    }
    sources = {
        "tools/run_current_reserved_decontamination_v1.py": Path(__file__).resolve()
    }
    for relative, module_path in module_paths.items():
        if module_path is None:
            raise RuntimeError(
                f"loaded implementation source has no origin: {relative}"
            )
        sources[relative] = Path(module_path).resolve()
    return sources


def _require_loaded_sources_from_checkout(repo_root: Path) -> None:
    root = repo_root.resolve()
    loaded = _loaded_implementation_sources()
    if set(loaded) != set(_IMPLEMENTATION_RELATIVE_PATHS):
        raise RuntimeError("loaded implementation source set drift")
    for relative in _IMPLEMENTATION_RELATIVE_PATHS:
        checkout_path = root / relative
        expected_path = checkout_path.resolve()
        actual_path = loaded[relative].resolve()
        if actual_path != expected_path:
            raise RuntimeError(
                "loaded implementation source is not from the exact checkout: "
                f"{relative} expected={expected_path} actual={actual_path}"
            )
        if checkout_path.is_symlink() or not expected_path.is_file():
            raise RuntimeError(
                "loaded implementation source is not a regular checkout file: "
                f"{relative}"
            )
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", relative],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        if tracked.returncode != 0:
            raise RuntimeError(
                "loaded implementation source is not tracked at expected head: "
                f"{relative}"
            )
        blob = subprocess.run(
            ["git", "rev-parse", f"HEAD:{relative}"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        if blob.returncode != 0 or not blob.stdout.strip():
            raise RuntimeError(
                "unable to bind implementation source blob at expected head: "
                f"{relative}"
            )
    _require_behavior_closure(root)


def require_exact_checkout(repo_root: Path, expected_git_sha: str) -> str:
    expected = _require_git_sha(expected_git_sha, "expected implementation Git SHA")
    actual = _git_head(repo_root)
    if actual != expected:
        raise RuntimeError(
            "checked-out decontamination implementation Git head differs from "
            f"independent expectation: expected={expected} actual={actual}"
        )
    if not _tracked_tree_is_clean(repo_root):
        raise RuntimeError(
            "tracked working tree differs from the expected implementation head"
        )
    return actual


def require_exact_implementation(repo_root: Path, expected_git_sha: str) -> str:
    actual = require_exact_checkout(repo_root, expected_git_sha)
    _require_loaded_sources_from_checkout(repo_root)
    return actual


def build_run_receipt(
    *,
    implementation_git_sha: str,
    input_file_sha256: Mapping[str, str],
    report: Mapping[str, Any],
    evidence: Mapping[str, Any],
    report_file_bytes: bytes,
    evidence_file_bytes: bytes,
) -> dict[str, Any]:
    _verify_durable_bundle_contract(report, evidence)
    git_sha = _require_git_sha(implementation_git_sha, "implementation_git_sha")
    expected_input_names = {
        "training_records_jsonl",
        "training_handoff_json",
        "evaluation_records_jsonl",
        "reserved_binding_json",
    }
    if set(input_file_sha256) != expected_input_names:
        raise ValueError("receipt input-file hash set drift")
    normalized_inputs = {
        key: _require_sha256(input_file_sha256[key], f"input_file_sha256.{key}")
        for key in sorted(expected_input_names)
    }

    core: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "decontamination_implementation_git_sha": git_sha,
        "input_files_sha256": normalized_inputs,
        "output_files_sha256": {
            REPORT_NAME: _sha256_bytes(report_file_bytes),
            EVIDENCE_NAME: _sha256_bytes(evidence_file_bytes),
        },
        "status": evidence.get("status"),
        "training_corpus_identity_sha256": evidence.get(
            "training_corpus_identity_sha256"
        ),
        "input_survivor_authority_sha256": evidence.get(
            "input_survivor_authority_sha256"
        ),
        "training_handoff_identity_sha256": evidence.get(
            "training_handoff_identity_sha256"
        ),
        "reserved_payload_binding_identity_sha256": evidence.get(
            "reserved_payload_binding_identity_sha256"
        ),
        "selection_validation_identity_sha256": evidence.get(
            "selection_validation_identity_sha256"
        ),
        "final_test_identity_sha256": evidence.get("final_test_identity_sha256"),
        "decontamination_report_sha256": report.get("report_sha256"),
        "execution_identity_sha256": evidence.get("execution_identity_sha256"),
        "final_test_payload_accessed_for_decontamination": evidence.get(
            "final_test_payload_accessed_for_decontamination"
        ),
        "final_test_outcomes_read": False,
        "durable_bundle_hash_only": True,
        "raw_text_persisted_in_bundle": False,
        "record_ids_persisted_in_bundle": False,
        "authorized_training_exposure": 0,
        "tokenizer_fit_authorized": False,
        "training_executed": False,
        "paid_compute_used": False,
    }
    receipt = deepcopy(core)
    receipt["receipt_identity_sha256"] = _sha256_bytes(_canonical_bytes(core))
    return receipt


def verify_run_receipt(
    receipt: Mapping[str, Any],
    report: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> None:
    if set(receipt) != _RECEIPT_TOP_LEVEL_KEYS:
        raise ValueError("run receipt top-level key set drift")
    if receipt.get("schema_version") != RECEIPT_SCHEMA:
        raise ValueError("run receipt schema drift")
    claimed = _require_sha256(
        receipt.get("receipt_identity_sha256"), "receipt_identity_sha256"
    )
    body = deepcopy(dict(receipt))
    body.pop("receipt_identity_sha256", None)
    if _sha256_bytes(_canonical_bytes(body)) != claimed:
        raise ValueError("run receipt self-hash mismatch")
    _require_git_sha(
        receipt.get("decontamination_implementation_git_sha"),
        "decontamination_implementation_git_sha",
    )

    inputs = receipt.get("input_files_sha256")
    if not isinstance(inputs, Mapping) or set(inputs) != {
        "training_records_jsonl",
        "training_handoff_json",
        "evaluation_records_jsonl",
        "reserved_binding_json",
    }:
        raise ValueError("run receipt input-file hash set drift")
    for name, value in inputs.items():
        _require_sha256(value, f"input_files_sha256.{name}")

    outputs = receipt.get("output_files_sha256")
    if not isinstance(outputs, Mapping) or set(outputs) != {
        REPORT_NAME,
        EVIDENCE_NAME,
    }:
        raise ValueError("run receipt output-file hash set drift")
    expected_output_hashes = {
        REPORT_NAME: _sha256_bytes(_file_bytes(report)),
        EVIDENCE_NAME: _sha256_bytes(_file_bytes(evidence)),
    }
    if dict(outputs) != expected_output_hashes:
        raise ValueError("run receipt output-file hash mismatch")

    _verify_durable_bundle_contract(report, evidence)
    bindings = {
        "status": evidence.get("status"),
        "training_corpus_identity_sha256": evidence.get(
            "training_corpus_identity_sha256"
        ),
        "input_survivor_authority_sha256": evidence.get(
            "input_survivor_authority_sha256"
        ),
        "training_handoff_identity_sha256": evidence.get(
            "training_handoff_identity_sha256"
        ),
        "reserved_payload_binding_identity_sha256": evidence.get(
            "reserved_payload_binding_identity_sha256"
        ),
        "selection_validation_identity_sha256": evidence.get(
            "selection_validation_identity_sha256"
        ),
        "final_test_identity_sha256": evidence.get("final_test_identity_sha256"),
        "decontamination_report_sha256": report.get("report_sha256"),
        "execution_identity_sha256": evidence.get("execution_identity_sha256"),
        "final_test_payload_accessed_for_decontamination": True,
    }
    for key, expected in bindings.items():
        if receipt.get(key) != expected:
            raise ValueError(f"run receipt binding drift: {key}")

    required_false = (
        "final_test_outcomes_read",
        "raw_text_persisted_in_bundle",
        "record_ids_persisted_in_bundle",
        "tokenizer_fit_authorized",
        "training_executed",
        "paid_compute_used",
    )
    for key in required_false:
        if receipt.get(key) is not False:
            raise ValueError(f"run receipt truth boundary widened: {key}")
    if receipt.get("durable_bundle_hash_only") is not True:
        raise ValueError("run receipt durable bundle is not hash-only")
    if receipt.get("authorized_training_exposure") != 0 or isinstance(
        receipt.get("authorized_training_exposure"), bool
    ):
        raise ValueError("run receipt authorized training exposure is not exact zero")


def _rename_directory_no_replace(source: Path, destination: Path) -> None:
    if os.name == "nt":
        os.rename(source, destination)
        return

    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
    except AttributeError as exc:
        raise RuntimeError(
            "atomic no-replace directory publication is unsupported on this platform"
        ) from exc
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(
            error_number,
            os.strerror(error_number),
            destination,
        )
    if error_number in (errno.ENOSYS, errno.EINVAL):
        raise RuntimeError(
            "atomic no-replace directory publication is unsupported on this platform"
        )
    raise OSError(error_number, os.strerror(error_number), destination)


def _publish_bundle(output_dir: Path, files: Mapping[str, bytes]) -> None:
    if output_dir.exists():
        raise FileExistsError(f"output bundle already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        for name in (REPORT_NAME, EVIDENCE_NAME, RECEIPT_NAME):
            payload = files[name]
            target = temporary / name
            with target.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        if output_dir.exists():
            raise FileExistsError(
                f"output bundle appeared during publication: {output_dir}"
            )
        _rename_directory_no_replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run current exact-authority reserved-evaluation decontamination and "
            "publish one atomic hash-only evidence bundle"
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--training-records-jsonl", type=Path, required=True)
    parser.add_argument("--training-handoff-json", type=Path, required=True)
    parser.add_argument("--evaluation-records-jsonl", type=Path, required=True)
    parser.add_argument("--reserved-binding-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-inventory-identity-sha256", required=True)
    parser.add_argument("--expected-survivor-authority-sha256", required=True)
    parser.add_argument("--expected-training-handoff-identity-sha256", required=True)
    parser.add_argument("--expected-reserved-binding-identity-sha256", required=True)
    parser.add_argument(
        "--expected-selection-validation-identity-sha256", required=True
    )
    parser.add_argument("--expected-final-test-identity-sha256", required=True)
    parser.add_argument(
        "--expected-decontamination-implementation-git-sha", required=True
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # This gate intentionally precedes all payload reads.
    actual_head = require_exact_implementation(
        args.repo_root, args.expected_decontamination_implementation_git_sha
    )
    if args.output_dir.exists():
        raise FileExistsError(f"output bundle already exists: {args.output_dir}")

    training_records, training_records_sha256 = _load_jsonl_with_sha(
        args.training_records_jsonl
    )
    training_handoff, training_handoff_sha256 = _load_json_with_sha(
        args.training_handoff_json
    )
    evaluation_records, evaluation_records_sha256 = _load_jsonl_with_sha(
        args.evaluation_records_jsonl
    )
    reserved_binding, reserved_binding_sha256 = _load_json_with_sha(
        args.reserved_binding_json
    )
    input_hashes = {
        "training_records_jsonl": training_records_sha256,
        "training_handoff_json": training_handoff_sha256,
        "evaluation_records_jsonl": evaluation_records_sha256,
        "reserved_binding_json": reserved_binding_sha256,
    }

    report, evidence = execute_reserved_decontamination(
        training_records,
        evaluation_records,
        training_handoff_evidence=training_handoff,
        reserved_payload_binding=reserved_binding,
        expected_inventory_identity_sha256=args.expected_inventory_identity_sha256,
        expected_survivor_authority_sha256=args.expected_survivor_authority_sha256,
        expected_training_handoff_identity_sha256=(
            args.expected_training_handoff_identity_sha256
        ),
        expected_reserved_binding_identity_sha256=(
            args.expected_reserved_binding_identity_sha256
        ),
        expected_selection_validation_identity_sha256=(
            args.expected_selection_validation_identity_sha256
        ),
        expected_final_test_identity_sha256=args.expected_final_test_identity_sha256,
        quarantine_cross_source_families=True,
    )
    _verify_durable_bundle_contract(report, evidence)

    report_bytes = _file_bytes(report)
    evidence_bytes = _file_bytes(evidence)
    receipt = build_run_receipt(
        implementation_git_sha=actual_head,
        input_file_sha256=input_hashes,
        report=report,
        evidence=evidence,
        report_file_bytes=report_bytes,
        evidence_file_bytes=evidence_bytes,
    )
    verify_run_receipt(receipt, report, evidence)
    receipt_bytes = _file_bytes(receipt)
    _publish_bundle(
        args.output_dir,
        {
            REPORT_NAME: report_bytes,
            EVIDENCE_NAME: evidence_bytes,
            RECEIPT_NAME: receipt_bytes,
        },
    )
    print(receipt["receipt_identity_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
