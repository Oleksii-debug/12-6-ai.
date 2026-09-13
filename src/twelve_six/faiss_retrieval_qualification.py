from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
import tempfile
from pathlib import Path
from typing import Any

EXPECTED_PROJECT_REPO = "Oleksii-debug/12-6-ai."
EXPECTED_CONTROL_ISSUE = 723
EXPECTED_PARENT_ISSUE = 720
EXPECTED_WORKER_ISSUE = 751
EXPECTED_BASE_SHA = "5020afd671a3885c1b738c8b4eafe7525f630546"
EXPECTED_REGISTRY_PATH = "configs/research/open_source_reuse_registry_v2.json"
EXPECTED_REGISTRY_BLOB_SHA = "d80a60357c56eacac135f948b8a72556bb849e5a"
EXPECTED_UPSTREAM_REPO = "https://github.com/facebookresearch/faiss"
EXPECTED_UPSTREAM_TAG = "v1.15.0"
EXPECTED_UPSTREAM_COMMIT = "20f14b31a6d54e243a3d1de6ae193fc4c3ec18ed"
EXPECTED_UPSTREAM_LICENSE = "MIT"
EXPECTED_PACKAGE = "faiss-cpu"
EXPECTED_PACKAGE_VERSION = "1.15.0"
EXPECTED_WHEEL_SHA256 = "ec9b29aae29e428c085c2d49dbb02e4673cdea75db418d420f9e60e0b4184498"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class QualificationError(ValueError):
    """Raised when a retrieval qualification contract fails closed."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise QualificationError(message)


def _require_hex(value: Any, regex: re.Pattern[str], field: str) -> None:
    _require(isinstance(value, str) and regex.fullmatch(value) is not None, f"invalid {field}")


def _validate_vector(vector: Any, dimension: int, field: str) -> list[float]:
    _require(isinstance(vector, list) and len(vector) == dimension, f"{field} dimension mismatch")
    converted: list[float] = []
    for item in vector:
        _require(isinstance(item, (int, float)) and not isinstance(item, bool), f"{field} non-numeric")
        number = float(item)
        _require(math.isfinite(number), f"{field} must be finite")
        converted.append(number)
    return converted


def validate_contract(contract: dict[str, Any]) -> None:
    _require(contract.get("schema_version") == 1, "unsupported schema_version")
    _require(contract.get("qualification_id") == "FAISS-RETRIEVAL-QUALIFICATION-V1", "bad qualification_id")

    authority = contract.get("authority")
    _require(isinstance(authority, dict), "authority must be an object")
    _require(authority.get("repository") == EXPECTED_PROJECT_REPO, "project repository drift")
    _require(authority.get("swarm_control_issue") == EXPECTED_CONTROL_ISSUE, "swarm control drift")
    _require(authority.get("parent_issue") == EXPECTED_PARENT_ISSUE, "parent issue drift")
    _require(authority.get("worker_issue") == EXPECTED_WORKER_ISSUE, "worker issue drift")
    _require_hex(authority.get("base_git_sha"), HEX40, "base_git_sha")
    _require(authority.get("base_git_sha") == EXPECTED_BASE_SHA, "base Git SHA drift")
    _require(authority.get("registry_path") == EXPECTED_REGISTRY_PATH, "registry path drift")
    _require(authority.get("registry_git_blob_sha") == EXPECTED_REGISTRY_BLOB_SHA, "registry blob drift")

    upstream = contract.get("upstream")
    _require(isinstance(upstream, dict), "upstream must be an object")
    _require(upstream.get("repository") == EXPECTED_UPSTREAM_REPO, "FAISS repository drift")
    _require(upstream.get("release_tag") == EXPECTED_UPSTREAM_TAG, "FAISS release drift")
    _require(upstream.get("release_commit") == EXPECTED_UPSTREAM_COMMIT, "FAISS release commit drift")
    _require(upstream.get("license_spdx") == EXPECTED_UPSTREAM_LICENSE, "FAISS license drift")
    _require(upstream.get("registry_decision") == "P1_OPTIONAL_LOCAL_DENSE_RETRIEVAL", "registry decision drift")

    package = contract.get("package")
    _require(isinstance(package, dict), "package must be an object")
    _require(package.get("distribution") == EXPECTED_PACKAGE, "FAISS package name drift")
    _require(package.get("version") == EXPECTED_PACKAGE_VERSION, "FAISS package version drift")
    _require(package.get("python_requires") == ">=3.10", "FAISS Python requirement drift")
    _require(package.get("license_spdx") == EXPECTED_UPSTREAM_LICENSE, "FAISS package license drift")
    _require_hex(package.get("qualified_linux_x86_64_wheel_sha256"), HEX64, "qualified wheel sha256")
    _require(
        package.get("qualified_linux_x86_64_wheel_sha256") == EXPECTED_WHEEL_SHA256,
        "qualified FAISS wheel identity drift",
    )

    source = contract.get("vector_source")
    _require(isinstance(source, dict), "vector_source must be an object")
    _require(source.get("kind") in {"PROJECT_OWNED_NUMERIC_FIXTURE", "EXPLICIT_EXTERNAL_VECTORS"}, "unsupported vector source")
    _require(source.get("foreign_pretrained_model") is False, "foreign pretrained embeddings are not authorized")
    for forbidden in ("embedding_model", "pretrained_model", "teacher_model", "teacher_logits"):
        _require(not source.get(forbidden), f"hidden model provenance field is forbidden: {forbidden}")
    _require(isinstance(source.get("identity"), str) and source["identity"], "missing vector-source identity")
    if source["kind"] == "EXPLICIT_EXTERNAL_VECTORS":
        _require(isinstance(source.get("producer_authority"), str) and source["producer_authority"], "external vectors require producer authority")

    index = contract.get("index")
    _require(isinstance(index, dict), "index must be an object")
    dimension = index.get("dimension")
    _require(isinstance(dimension, int) and not isinstance(dimension, bool) and dimension > 0, "invalid dimension")
    _require(index.get("dtype") == "float32", "only float32 is qualified")
    _require(index.get("metric") in {"L2", "IP"}, "unsupported metric")
    _require(index.get("index_family") == "FLAT_EXACT", "only exact Flat indexes are qualified in V1")
    _require(index.get("deterministic_tie_break") == "record_id_ascending", "tie policy drift")

    records = contract.get("records")
    _require(isinstance(records, list) and records, "records must be non-empty")
    seen_ids: set[str] = set()
    for pos, record in enumerate(records):
        _require(isinstance(record, dict), f"record {pos} must be an object")
        record_id = record.get("id")
        _require(isinstance(record_id, str) and record_id, f"record {pos} missing id")
        _require(record_id not in seen_ids, f"duplicate record id: {record_id}")
        seen_ids.add(record_id)
        _validate_vector(record.get("vector"), dimension, f"record {record_id}")

    queries = contract.get("queries")
    _require(isinstance(queries, list) and queries, "queries must be non-empty")
    query_ids: set[str] = set()
    for pos, query in enumerate(queries):
        _require(isinstance(query, dict), f"query {pos} must be an object")
        query_id = query.get("id")
        _require(isinstance(query_id, str) and query_id, f"query {pos} missing id")
        _require(query_id not in query_ids, f"duplicate query id: {query_id}")
        query_ids.add(query_id)
        _validate_vector(query.get("vector"), dimension, f"query {query_id}")
        top_k = query.get("top_k")
        _require(isinstance(top_k, int) and not isinstance(top_k, bool) and 1 <= top_k <= len(records), f"invalid top_k for {query_id}")
        expected = query.get("expected_record_ids")
        _require(isinstance(expected, list) and len(expected) == top_k, f"missing expected results for {query_id}")
        _require(len(set(expected)) == len(expected) and all(item in seen_ids for item in expected), f"invalid expected result ids for {query_id}")

    persistence = contract.get("persistence")
    _require(isinstance(persistence, dict), "persistence must be an object")
    _require(persistence.get("trust_policy") == "HASH_BOUND_TRUSTED_LOCAL_ONLY", "unsafe index load policy")
    _require(persistence.get("allow_untrusted_load") is False, "untrusted FAISS loading is forbidden")

    execution = contract.get("backend_execution")
    _require(isinstance(execution, dict), "backend_execution must be an object")
    status = execution.get("status")
    _require(status in {"NOT_EXECUTED_DEPENDENCY_ABSENT", "EXECUTED_PASS", "EXECUTED_FAIL"}, "invalid backend status")
    if status == "EXECUTED_PASS":
        _require(execution.get("import_version") == "1.15.0", "executed FAISS version is not the qualified release")
        _require_hex(persistence.get("index_sha256"), HEX64, "persistence.index_sha256")
        _require(execution.get("reference_parity") is True, "executed backend lacks reference parity")
    else:
        _require(persistence.get("index_sha256") is None, "unexecuted/failed backend may not claim persisted index identity")
        _require(execution.get("reference_parity") is False, "unexecuted/failed backend may not claim parity")

    policy = contract.get("policy")
    _require(isinstance(policy, dict), "policy must be an object")
    _require(policy.get("canonical_base_dependency") is False, "FAISS may not become a canonical Base dependency")
    _require(policy.get("foreign_embedding_required") is False, "foreign embedding model may not be required")
    _require(policy.get("training_authorized") is False, "retrieval qualification is not training authority")
    _require(policy.get("stage_promotion_authorized") is False, "retrieval qualification is not stage-promotion authority")
    _require(policy.get("benchmark_or_final_test_payload") is False, "benchmark/final-test material is forbidden")
    requested = policy.get("requested_promotion_state")
    _require(requested in {"DISCOVERED", "CANDIDATE", "PARITY_PROVEN"}, "ADOPTED/self-promotion is forbidden")
    if requested == "PARITY_PROVEN":
        _require(status == "EXECUTED_PASS", "PARITY_PROVEN requires actual backend execution")


def _score(a: list[float], b: list[float], metric: str) -> float:
    if metric == "L2":
        return sum((x - y) ** 2 for x, y in zip(a, b, strict=True))
    return sum(x * y for x, y in zip(a, b, strict=True))


def brute_force_search(contract: dict[str, Any], query_vector: list[float], top_k: int) -> list[dict[str, Any]]:
    validate_contract(contract)
    dimension = contract["index"]["dimension"]
    vector = _validate_vector(query_vector, dimension, "ad-hoc query")
    metric = contract["index"]["metric"]
    rows = [
        {"id": record["id"], "score": _score(vector, [float(x) for x in record["vector"]], metric)}
        for record in contract["records"]
    ]
    if metric == "L2":
        rows.sort(key=lambda row: (row["score"], row["id"]))
    else:
        rows.sort(key=lambda row: (-row["score"], row["id"]))
    return rows[:top_k]


def build_evidence(contract: dict[str, Any]) -> dict[str, Any]:
    validate_contract(contract)
    reference: dict[str, Any] = {}
    for query in contract["queries"]:
        rows = brute_force_search(contract, query["vector"], query["top_k"])
        ids = [row["id"] for row in rows]
        _require(ids == query["expected_record_ids"], f"reference result drift for {query['id']}")
        reference[query["id"]] = rows

    fixture_identity = {
        "vector_source": contract["vector_source"],
        "index": contract["index"],
        "records": contract["records"],
        "queries": contract["queries"],
    }
    executed = contract["backend_execution"]["status"] == "EXECUTED_PASS"
    return {
        "schema_version": 1,
        "qualification_id": contract["qualification_id"],
        "worker_issue": EXPECTED_WORKER_ISSUE,
        "contract_sha256": sha256_json(contract),
        "fixture_sha256": sha256_json(fixture_identity),
        "reference_results": reference,
        "backend_execution": contract["backend_execution"],
        "persistence": contract["persistence"],
        "parity_proven": executed and contract["backend_execution"]["reference_parity"] is True,
        "promotion_state": "PARITY_PROVEN" if executed else "CANDIDATE",
        "canonical_base_changed": False,
        "training_authorized": False,
    }


def probe_faiss(contract: dict[str, Any]) -> dict[str, Any]:
    """Run a trusted local Flat-index probe when faiss and numpy are installed.

    The function never downloads a model or data. It only indexes the project-owned numeric fixture.
    """
    validate_contract(contract)
    try:
        faiss = importlib.import_module("faiss")
        np = importlib.import_module("numpy")
    except ModuleNotFoundError as exc:
        return {"status": "NOT_EXECUTED_DEPENDENCY_ABSENT", "missing": exc.name}

    version = str(getattr(faiss, "__version__", "UNKNOWN"))
    if version != "1.15.0":
        return {"status": "EXECUTED_FAIL", "reason": "FAISS_VERSION_DRIFT", "import_version": version}

    vectors = np.asarray([record["vector"] for record in contract["records"]], dtype="float32")
    ids = np.arange(len(contract["records"]), dtype="int64")
    metric = contract["index"]["metric"]
    base_index = faiss.IndexFlatL2(contract["index"]["dimension"]) if metric == "L2" else faiss.IndexFlatIP(contract["index"]["dimension"])
    index = faiss.IndexIDMap2(base_index)
    index.add_with_ids(vectors, ids)

    with tempfile.TemporaryDirectory(prefix="twelve-six-faiss-") as tmpdir:
        path = Path(tmpdir) / "fixture.faiss"
        faiss.write_index(index, str(path))
        payload = path.read_bytes()
        index_sha256 = hashlib.sha256(payload).hexdigest()
        # The only read performed is the just-written, hash-bound local fixture.
        if hashlib.sha256(path.read_bytes()).hexdigest() != index_sha256:
            return {"status": "EXECUTED_FAIL", "reason": "PERSISTENCE_HASH_DRIFT", "import_version": version}
        loaded = faiss.read_index(str(path))
        results: dict[str, list[str]] = {}
        for query in contract["queries"]:
            query_array = np.asarray([query["vector"]], dtype="float32")
            _, found = loaded.search(query_array, query["top_k"])
            names = [contract["records"][int(i)]["id"] for i in found[0]]
            reference = [row["id"] for row in brute_force_search(contract, query["vector"], query["top_k"])]
            if names != reference:
                return {
                    "status": "EXECUTED_FAIL",
                    "reason": "REFERENCE_PARITY_MISMATCH",
                    "import_version": version,
                    "query_id": query["id"],
                    "faiss": names,
                    "reference": reference,
                }
            results[query["id"]] = names

    return {
        "status": "EXECUTED_PASS",
        "import_version": version,
        "reference_parity": True,
        "index_sha256": index_sha256,
        "results": results,
    }
