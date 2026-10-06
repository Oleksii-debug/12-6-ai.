"""Bind post-G06 current-Rada physical capacity to canonical NEXT100-106 balance.

Execution-only bridge. It authenticates the already-terminal current-Rada global
dedup evidence, the post-G06 materialization receipt, and the family vector;
creates a zero-credit global-unique composition proof; then delegates all
balance mathematics and binding semantics to the merged canonical authorities.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

POST_G06_SCHEMA = "12-6.d03-rada-current-postdata232-g05-g06-execution.v1"
G05_G06_TWO_CLEAN_SCHEMA = (
    "12-6.d03-rada-current-postdata232-g05-g06-two-clean.v1"
)
PARENT_DATA232_HEAD = "a7982bdfd1650b062808024856e13b1d8916a634"
G05_G06_NEXT_GATE = "CURRENT_RADA_BALANCE_DIVERSITY_FAMILY_CAP_RETEST"
G05_G06_BUNDLE_FILES = {
    "evidence": "post-g05-g06-evidence.json",
    "quality": "g05-authority.json",
    "privacy": "g06-authority.json",
    "survivor_inventory": "survivor-inventory.json",
}
FAMILY_VECTOR_SCHEMA = "12-6.d03-postmaterialization-family-vector.v1"
COMPOSITION_SCHEMA = "12-6.d03-rada-post-g06-global-unique-composition.v2"
RECEIPT_SCHEMA = "12-6.d03-rada-postg06-next100-106-execution.v2"
REPEAT_SCHEMA = "12-6.d03-rada-postg06-next100-106-two-clean.v2"
DEDUP_WORKER_ID = "D03-RADA-POST-G06-GLOBAL-UNIQUE-COMPOSITION-V2"
COMPOSITION_SEMANTICS = (
    "UPSTREAM_GLOBAL_DEDUP_PASS_PLUS_POST_G06_EXACT_PAYLOAD_UNIQUENESS_"
    "PLUS_G05_G06_TWO_CLEAN_PLUS_PHYSICAL_FAMILY_VECTOR_NO_REPLAY"
)

STACK_BASE_HEAD = "039ae67cfd0a1393936c49a7bf463f5f68927a03"
UPSTREAM_GLOBAL_DEDUP_HEAD = "a4663e87b010b190343caf1d42784f5dc7984601"
UPSTREAM_GLOBAL_DEDUP_EVIDENCE_ID = (
    "a3f7e396cb13a6b107aaa0eb330edcbdc61bead6fa12eb68542ff5c3a3ed9301"
)
UPSTREAM_GLOBAL_DEDUP_EVIDENCE_FILE_SHA256 = (
    "43a9621ab11fd82232e8251ab26aae4126cf0b4854878af1e97184d587740caa"
)
UPSTREAM_SURVIVOR_AUTHORITY_ID = (
    "f103a3f18216519bd9228e586bd673d49f73cace03b3b0278f2dd0a33383bffb"
)
UPSTREAM_TWO_CLEAN_ID = (
    "f24f4b2d23bee6cb1273a4680297d942aa59030dab50d5c5de7a4d659ff99a5e"
)
UPSTREAM_TWO_CLEAN_FILE_SHA256 = (
    "748adad71a730f7daf18fa52c9e78e3c249ae683e6f3d9cb865b1bbfc9365aa9"
)
UPSTREAM_ARTIFACT_ZIP_SHA256 = (
    "63f9e1bf5713989a155429c8862196cacaff3207a7e0fe97586f7dd2c98881f9"
)
POLICY_IDENTITY_SHA256 = (
    "9a9242f47981c25e754fc95e2650050da4e4195aa1ef3a78f2c293f9e25d7ff7"
)

PINNED_BLOBS = {
    "tools/next100_106_balance_gate.py":
        "ae4f9ccdc3cfe3e053dd57d46f80aebe121268c5",
    "configs/data/next100_106_balance_gate_policy_v1.json":
        "b5a2577aeb1a2e56ebff1a4b46ac325d99dd8f8f",
    "src/twelve_six/__init__.py":
        "5433166c507bc845bd12d8d5c4145f1fbedda204",
    "src/twelve_six/data/trusted_family_authority_v1.py":
        "9382332d2d0d5c09aa5582e6f949b1630c459cbb",
    "src/twelve_six/data/postdecontam_balance_projection_v1.py":
        "8512a7363cb3953c46d7072cb4b3c09c86751236",
    "src/twelve_six/data/postmaterialization_balance_projection_v1.py":
        "62a6640ad5911e21f406ef12c1b0d7e5e5b1afef",
}

SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA64 = re.compile(r"^[0-9a-f]{64}$")

ZERO_CREDIT = {
    "canonical_capacity_credited": 0,
    "training_authorized_bytes": 0,
    "authorized_unique_loss_positions": 0,
    "authorized_optimized_target_exposure": 0,
    "tokenizer_fit_authorized": False,
    "model_training_authorized": False,
    "optimizer_updates_executed_on_real_targets": 0,
    "training_executed": False,
    "learned_weights_created": False,
    "final_test_outcomes_read": False,
    "paid_compute_used": False,
    "foreign_pretrained_weights_used": False,
    "scale_promotion_authorized": False,
}

POST_G06_EXPECTED_CONTENT_BOUNDARY = {
    "raw_training_text_persisted": False,
    "raw_evaluation_text_persisted": False,
    "raw_survivor_text_persisted": False,
    "durable_output_text_free": True,
}

POST_G06_EXPECTED_TRUTH_BOUNDARY = {
    "current_rada_data232_parent_two_clean_complete": True,
    "canonical_quality_privacy_executed": True,
    "balance_diversity_retest_complete": False,
    "family_caps_complete": False,
    "cluster_safe_split_complete": False,
    "deterministic_pack_two_clean_complete": False,
    "positive_exact_unique_loss_ledger": False,
    "canonical_capacity_credited": 0,
    "training_authorized_bytes": 0,
    "authorized_unique_loss_positions": 0,
    "authorized_optimized_target_exposure": 0,
    "tokenizer_fit_authorized": False,
    "optimizer_updates_executed_on_real_targets": 0,
    "training_executed": False,
    "learned_weights_created": False,
    "final_test_outcomes_read": False,
    "paid_compute_used": False,
    "foreign_pretrained_weights_used": False,
    "scale_promotion_authorized": False,
}


class RadaPostG06BalanceError(RuntimeError):
    """Raised when the physical-to-balance lineage cannot be proven."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RadaPostG06BalanceError(message)


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_line(value: Any) -> bytes:
    return canonical(value) + b"\n"


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def self_hash(document: Mapping[str, Any], field: str) -> str:
    core = dict(document)
    core.pop(field, None)
    return sha256(canonical(core))


def require_sha256(value: Any, label: str) -> str:
    require(
        isinstance(value, str) and SHA64.fullmatch(value) is not None,
        f"{label} must be 64 lowercase hex",
    )
    return value


def require_git_sha(value: Any, label: str) -> str:
    require(
        isinstance(value, str) and SHA40.fullmatch(value) is not None,
        f"{label} must be 40 lowercase hex",
    )
    return value


def require_positive_int(value: Any, label: str) -> int:
    require(type(value) is int and value > 0, f"{label} must be positive integer")
    return value


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key rejected: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise RadaPostG06BalanceError(f"non-finite JSON constant rejected: {value}")


def load_json_bytes(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RadaPostG06BalanceError(f"{label} strict JSON decode failed") from exc
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def load_pinned_json(
    path: Path,
    *,
    expected_file_sha256: str,
    label: str,
) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"{label} file invalid")
    raw = path.read_bytes()
    require(
        sha256(raw) == require_sha256(expected_file_sha256, f"{label} file SHA"),
        f"{label} file SHA-256 drift",
    )
    return load_json_bytes(raw, label)


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            text=True,
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        operation = args[0] if args else "command"
        raise RadaPostG06BalanceError(
            f"git {operation} failed"
        ) from exc


def verify_source_head(source_git_sha: str) -> str:
    expected = require_git_sha(source_git_sha, "source_git_sha")
    require(_git("rev-parse", "HEAD") == expected, "execution HEAD drift")
    try:
        ancestry = subprocess.run(
            ["git", "merge-base", "--is-ancestor", STACK_BASE_HEAD, expected],
            cwd=ROOT,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        raise RadaPostG06BalanceError("git merge-base failed") from exc
    require(
        ancestry.returncode == 0,
        "execution HEAD is outside the exact post-G06 balance-adapter stack",
    )
    return expected


def verify_dependency_blobs() -> None:
    for relative, expected in PINNED_BLOBS.items():
        observed = _git("hash-object", str(ROOT / relative))
        require(observed == expected, f"canonical dependency blob drift: {relative}")


def _resolve_existing_path(raw_path: str, error: str) -> Path:
    try:
        return Path(raw_path).resolve(strict=True)
    except OSError as exc:
        raise RadaPostG06BalanceError(error) from exc


def write_immutable_bytes(path: Path, payload: bytes, *, label: str) -> None:
    """Atomically create deterministic evidence or resume an identical write."""
    require(not path.is_symlink(), f"{label}: output path must not be a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.is_file(), f"{label}: output path is not a regular file")
        require(
            path.read_bytes() == payload,
            f"{label}: refusing to overwrite divergent durable evidence",
        )
        return

    temp = path.with_name(path.name + ".tmp")
    require(not temp.is_symlink(), f"{label}: temp path must not be a symlink")
    if temp.exists():
        require(temp.is_file(), f"{label}: temp path is not a regular file")
        require(
            temp.read_bytes() == payload,
            f"{label}: divergent interrupted temp evidence",
        )
        temp.replace(path)
        return

    created_temp = False
    try:
        with temp.open("xb") as handle:
            created_temp = True
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temp.replace(path)
    except OSError:
        if created_temp:
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def _read_regular_bytes(path: Path, *, label: str) -> bytes:
    require(
        path.is_file() and not path.is_symlink(),
        f"{label}: file invalid",
    )
    return path.read_bytes()


def verify_module_provenance(module: Any, relative: str) -> None:
    raw = getattr(module, "__file__", None)
    require(isinstance(raw, str) and raw, f"module path missing: {relative}")
    observed = _resolve_existing_path(
        raw,
        f"module provenance drift: {relative}",
    )
    expected = (ROOT / relative).resolve(strict=True)
    require(
        observed == expected,
        f"module provenance drift: {relative}",
    )


def _clear_canonical_module_cache() -> None:
    for module_name in (
        "tools.next100_106_balance_gate",
        "twelve_six.data.postmaterialization_balance_projection_v1",
        "twelve_six.data.postdecontam_balance_projection_v1",
        "twelve_six.data.trusted_family_authority_v1",
    ):
        cached = sys.modules.pop(module_name, None)
        if cached is None or "." not in module_name:
            continue
        parent_name, attribute = module_name.rsplit(".", 1)
        parent = sys.modules.get(parent_name)
        if parent is not None and getattr(parent, attribute, None) is cached:
            delattr(parent, attribute)
    importlib.invalidate_caches()


def load_canonical_authorities() -> tuple[Any, Any, dict[str, Any]]:
    verify_dependency_blobs()
    import twelve_six

    verify_module_provenance(twelve_six, "src/twelve_six/__init__.py")
    expected_package = (ROOT / "src/twelve_six").resolve(strict=True)
    package_paths = [
        _resolve_existing_path(
            value,
            "canonical twelve_six package search path drift",
        )
        for value in twelve_six.__path__
    ]
    require(
        package_paths == [expected_package],
        "canonical twelve_six package search path drift",
    )
    data_package = importlib.import_module("twelve_six.data")
    expected_data = (ROOT / "src/twelve_six/data").resolve(strict=True)
    data_paths = [
        _resolve_existing_path(
            value,
            "canonical twelve_six.data package search path drift",
        )
        for value in data_package.__path__
    ]
    require(
        data_paths == [expected_data],
        "canonical twelve_six.data package search path drift",
    )

    _clear_canonical_module_cache()
    gate = importlib.import_module("tools.next100_106_balance_gate")
    trusted = importlib.import_module(
        "twelve_six.data.trusted_family_authority_v1"
    )
    projection = importlib.import_module(
        "twelve_six.data.postdecontam_balance_projection_v1"
    )
    bridge = importlib.import_module(
        "twelve_six.data.postmaterialization_balance_projection_v1"
    )
    for module, relative in (
        (gate, "tools/next100_106_balance_gate.py"),
        (trusted, "src/twelve_six/data/trusted_family_authority_v1.py"),
        (
            projection,
            "src/twelve_six/data/postdecontam_balance_projection_v1.py",
        ),
        (
            bridge,
            "src/twelve_six/data/postmaterialization_balance_projection_v1.py",
        ),
    ):
        verify_module_provenance(module, relative)
    require(
        bridge.TRUSTED_FAMILY_SEMANTICS is trusted.TRUSTED_FAMILY_SEMANTICS
        and projection.TRUSTED_FAMILY_SEMANTICS is trusted.TRUSTED_FAMILY_SEMANTICS,
        "canonical trusted-family object split across balance modules",
    )
    require(
        bridge.ProjectionError is projection.ProjectionError,
        "canonical projection exception identity drift",
    )

    policy_path = ROOT / "configs/data/next100_106_balance_gate_policy_v1.json"
    policy = load_json_bytes(policy_path.read_bytes(), "NEXT100-106 policy")
    gate.validate_policy(policy)
    require(
        policy.get("policy_identity_sha256") == POLICY_IDENTITY_SHA256,
        "canonical balance policy identity drift",
    )
    return gate, bridge, policy


def verify_global_dedup(
    evidence: Mapping[str, Any],
    two_clean: Mapping[str, Any],
) -> None:
    require(
        evidence.get("schema_version")
        == "12-6.d03-rada-current-global-dedup-execution.v1",
        "upstream global-dedup evidence schema drift",
    )
    require(
        evidence.get("execution_head_sha") == UPSTREAM_GLOBAL_DEDUP_HEAD,
        "upstream global-dedup head drift",
    )
    claimed = require_sha256(
        evidence.get("evidence_identity_sha256"),
        "upstream global-dedup evidence identity",
    )
    require(claimed == self_hash(evidence, "evidence_identity_sha256"),
            "upstream global-dedup evidence self-hash mismatch")
    require(claimed == UPSTREAM_GLOBAL_DEDUP_EVIDENCE_ID,
            "upstream global-dedup evidence identity drift")
    require(
        evidence.get("survivor_authority_sha256") == UPSTREAM_SURVIVOR_AUTHORITY_ID,
        "upstream survivor authority drift",
    )
    require(evidence.get("raw_text_persisted") is False,
            "upstream dedup persisted raw text")
    require(evidence.get("canonical_capacity_credited") == 0,
            "upstream dedup widened capacity")
    require(evidence.get("training_authorized_bytes") == 0,
            "upstream dedup widened training bytes")
    require(evidence.get("tokenizer_fit_authorized") is False,
            "upstream dedup widened tokenizer authority")
    require(evidence.get("training_executed") is False,
            "upstream dedup executed training")
    require(evidence.get("final_test_outcomes_read") is False,
            "upstream dedup read final-test outcomes")
    require(evidence.get("paid_compute_used") is False,
            "upstream dedup used paid compute")

    require(
        two_clean.get("schema_version")
        == "12-6.d03-rada-current-global-dedup-two-clean.v1",
        "upstream two-clean schema drift",
    )
    require(two_clean.get("execution_head_sha") == UPSTREAM_GLOBAL_DEDUP_HEAD,
            "upstream two-clean head drift")
    two_id = require_sha256(two_clean.get("two_clean_identity_sha256"),
                            "upstream two-clean identity")
    require(two_id == self_hash(two_clean, "two_clean_identity_sha256"),
            "upstream two-clean self-hash mismatch")
    require(two_id == UPSTREAM_TWO_CLEAN_ID, "upstream two-clean identity drift")
    require(two_clean.get("fresh_process_count") == 2,
            "upstream two-clean process count drift")
    require(two_clean.get("byte_identical_outputs") is True,
            "upstream two-clean outputs are not byte-identical")
    require(
        two_clean.get("survivor_authority_sha256")
        == evidence.get("survivor_authority_sha256"),
        "upstream two-clean survivor authority mismatch",
    )
    require(
        two_clean.get("matcher_report_sha256") == evidence.get("matcher_report_sha256"),
        "upstream two-clean matcher report mismatch",
    )
    require(two_clean.get("canonical_capacity_credited") == 0,
            "upstream two-clean widened capacity")
    require(two_clean.get("training_authorized_bytes") == 0,
            "upstream two-clean widened training bytes")
    require(two_clean.get("tokenizer_fit_authorized") is False,
            "upstream two-clean widened tokenizer authority")


def verify_post_g06_receipt(
    evidence: Mapping[str, Any],
    family_vector: Mapping[str, Any],
    *,
    expected_evidence_identity_sha256: str,
) -> dict[str, Any]:
    require(evidence.get("schema_version") == POST_G06_SCHEMA,
            "post-G06 evidence schema drift")
    claimed = require_sha256(
        evidence.get("evidence_identity_sha256"),
        "post-G06 evidence identity",
    )
    require(claimed == self_hash(evidence, "evidence_identity_sha256"),
            "post-G06 evidence self-hash mismatch")
    require(
        claimed == require_sha256(
            expected_evidence_identity_sha256,
            "expected post-G06 evidence identity",
        ),
        "post-G06 evidence identity differs from external expectation",
    )
    require(
        family_vector.get("materialization_identity_sha256") == claimed,
        "family vector materialization identity differs from post-G06 receipt",
    )
    require(
        family_vector.get("materialization_execution_head_sha")
        == evidence.get("execution_head_sha"),
        "family vector materialization head differs from post-G06 receipt",
    )
    g06 = evidence.get("g06")
    require(isinstance(g06, Mapping), "post-G06 receipt lacks G06")
    require(g06.get("exact_payload_collision_free") is True,
            "post-G06 exact payload collision proof is not terminal")
    unique_count = require_positive_int(
        g06.get("unique_payload_count"), "post-G06 unique payload count"
    )
    payload_set_id = require_sha256(
        g06.get("payload_set_identity_sha256"), "post-G06 payload-set identity"
    )
    survivor = evidence.get("survivor_inventory")
    require(isinstance(survivor, Mapping), "post-G06 survivor receipt missing")
    require(unique_count == family_vector.get("record_count"),
            "post-G06 unique payload count differs from family vector")
    for field in (
        "record_count",
        "total_payload_bytes",
        "record_inventory_digest_sha256",
        "payload_inventory_digest_sha256",
        "record_payload_jsonl_sha256",
    ):
        require(
            survivor.get(field) == family_vector.get(field),
            f"post-G06/family-vector cross-bind drift: {field}",
        )
    require(
        evidence.get("content_boundary") == POST_G06_EXPECTED_CONTENT_BOUNDARY,
        "post-G06 content boundary drift",
    )
    truth = evidence.get("truth_boundary")
    require(
        truth == POST_G06_EXPECTED_TRUTH_BOUNDARY,
        "post-G06 truth boundary drift",
    )
    require(
        evidence.get("next_gate") == "CURRENT_RADA_BALANCE_DIVERSITY_FAMILY_CAP_RETEST",
        "post-G06 next gate drift",
    )
    return {
        "execution_head_sha": evidence["execution_head_sha"],
        "evidence_identity_sha256": claimed,
        "payload_set_identity_sha256": payload_set_id,
        "unique_payload_count": unique_count,
        "record_inventory_digest_sha256": survivor[
            "record_inventory_digest_sha256"
        ],
        "payload_inventory_digest_sha256": survivor[
            "payload_inventory_digest_sha256"
        ],
    }



def verify_post_g06_two_clean_proof(
    proof: Mapping[str, Any],
    evidence: Mapping[str, Any],
    family_vector: Mapping[str, Any],
    *,
    expected_proof_identity_sha256: str,
    expected_evidence_file_sha256: str,
) -> str:
    expected_fields = {
        "schema_version",
        "execution_head_sha",
        "parent_execution_head_sha",
        "parent_artifact_id",
        "parent_artifact_zip_sha256",
        "fresh_execution_count",
        "independent_runner_jobs",
        "byte_identical_outputs",
        "output_file_sha256",
        "evidence_identity_sha256",
        "g05_execution_identity_sha256",
        "g06_execution_identity_sha256",
        "record_payload_jsonl_sha256",
        "record_inventory_digest_sha256",
        "payload_inventory_digest_sha256",
        "payload_set_identity_sha256",
        "record_count",
        "total_payload_bytes",
        "canonical_capacity_credited",
        "training_authorized_bytes",
        "authorized_unique_loss_positions",
        "authorized_optimized_target_exposure",
        "tokenizer_fit_authorized",
        "training_executed",
        "learned_weights_created",
        "final_test_outcomes_read",
        "paid_compute_used",
        "scale_promotion_authorized",
        "next_gate",
        "proof_identity_sha256",
    }
    require(set(proof) == expected_fields, "post-G06 two-clean proof fields drift")
    require(
        proof.get("schema_version") == G05_G06_TWO_CLEAN_SCHEMA,
        "post-G06 two-clean proof schema drift",
    )
    claimed = require_sha256(
        proof.get("proof_identity_sha256"),
        "post-G06 two-clean proof identity",
    )
    require(
        claimed == self_hash(proof, "proof_identity_sha256"),
        "post-G06 two-clean proof self-hash mismatch",
    )
    require(
        claimed
        == require_sha256(
            expected_proof_identity_sha256,
            "expected post-G06 two-clean proof identity",
        ),
        "post-G06 two-clean proof identity differs from external expectation",
    )
    require(
        proof.get("execution_head_sha") == evidence.get("execution_head_sha")
        == family_vector.get("materialization_execution_head_sha"),
        "post-G06 two-clean execution head drift",
    )
    require(
        proof.get("parent_execution_head_sha") == PARENT_DATA232_HEAD,
        "post-G06 two-clean DATA232 parent head drift",
    )
    require(proof.get("fresh_execution_count") == 2,
            "post-G06 two-clean fresh execution count drift")
    require(proof.get("independent_runner_jobs") is True,
            "post-G06 two-clean independent-runner proof missing")
    require(proof.get("byte_identical_outputs") is True,
            "post-G06 two-clean byte identity missing")

    parent = evidence.get("parent")
    g05 = evidence.get("g05")
    g06 = evidence.get("g06")
    survivor = evidence.get("survivor_inventory")
    artifacts = evidence.get("durable_artifacts")
    require(isinstance(parent, Mapping), "post-G06 parent binding missing")
    require(isinstance(g05, Mapping), "post-G06 G05 binding missing")
    require(isinstance(g06, Mapping), "post-G06 G06 binding missing")
    require(isinstance(survivor, Mapping), "post-G06 survivor binding missing")
    require(isinstance(artifacts, Mapping), "post-G06 artifact binding missing")
    require(
        proof.get("parent_artifact_id") == parent.get("artifact_id"),
        "post-G06 two-clean parent artifact ID drift",
    )
    require(
        proof.get("parent_artifact_zip_sha256")
        == parent.get("artifact_zip_sha256"),
        "post-G06 two-clean parent artifact ZIP drift",
    )
    expected_outputs = {
        G05_G06_BUNDLE_FILES["evidence"]: require_sha256(
            expected_evidence_file_sha256,
            "expected post-G06 evidence file SHA-256",
        ),
        G05_G06_BUNDLE_FILES["quality"]: require_sha256(
            artifacts.get("g05_authority_file_sha256"),
            "G05 authority file SHA-256",
        ),
        G05_G06_BUNDLE_FILES["privacy"]: require_sha256(
            artifacts.get("g06_authority_file_sha256"),
            "G06 authority file SHA-256",
        ),
        G05_G06_BUNDLE_FILES["survivor_inventory"]: require_sha256(
            artifacts.get("survivor_inventory_file_sha256"),
            "survivor inventory file SHA-256",
        ),
    }
    require(
        proof.get("output_file_sha256") == expected_outputs,
        "post-G06 two-clean output-file roots drift",
    )

    bindings = {
        "evidence_identity_sha256": evidence.get("evidence_identity_sha256"),
        "g05_execution_identity_sha256": g05.get("execution_identity_sha256"),
        "g06_execution_identity_sha256": g06.get("execution_identity_sha256"),
        "record_payload_jsonl_sha256": survivor.get(
            "record_payload_jsonl_sha256"
        ),
        "record_inventory_digest_sha256": survivor.get(
            "record_inventory_digest_sha256"
        ),
        "payload_inventory_digest_sha256": survivor.get(
            "payload_inventory_digest_sha256"
        ),
        "payload_set_identity_sha256": g06.get("payload_set_identity_sha256"),
        "record_count": family_vector.get("record_count"),
        "total_payload_bytes": family_vector.get("total_payload_bytes"),
    }
    for field, expected in bindings.items():
        require(
            proof.get(field) == expected,
            f"post-G06 two-clean binding drift: {field}",
        )

    for field in (
        "canonical_capacity_credited",
        "training_authorized_bytes",
        "authorized_unique_loss_positions",
        "authorized_optimized_target_exposure",
    ):
        require(
            type(proof.get(field)) is int and proof.get(field) == 0,
            f"post-G06 two-clean authority widened: {field}",
        )
    for field in (
        "tokenizer_fit_authorized",
        "training_executed",
        "learned_weights_created",
        "final_test_outcomes_read",
        "paid_compute_used",
        "scale_promotion_authorized",
    ):
        require(
            proof.get(field) is False,
            f"post-G06 two-clean truth widened: {field}",
        )
    require(
        proof.get("next_gate") == G05_G06_NEXT_GATE,
        "post-G06 two-clean next gate drift",
    )
    return claimed


def build_composition_proof(
    *,
    source_git_sha: str,
    family_vector: Mapping[str, Any],
    post_g06: Mapping[str, Any],
) -> dict[str, Any]:
    core: dict[str, Any] = {
        "schema": COMPOSITION_SCHEMA,
        "execution_head_sha": source_git_sha,
        "upstream_global_dedup": {
            "schema_version": "12-6.d03-rada-current-global-dedup-execution.v1",
            "head_sha": UPSTREAM_GLOBAL_DEDUP_HEAD,
            "evidence_identity_sha256": UPSTREAM_GLOBAL_DEDUP_EVIDENCE_ID,
            "survivor_authority_sha256": UPSTREAM_SURVIVOR_AUTHORITY_ID,
            "two_clean_identity_sha256": UPSTREAM_TWO_CLEAN_ID,
            "artifact_zip_sha256": UPSTREAM_ARTIFACT_ZIP_SHA256,
            "terminal_verdict": "PASS",
        },
        "post_g06_physical_uniqueness": dict(post_g06),
        "family_vector_identity_sha256": family_vector[
            "family_vector_identity_sha256"
        ],
        "materialization_identity_sha256": family_vector[
            "materialization_identity_sha256"
        ],
        "record_count": family_vector["record_count"],
        "total_payload_bytes": family_vector["total_payload_bytes"],
        "cross_transform_exact_payload_collision_free": True,
        "semantics": COMPOSITION_SEMANTICS,
        "terminal_verdict": "PASS",
        **ZERO_CREDIT,
    }
    return {
        **core,
        "evidence_identity_sha256": sha256(canonical(core)),
    }


def execute(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    source_git_sha = verify_source_head(args.source_git_sha)
    gate, bridge, policy = load_canonical_authorities()

    family_vector = load_pinned_json(
        args.family_vector,
        expected_file_sha256=args.expected_family_vector_file_sha256,
        label="family vector",
    )
    family_identity = bridge.verify_postmaterialization_family_vector(
        family_vector,
        expected_identity_sha256=args.expected_family_vector_identity_sha256,
    )
    require(family_vector.get("schema") == FAMILY_VECTOR_SCHEMA,
            "family vector schema drift")

    post_g06_evidence = load_pinned_json(
        args.post_g06_evidence,
        expected_file_sha256=args.expected_post_g06_evidence_file_sha256,
        label="post-G06 evidence",
    )
    post_g06 = verify_post_g06_receipt(
        post_g06_evidence,
        family_vector,
        expected_evidence_identity_sha256=args.expected_post_g06_evidence_identity_sha256,
    )
    post_g06_two_clean = load_pinned_json(
        args.post_g06_two_clean_proof,
        expected_file_sha256=args.expected_post_g06_two_clean_proof_file_sha256,
        label="post-G06 two-clean proof",
    )
    post_g06_two_clean_identity = verify_post_g06_two_clean_proof(
        post_g06_two_clean,
        post_g06_evidence,
        family_vector,
        expected_proof_identity_sha256=(
            args.expected_post_g06_two_clean_proof_identity_sha256
        ),
        expected_evidence_file_sha256=args.expected_post_g06_evidence_file_sha256,
    )
    post_g06 = {
        **post_g06,
        "two_clean_proof_identity_sha256": post_g06_two_clean_identity,
    }

    upstream_evidence = load_pinned_json(
        args.upstream_global_dedup_evidence,
        expected_file_sha256=UPSTREAM_GLOBAL_DEDUP_EVIDENCE_FILE_SHA256,
        label="upstream global-dedup evidence",
    )
    upstream_two_clean = load_pinned_json(
        args.upstream_global_dedup_two_clean,
        expected_file_sha256=UPSTREAM_TWO_CLEAN_FILE_SHA256,
        label="upstream global-dedup two-clean",
    )
    verify_global_dedup(upstream_evidence, upstream_two_clean)

    composition = build_composition_proof(
        source_git_sha=source_git_sha,
        family_vector=family_vector,
        post_g06=post_g06,
    )
    dedup_authority = {
        "worker_id": DEDUP_WORKER_ID,
        "head_sha": source_git_sha,
        "evidence_identity_sha256": composition["evidence_identity_sha256"],
        "terminal_verdict": "PASS",
    }
    next100 = bridge.adapt_postmaterialization_family_vector_to_next100_106(
        family_vector,
        expected_family_vector_identity_sha256=family_identity,
        dedup_authority=dedup_authority,
        expected_dedup_worker_id=DEDUP_WORKER_ID,
        expected_dedup_head_sha=source_git_sha,
        expected_dedup_evidence_identity_sha256=composition[
            "evidence_identity_sha256"
        ],
    )
    gate.validate_vector(next100)
    balance = gate.evaluate(policy, next100)
    result_identity = bridge.verify_balance_result(
        balance,
        expected_result_identity_sha256=balance["result_identity_sha256"],
    )
    binding = bridge.build_balance_result_binding(
        family_vector=family_vector,
        expected_family_vector_identity_sha256=family_identity,
        next100_input=next100,
        balance_result=balance,
        expected_policy_identity_sha256=POLICY_IDENTITY_SHA256,
        expected_result_identity_sha256=result_identity,
    )
    status = balance["status"]
    next_gate = (
        "CLUSTER_SAFE_SPLIT_AND_DETERMINISTIC_PACK"
        if status == "TARGET_20M_SOURCE_MIX_FEASIBLE"
        else "ACQUIRE_MORE_DIVERSE_LAWFUL_SOURCE_CAPACITY"
    )
    receipt_core: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "execution_profile": "LOCAL_FREE",
        "execution_head_sha": source_git_sha,
        "upstream_global_dedup_evidence_identity_sha256":
            UPSTREAM_GLOBAL_DEDUP_EVIDENCE_ID,
        "upstream_global_dedup_two_clean_identity_sha256": UPSTREAM_TWO_CLEAN_ID,
        "post_g06_evidence_identity_sha256": post_g06[
            "evidence_identity_sha256"
        ],
        "post_g06_two_clean_proof_identity_sha256": post_g06[
            "two_clean_proof_identity_sha256"
        ],
        "family_vector_identity_sha256": family_identity,
        "composition_dedup_identity_sha256": composition[
            "evidence_identity_sha256"
        ],
        "next100_input_identity_sha256": sha256(canonical(next100)),
        "balance_policy_identity_sha256": POLICY_IDENTITY_SHA256,
        "balance_result_identity_sha256": result_identity,
        "balance_binding_identity_sha256": binding["binding_identity_sha256"],
        "balance_status": status,
        "maximum_feasible_total_source_bytes": balance[
            "maximum_feasible_total_source_bytes"
        ],
        "raw_capacity_by_stratum": balance["raw_capacity_by_stratum"],
        "raw_gap_to_target_by_stratum": balance["raw_gap_to_target_by_stratum"],
        "family_minimum": balance["family_minimum"],
        "next_scientific_gate": next_gate,
        **ZERO_CREDIT,
    }
    receipt = {
        **receipt_core,
        "receipt_identity_sha256": sha256(canonical(receipt_core)),
    }
    return {
        "composition-dedup-proof": composition,
        "next100-input": next100,
        "balance-result": balance,
        "balance-binding": binding,
        "execution-receipt": receipt,
    }


OUTPUT_NAMES = (
    "composition-dedup-proof",
    "next100-input",
    "balance-result",
    "balance-binding",
    "execution-receipt",
)


def _verify_output_directory(
    path: Path,
    *,
    allow_interrupted_temps: bool,
    require_complete: bool,
) -> None:
    expected = {f"{name}.json" for name in OUTPUT_NAMES}
    temp_names = {name + ".tmp" for name in expected}
    allowed = expected | temp_names if allow_interrupted_temps else expected
    observed: set[str] = set()
    for entry in path.iterdir():
        require(
            entry.is_file() and not entry.is_symlink(),
            "output directory contains non-regular entry",
        )
        observed.add(entry.name)
    require(
        observed <= allowed,
        "output directory contains unexpected entries",
    )
    if require_complete:
        require(
            observed == expected,
            "output directory bundle is incomplete",
        )


def write_output_dir(path: Path, values: Mapping[str, Mapping[str, Any]]) -> None:
    require(set(values) == set(OUTPUT_NAMES), "output bundle key set drift")
    require(not path.is_symlink(), "output directory must not be a symlink")
    if path.exists():
        require(path.is_dir(), "output path is not a directory")
    else:
        path.mkdir(parents=True)
    _verify_output_directory(
        path,
        allow_interrupted_temps=True,
        require_complete=False,
    )
    if (path / "execution-receipt.json").exists():
        _verify_output_directory(
            path,
            allow_interrupted_temps=False,
            require_complete=True,
        )

    # Commit child artifacts first and the bound receipt last. This permits
    # deterministic restart after an interrupted partial bundle without ever
    # publishing a receipt before its children are durable.
    for name in OUTPUT_NAMES[:-1]:
        write_immutable_bytes(
            path / f"{name}.json",
            canonical_line(values[name]),
            label=name,
        )
    write_immutable_bytes(
        path / "execution-receipt.json",
        canonical_line(values["execution-receipt"]),
        label="execution receipt",
    )
    _verify_output_directory(
        path,
        allow_interrupted_temps=False,
        require_complete=True,
    )


def compare_outputs(
    output_a: Path,
    output_b: Path,
    proof_path: Path,
    *,
    runner_a_identity: str | None = None,
    runner_b_identity: str | None = None,
    independent_runner_jobs: bool = False,
    enforce_checkout_provenance: bool = False,
) -> dict[str, Any]:
    runner_pattern = re.compile(r"^[A-Za-z0-9._:/-]{1,200}$")
    require(
        isinstance(runner_a_identity, str)
        and runner_pattern.fullmatch(runner_a_identity) is not None,
        "two-clean runner A identity missing or invalid",
    )
    require(
        isinstance(runner_b_identity, str)
        and runner_pattern.fullmatch(runner_b_identity) is not None,
        "two-clean runner B identity missing or invalid",
    )
    require(
        runner_a_identity != runner_b_identity,
        "two-clean runner identities must be distinct",
    )
    require(
        independent_runner_jobs is True,
        "two-clean independent runner jobs are not attested",
    )
    for path, label in (
        (output_a, "two-clean output A"),
        (output_b, "two-clean output B"),
    ):
        require(path.is_dir() and not path.is_symlink(), f"{label}: directory invalid")
    require(
        _resolve_existing_path(
            str(output_a),
            "two-clean output A: directory invalid",
        )
        != _resolve_existing_path(
            str(output_b),
            "two-clean output B: directory invalid",
        ),
        "two-clean output directories must be distinct",
    )

    _verify_output_directory(
        output_a,
        allow_interrupted_temps=False,
        require_complete=True,
    )
    _verify_output_directory(
        output_b,
        allow_interrupted_temps=False,
        require_complete=True,
    )
    hashes: dict[str, str] = {}
    parsed: dict[str, dict[str, Any]] = {}
    for name in OUTPUT_NAMES:
        a = _read_regular_bytes(
            output_a / f"{name}.json",
            label=f"two-clean A {name}",
        )
        b = _read_regular_bytes(
            output_b / f"{name}.json",
            label=f"two-clean B {name}",
        )
        require(a == b, f"two-clean output differs: {name}")
        value = load_json_bytes(a, f"two-clean {name}")
        require(
            canonical_line(value) == a,
            f"two-clean output is not canonical JSON: {name}",
        )
        parsed[name] = value
        hashes[f"{name}.json"] = sha256(a)

    composition = parsed["composition-dedup-proof"]
    require(
        composition.get("schema") == COMPOSITION_SCHEMA,
        "composition schema mismatch",
    )
    require(
        composition.get("terminal_verdict") == "PASS",
        "composition terminal verdict is not PASS",
    )
    require(
        composition.get("cross_transform_exact_payload_collision_free") is True,
        "composition post-transform uniqueness is not terminal",
    )
    require(
        composition.get("semantics") == COMPOSITION_SEMANTICS,
        "composition semantics drift",
    )
    composition_identity = require_sha256(
        composition.get("evidence_identity_sha256"),
        "composition dedup identity",
    )
    require(
        composition_identity == self_hash(composition, "evidence_identity_sha256"),
        "composition dedup self-hash mismatch",
    )
    for field, expected in ZERO_CREDIT.items():
        require(
            composition.get(field) == expected,
            f"composition zero-credit drift: {field}",
        )

    next100_input = parsed["next100-input"]
    next100_identity = sha256(canonical(next100_input))
    dedup_authority = next100_input.get("dedup_authority")
    require(
        isinstance(dedup_authority, Mapping),
        "NEXT100 input lacks dedup authority",
    )
    require(
        dedup_authority.get("worker_id") == DEDUP_WORKER_ID,
        "NEXT100 dedup worker drift",
    )
    require(
        dedup_authority.get("evidence_identity_sha256") == composition_identity,
        "NEXT100 dedup evidence differs from composition",
    )
    require(
        dedup_authority.get("terminal_verdict") == "PASS",
        "NEXT100 dedup authority is not PASS",
    )

    balance_result = parsed["balance-result"]
    result_identity = require_sha256(
        balance_result.get("result_identity_sha256"),
        "balance result identity",
    )
    require(
        result_identity == self_hash(balance_result, "result_identity_sha256"),
        "balance result self-hash mismatch",
    )

    balance_binding = parsed["balance-binding"]
    binding_identity = require_sha256(
        balance_binding.get("binding_identity_sha256"),
        "balance binding identity",
    )
    require(
        binding_identity == self_hash(balance_binding, "binding_identity_sha256"),
        "balance binding self-hash mismatch",
    )
    require(
        balance_binding.get("next100_input_identity_sha256") == next100_identity,
        "balance binding next100-input identity mismatch",
    )
    require(
        balance_result.get("dedup_authority") == next100_input.get("dedup_authority"),
        "balance result dedup authority differs from next100 input",
    )
    require(
        balance_result.get("input_totals") == next100_input.get("totals"),
        "balance result totals differ from next100 input",
    )

    receipt = parsed["execution-receipt"]
    expected_receipt_fields = {
        "schema",
        "execution_profile",
        "execution_head_sha",
        "upstream_global_dedup_evidence_identity_sha256",
        "upstream_global_dedup_two_clean_identity_sha256",
        "post_g06_evidence_identity_sha256",
        "post_g06_two_clean_proof_identity_sha256",
        "family_vector_identity_sha256",
        "composition_dedup_identity_sha256",
        "next100_input_identity_sha256",
        "balance_policy_identity_sha256",
        "balance_result_identity_sha256",
        "balance_binding_identity_sha256",
        "balance_status",
        "maximum_feasible_total_source_bytes",
        "raw_capacity_by_stratum",
        "raw_gap_to_target_by_stratum",
        "family_minimum",
        "next_scientific_gate",
        *ZERO_CREDIT.keys(),
        "receipt_identity_sha256",
    }
    require(
        set(receipt) == expected_receipt_fields,
        "execution receipt fields drift",
    )
    require(receipt.get("schema") == RECEIPT_SCHEMA, "execution receipt schema mismatch")
    require(
        receipt.get("execution_profile") == "LOCAL_FREE",
        "execution receipt profile drift",
    )
    receipt_identity = require_sha256(
        receipt.get("receipt_identity_sha256"),
        "execution receipt identity",
    )
    require(
        receipt_identity == self_hash(receipt, "receipt_identity_sha256"),
        "execution receipt self-hash mismatch",
    )
    receipt_head = require_git_sha(
        receipt.get("execution_head_sha"),
        "execution receipt head",
    )
    require(
        composition.get("execution_head_sha") == receipt_head,
        "composition execution head differs from receipt",
    )
    require(
        dedup_authority.get("head_sha") == receipt_head,
        "NEXT100 dedup head differs from receipt",
    )
    require(
        receipt.get("composition_dedup_identity_sha256") == composition_identity,
        "execution receipt composition identity mismatch",
    )

    composition_family = require_sha256(
        composition.get("family_vector_identity_sha256"),
        "composition family-vector identity",
    )
    receipt_family = require_sha256(
        receipt.get("family_vector_identity_sha256"),
        "execution receipt family-vector identity",
    )
    require(
        composition_family
        == receipt_family
        == balance_binding.get("family_vector_identity_sha256"),
        "family-vector identity differs across evidence chain",
    )
    physical = next100_input.get("physical_authority")
    require(
        isinstance(physical, Mapping)
        and physical.get("family_vector_identity_sha256") == receipt_family,
        "NEXT100 physical family-vector identity mismatch",
    )

    post_g06 = composition.get("post_g06_physical_uniqueness")
    require(
        isinstance(post_g06, Mapping),
        "composition lacks post-G06 physical evidence",
    )
    require(
        post_g06.get("evidence_identity_sha256")
        == receipt.get("post_g06_evidence_identity_sha256"),
        "post-G06 identity differs across evidence chain",
    )
    post_g06_two_clean_id = require_sha256(
        post_g06.get("two_clean_proof_identity_sha256"),
        "composition post-G06 two-clean identity",
    )
    require(
        post_g06_two_clean_id
        == receipt.get("post_g06_two_clean_proof_identity_sha256"),
        "post-G06 two-clean identity differs across evidence chain",
    )

    upstream = composition.get("upstream_global_dedup")
    require(
        isinstance(upstream, Mapping),
        "composition lacks upstream global-dedup evidence",
    )
    require(
        upstream.get("evidence_identity_sha256")
        == receipt.get("upstream_global_dedup_evidence_identity_sha256")
        == UPSTREAM_GLOBAL_DEDUP_EVIDENCE_ID,
        "upstream global-dedup evidence identity mismatch",
    )
    require(
        upstream.get("two_clean_identity_sha256")
        == receipt.get("upstream_global_dedup_two_clean_identity_sha256")
        == UPSTREAM_TWO_CLEAN_ID,
        "upstream global-dedup two-clean identity mismatch",
    )
    require(
        receipt.get("next100_input_identity_sha256") == next100_identity,
        "execution receipt next100-input identity mismatch",
    )
    require(
        receipt.get("family_vector_identity_sha256")
        == balance_binding.get("family_vector_identity_sha256"),
        "execution receipt family-vector identity mismatch",
    )
    require(
        receipt.get("balance_policy_identity_sha256")
        == balance_binding.get("balance_policy_identity_sha256")
        == balance_result.get("policy_identity_sha256"),
        "execution receipt balance-policy identity mismatch",
    )
    require(
        receipt.get("balance_status")
        == balance_binding.get("balance_status")
        == balance_result.get("status"),
        "execution receipt balance status mismatch",
    )
    for field in (
        "maximum_feasible_total_source_bytes",
        "raw_capacity_by_stratum",
        "raw_gap_to_target_by_stratum",
        "family_minimum",
    ):
        require(
            receipt.get(field) == balance_result.get(field),
            f"execution receipt balance summary mismatch: {field}",
        )
    expected_next_gate = (
        "CLUSTER_SAFE_SPLIT_AND_DETERMINISTIC_PACK"
        if balance_result.get("status") == "TARGET_20M_SOURCE_MIX_FEASIBLE"
        else "ACQUIRE_MORE_DIVERSE_LAWFUL_SOURCE_CAPACITY"
    )
    require(
        receipt.get("next_scientific_gate") == expected_next_gate,
        "execution receipt next scientific gate drift",
    )
    require(
        receipt.get("balance_result_identity_sha256") == result_identity,
        "execution receipt balance-result identity mismatch",
    )
    require(
        receipt.get("balance_binding_identity_sha256") == binding_identity,
        "execution receipt balance-binding identity mismatch",
    )
    for field, expected in ZERO_CREDIT.items():
        require(
            receipt.get(field) == expected,
            f"execution receipt zero-credit drift: {field}",
        )

    if enforce_checkout_provenance:
        verify_dependency_blobs()
        verify_source_head(str(receipt["execution_head_sha"]))
        gate, _bridge, policy = load_canonical_authorities()
        gate.validate_vector(next100_input)
        replayed_result = gate.evaluate(policy, next100_input)
        require(
            canonical(replayed_result) == canonical(balance_result),
            "balance result is not a deterministic replay of canonical gate",
        )

    core: dict[str, Any] = {
        "schema": REPEAT_SCHEMA,
        "execution_head_sha": receipt["execution_head_sha"],
        "checkout_provenance_verified": enforce_checkout_provenance,
        "terminal_verdict": (
            "PASS" if enforce_checkout_provenance else "UNVERIFIED_TEST_ONLY"
        ),
        "fresh_process_count": 2,
        "independent_runner_jobs": True,
        "runner_instance_identities": {
            "a": runner_a_identity,
            "b": runner_b_identity,
        },
        "byte_identical_outputs": True,
        "output_file_sha256": hashes,
        "receipt_identity_sha256": receipt_identity,
        "post_g06_two_clean_proof_identity_sha256": post_g06_two_clean_id,
        "next100_input_identity_sha256": next100_identity,
        "balance_result_identity_sha256": receipt[
            "balance_result_identity_sha256"
        ],
        "balance_binding_identity_sha256": receipt[
            "balance_binding_identity_sha256"
        ],
        **ZERO_CREDIT,
    }
    proof = {**core, "proof_identity_sha256": sha256(canonical(core))}
    write_immutable_bytes(
        proof_path,
        canonical_line(proof),
        label="repeat proof",
    )
    return proof


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    sub = result.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", allow_abbrev=False)
    run.add_argument("--source-git-sha", required=True)
    run.add_argument("--family-vector", type=Path, required=True)
    run.add_argument("--expected-family-vector-file-sha256", required=True)
    run.add_argument("--expected-family-vector-identity-sha256", required=True)
    run.add_argument("--post-g06-evidence", type=Path, required=True)
    run.add_argument("--expected-post-g06-evidence-file-sha256", required=True)
    run.add_argument("--expected-post-g06-evidence-identity-sha256", required=True)
    run.add_argument("--post-g06-two-clean-proof", type=Path, required=True)
    run.add_argument(
        "--expected-post-g06-two-clean-proof-file-sha256",
        required=True,
    )
    run.add_argument(
        "--expected-post-g06-two-clean-proof-identity-sha256",
        required=True,
    )
    run.add_argument("--upstream-global-dedup-evidence", type=Path, required=True)
    run.add_argument("--upstream-global-dedup-two-clean", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)

    compare = sub.add_parser("compare", allow_abbrev=False)
    compare.add_argument("--output-a", type=Path, required=True)
    compare.add_argument("--output-b", type=Path, required=True)
    compare.add_argument("--proof", type=Path, required=True)
    compare.add_argument("--runner-a-identity", required=True)
    compare.add_argument("--runner-b-identity", required=True)
    compare.add_argument(
        "--independent-runner-jobs-attested",
        action="store_true",
    )
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        if args.command == "run":
            values = execute(args)
            write_output_dir(args.output_dir, values)
            print(
                "D03_RADA_POSTG06_NEXT100_BALANCE=PASS_ZERO_CREDIT "
                + values["execution-receipt"]["balance_status"]
            )
        else:
            proof = compare_outputs(
                args.output_a,
                args.output_b,
                args.proof,
                runner_a_identity=args.runner_a_identity,
                runner_b_identity=args.runner_b_identity,
                independent_runner_jobs=args.independent_runner_jobs_attested,
                enforce_checkout_provenance=True,
            )
            print(
                "D03_RADA_POSTG06_NEXT100_TWO_CLEAN=PASS "
                + proof["proof_identity_sha256"]
            )
    except (
        ImportError,
        OSError,
        RuntimeError,
        UnicodeError,
        ValueError,
    ) as exc:
        detail = " ".join(str(exc).split())[:500]
        print(f"D03_RADA_POSTG06_NEXT100_BALANCE=BLOCKED: {detail}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
