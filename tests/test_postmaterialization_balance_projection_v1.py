from __future__ import annotations

import copy
import hashlib
import json

import pytest

import tools.next100_106_balance_gate as next100_gate
from twelve_six.data.current_clean_balanced_selection_v1 import (
    SELECTION_REALIZATION_POLICY,
    _exact_record_subset,
    build_current_clean_balanced_selection,
    project_selected_current_clean_raw_records,
)
from twelve_six.data.postdecontam_balance_projection_v1 import ProjectionError
from twelve_six.data.postmaterialization_balance_projection_v1 import (
    BALANCE_BINDING_SCHEMA,
    CURRENT_CLEAN_BALANCE_BINDING_SCHEMA,
    CURRENT_CLEAN_FAMILY_VECTOR_SCHEMA,
    FAMILY_VECTOR_SCHEMA,
    _verify_current_clean_receipt,
    adapt_postmaterialization_family_vector_to_next100_106,
    build_balance_result_binding,
    build_current_clean_family_vector,
    build_postmaterialization_family_vector,
    load_strict_json_object,
    require_balanced_selection_ready,
    verify_postmaterialization_family_vector,
)

GIT_SHA = "2" * 40
MATERIALIZATION_SHA = "3" * 64
JSONL_SHA = "4" * 64
DEDUP_HEAD_SHA = "5" * 40
DEDUP_EVIDENCE_SHA = "6" * 64
DEDUP_WORKER = "NEXT100-065F-CURRENT-MAIN-GLOBAL-DEDUP-V8"

UA_A = "ua.kmu.portal.secretariat-news"
UA_B = "ua.verba.public-domain.nomis1864"
EN_A = "en.mdn.webdocs.prose"
EN_B = "en.project-gutenberg.public-domain-books"
CODE_A = "github:agronholm/anyio"
CODE_B = "github:pytest-dev/pytest"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _row(
    record_id: str,
    family: str,
    modality: str,
    payload_bytes: int,
    *,
    source_id: str | None = None,
) -> dict:
    return {
        "record_id": record_id,
        "source_id": source_id or record_id,
        "family": family,
        "modality": modality,
        "payload_sha256": hashlib.sha256(record_id.encode()).hexdigest(),
        "payload_bytes": payload_bytes,
    }


def _inventory(rows: list[dict]) -> dict:
    rows = sorted(copy.deepcopy(rows), key=lambda item: item["record_id"])
    payload_projection = [
        {
            "record_id": row["record_id"],
            "payload_sha256": row["payload_sha256"],
            "payload_bytes": row["payload_bytes"],
        }
        for row in rows
    ]
    return {
        "schema_version": "12-6.data526-record-inventory.v1",
        "record_count": len(rows),
        "total_payload_bytes": sum(row["payload_bytes"] for row in rows),
        "record_inventory_digest_sha256": hashlib.sha256(
            _canonical(rows)
        ).hexdigest(),
        "payload_inventory_digest_sha256": hashlib.sha256(
            _canonical(payload_projection)
        ).hexdigest(),
        "records": rows,
    }


def _truth() -> dict:
    return {
        "authorized_optimized_target_exposure": 0,
        "authorized_unique_loss_positions": 0,
        "current_corpus_eligible": False,
        "whole_corpus_external_llm_cleanliness_claimed": False,
        "final_test_outcomes_read": False,
        "foreign_pretrained_weights": False,
        "learned_weights_created": False,
        "optimizer_updates_executed_on_real_targets": 0,
        "paid_compute_used": False,
        "tokenizer_fit_authorized": False,
        "training_authorized_bytes": 0,
        "training_executed": False,
    }


def _evidence(inventory: dict) -> dict:
    return {
        "schema_version": "12-6.d03-post-g05-g06-materialization.v1",
        "status": "MATERIALIZED_ZERO_CREDIT",
        "execution_profile": "LOCAL_FREE",
        "execution_head_sha": GIT_SHA,
        "materialization_identity_sha256": MATERIALIZATION_SHA,
        "materializer_implementation_git_blob_sha1": "7" * 40,
        "repeat_materialization_byte_identical": True,
        "remaining_materialization_blockers": [],
        "input": {},
        "result": {
            "record_payload_jsonl_sha256": JSONL_SHA,
            "record_count": inventory["record_count"],
            "total_payload_bytes": inventory["total_payload_bytes"],
            "source_object_count": len(
                {row["source_id"] for row in inventory["records"]}
            ),
            "record_inventory_digest_sha256": inventory[
                "record_inventory_digest_sha256"
            ],
            "payload_inventory_digest_sha256": inventory[
                "payload_inventory_digest_sha256"
            ],
        },
        "transform": {},
        "consumed_blockers": [],
        "truth_boundary": _truth(),
    }


def _build(rows: list[dict]) -> tuple[dict, dict, dict]:
    inventory = _inventory(rows)
    evidence = _evidence(inventory)
    vector = build_postmaterialization_family_vector(
        inventory=inventory,
        materialization_evidence=evidence,
        expected_execution_head_sha=GIT_SHA,
        expected_materialization_identity_sha256=MATERIALIZATION_SHA,
        expected_result_jsonl_sha256=JSONL_SHA,
        expected_record_count=inventory["record_count"],
        expected_total_payload_bytes=inventory["total_payload_bytes"],
        expected_source_object_count=len(
            {row["source_id"] for row in inventory["records"]}
        ),
        expected_record_inventory_digest_sha256=inventory[
            "record_inventory_digest_sha256"
        ],
        expected_payload_inventory_digest_sha256=inventory[
            "payload_inventory_digest_sha256"
        ],
        source_git_sha=GIT_SHA,
    )
    return vector, inventory, evidence


def _dedup() -> dict:
    return {
        "worker_id": DEDUP_WORKER,
        "head_sha": DEDUP_HEAD_SHA,
        "evidence_identity_sha256": DEDUP_EVIDENCE_SHA,
        "terminal_verdict": "PASS",
    }


def _adapt(vector: dict) -> dict:
    return adapt_postmaterialization_family_vector_to_next100_106(
        vector,
        expected_family_vector_identity_sha256=vector[
            "family_vector_identity_sha256"
        ],
        dedup_authority=_dedup(),
        expected_dedup_worker_id=DEDUP_WORKER,
        expected_dedup_head_sha=DEDUP_HEAD_SHA,
        expected_dedup_evidence_identity_sha256=DEDUP_EVIDENCE_SHA,
    )


def _partial_rows() -> list[dict]:
    return [
        _row("ua-a", UA_A, "uk", 1_000),
        _row("ua-b", UA_B, "uk", 1_000),
        _row("en-a", EN_A, "en", 1_000),
        _row("en-b", EN_B, "en", 1_000),
        _row("code-a", CODE_A, "code", 1_000),
        _row("code-b", CODE_B, "code", 1_000),
    ]


def _target_rows() -> list[dict]:
    return [
        _row("ua-a", UA_A, "uk", 4_500_000),
        _row("ua-b", UA_B, "uk", 4_500_000),
        _row("en-a", EN_A, "en", 3_500_000),
        _row("en-b", EN_B, "en", 3_500_000),
        _row("code-a", CODE_A, "code", 2_000_000),
        _row("code-b", CODE_B, "code", 2_000_000),
    ]


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("source_object_count", True, "source_object_count"),
        ("source_object_count", 6.0, "source_object_count"),
        ("record_count", 6.0, "record_count"),
        ("total_payload_bytes", 6000.0, "total_payload_bytes"),
        ("stratum_capacity_bytes.code", 2000.0, "stratum capacity"),
        ("stratum_capacity_bytes.en", True, "stratum capacity"),
        ("stratum_family_counts.uk", 2.0, "stratum family-count"),
        ("stratum_family_counts.code", True, "stratum family-count"),
    ],
)
def test_family_vector_rejects_self_resealed_scalar_type_aliases(
    field: str, replacement: object, message: str,
) -> None:
    vector, _inventory, _evidence_doc = _build(_partial_rows())
    if "." in field:
        name, stratum = field.split(".", 1)
        vector[name][stratum] = replacement
    else:
        vector[field] = replacement
    vector["family_vector_identity_sha256"] = hashlib.sha256(
        _canonical(
            {key: value for key, value in vector.items()
             if key != "family_vector_identity_sha256"}
        )
    ).hexdigest()
    with pytest.raises(ProjectionError, match=message):
        verify_postmaterialization_family_vector(
            vector,
            expected_identity_sha256=vector["family_vector_identity_sha256"],
        )


def test_postmaterialization_vector_binds_machine_inventory() -> None:
    vector, inventory, evidence = _build(_partial_rows())
    assert vector["schema"] == FAMILY_VECTOR_SCHEMA
    assert vector["record_count"] == 6
    assert vector["total_payload_bytes"] == 6_000
    assert vector["source_object_count"] == 6
    assert vector["record_inventory_digest_sha256"] == inventory[
        "record_inventory_digest_sha256"
    ]
    assert evidence["truth_boundary"][
        "whole_corpus_external_llm_cleanliness_claimed"
    ] is False
    assert (
        "external_llm_or_api_used_for_data_or_intelligence"
        not in evidence["truth_boundary"]
    )
    assert vector["stratum_capacity_bytes"] == {
        "code": 2_000,
        "en": 2_000,
        "uk": 2_000,
    }
    assert verify_postmaterialization_family_vector(
        vector,
        expected_identity_sha256=vector["family_vector_identity_sha256"],
    ) == vector["family_vector_identity_sha256"]


def test_inventory_rows_are_rehashed_not_trusted() -> None:
    _, inventory, evidence = _build(_partial_rows())
    tampered = copy.deepcopy(inventory)
    tampered["records"][0]["payload_bytes"] += 1
    with pytest.raises(ProjectionError, match="inventory .* drift"):
        build_postmaterialization_family_vector(
            inventory=tampered,
            materialization_evidence=evidence,
            expected_execution_head_sha=GIT_SHA,
            expected_materialization_identity_sha256=MATERIALIZATION_SHA,
            expected_result_jsonl_sha256=JSONL_SHA,
            expected_record_count=inventory["record_count"],
            expected_total_payload_bytes=inventory["total_payload_bytes"],
            expected_source_object_count=6,
            expected_record_inventory_digest_sha256=inventory[
                "record_inventory_digest_sha256"
            ],
            expected_payload_inventory_digest_sha256=inventory[
                "payload_inventory_digest_sha256"
            ],
            source_git_sha=GIT_SHA,
        )


def test_coherently_resealed_wrong_inventory_still_fails_external_roots() -> None:
    _, inventory, _ = _build(_partial_rows())
    tampered = copy.deepcopy(inventory)
    tampered["records"][0]["payload_bytes"] += 1
    tampered = _inventory(tampered["records"])
    forged_evidence = _evidence(tampered)
    with pytest.raises(ProjectionError, match="independently expected value"):
        build_postmaterialization_family_vector(
            inventory=tampered,
            materialization_evidence=forged_evidence,
            expected_execution_head_sha=GIT_SHA,
            expected_materialization_identity_sha256=MATERIALIZATION_SHA,
            expected_result_jsonl_sha256=JSONL_SHA,
            expected_record_count=inventory["record_count"],
            expected_total_payload_bytes=inventory["total_payload_bytes"],
            expected_source_object_count=6,
            expected_record_inventory_digest_sha256=inventory[
                "record_inventory_digest_sha256"
            ],
            expected_payload_inventory_digest_sha256=inventory[
                "payload_inventory_digest_sha256"
            ],
            source_git_sha=GIT_SHA,
        )


def test_materialization_truth_drift_fails_closed() -> None:
    _, inventory, evidence = _build(_partial_rows())
    evidence["truth_boundary"]["tokenizer_fit_authorized"] = True
    with pytest.raises(ProjectionError, match="truth boundary drift"):
        build_postmaterialization_family_vector(
            inventory=inventory,
            materialization_evidence=evidence,
            expected_execution_head_sha=GIT_SHA,
            expected_materialization_identity_sha256=MATERIALIZATION_SHA,
            expected_result_jsonl_sha256=JSONL_SHA,
            expected_record_count=inventory["record_count"],
            expected_total_payload_bytes=inventory["total_payload_bytes"],
            expected_source_object_count=6,
            expected_record_inventory_digest_sha256=inventory[
                "record_inventory_digest_sha256"
            ],
            expected_payload_inventory_digest_sha256=inventory[
                "payload_inventory_digest_sha256"
            ],
            source_git_sha=GIT_SHA,
        )


def test_legacy_global_external_llm_negative_is_rejected() -> None:
    _, inventory, evidence = _build(_partial_rows())
    del evidence["truth_boundary"]["whole_corpus_external_llm_cleanliness_claimed"]
    evidence["truth_boundary"][
        "external_llm_or_api_used_for_data_or_intelligence"
    ] = False
    with pytest.raises(ProjectionError, match="truth boundary drift"):
        build_postmaterialization_family_vector(
            inventory=inventory,
            materialization_evidence=evidence,
            expected_execution_head_sha=GIT_SHA,
            expected_materialization_identity_sha256=MATERIALIZATION_SHA,
            expected_result_jsonl_sha256=JSONL_SHA,
            expected_record_count=inventory["record_count"],
            expected_total_payload_bytes=inventory["total_payload_bytes"],
            expected_source_object_count=6,
            expected_record_inventory_digest_sha256=inventory[
                "record_inventory_digest_sha256"
            ],
            expected_payload_inventory_digest_sha256=inventory[
                "payload_inventory_digest_sha256"
            ],
            source_git_sha=GIT_SHA,
        )


def test_positive_whole_corpus_external_llm_cleanliness_claim_is_rejected() -> None:
    _, inventory, evidence = _build(_partial_rows())
    evidence["truth_boundary"]["whole_corpus_external_llm_cleanliness_claimed"] = True
    with pytest.raises(ProjectionError, match="truth boundary drift"):
        build_postmaterialization_family_vector(
            inventory=inventory,
            materialization_evidence=evidence,
            expected_execution_head_sha=GIT_SHA,
            expected_materialization_identity_sha256=MATERIALIZATION_SHA,
            expected_result_jsonl_sha256=JSONL_SHA,
            expected_record_count=inventory["record_count"],
            expected_total_payload_bytes=inventory["total_payload_bytes"],
            expected_source_object_count=6,
            expected_record_inventory_digest_sha256=inventory[
                "record_inventory_digest_sha256"
            ],
            expected_payload_inventory_digest_sha256=inventory[
                "payload_inventory_digest_sha256"
            ],
            source_git_sha=GIT_SHA,
        )


def test_adapter_preserves_physical_authority_for_external_binding() -> None:
    vector, _, _ = _build(_partial_rows())
    adapted = _adapt(vector)
    next100_gate.validate_vector(adapted)
    assert adapted["physical_authority"]["family_vector_identity_sha256"] == vector[
        "family_vector_identity_sha256"
    ]
    assert adapted["physical_authority"]["materialization_identity_sha256"] == (
        MATERIALIZATION_SHA
    )


@pytest.mark.parametrize(
    "mutation",
    ("total_unique_bytes_float", "stratum_bytes_float", "family_count_float"),
)
def test_partial_result_rejects_resealed_input_total_numeric_alias(
    mutation: str,
) -> None:
    vector, _, _ = _build(_partial_rows())
    adapted = _adapt(vector)
    policy = next100_gate.load_json(next100_gate.POLICY_PATH)
    balance = next100_gate.evaluate(policy, adapted)
    assert balance["status"] == "PARTIAL_MIX_FEASIBLE_ACQUIRE_MORE_DATA"
    altered = copy.deepcopy(balance)
    totals = altered["input_totals"]
    if mutation == "total_unique_bytes_float":
        totals["total_unique_bytes"] = float(totals["total_unique_bytes"])
    elif mutation == "stratum_bytes_float":
        totals["by_stratum"]["ua"] = float(totals["by_stratum"]["ua"])
    else:
        totals["family_count"]["en"] = float(totals["family_count"]["en"])
    assert altered["input_totals"] == balance["input_totals"]
    altered["result_identity_sha256"] = next100_gate.canonical_sha(
        altered, "result_identity_sha256"
    )
    with pytest.raises(ProjectionError, match="result totals differ"):
        build_balance_result_binding(
            family_vector=vector,
            expected_family_vector_identity_sha256=vector[
                "family_vector_identity_sha256"
            ],
            next100_input=adapted,
            balance_result=altered,
            expected_policy_identity_sha256=policy["policy_identity_sha256"],
            expected_result_identity_sha256=altered["result_identity_sha256"],
        )


@pytest.mark.parametrize("current_clean", (False, True))
@pytest.mark.parametrize(
    "mutation", ("extra_training_claim", "missing_next_step", "promoted_next_step")
)
def test_balance_result_rejects_resealed_schema_and_gate_claims(
    current_clean: bool, mutation: str,
) -> None:
    if current_clean:
        vector, _, _ = _build_current_clean(_partial_rows())
    else:
        vector, _, _ = _build(_partial_rows())
    adapted = _adapt(vector)
    policy = next100_gate.load_json(next100_gate.POLICY_PATH)
    balance = next100_gate.evaluate(policy, adapted)
    assert balance["status"] == "PARTIAL_MIX_FEASIBLE_ACQUIRE_MORE_DATA"
    altered = copy.deepcopy(balance)
    if mutation == "extra_training_claim":
        altered["training_ready"] = True
    elif mutation == "missing_next_step":
        del altered["next_step"]
    else:
        altered["next_step"] = "MODEL_TRAINING_AUTHORIZED"
    altered["result_identity_sha256"] = next100_gate.canonical_sha(
        altered, "result_identity_sha256"
    )
    with pytest.raises(
        ProjectionError,
        match="fields are not closed-world|cannot promote downstream gates",
    ):
        build_balance_result_binding(
            family_vector=vector,
            expected_family_vector_identity_sha256=vector[
                "family_vector_identity_sha256"
            ],
            next100_input=adapted,
            balance_result=altered,
            expected_policy_identity_sha256=policy["policy_identity_sha256"],
            expected_result_identity_sha256=altered["result_identity_sha256"],
        )


def test_partial_balance_is_bound_but_cannot_authorize_selection() -> None:
    vector, _, _ = _build(_partial_rows())
    adapted = _adapt(vector)
    policy = next100_gate.load_json(next100_gate.POLICY_PATH)
    balance = next100_gate.evaluate(policy, adapted)
    assert balance["status"] == "PARTIAL_MIX_FEASIBLE_ACQUIRE_MORE_DATA"

    binding = build_balance_result_binding(
        family_vector=vector,
        expected_family_vector_identity_sha256=vector[
            "family_vector_identity_sha256"
        ],
        next100_input=adapted,
        balance_result=balance,
        expected_policy_identity_sha256=policy["policy_identity_sha256"],
        expected_result_identity_sha256=balance["result_identity_sha256"],
    )
    assert binding["schema"] == BALANCE_BINDING_SCHEMA
    assert binding["balanced_selection_authorized"] is False
    with pytest.raises(
        ProjectionError,
        match="balanced selection blocked",
    ):
        require_balanced_selection_ready(
            binding,
            expected_binding_identity_sha256=binding[
                "binding_identity_sha256"
            ],
        )


def test_target_feasible_balance_can_cross_selection_readiness_gate() -> None:
    vector, _, _ = _build(_target_rows())
    adapted = _adapt(vector)
    policy = next100_gate.load_json(next100_gate.POLICY_PATH)
    balance = next100_gate.evaluate(policy, adapted)
    assert balance["status"] == "TARGET_20M_SOURCE_MIX_FEASIBLE"
    assert balance["maximum_feasible_total_source_bytes"] == 20_000_000

    binding = build_balance_result_binding(
        family_vector=vector,
        expected_family_vector_identity_sha256=vector[
            "family_vector_identity_sha256"
        ],
        next100_input=adapted,
        balance_result=balance,
        expected_policy_identity_sha256=policy["policy_identity_sha256"],
        expected_result_identity_sha256=balance["result_identity_sha256"],
    )
    require_balanced_selection_ready(
        binding,
        expected_binding_identity_sha256=binding["binding_identity_sha256"],
    )


def test_balance_result_substitution_fails_binding() -> None:
    vector, _, _ = _build(_partial_rows())
    adapted = _adapt(vector)
    policy = next100_gate.load_json(next100_gate.POLICY_PATH)
    balance = next100_gate.evaluate(policy, adapted)
    tampered = copy.deepcopy(balance)
    tampered["maximum_feasible_total_source_bytes"] += 100
    with pytest.raises(ProjectionError, match="self-hash mismatch"):
        build_balance_result_binding(
            family_vector=vector,
            expected_family_vector_identity_sha256=vector[
                "family_vector_identity_sha256"
            ],
            next100_input=adapted,
            balance_result=tampered,
            expected_policy_identity_sha256=policy["policy_identity_sha256"],
            expected_result_identity_sha256=balance["result_identity_sha256"],
        )


def _rebuild_documents(inventory: dict, evidence: dict) -> dict:
    expected = _inventory(inventory["records"])
    return build_postmaterialization_family_vector(
        inventory=inventory,
        materialization_evidence=evidence,
        expected_execution_head_sha=GIT_SHA,
        expected_materialization_identity_sha256=MATERIALIZATION_SHA,
        expected_result_jsonl_sha256=JSONL_SHA,
        expected_record_count=expected["record_count"],
        expected_total_payload_bytes=expected["total_payload_bytes"],
        expected_source_object_count=len(
            {row["source_id"] for row in expected["records"]}
        ),
        expected_record_inventory_digest_sha256=expected[
            "record_inventory_digest_sha256"
        ],
        expected_payload_inventory_digest_sha256=expected[
            "payload_inventory_digest_sha256"
        ],
        source_git_sha=GIT_SHA,
    )


@pytest.mark.parametrize(
    ("field", "alias"),
    [
        ("whole_corpus_external_llm_cleanliness_claimed", 0),
        ("authorized_optimized_target_exposure", False),
        ("authorized_optimized_target_exposure", 0.0),
    ],
)
def test_zero_credit_truth_rejects_scalar_type_aliases(
    field: str,
    alias: object,
) -> None:
    _, inventory, evidence = _build(_partial_rows())
    evidence = copy.deepcopy(evidence)
    evidence["truth_boundary"][field] = alias
    with pytest.raises(ProjectionError, match="truth boundary drift"):
        _rebuild_documents(inventory, evidence)


@pytest.mark.parametrize(
    "field",
    ["record_count", "total_payload_bytes", "source_object_count"],
)
def test_materialization_result_rejects_integer_to_float_aliases(field: str) -> None:
    _, inventory, evidence = _build(_partial_rows())
    evidence = copy.deepcopy(evidence)
    evidence["result"][field] = float(evidence["result"][field])
    with pytest.raises(
        ProjectionError,
        match=rf"materialization result\.{field} does not match independently expected value",
    ):
        _rebuild_documents(inventory, evidence)


@pytest.mark.parametrize("field", ["record_count", "total_payload_bytes"])
def test_inventory_totals_reject_integer_to_float_aliases(field: str) -> None:
    _, inventory, evidence = _build(_partial_rows())
    inventory = copy.deepcopy(inventory)
    inventory[field] = float(inventory[field])
    with pytest.raises(
        ProjectionError,
        match=rf"post-materialization inventory {field} drift",
    ):
        _rebuild_documents(inventory, evidence)


def _current_clean_bytes(
    rows: list[dict],
) -> tuple[dict, dict[str, bytes], dict[str, object]]:
    survivors: list[dict[str, str]] = []
    inventory_rows: list[dict] = []
    for index, row in enumerate(rows):
        payload = chr(ord("a") + index) * row["payload_bytes"]
        survivor = {
            "record_id": row["record_id"],
            "source_id": row["source_id"],
            "family": row["family"],
            "modality": row["modality"],
            "normalized_payload": payload,
        }
        survivors.append(survivor)
        payload_raw = payload.encode()
        inventory_rows.append(
            {
                "record_id": row["record_id"],
                "source_id": row["source_id"],
                "family": row["family"],
                "modality": row["modality"],
                "payload_sha256": hashlib.sha256(payload_raw).hexdigest(),
                "payload_bytes": len(payload_raw),
            }
        )
    inventory_rows.sort(key=lambda row: row["record_id"])
    payload_projection = [
        {
            "record_id": row["record_id"],
            "payload_sha256": row["payload_sha256"],
            "payload_bytes": row["payload_bytes"],
        }
        for row in inventory_rows
    ]
    inventory = {
        "schema_version": "12-6.data526-record-inventory.v1",
        "record_count": len(inventory_rows),
        "total_payload_bytes": sum(row["payload_bytes"] for row in inventory_rows),
        "record_inventory_digest_sha256": hashlib.sha256(
            _canonical(inventory_rows)
        ).hexdigest(),
        "payload_inventory_digest_sha256": hashlib.sha256(
            _canonical(payload_projection)
        ).hexdigest(),
        "records": inventory_rows,
    }
    survivor_raw = b"".join(_canonical(record) + b"\n" for record in survivors)
    survivor_sha = hashlib.sha256(survivor_raw).hexdigest()
    source_objects = len({row["source_id"] for row in inventory["records"]})
    count = inventory["record_count"]
    receipt: dict[str, object] = {
        "schema_version": "12-6.current-clean-decontam-quality-privacy-survivor.v2",
        "status": "CLEAN_SURVIVOR_MATERIALIZED_PENDING_INDEPENDENT_QUALIFICATION",
        "clean_training_records_sha256": "1" * 64,
        "clean_training_handoff_sha256": "2" * 64,
        "data232_report_sha256": "3" * 64,
        "decontamination_execution_identity_sha256": "4" * 64,
        "eval647_execution_receipt_identity_sha256": "5" * 64,
        "quality_execution_identity_sha256": "6" * 64,
        "privacy_execution_identity_sha256": "7" * 64,
        "post_decontamination_input_rows_sha256": "8" * 64,
        "post_quality_input_rows_sha256": "9" * 64,
        "survivor_jsonl_sha256": survivor_sha,
        "survivor_record_inventory_digest_sha256": inventory[
            "record_inventory_digest_sha256"
        ],
        "survivor_payload_inventory_digest_sha256": inventory[
            "payload_inventory_digest_sha256"
        ],
        "input_training_records": count,
        "post_decontamination_records": count,
        "post_quality_records": count,
        "survivor_records": count,
        "survivor_payload_bytes": inventory["total_payload_bytes"],
        "survivor_source_objects": source_objects,
        "rejection_counts": {},
        "privacy_detector_counts": {},
        "dependency_git_blobs": {"src/twelve_six/data/example.py": "a" * 40},
        "durable_evidence_hash_only": True,
        "terminal_post_g05_g06_authority": False,
        "independent_qualification_required": True,
        "current_corpus_launch_authority_promoted": False,
        "authorized_optimized_target_exposure": 0,
        "tokenizer_fit_authorized": False,
        "optimizer_updates_executed_on_real_targets": 0,
        "training_executed": False,
        "learned_weights_created": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
        "foreign_pretrained_weights": False,
    }
    receipt["receipt_identity_sha256"] = hashlib.sha256(
        _canonical(receipt) + b"\n"
    ).hexdigest()
    receipt_raw = _canonical(receipt) + b"\n"
    inventory_raw = _canonical(inventory) + b"\n"
    roots = {
        "composition_receipt.json": hashlib.sha256(receipt_raw).hexdigest(),
        "data232_report.json": "b" * 64,
        "decontamination_execution.json": "c" * 64,
        "eval647_execution_receipt.json": "d" * 64,
        "privacy_execution.json": "e" * 64,
        "quality_execution.json": "f" * 64,
        "survivor_inventory.json": hashlib.sha256(inventory_raw).hexdigest(),
        "survivor_records.jsonl": survivor_sha,
    }
    repeat: dict[str, object] = {
        "schema_version": "12-6.d03-current-clean-composition-physical-repeat.v1",
        "status": "PHYSICAL_EXECUTION_COMPLETE_PENDING_INDEPENDENT_QUALIFICATION",
        "execution_head_sha": GIT_SHA,
        "execution_profile": "LOCAL_FREE",
        "output_files_sha256": roots,
        "survivor_jsonl_sha256": survivor_sha,
        "survivor_record_inventory_digest_sha256": inventory[
            "record_inventory_digest_sha256"
        ],
        "survivor_payload_inventory_digest_sha256": inventory[
            "payload_inventory_digest_sha256"
        ],
        "survivor_records": count,
        "survivor_payload_bytes": inventory["total_payload_bytes"],
        "two_fresh_executions_byte_identical": True,
        "terminal_post_g05_g06_authority": False,
        "independent_qualification_required": True,
        "authorized_optimized_target_exposure": 0,
        "tokenizer_fit_authorized": False,
        "optimizer_updates_executed_on_real_targets": 0,
        "training_executed": False,
        "learned_weights_created": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
        "foreign_pretrained_weights": False,
    }
    repeat["proof_identity_sha256"] = hashlib.sha256(_canonical(repeat)).hexdigest()
    repeat_raw = _canonical(repeat) + b"\n"
    raw = {
        "composition_receipt": receipt_raw,
        "survivor_inventory": inventory_raw,
        "repeat_proof": repeat_raw,
        "survivor_records": survivor_raw,
    }
    expected = {
        "expected_composition_receipt_json_sha256": hashlib.sha256(
            receipt_raw
        ).hexdigest(),
        "expected_survivor_inventory_json_sha256": hashlib.sha256(
            inventory_raw
        ).hexdigest(),
        "expected_repeat_proof_json_sha256": hashlib.sha256(repeat_raw).hexdigest(),
        "expected_survivor_records_jsonl_sha256": survivor_sha,
        "expected_receipt_identity_sha256": receipt["receipt_identity_sha256"],
        "expected_repeat_proof_identity_sha256": repeat["proof_identity_sha256"],
        "expected_execution_head_sha": GIT_SHA,
        "expected_record_count": count,
        "expected_total_payload_bytes": inventory["total_payload_bytes"],
        "expected_source_object_count": source_objects,
        "expected_record_inventory_digest_sha256": inventory[
            "record_inventory_digest_sha256"
        ],
        "expected_payload_inventory_digest_sha256": inventory[
            "payload_inventory_digest_sha256"
        ],
        "source_git_sha": GIT_SHA,
    }
    return inventory, raw, expected


def _build_current_clean(rows: list[dict]) -> tuple[dict, dict[str, bytes], dict[str, object]]:
    _, raw, expected = _current_clean_bytes(rows)
    vector = build_current_clean_family_vector(
        composition_receipt_raw=raw["composition_receipt"],
        survivor_inventory_raw=raw["survivor_inventory"],
        repeat_proof_raw=raw["repeat_proof"],
        survivor_records_raw=raw["survivor_records"],
        **expected,
    )
    return vector, raw, expected


def _reseal(document: dict, identity_field: str) -> bytes:
    document = copy.deepcopy(document)
    document.pop(identity_field, None)
    identity_bytes = _canonical(document)
    if identity_field == "receipt_identity_sha256":
        identity_bytes += b"\n"
    document[identity_field] = hashlib.sha256(identity_bytes).hexdigest()
    return _canonical(document) + b"\n"


@pytest.mark.parametrize("current_clean", (False, True))
@pytest.mark.parametrize(
    "field", ("record_count", "total_payload_bytes", "source_object_count")
)
def test_next100_binding_rejects_physical_authority_numeric_alias(
    current_clean: bool, field: str,
) -> None:
    if current_clean:
        vector, _, _ = _build_current_clean(_partial_rows())
    else:
        vector, _, _ = _build(_partial_rows())
    adapted = _adapt(vector)
    policy = next100_gate.load_json(next100_gate.POLICY_PATH)
    balance = next100_gate.evaluate(policy, adapted)
    assert balance["status"] == "PARTIAL_MIX_FEASIBLE_ACQUIRE_MORE_DATA"
    altered = copy.deepcopy(adapted)
    physical = altered["physical_authority"]
    physical[field] = float(physical[field])
    assert altered == adapted
    with pytest.raises(ProjectionError, match="physical authority mismatch"):
        build_balance_result_binding(
            family_vector=vector,
            expected_family_vector_identity_sha256=vector[
                "family_vector_identity_sha256"
            ],
            next100_input=altered,
            balance_result=balance,
            expected_policy_identity_sha256=policy["policy_identity_sha256"],
            expected_result_identity_sha256=balance["result_identity_sha256"],
        )


@pytest.mark.parametrize("current_clean", (False, True))
@pytest.mark.parametrize("mutation", ("extra_training_claim", "missing_next_gate"))
def test_family_vector_rejects_self_resealed_schema_extensions(
    current_clean: bool, mutation: str,
) -> None:
    if current_clean:
        vector, _, _ = _build_current_clean(_partial_rows())
    else:
        vector, _, _ = _build(_partial_rows())
    assert verify_postmaterialization_family_vector(
        vector, expected_identity_sha256=vector["family_vector_identity_sha256"]
    ) == vector["family_vector_identity_sha256"]
    altered = copy.deepcopy(vector)
    if mutation == "extra_training_claim":
        altered["additional_training_claim"] = True
    else:
        del altered["next_gate"]
    altered["family_vector_identity_sha256"] = hashlib.sha256(
        _canonical(
            {
                key: value
                for key, value in altered.items()
                if key != "family_vector_identity_sha256"
            }
        )
    ).hexdigest()
    with pytest.raises(ProjectionError, match="fields are not closed-world"):
        verify_postmaterialization_family_vector(
            altered,
            expected_identity_sha256=altered["family_vector_identity_sha256"],
        )


def test_current_clean_vector_binds_receipt_repeat_and_raw_files() -> None:
    vector, raw, expected = _build_current_clean(_partial_rows())
    assert vector["schema"] == CURRENT_CLEAN_FAMILY_VECTOR_SCHEMA
    assert vector["current_clean_receipt_identity_sha256"] == (
        expected["expected_receipt_identity_sha256"]
    )
    assert vector["current_clean_repeat_proof_identity_sha256"] == (
        expected["expected_repeat_proof_identity_sha256"]
    )
    assert vector["composition_receipt_json_sha256"] == hashlib.sha256(
        raw["composition_receipt"]
    ).hexdigest()
    assert vector["repeat_proof_json_sha256"] == hashlib.sha256(
        raw["repeat_proof"]
    ).hexdigest()
    assert verify_postmaterialization_family_vector(
        vector,
        expected_identity_sha256=vector["family_vector_identity_sha256"],
    ) == vector["family_vector_identity_sha256"]


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity", "1e400"])
def test_strict_current_clean_json_rejects_nonfinite_numbers(literal: str) -> None:
    with pytest.raises(ProjectionError):
        load_strict_json_object(
            ('{"outer":{"value":' + literal + "}}").encode(),
            label="adversarial",
        )


@pytest.mark.parametrize("literal", ["1e-9999", "-1e-9999", "5.4e-9999"])
def test_strict_current_clean_json_rejects_nonzero_underflow(literal: str) -> None:
    with pytest.raises(ProjectionError, match="underflowed"):
        load_strict_json_object(
            ('{"value":' + literal + "}").encode(),
            label="adversarial",
        )


@pytest.mark.parametrize("literal", ["0e-9999", "-0.000e-9999", "0.0", "1.25"])
def test_strict_current_clean_json_keeps_valid_finite_float(literal: str) -> None:
    decoded = load_strict_json_object(
        ('{"value":' + literal + "}").encode(),
        label="valid",
    )
    assert decoded["value"] == float(literal)


@pytest.mark.parametrize(
    "raw",
    (
        b'{"value":"\\ud800"}',
        b'{"value":"\\udc00"}',
        b'{"\\ud800":"bad-key"}',
        b'{"nested":{"value":["\\udfff"]}}',
    ),
)
def test_strict_current_clean_json_rejects_unpaired_surrogates(raw: bytes) -> None:
    with pytest.raises(ProjectionError, match="invalid Unicode JSON"):
        load_strict_json_object(raw, label="adversarial")


def test_strict_current_clean_json_accepts_valid_surrogate_pair() -> None:
    decoded = load_strict_json_object(
        b'{"value":"\\ud83d\\ude00"}',
        label="valid",
    )
    assert decoded["value"] == "\U0001f600"


def test_strict_current_clean_json_rejects_nested_duplicate_keys() -> None:
    with pytest.raises(ProjectionError):
        load_strict_json_object(
            b'{"outer":{"identity":"a","identity":"b"}}',
            label="adversarial",
        )


@pytest.mark.parametrize("container", ["array", "object"])
def test_strict_current_clean_json_rejects_deeply_nested_input(container: str) -> None:
    opening, closing = (b"[", b"]") if container == "array" else (b'{"x":', b"}")
    raw = b'{"v":' + opening * 12_000 + b"0" + closing * 12_000 + b"}"
    with pytest.raises(ProjectionError, match="not strict JSON"):
        load_strict_json_object(raw, label="adversarial")


def test_strict_current_clean_json_rejects_oversized_integer() -> None:
    raw = b'{"v":' + b"9" * 10_000 + b"}"
    with pytest.raises(ProjectionError, match="not strict JSON"):
        load_strict_json_object(raw, label="adversarial")


def test_current_clean_rejects_raw_receipt_byte_substitution() -> None:
    _, raw, expected = _current_clean_bytes(_partial_rows())
    raw["composition_receipt"] += b" "
    with pytest.raises(ProjectionError):
        build_current_clean_family_vector(
            composition_receipt_raw=raw["composition_receipt"],
            survivor_inventory_raw=raw["survivor_inventory"],
            repeat_proof_raw=raw["repeat_proof"],
            survivor_records_raw=raw["survivor_records"],
            **expected,
        )


def test_current_clean_rejects_coherent_receipt_reseal_under_fixed_identity() -> None:
    _, raw, expected = _current_clean_bytes(_partial_rows())
    receipt = json.loads(raw["composition_receipt"])
    receipt["survivor_payload_bytes"] += 1
    raw["composition_receipt"] = _reseal(receipt, "receipt_identity_sha256")
    expected["expected_composition_receipt_json_sha256"] = hashlib.sha256(
        raw["composition_receipt"]
    ).hexdigest()
    with pytest.raises(ProjectionError, match="not independently expected"):
        build_current_clean_family_vector(
            composition_receipt_raw=raw["composition_receipt"],
            survivor_inventory_raw=raw["survivor_inventory"],
            repeat_proof_raw=raw["repeat_proof"],
            survivor_records_raw=raw["survivor_records"],
            **expected,
        )


def test_current_clean_rejects_repeat_raw_root_substitution_even_if_resealed() -> None:
    _, raw, expected = _current_clean_bytes(_partial_rows())
    repeat = json.loads(raw["repeat_proof"])
    repeat["output_files_sha256"]["survivor_inventory.json"] = "0" * 64
    raw["repeat_proof"] = _reseal(repeat, "proof_identity_sha256")
    resealed = json.loads(raw["repeat_proof"])
    expected["expected_repeat_proof_identity_sha256"] = resealed[
        "proof_identity_sha256"
    ]
    expected["expected_repeat_proof_json_sha256"] = hashlib.sha256(
        raw["repeat_proof"]
    ).hexdigest()
    with pytest.raises(ProjectionError, match="raw root"):
        build_current_clean_family_vector(
            composition_receipt_raw=raw["composition_receipt"],
            survivor_inventory_raw=raw["survivor_inventory"],
            repeat_proof_raw=raw["repeat_proof"],
            survivor_records_raw=raw["survivor_records"],
            **expected,
        )


def test_current_clean_rejects_source_object_count_drift() -> None:
    _, raw, expected = _current_clean_bytes(_partial_rows())
    receipt = json.loads(raw["composition_receipt"])
    receipt["survivor_source_objects"] -= 1
    raw["composition_receipt"] = _reseal(receipt, "receipt_identity_sha256")
    resealed = json.loads(raw["composition_receipt"])
    expected["expected_receipt_identity_sha256"] = resealed[
        "receipt_identity_sha256"
    ]
    expected["expected_composition_receipt_json_sha256"] = hashlib.sha256(
        raw["composition_receipt"]
    ).hexdigest()

    # Keep the upstream repeat proof coherently bound to the resealed receipt so
    # this adversarial fixture reaches the intended independent inventory-count
    # check instead of failing earlier at the stricter raw-file cross-bind.
    repeat = json.loads(raw["repeat_proof"])
    repeat["output_files_sha256"]["composition_receipt.json"] = expected[
        "expected_composition_receipt_json_sha256"
    ]
    raw["repeat_proof"] = _reseal(repeat, "proof_identity_sha256")
    resealed_repeat = json.loads(raw["repeat_proof"])
    expected["expected_repeat_proof_identity_sha256"] = resealed_repeat[
        "proof_identity_sha256"
    ]
    expected["expected_repeat_proof_json_sha256"] = hashlib.sha256(
        raw["repeat_proof"]
    ).hexdigest()

    expected["expected_source_object_count"] = receipt["survivor_source_objects"]
    with pytest.raises(ProjectionError, match="source_object_count"):
        build_current_clean_family_vector(
            composition_receipt_raw=raw["composition_receipt"],
            survivor_inventory_raw=raw["survivor_inventory"],
            repeat_proof_raw=raw["repeat_proof"],
            survivor_records_raw=raw["survivor_records"],
            **expected,
        )


def test_current_clean_rejects_boolean_alias_for_zero_exposure() -> None:
    _, raw, expected = _current_clean_bytes(_partial_rows())
    receipt = json.loads(raw["composition_receipt"])
    receipt["authorized_optimized_target_exposure"] = False
    raw["composition_receipt"] = _reseal(receipt, "receipt_identity_sha256")
    resealed = json.loads(raw["composition_receipt"])
    expected["expected_receipt_identity_sha256"] = resealed[
        "receipt_identity_sha256"
    ]
    expected["expected_composition_receipt_json_sha256"] = hashlib.sha256(
        raw["composition_receipt"]
    ).hexdigest()
    with pytest.raises(ProjectionError, match="zero-credit"):
        build_current_clean_family_vector(
            composition_receipt_raw=raw["composition_receipt"],
            survivor_inventory_raw=raw["survivor_inventory"],
            repeat_proof_raw=raw["repeat_proof"],
            survivor_records_raw=raw["survivor_records"],
            **expected,
        )


def test_current_clean_rejects_legacy_schema_impersonation() -> None:
    _, raw, expected = _current_clean_bytes(_partial_rows())
    receipt = json.loads(raw["composition_receipt"])
    receipt["schema_version"] = "12-6.d03-post-g05-g06-materialization.v1"
    raw["composition_receipt"] = _reseal(receipt, "receipt_identity_sha256")
    resealed = json.loads(raw["composition_receipt"])
    expected["expected_receipt_identity_sha256"] = resealed[
        "receipt_identity_sha256"
    ]
    expected["expected_composition_receipt_json_sha256"] = hashlib.sha256(
        raw["composition_receipt"]
    ).hexdigest()
    with pytest.raises(ProjectionError, match="current-clean receipt schema"):
        build_current_clean_family_vector(
            composition_receipt_raw=raw["composition_receipt"],
            survivor_inventory_raw=raw["survivor_inventory"],
            repeat_proof_raw=raw["repeat_proof"],
            survivor_records_raw=raw["survivor_records"],
            **expected,
        )


def test_current_clean_balance_binding_preserves_explicit_physical_authority() -> None:
    vector, _, _ = _build_current_clean(_partial_rows())
    adapted = _adapt(vector)
    physical = adapted["physical_authority"]
    assert "materialization_identity_sha256" not in physical
    assert physical["current_clean_receipt_identity_sha256"] == vector[
        "current_clean_receipt_identity_sha256"
    ]
    assert physical["current_clean_repeat_proof_identity_sha256"] == vector[
        "current_clean_repeat_proof_identity_sha256"
    ]
    policy = next100_gate.load_json(next100_gate.POLICY_PATH)
    balance = next100_gate.evaluate(policy, adapted)
    binding = build_balance_result_binding(
        family_vector=vector,
        expected_family_vector_identity_sha256=vector[
            "family_vector_identity_sha256"
        ],
        next100_input=adapted,
        balance_result=balance,
        expected_policy_identity_sha256=policy["policy_identity_sha256"],
        expected_result_identity_sha256=balance["result_identity_sha256"],
    )
    assert binding["schema"] == CURRENT_CLEAN_BALANCE_BINDING_SCHEMA
    assert binding["current_clean_physical_authority"] == physical
    assert binding["authorized_optimized_target_exposure"] == 0


def test_current_clean_rebuilds_inventory_from_survivor_jsonl() -> None:
    inventory, raw, expected = _current_clean_bytes(_partial_rows())
    parsed = [
        json.loads(line)
        for line in raw["survivor_records"].decode("utf-8").splitlines()
    ]
    assert len(parsed) == inventory["record_count"]
    assert sum(len(row["normalized_payload"].encode()) for row in parsed) == (
        inventory["total_payload_bytes"]
    )
    vector = build_current_clean_family_vector(
        composition_receipt_raw=raw["composition_receipt"],
        survivor_inventory_raw=raw["survivor_inventory"],
        repeat_proof_raw=raw["repeat_proof"],
        survivor_records_raw=raw["survivor_records"],
        **expected,
    )
    assert vector["record_inventory_digest_sha256"] == inventory[
        "record_inventory_digest_sha256"
    ]


@pytest.mark.parametrize(
    "mutation",
    ("record_count_float", "total_bytes_float", "row_bytes_float"),
)
def test_current_clean_rejects_resealed_inventory_numeric_alias(
    mutation: str,
) -> None:
    original, raw, expected = _current_clean_bytes(_partial_rows())
    inventory = copy.deepcopy(original)
    if mutation == "record_count_float":
        inventory["record_count"] = float(inventory["record_count"])
    elif mutation == "total_bytes_float":
        inventory["total_payload_bytes"] = float(
            inventory["total_payload_bytes"]
        )
    else:
        first = inventory["records"][0]
        first["payload_bytes"] = float(first["payload_bytes"])
    assert inventory == original
    raw["survivor_inventory"] = _canonical(inventory) + b"\n"
    inventory_sha = hashlib.sha256(raw["survivor_inventory"]).hexdigest()
    repeat = json.loads(raw["repeat_proof"])
    repeat["output_files_sha256"]["survivor_inventory.json"] = inventory_sha
    raw["repeat_proof"] = _reseal(repeat, "proof_identity_sha256")
    expected["expected_survivor_inventory_json_sha256"] = inventory_sha
    expected["expected_repeat_proof_json_sha256"] = hashlib.sha256(
        raw["repeat_proof"]
    ).hexdigest()
    expected["expected_repeat_proof_identity_sha256"] = json.loads(
        raw["repeat_proof"]
    )["proof_identity_sha256"]
    with pytest.raises(
        ProjectionError, match="differs from rebuilt survivor JSONL"
    ):
        build_current_clean_family_vector(
            composition_receipt_raw=raw["composition_receipt"],
            survivor_inventory_raw=raw["survivor_inventory"],
            repeat_proof_raw=raw["repeat_proof"],
            survivor_records_raw=raw["survivor_records"],
            **expected,
        )


def test_current_clean_rejects_coherently_resealed_record_inventory_mismatch() -> None:
    _, raw, expected = _current_clean_bytes(_partial_rows())
    records = [
        json.loads(line)
        for line in raw["survivor_records"].decode("utf-8").splitlines()
    ]
    records[0]["normalized_payload"] = "z" + records[0]["normalized_payload"][1:]
    raw["survivor_records"] = b"".join(
        _canonical(record) + b"\n" for record in records
    )
    survivor_sha = hashlib.sha256(raw["survivor_records"]).hexdigest()

    receipt = json.loads(raw["composition_receipt"])
    receipt["survivor_jsonl_sha256"] = survivor_sha
    raw["composition_receipt"] = _reseal(receipt, "receipt_identity_sha256")
    resealed_receipt = json.loads(raw["composition_receipt"])

    repeat = json.loads(raw["repeat_proof"])
    repeat["survivor_jsonl_sha256"] = survivor_sha
    repeat["output_files_sha256"]["survivor_records.jsonl"] = survivor_sha
    repeat["output_files_sha256"]["composition_receipt.json"] = hashlib.sha256(
        raw["composition_receipt"]
    ).hexdigest()
    raw["repeat_proof"] = _reseal(repeat, "proof_identity_sha256")
    resealed_repeat = json.loads(raw["repeat_proof"])

    expected["expected_survivor_records_jsonl_sha256"] = survivor_sha
    expected["expected_composition_receipt_json_sha256"] = hashlib.sha256(
        raw["composition_receipt"]
    ).hexdigest()
    expected["expected_repeat_proof_json_sha256"] = hashlib.sha256(
        raw["repeat_proof"]
    ).hexdigest()
    expected["expected_receipt_identity_sha256"] = resealed_receipt[
        "receipt_identity_sha256"
    ]
    expected["expected_repeat_proof_identity_sha256"] = resealed_repeat[
        "proof_identity_sha256"
    ]

    with pytest.raises(ProjectionError, match="differs from rebuilt survivor JSONL"):
        build_current_clean_family_vector(
            composition_receipt_raw=raw["composition_receipt"],
            survivor_inventory_raw=raw["survivor_inventory"],
            repeat_proof_raw=raw["repeat_proof"],
            survivor_records_raw=raw["survivor_records"],
            **expected,
        )


def test_current_clean_rejects_noncanonical_survivor_jsonl() -> None:
    _, raw, expected = _current_clean_bytes(_partial_rows())
    first, *rest = raw["survivor_records"].splitlines(keepends=True)
    record = json.loads(first)
    noncanonical = json.dumps(record, ensure_ascii=False).encode() + b"\n"
    raw["survivor_records"] = noncanonical + b"".join(rest)
    survivor_sha = hashlib.sha256(raw["survivor_records"]).hexdigest()

    receipt = json.loads(raw["composition_receipt"])
    receipt["survivor_jsonl_sha256"] = survivor_sha
    raw["composition_receipt"] = _reseal(receipt, "receipt_identity_sha256")
    resealed_receipt = json.loads(raw["composition_receipt"])

    repeat = json.loads(raw["repeat_proof"])
    repeat["survivor_jsonl_sha256"] = survivor_sha
    repeat["output_files_sha256"]["survivor_records.jsonl"] = survivor_sha
    repeat["output_files_sha256"]["composition_receipt.json"] = hashlib.sha256(
        raw["composition_receipt"]
    ).hexdigest()
    raw["repeat_proof"] = _reseal(repeat, "proof_identity_sha256")
    resealed_repeat = json.loads(raw["repeat_proof"])

    expected["expected_survivor_records_jsonl_sha256"] = survivor_sha
    expected["expected_composition_receipt_json_sha256"] = hashlib.sha256(
        raw["composition_receipt"]
    ).hexdigest()
    expected["expected_repeat_proof_json_sha256"] = hashlib.sha256(
        raw["repeat_proof"]
    ).hexdigest()
    expected["expected_receipt_identity_sha256"] = resealed_receipt[
        "receipt_identity_sha256"
    ]
    expected["expected_repeat_proof_identity_sha256"] = resealed_repeat[
        "proof_identity_sha256"
    ]

    with pytest.raises(ProjectionError, match="not canonical JSON"):
        build_current_clean_family_vector(
            composition_receipt_raw=raw["composition_receipt"],
            survivor_inventory_raw=raw["survivor_inventory"],
            repeat_proof_raw=raw["repeat_proof"],
            survivor_records_raw=raw["survivor_records"],
            **expected,
        )


def _current_clean_target_selection_fixture() -> tuple[
    dict, dict[str, bytes], dict[str, object], dict, dict, dict
]:
    vector, raw, expected = _build_current_clean(_target_rows())
    adapted = _adapt(vector)
    policy = next100_gate.load_json(next100_gate.POLICY_PATH)
    balance = next100_gate.evaluate(policy, adapted)
    assert balance["status"] == "TARGET_20M_SOURCE_MIX_FEASIBLE"
    binding = build_balance_result_binding(
        family_vector=vector,
        expected_family_vector_identity_sha256=vector[
            "family_vector_identity_sha256"
        ],
        next100_input=adapted,
        balance_result=balance,
        expected_policy_identity_sha256=policy["policy_identity_sha256"],
        expected_result_identity_sha256=balance["result_identity_sha256"],
    )
    return vector, raw, expected, adapted, balance, binding


def _project_current_clean_selection(
    selection: dict,
    survivor_records_raw: bytes,
    *,
    expected_authority: dict | None = None,
) -> list[dict]:
    authority = selection if expected_authority is None else expected_authority
    return project_selected_current_clean_raw_records(
        selection,
        survivor_records_raw,
        expected_selection_identity_sha256=authority[
            "balanced_selection_identity_sha256"
        ],
        expected_retained_inventory_identity_sha256=authority[
            "retained_inventory_identity_sha256"
        ],
        expected_decontamination_authority_sha256=authority[
            "decontamination_authority_sha256"
        ],
        expected_dedup_authority_sha256=authority["dedup_authority_sha256"],
        expected_balance_policy_identity_sha256=authority[
            "balance_policy_identity_sha256"
        ],
        expected_balance_result_identity_sha256=authority[
            "balance_result_identity_sha256"
        ],
    )


def _build_current_clean_target_selection() -> tuple[dict, list[dict], dict]:
    vector, raw, _expected, adapted, balance, binding = (
        _current_clean_target_selection_fixture()
    )
    policy = next100_gate.load_json(next100_gate.POLICY_PATH)
    selection = build_current_clean_balanced_selection(
        family_vector=vector,
        next100_input=adapted,
        balance_result=balance,
        balance_binding=binding,
        composition_receipt_raw=raw["composition_receipt"],
        survivor_records_raw=raw["survivor_records"],
        expected_family_vector_identity_sha256=vector[
            "family_vector_identity_sha256"
        ],
        expected_balance_binding_identity_sha256=binding[
            "binding_identity_sha256"
        ],
        expected_policy_identity_sha256=policy["policy_identity_sha256"],
        expected_result_identity_sha256=balance["result_identity_sha256"],
    )
    projected = _project_current_clean_selection(
        selection,
        raw["survivor_records"],
    )
    return selection, projected, balance


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("model_training_authorized", 0),
        ("tokenizer_fit_authorized", 0),
        ("authorized_training_exposure_loss_positions", False),
        ("source_bytes_are_loss_positions", 0),
    ],
)
def test_balance_binding_rejects_resealed_zero_credit_type_alias(
    field: str, replacement: object,
) -> None:
    vector, _raw, _expected, adapted, balance, _binding = (
        _current_clean_target_selection_fixture()
    )
    altered = copy.deepcopy(balance)
    altered["claim_boundary"][field] = replacement
    altered["result_identity_sha256"] = next100_gate.canonical_sha(
        altered, "result_identity_sha256"
    )
    policy = next100_gate.load_json(next100_gate.POLICY_PATH)
    with pytest.raises(ProjectionError, match="claim boundary drift"):
        build_balance_result_binding(
            family_vector=vector,
            expected_family_vector_identity_sha256=vector["family_vector_identity_sha256"],
            next100_input=adapted,
            balance_result=altered,
            expected_policy_identity_sha256=policy["policy_identity_sha256"],
            expected_result_identity_sha256=altered["result_identity_sha256"],
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "stratum_mix", "inflated_family_cap", "noncanonical_family_order",
        "source_family_substitution", "minimum_family_count", "physical_capacity",
    ],
)
def test_target_binding_rejects_resealed_physical_or_policy_substitution(
    mutation: str,
) -> None:
    vector, _raw, _expected, adapted, balance, _binding = (
        _current_clean_target_selection_fixture()
    )
    altered = copy.deepcopy(balance)
    input_value = copy.deepcopy(adapted)
    if mutation == "stratum_mix":
        altered["maximum_feasible_stratum_bytes"]["ua"] -= 100
        altered["maximum_feasible_stratum_bytes"]["en"] += 100
    elif mutation == "inflated_family_cap":
        altered["deterministic_maximum_allocation"][0]["effective_family_cap_bytes"] += 100
    elif mutation == "noncanonical_family_order":
        altered["deterministic_maximum_allocation"].reverse()
    elif mutation == "source_family_substitution":
        input_value["families"][0]["family_id"] = "forged-family"
    elif mutation == "minimum_family_count":
        altered["family_minimum"]["observed"]["ua"] += 1
    elif mutation == "physical_capacity":
        altered["raw_capacity_by_stratum"]["ua"] += 100
    else:  # pragma: no cover - closed parametrization.
        raise AssertionError(mutation)
    altered["result_identity_sha256"] = next100_gate.canonical_sha(
        altered, "result_identity_sha256"
    )
    policy = next100_gate.load_json(next100_gate.POLICY_PATH)
    with pytest.raises(ProjectionError, match="target balance"):
        build_balance_result_binding(
            family_vector=vector,
            expected_family_vector_identity_sha256=vector["family_vector_identity_sha256"],
            next100_input=input_value,
            balance_result=altered,
            expected_policy_identity_sha256=policy["policy_identity_sha256"],
            expected_result_identity_sha256=altered["result_identity_sha256"],
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "forged_policy", "input_family_float", "input_total_float",
        "result_total_float", "minimum_count_float", "allocation_float",
    ],
)
def test_target_binding_rejects_resealed_numeric_alias_and_policy_drift(
    mutation: str,
) -> None:
    vector, _raw, _expected, adapted, balance, _binding = (
        _current_clean_target_selection_fixture()
    )
    input_value = copy.deepcopy(adapted)
    altered = copy.deepcopy(balance)
    policy = next100_gate.load_json(next100_gate.POLICY_PATH)
    expected_policy = policy["policy_identity_sha256"]
    if mutation == "forged_policy":
        expected_policy = "f" * 64
        altered["policy_identity_sha256"] = expected_policy
    elif mutation == "input_family_float":
        input_value["families"][0]["unique_bytes"] = float(
            input_value["families"][0]["unique_bytes"]
        )
    elif mutation == "input_total_float":
        input_value["totals"]["family_count"]["ua"] = 2.0
        altered["input_totals"] = copy.deepcopy(input_value["totals"])
    elif mutation == "result_total_float":
        observed_count = altered["input_totals"]["family_count"]["en"]
        assert type(observed_count) is int
        altered["input_totals"]["family_count"]["en"] = float(observed_count)
    elif mutation == "minimum_count_float":
        altered["family_minimum"]["observed"]["ua"] = 2.0
    elif mutation == "allocation_float":
        allocated = altered["deterministic_maximum_allocation"][0]
        allocated["allocated_bytes"] = float(allocated["allocated_bytes"])
    else:  # pragma: no cover - closed parametrization.
        raise AssertionError(mutation)
    altered["result_identity_sha256"] = next100_gate.canonical_sha(
        altered, "result_identity_sha256"
    )
    expected_error = (
        "NEXT100-106 result totals differ from physical input"
        if mutation == "result_total_float" else "target balance"
    )
    with pytest.raises(ProjectionError, match=expected_error):
        build_balance_result_binding(
            family_vector=vector,
            expected_family_vector_identity_sha256=vector["family_vector_identity_sha256"],
            next100_input=input_value,
            balance_result=altered,
            expected_policy_identity_sha256=expected_policy,
            expected_result_identity_sha256=altered["result_identity_sha256"],
        )


@pytest.mark.parametrize(
    ("field", "replacement", "error"),
    [
        ("authorized_optimized_target_exposure", False, "optimized-target exposure"),
        ("balance_policy_identity_sha256", "f" * 64, "independently pinned"),
        ("target_total_source_bytes", 20_000_000.0, "target total drift"),
        ("maximum_feasible_total_source_bytes", 20_000_000.0, "maximum"),
    ],
)
def test_standalone_balance_readiness_rejects_resealed_alias_and_policy(
    field: str, replacement: object, error: str,
) -> None:
    _vector, _raw, _expected, _adapted, _balance, binding = (
        _current_clean_target_selection_fixture()
    )
    altered = copy.deepcopy(binding)
    altered[field] = replacement
    altered["binding_identity_sha256"] = hashlib.sha256(
        _canonical(
            {
                key: value for key, value in altered.items()
                if key != "binding_identity_sha256"
            }
        )
    ).hexdigest()
    with pytest.raises(ProjectionError, match=error):
        require_balanced_selection_ready(
            altered,
            expected_binding_identity_sha256=altered["binding_identity_sha256"],
        )


def test_selection_rejects_resealed_false_current_clean_source_count() -> None:
    vector, raw, _expected, _adapted, _balance, _binding = (
        _current_clean_target_selection_fixture()
    )
    assert vector["source_object_count"] == 6
    vector["source_object_count"] = 5  # Remains positive and below record count.
    vector["family_vector_identity_sha256"] = hashlib.sha256(
        _canonical(
            {key: value for key, value in vector.items()
             if key != "family_vector_identity_sha256"}
        )
    ).hexdigest()
    adapted = _adapt(vector)
    policy = next100_gate.load_json(next100_gate.POLICY_PATH)
    balance = next100_gate.evaluate(policy, adapted)
    binding = build_balance_result_binding(
        family_vector=vector,
        expected_family_vector_identity_sha256=vector["family_vector_identity_sha256"],
        next100_input=adapted,
        balance_result=balance,
        expected_policy_identity_sha256=policy["policy_identity_sha256"],
        expected_result_identity_sha256=balance["result_identity_sha256"],
    )
    with pytest.raises(ProjectionError, match="source-object count differs"):
        build_current_clean_balanced_selection(
            family_vector=vector,
            next100_input=adapted,
            balance_result=balance,
            balance_binding=binding,
            composition_receipt_raw=raw["composition_receipt"],
            survivor_records_raw=raw["survivor_records"],
            expected_family_vector_identity_sha256=vector["family_vector_identity_sha256"],
            expected_balance_binding_identity_sha256=binding["binding_identity_sha256"],
            expected_policy_identity_sha256=policy["policy_identity_sha256"],
            expected_result_identity_sha256=balance["result_identity_sha256"],
        )



def _assert_resealed_vector_rejected_against_raw_survivors(
    vector: dict, raw: dict[str, bytes], *, rejected_field: str,
) -> None:
    # Intentionally allow the attacker to supply a freshly resealed expected
    # vector and matching derived NEXT100 input/result/binding. The immutable
    # physical rows and producer receipt must independently reject the alias.
    vector["family_vector_identity_sha256"] = hashlib.sha256(
        _canonical(
            {
                key: value for key, value in vector.items()
                if key != "family_vector_identity_sha256"
            }
        )
    ).hexdigest()
    adapted = _adapt(vector)
    policy = next100_gate.load_json(next100_gate.POLICY_PATH)
    balance = next100_gate.evaluate(policy, adapted)
    assert balance["status"] == "TARGET_20M_SOURCE_MIX_FEASIBLE"
    binding = build_balance_result_binding(
        family_vector=vector,
        expected_family_vector_identity_sha256=vector["family_vector_identity_sha256"],
        next100_input=adapted,
        balance_result=balance,
        expected_policy_identity_sha256=policy["policy_identity_sha256"],
        expected_result_identity_sha256=balance["result_identity_sha256"],
    )
    with pytest.raises(ProjectionError, match=rejected_field):
        build_current_clean_balanced_selection(
            family_vector=vector,
            next100_input=adapted,
            balance_result=balance,
            balance_binding=binding,
            composition_receipt_raw=raw["composition_receipt"],
            survivor_records_raw=raw["survivor_records"],
            expected_family_vector_identity_sha256=vector["family_vector_identity_sha256"],
            expected_balance_binding_identity_sha256=binding["binding_identity_sha256"],
            expected_policy_identity_sha256=policy["policy_identity_sha256"],
            expected_result_identity_sha256=balance["result_identity_sha256"],
        )


def test_selection_rejects_resealed_wrong_physical_record_membership() -> None:
    vector, raw, _expected, _adapted, _balance, _binding = (
        _current_clean_target_selection_fixture()
    )
    physical_membership = vector["record_membership_sha256"]
    vector["record_membership_sha256"] = (
        "0" * 64 if physical_membership != "0" * 64 else "1" * 64
    )
    _assert_resealed_vector_rejected_against_raw_survivors(
        vector, raw, rejected_field="record_membership_sha256",
    )


def test_selection_rejects_resealed_per_family_record_redistribution() -> None:
    # Two families have two real records each. Reallocate counts 2/2 -> 3/1
    # while retaining both positive, all bytes, stratum totals and source count.
    rows = [
        _row("ua-a-1", UA_A, "uk", 2_000_000),
        _row("ua-a-2", UA_A, "uk", 2_500_000),
        _row("ua-b-1", UA_B, "uk", 2_000_000),
        _row("ua-b-2", UA_B, "uk", 2_500_000),
        *_target_rows()[2:],
    ]
    vector, raw, _expected = _build_current_clean(rows)
    family_counts = {
        item["family"]: item["record_count"] for item in vector["families"]
    }
    assert family_counts[UA_A] == family_counts[UA_B] == 2
    for family in vector["families"]:
        if family["family"] == UA_A:
            family["record_count"] = 3
        elif family["family"] == UA_B:
            family["record_count"] = 1
    _assert_resealed_vector_rejected_against_raw_survivors(
        vector, raw, rejected_field="families",
    )


@pytest.mark.parametrize(
    "mutation",
    ("physical_capacity_float", "zero_gap_float", "allocation_float"),
)
def test_selection_rejects_self_resealed_binding_numeric_alias(
    mutation: str,
) -> None:
    vector, raw, _expected, adapted, balance, binding = (
        _current_clean_target_selection_fixture()
    )
    altered = copy.deepcopy(binding)
    if mutation == "physical_capacity_float":
        physical = altered["raw_capacity_by_stratum"]
        physical["ua"] = float(physical["ua"])
    elif mutation == "zero_gap_float":
        gap = altered["raw_gap_to_target_by_stratum"]
        gap["ua"] = float(gap["ua"])
    else:
        allocation = altered["deterministic_maximum_allocation"][0]
        allocation["allocated_bytes"] = float(allocation["allocated_bytes"])
    # Python's value-only mapping equality masks these typed JSON changes.
    assert altered == binding
    altered["binding_identity_sha256"] = hashlib.sha256(
        _canonical(
            {
                key: value
                for key, value in altered.items()
                if key != "binding_identity_sha256"
            }
        )
    ).hexdigest()
    policy = next100_gate.load_json(next100_gate.POLICY_PATH)
    with pytest.raises(ProjectionError, match="deterministic authenticated rebuild"):
        build_current_clean_balanced_selection(
            family_vector=vector,
            next100_input=adapted,
            balance_result=balance,
            balance_binding=altered,
            composition_receipt_raw=raw["composition_receipt"],
            survivor_records_raw=raw["survivor_records"],
            expected_family_vector_identity_sha256=vector[
                "family_vector_identity_sha256"
            ],
            expected_balance_binding_identity_sha256=altered[
                "binding_identity_sha256"
            ],
            expected_policy_identity_sha256=policy["policy_identity_sha256"],
            expected_result_identity_sha256=balance["result_identity_sha256"],
        )


def test_current_clean_balanced_selection_materializes_exact_target() -> None:
    selection, projected, balance = _build_current_clean_target_selection()
    assert selection["schema"] == "12-6.d03-balanced-selection-authority.v1"
    assert selection["terminal"] is True
    assert selection["status"] == "PASS"
    assert selection["totals"]["record_count"] == 6
    assert selection["totals"]["source_bytes"] == 20_000_000
    assert selection["totals"]["source_bytes"] == balance[
        "maximum_feasible_total_source_bytes"
    ]
    assert sum(
        selection["totals"]["family_source_bytes"].values()
    ) == 20_000_000
    assert len(projected) == 6
    assert all(row["training_eligible"] is False for row in projected)
    assert all(row["evaluation_eligible"] is False for row in projected)
    assert all(row["evaluation_reserved"] is False for row in projected)



def test_selected_projection_rejects_extra_unselected_physical_survivor() -> None:
    selection, _projected, _balance = _build_current_clean_target_selection()
    _vector, raw, _expected, _adapted, _result, _binding = (
        _current_clean_target_selection_fixture()
    )
    extra = json.loads(raw["survivor_records"].split(b"\n", 1)[0])
    extra["record_id"] = "zz-unselected-record"
    extra["source_id"] = "zz-unselected-source"
    extra["normalized_payload"] = "separate-valid-physical-record"
    substituted_raw = raw["survivor_records"] + _canonical(extra) + b"\n"
    # All original selected records are unchanged. The substituted full raw
    # corpus must still be refused under the retained inventory authority.
    with pytest.raises(ProjectionError, match="raw survivor inventory"):
        _project_current_clean_selection(selection, substituted_raw)


def test_current_clean_balanced_selection_is_deterministic() -> None:
    first, first_projected, _ = _build_current_clean_target_selection()
    second, second_projected, _ = _build_current_clean_target_selection()
    assert second == first
    assert second_projected == first_projected


def test_current_clean_balanced_selection_groups_same_source_as_one_cluster() -> None:
    rows = _target_rows()
    rows[1]["source_id"] = rows[0]["source_id"]
    vector, raw, _expected = _build_current_clean(rows)
    adapted = _adapt(vector)
    policy = next100_gate.load_json(next100_gate.POLICY_PATH)
    balance = next100_gate.evaluate(policy, adapted)
    binding = build_balance_result_binding(
        family_vector=vector,
        expected_family_vector_identity_sha256=vector[
            "family_vector_identity_sha256"
        ],
        next100_input=adapted,
        balance_result=balance,
        expected_policy_identity_sha256=policy["policy_identity_sha256"],
        expected_result_identity_sha256=balance["result_identity_sha256"],
    )
    selection = build_current_clean_balanced_selection(
        family_vector=vector,
        next100_input=adapted,
        balance_result=balance,
        balance_binding=binding,
        composition_receipt_raw=raw["composition_receipt"],
        survivor_records_raw=raw["survivor_records"],
        expected_family_vector_identity_sha256=vector[
            "family_vector_identity_sha256"
        ],
        expected_balance_binding_identity_sha256=binding[
            "binding_identity_sha256"
        ],
        expected_policy_identity_sha256=policy["policy_identity_sha256"],
        expected_result_identity_sha256=balance["result_identity_sha256"],
    )
    selected = {row["record_id"]: row for row in selection["records"]}
    assert selected["ua-a"]["near_duplicate_cluster_id"] == selected["ua-b"][
        "near_duplicate_cluster_id"
    ]


def test_exact_record_realization_rejects_unrepresentable_allocation() -> None:
    rows = [
        {"record_id": "a", "payload_bytes": 2},
        {"record_id": "b", "payload_bytes": 4},
    ]
    with pytest.raises(ProjectionError, match="no exact whole-record realization"):
        _exact_record_subset(rows, target_bytes=3, family="family")
    assert SELECTION_REALIZATION_POLICY == (
        "record-id-ascending-exact-family-byte-subset-v1"
    )


def test_exact_record_realization_fails_closed_on_state_budget() -> None:
    rows = [
        {"record_id": "a", "payload_bytes": 1},
        {"record_id": "b", "payload_bytes": 2},
        {"record_id": "c", "payload_bytes": 4},
        {"record_id": "d", "payload_bytes": 8},
    ]
    with pytest.raises(ProjectionError, match="exact-subset state budget exceeded"):
        _exact_record_subset(
            rows,
            target_bytes=14,
            family="family",
            max_states=3,
            max_expansions=100,
        )


def test_exact_record_realization_fails_closed_on_work_budget() -> None:
    rows = [
        {"record_id": "a", "payload_bytes": 2},
        {"record_id": "b", "payload_bytes": 4},
        {"record_id": "c", "payload_bytes": 6},
    ]
    with pytest.raises(ProjectionError, match="exact-subset work budget exceeded"):
        _exact_record_subset(
            rows,
            target_bytes=11,
            family="family",
            max_states=100,
            max_expansions=1,
        )


def test_current_clean_balanced_selection_rejects_raw_survivor_substitution() -> None:
    vector, raw, _expected, adapted, balance, binding = (
        _current_clean_target_selection_fixture()
    )
    policy = next100_gate.load_json(next100_gate.POLICY_PATH)
    tampered = raw["survivor_records"].replace(b'"ua-a"', b'"ua-x"', 1)
    with pytest.raises(ProjectionError, match="survivor JSONL bytes differ"):
        build_current_clean_balanced_selection(
            family_vector=vector,
            next100_input=adapted,
            balance_result=balance,
            balance_binding=binding,
            composition_receipt_raw=raw["composition_receipt"],
            survivor_records_raw=tampered,
            expected_family_vector_identity_sha256=vector[
                "family_vector_identity_sha256"
            ],
            expected_balance_binding_identity_sha256=binding[
                "binding_identity_sha256"
            ],
            expected_policy_identity_sha256=policy["policy_identity_sha256"],
            expected_result_identity_sha256=balance["result_identity_sha256"],
        )


def test_current_clean_balanced_selection_rejects_result_reseal_under_external_root() -> None:
    vector, raw, _expected, adapted, balance, binding = (
        _current_clean_target_selection_fixture()
    )
    policy = next100_gate.load_json(next100_gate.POLICY_PATH)
    tampered = copy.deepcopy(balance)
    tampered["deterministic_maximum_allocation"][0]["allocated_bytes"] -= 1
    tampered["result_identity_sha256"] = hashlib.sha256(
        _canonical(
            {
                key: value
                for key, value in tampered.items()
                if key != "result_identity_sha256"
            }
        )
    ).hexdigest()
    with pytest.raises(ProjectionError, match="external expectation"):
        build_current_clean_balanced_selection(
            family_vector=vector,
            next100_input=adapted,
            balance_result=tampered,
            balance_binding=binding,
            composition_receipt_raw=raw["composition_receipt"],
            survivor_records_raw=raw["survivor_records"],
            expected_family_vector_identity_sha256=vector[
                "family_vector_identity_sha256"
            ],
            expected_balance_binding_identity_sha256=binding[
                "binding_identity_sha256"
            ],
            expected_policy_identity_sha256=policy["policy_identity_sha256"],
            expected_result_identity_sha256=balance["result_identity_sha256"],
        )


def test_current_clean_balanced_selection_rejects_nonterminal_balance() -> None:
    vector, raw, _expected = _build_current_clean(_partial_rows())
    adapted = _adapt(vector)
    policy = next100_gate.load_json(next100_gate.POLICY_PATH)
    balance = next100_gate.evaluate(policy, adapted)
    binding = build_balance_result_binding(
        family_vector=vector,
        expected_family_vector_identity_sha256=vector[
            "family_vector_identity_sha256"
        ],
        next100_input=adapted,
        balance_result=balance,
        expected_policy_identity_sha256=policy["policy_identity_sha256"],
        expected_result_identity_sha256=balance["result_identity_sha256"],
    )
    with pytest.raises(
        ProjectionError,
        match="balanced selection blocked until TARGET_20M_SOURCE_MIX_FEASIBLE",
    ):
        build_current_clean_balanced_selection(
            family_vector=vector,
            next100_input=adapted,
            balance_result=balance,
            balance_binding=binding,
            composition_receipt_raw=raw["composition_receipt"],
            survivor_records_raw=raw["survivor_records"],
            expected_family_vector_identity_sha256=vector[
                "family_vector_identity_sha256"
            ],
            expected_balance_binding_identity_sha256=binding[
                "binding_identity_sha256"
            ],
            expected_policy_identity_sha256=policy["policy_identity_sha256"],
            expected_result_identity_sha256=balance["result_identity_sha256"],
        )


@pytest.mark.parametrize(
    ("mutation"),
    (
        "terminal",
        "status",
        "claim_boundary",
        "totals",
    ),
)
def test_selected_raw_projection_rejects_self_resealed_authority_semantics(
    mutation: str,
) -> None:
    selection, _projected, _balance = _build_current_clean_target_selection()
    _, raw, _expected = _current_clean_bytes(_target_rows())
    tampered = copy.deepcopy(selection)
    if mutation == "terminal":
        tampered["terminal"] = False
    elif mutation == "status":
        tampered["status"] = "FAIL"
    elif mutation == "claim_boundary":
        tampered["claim_boundary"]["model_training_authorized"] = True
    elif mutation == "totals":
        tampered["totals"]["source_bytes"] += 1
    else:  # pragma: no cover - parametrization is closed above.
        raise AssertionError(mutation)
    tampered["balanced_selection_identity_sha256"] = hashlib.sha256(
        _canonical(
            {
                key: value
                for key, value in tampered.items()
                if key != "balanced_selection_identity_sha256"
            }
        )
    ).hexdigest()
    with pytest.raises(
        ProjectionError,
        match="canonical balanced selection verification failed",
    ):
        _project_current_clean_selection(
            tampered,
            raw["survivor_records"],
            expected_authority=selection,
        )


def test_selected_raw_projection_rejects_self_resealed_metadata_substitution() -> None:
    selection, _projected, _balance = _build_current_clean_target_selection()
    _, raw, _expected = _current_clean_bytes(_target_rows())
    tampered = copy.deepcopy(selection)
    tampered["records"][0]["source_id"] = "substituted-source"
    tampered["records"][0]["near_duplicate_cluster_id"] = "substituted-source"
    tampered["balanced_selection_identity_sha256"] = hashlib.sha256(
        _canonical(
            {
                key: value
                for key, value in tampered.items()
                if key != "balanced_selection_identity_sha256"
            }
        )
    ).hexdigest()
    with pytest.raises(
        ProjectionError,
        match="canonical balanced selection verification failed",
    ):
        _project_current_clean_selection(
            tampered,
            raw["survivor_records"],
            expected_authority=selection,
        )


# Hash-only receipt from independently verified PR2211 artifact 11181030848.
# This fixed producer output catches canonical-byte mismatches hidden by mocks.
_PRODUCTION_CURRENT_CLEAN_RECEIPT = {'authorized_optimized_target_exposure': 0,
 'clean_training_handoff_sha256': '80bcf2dd28f0d13795ceea01b358c7149b636f55f17b29575d14313b5cee99ee',
 'clean_training_records_sha256': '3458afe0380ea45d328ad3f004b21845a188ca69f2f7a83c24999e9e53268e53',
 'current_corpus_launch_authority_promoted': False,
 'data232_report_sha256': '7176e069fb23a834353e578bce4e0381a037ccb4e5c77c24d89943e749517863',
 'decontamination_execution_identity_sha256': '9147fc688ce328503b4ab0bed3e1cf1f1d4250eb97daebd4cfc90cdf136cce59',
 'dependency_git_blobs': {'current_reserved_decontamination_v1.py': 'e5c555e3cd27844e98d4ae91af0b746e427f36c9',
                          'eval647_reserved_decontamination_v1.py': 'ce33771c9fb4a6cc421e2f8e1f6f232c119bec71',
                          'post_g05_g06_materialization_v1.py': '830087f91d1fa24385c5cbc8d2f687e4cc46b419',
                          'privacy_execution_authority.py': '9215287e81c0a82f05ec8405dc4f34c60313c193',
                          'privacy_filter_v3.py': 'bcc5938395724f6728ab212f98b39f2334b0f37d',
                          'quality_execution_authority.py': '4659a9d4aba49908f372250904a54361c8d8cf46'},
 'durable_evidence_hash_only': True,
 'eval647_execution_receipt_identity_sha256': 'c54c3f2d74ba5c9dcbb6913042d24cce091c4e183ff72ae00b4bc9131df600c4',
 'final_test_outcomes_read': False,
 'foreign_pretrained_weights': False,
 'independent_qualification_required': True,
 'input_training_records': 257,
 'learned_weights_created': False,
 'optimizer_updates_executed_on_real_targets': 0,
 'paid_compute_used': False,
 'post_decontamination_input_rows_sha256': 'e4cc685d7bc24442d151322a511d4067f6f387825a6b65ffa9d5e55acfe829e6',
 'post_decontamination_records': 254,
 'post_quality_input_rows_sha256': 'e4cc685d7bc24442d151322a511d4067f6f387825a6b65ffa9d5e55acfe829e6',
 'post_quality_records': 254,
 'privacy_detector_counts': {},
 'privacy_execution_identity_sha256': 'b3488f8186fb761d0e383547e8b082b827de93a3b0b3f5c902c5ed9ef5db6966',
 'quality_execution_identity_sha256': 'a109bf4ce98b962eede358e6ac8fb29e1999236fed49b283a4921f94cd3ae6bf',
 'receipt_identity_sha256': '2fe7c4c7ed85f158e8a2bf3ef927af03bf5dc32ecbc33ee97598e613faddb04c',
 'rejection_counts': {'data232_excluded_records': 3,
                      'g05_partial_documents': 0,
                      'g05_reject_documents': 0,
                      'g05_rejected_units': 0,
                      'g05_rejected_utf8_bytes': 0,
                      'g06_dropped_utf8_bytes': 0,
                      'g06_exclude_records': 0,
                      'g06_quarantine_records': 0,
                      'g06_redacted_records': 0},
 'schema_version': '12-6.current-clean-decontam-quality-privacy-survivor.v2',
 'status': 'CLEAN_SURVIVOR_MATERIALIZED_PENDING_INDEPENDENT_QUALIFICATION',
 'survivor_jsonl_sha256': '3aa6235e07da4fa50c9642639cc0d93d7da22a6b602d822c7c7ca6e1f4a7486a',
 'survivor_payload_bytes': 5428358,
 'survivor_payload_inventory_digest_sha256': '2d2f0f5695ee07dfbba2e49b2d14983ede91051644770804fb734235125353af',
 'survivor_record_inventory_digest_sha256': '0807f46418fb5a16a1c553c86a6f6967e5c2854d0016e3f6fe24c1d9c05c628b',
 'survivor_records': 254,
 'survivor_source_objects': 241,
 'terminal_post_g05_g06_authority': False,
 'tokenizer_fit_authorized': False,
 'training_executed': False}


def _verify_production_receipt(receipt: dict, expected_identity: str) -> None:
    _verify_current_clean_receipt(
        receipt,
        expected_receipt_identity_sha256=expected_identity,
        expected_survivor_jsonl_sha256="3aa6235e07da4fa50c9642639cc0d93d7da22a6b602d822c7c7ca6e1f4a7486a",
        expected_record_inventory_digest_sha256="0807f46418fb5a16a1c553c86a6f6967e5c2854d0016e3f6fe24c1d9c05c628b",
        expected_payload_inventory_digest_sha256="2d2f0f5695ee07dfbba2e49b2d14983ede91051644770804fb734235125353af",
        expected_record_count=254,
        expected_total_payload_bytes=5_428_358,
        expected_source_object_count=241,
    )


def test_current_clean_accepts_fixed_physical_producer_receipt() -> None:
    _verify_production_receipt(
        _PRODUCTION_CURRENT_CLEAN_RECEIPT,
        "2fe7c4c7ed85f158e8a2bf3ef927af03bf5dc32ecbc33ee97598e613faddb04c",
    )


def test_current_clean_rejects_resealed_receipt_without_producer_lf() -> None:
    receipt = copy.deepcopy(_PRODUCTION_CURRENT_CLEAN_RECEIPT)
    receipt.pop("receipt_identity_sha256")
    wrong_identity = hashlib.sha256(_canonical(receipt)).hexdigest()
    receipt["receipt_identity_sha256"] = wrong_identity
    with pytest.raises(ProjectionError, match="receipt self-hash mismatch"):
        _verify_production_receipt(receipt, wrong_identity)
