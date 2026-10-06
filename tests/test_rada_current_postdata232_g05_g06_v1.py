from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_d03_rada_current_postdata232_g05_g06_v1 as target


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _decision(label: str) -> str:
    return _sha(label.encode("utf-8"))


def _partial_authority(
    *,
    record_id: str,
    text: str,
    split: int,
) -> dict[str, object]:
    left = text[:split]
    right = text[split:]
    left_raw = left.encode("utf-8")
    right_raw = right.encode("utf-8")
    full_raw = text.encode("utf-8")
    return {
        "records": [
            {
                "record_id": record_id,
                "mode": "uk",
                "payload_sha256": _sha(full_raw),
                "utf8_bytes": len(full_raw),
                "authoritative_unit": "BOUNDED_NATURAL_LANGUAGE_WINDOW",
                "status": "RETAIN_PARTIAL",
                "retained_utf8_bytes": len(left_raw),
                "rejected_utf8_bytes": len(right_raw),
                "units": [
                    {
                        "unit_id": f"{record_id}#quality-window-0000",
                        "start_char": 0,
                        "end_char": split,
                        "payload_sha256": _sha(left_raw),
                        "utf8_bytes": len(left_raw),
                        "accepted": True,
                        "decision_sha256": _decision("accept"),
                    },
                    {
                        "unit_id": f"{record_id}#quality-window-0001",
                        "start_char": split,
                        "end_char": len(text),
                        "payload_sha256": _sha(right_raw),
                        "utf8_bytes": len(right_raw),
                        "accepted": False,
                        "decision_sha256": _decision("reject"),
                    },
                ],
            }
        ]
    }


def test_partial_window_materialization_is_exact_and_text_preserving() -> None:
    record_id = "rada.synthetic.1"
    text = "абвгДЕЖЗ"
    split = 4
    inputs = [{"id": record_id, "text": text, "mode": "uk"}]
    metadata = {
        record_id: {
            "source_id": "rada-source-1",
            "family": "rada-current",
            "mode": "uk",
        }
    }
    quality = _partial_authority(
        record_id=record_id,
        text=text,
        split=split,
    )

    output, stats, detail = target._materialize_quality_survivors_with_partial(
        inputs,
        metadata,
        quality,
    )

    retained = text[:split]
    retained_raw = retained.encode("utf-8")
    expected_id = (
        f"{record_id}#quality-window-0000:"
        f"{_sha(retained_raw)}"
    )
    assert output == [
        {
            "record_id": expected_id,
            "source_id": "rada-source-1",
            "family": "rada-current",
            "modality": "uk",
            "normalized_payload": retained,
        }
    ]
    assert stats == {
        "g05_reject_documents": 0,
        "g05_partial_documents": 1,
        "g05_rejected_units": 1,
        "g05_rejected_utf8_bytes": len(text[split:].encode("utf-8")),
    }
    assert detail["partial_documents"] == 1
    assert detail["partial_emitted_units"] == 1
    assert detail["partial_rejected_units"] == 1
    assert detail["input_utf8_bytes"] == len(text.encode("utf-8"))
    assert (
        detail["input_utf8_bytes"]
        == detail["retained_utf8_bytes"] + detail["rejected_utf8_bytes"]
    )
    assert detail["partial_unit_projection_count"] == 1


def test_partial_window_materialization_rejects_partition_gap() -> None:
    record_id = "rada.synthetic.2"
    text = "abcdefgh"
    quality = _partial_authority(
        record_id=record_id,
        text=text,
        split=4,
    )
    units = quality["records"][0]["units"]
    units[1]["start_char"] = 5

    with pytest.raises(
        target.RadaPostData232Error,
        match="unit partition drift",
    ):
        target._materialize_quality_survivors_with_partial(
            [{"id": record_id, "text": text, "mode": "uk"}],
            {
                record_id: {
                    "source_id": "rada-source-2",
                    "family": "rada-current",
                    "mode": "uk",
                }
            },
            quality,
        )


def test_partial_window_materialization_rejects_payload_hash_drift() -> None:
    record_id = "rada.synthetic.3"
    text = "abcdefgh"
    quality = _partial_authority(
        record_id=record_id,
        text=text,
        split=4,
    )
    quality["records"][0]["units"][0]["payload_sha256"] = "0" * 64

    with pytest.raises(
        target.RadaPostData232Error,
        match="unit payload drift",
    ):
        target._materialize_quality_survivors_with_partial(
            [{"id": record_id, "text": text, "mode": "uk"}],
            {
                record_id: {
                    "source_id": "rada-source-3",
                    "family": "rada-current",
                    "mode": "uk",
                }
            },
            quality,
        )


def test_strict_json_rejects_duplicate_members_and_nonfinite_values() -> None:
    with pytest.raises(
        target.RadaPostData232Error,
        match="duplicate JSON member",
    ):
        target.strict_loads(
            b'{"a":1,"a":2}',
            "duplicate",
        )

    with pytest.raises(
        target.RadaPostData232Error,
        match="non-finite JSON constant",
    ):
        target.strict_loads(
            b'{"a":NaN}',
            "nonfinite",
        )


def test_self_hash_rejects_caller_selected_reseal_against_external_pin() -> None:
    core = {
        "schema_version": "synthetic.v1",
        "value": 1,
    }
    document = {
        **core,
        "identity_sha256": _sha(target.canonical(core)),
    }
    target.verify_self_hash(
        document,
        "identity_sha256",
        label="synthetic",
        expected=document["identity_sha256"],
    )

    mutated = {
        "schema_version": "synthetic.v1",
        "value": 2,
    }
    resealed = {
        **mutated,
        "identity_sha256": _sha(target.canonical(mutated)),
    }
    with pytest.raises(
        target.RadaPostData232Error,
        match="not independently expected",
    ):
        target.verify_self_hash(
            resealed,
            "identity_sha256",
            label="synthetic",
            expected=document["identity_sha256"],
        )


def test_canonical_serialization_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError):
        json.loads(target.canonical({"x": float("nan")}).decode("utf-8"))


def test_bootstrap_has_no_eager_twelve_six_imports() -> None:
    source = Path(target.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    eager: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "twelve_six" or module.startswith("twelve_six."):
                eager.append(module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "twelve_six" or alias.name.startswith("twelve_six."):
                    eager.append(alias.name)
    assert eager == []


def test_bootstrap_pins_behavior_import_closure_before_runtime_load() -> None:
    expected = {
        "src/twelve_six/__init__.py":
            "5433166c507bc845bd12d8d5c4145f1fbedda204",
        "src/twelve_six/data/_data232_decontamination_matching.py":
            "afa70511f82dc81d9c9f85e3d0b67eba343004f9",
        "src/twelve_six/data/decontamination_authority_v2.py":
            "3ca8f21945c02f692c130a015e036673fa24e7af",
        "src/twelve_six/data/document_quality.py":
            "b1461263034b4fb9510479b20c9697e22faa5f97",
        "src/twelve_six/data/quality_granularity.py":
            "513523b86824c423cad97352b3abb3d1241531b9",
        "src/twelve_six/data/eval647_future_training_exclusion_v1.py":
            "5516577a0720150a7ec12c1bf8898972968e6970",
        "src/twelve_six/data/eval647_reserved_decontamination_v1.py":
            "ce33771c9fb4a6cc421e2f8e1f6f232c119bec71",
    }
    for path, blob in expected.items():
        assert target.EXPECTED_DEPENDENCY_BLOBS[path] == blob

    execute_source = ast.parse(
        Path(target.__file__).read_text(encoding="utf-8")
    )
    execute_node = next(
        node
        for node in execute_source.body
        if isinstance(node, ast.FunctionDef) and node.name == "execute"
    )
    calls = [
        node.func.id
        for node in ast.walk(execute_node)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert calls.index("verify_local_authority") < calls.index("load_bound_runtime")



def test_immutable_writer_resumes_identical_and_rejects_divergence(
    tmp_path: Path,
) -> None:
    output = tmp_path / "evidence.json"
    payload = b"{\"ok\":true}\n"
    target.write_immutable_bytes(output, payload, label="synthetic")
    assert output.read_bytes() == payload

    target.write_immutable_bytes(output, payload, label="synthetic")
    assert output.read_bytes() == payload

    with pytest.raises(
        target.RadaPostData232Error,
        match="refusing to overwrite divergent durable evidence",
    ):
        target.write_immutable_bytes(
            output,
            b"{\"ok\":false}\n",
            label="synthetic",
        )


def test_immutable_writer_recovers_matching_interrupted_temp(
    tmp_path: Path,
) -> None:
    output = tmp_path / "evidence.json"
    payload = b"{\"resume\":true}\n"
    temp = output.with_name(output.name + ".tmp")
    temp.write_bytes(payload)

    target.write_immutable_bytes(output, payload, label="synthetic")

    assert output.read_bytes() == payload
    assert not temp.exists()


def test_immutable_writer_rejects_divergent_interrupted_temp(
    tmp_path: Path,
) -> None:
    output = tmp_path / "evidence.json"
    temp = output.with_name(output.name + ".tmp")
    temp.write_bytes(b"stale\n")

    with pytest.raises(
        target.RadaPostData232Error,
        match="divergent interrupted temp evidence",
    ):
        target.write_immutable_bytes(
            output,
            b"fresh\n",
            label="synthetic",
        )



def _zero_credit_truth() -> dict[str, object]:
    return {
        "canonical_capacity_credited": 0,
        "authorized_optimized_target_exposure": 0,
        "authorized_unique_loss_positions": 0,
        "authorized_training_exposure": 0,
        "tokenizer_fit_authorized": False,
        "training_executed": False,
        "learned_weights_created": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
        "scale_promotion_authorized": False,
        "model_architecture_or_hyperparameters_selected": False,
    }


def test_zero_credit_truth_accepts_exact_parent_boundary() -> None:
    target.verify_zero_credit_truth(
        _zero_credit_truth(),
        label="synthetic",
        require_model_selection_false=True,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("canonical_capacity_credited", 1),
        ("authorized_optimized_target_exposure", 1),
        ("authorized_unique_loss_positions", 1),
        ("authorized_training_exposure", 1),
        ("tokenizer_fit_authorized", True),
        ("training_executed", True),
        ("learned_weights_created", True),
        ("final_test_outcomes_read", True),
        ("paid_compute_used", True),
        ("scale_promotion_authorized", True),
        ("model_architecture_or_hyperparameters_selected", True),
    ],
)
def test_zero_credit_truth_rejects_every_authority_widening(
    field: str,
    value: object,
) -> None:
    truth = _zero_credit_truth()
    truth[field] = value
    with pytest.raises(target.RadaPostData232Error):
        target.verify_zero_credit_truth(
            truth,
            label="synthetic",
            require_model_selection_false=True,
        )


def test_zero_credit_truth_rejects_boolean_zero_impostor() -> None:
    truth = _zero_credit_truth()
    truth["authorized_training_exposure"] = False
    with pytest.raises(
        target.RadaPostData232Error,
        match="authorized_training_exposure",
    ):
        target.verify_zero_credit_truth(
            truth,
            label="synthetic",
            require_model_selection_false=True,
        )



class _ReservedStub:
    @staticmethod
    def _verify_training_handoff(*args: object, **kwargs: object) -> None:
        return None

    @staticmethod
    def verify_execution_evidence(*args: object, **kwargs: object) -> None:
        return None


def _self_hashed(
    core: dict[str, object],
    field: str,
) -> dict[str, object]:
    return {**core, field: _sha(target.canonical(core))}


def _parent_fixture() -> dict[str, object]:
    counts = {
        "training_records": 3,
        "evaluation_records": 2,
        "excluded_training_records": 1,
        "quarantined_source_families": 0,
        "match_evidence_records": 1,
    }
    selection = _decision("selection")
    final_test = _decision("final-test")
    report_id = _decision("report")
    evidence_id = _decision("execution")

    inventory = _self_hashed(
        {
            "schema_version": "12-6.d03-rada-current-postdedup-inventory.v1",
            "full_selection_projection_sha256":
                target.EXPECTED_FULL_SELECTION_SHA256,
            "current_rada_slice_authority_sha256":
                target.EXPECTED_RADA_SLICE_SHA256,
            "raw_text_persisted": False,
        },
        "inventory_identity_sha256",
    )
    handoff = _self_hashed(
        {"schema_version": "synthetic-handoff.v1"},
        "handoff_identity_sha256",
    )
    report = {
        "report_sha256": report_id,
        "status": "PASS_WITH_EXCLUSIONS",
        "selection_validation_identity": selection,
        "final_test_identity": final_test,
        "counts": counts,
    }
    evidence = {
        "execution_identity_sha256": evidence_id,
        "status": "PASS_WITH_EXCLUSIONS",
        "counts": counts,
    }
    result_core = {
        "schema_version":
            "12-6.d03-rada-current-data232-execution-result.v1",
        "execution_head_sha": target.PARENT_EXECUTION_HEAD,
        "parent_execution_head_sha": target.UPSTREAM_GLOBAL_DEDUP_HEAD,
        "postdedup_inventory_identity_sha256":
            inventory["inventory_identity_sha256"],
        "full_selection_projection_sha256":
            target.EXPECTED_FULL_SELECTION_SHA256,
        "current_rada_slice_authority_sha256":
            target.EXPECTED_RADA_SLICE_SHA256,
        "training_handoff_identity_sha256":
            handoff["handoff_identity_sha256"],
        "reserved_payload_binding_identity_sha256":
            _decision("reserved-binding"),
        "selection_validation_identity_sha256": selection,
        "final_test_identity_sha256": final_test,
        "data232_report_sha256": report_id,
        "data232_execution_identity_sha256": evidence_id,
        "status": "PASS_WITH_EXCLUSIONS",
        "counts": counts,
        "final_test_payload_accessed_for_decontamination": True,
        "durable_evidence_hash_only": True,
        "next_gate": "CURRENT_RADA_POST_DATA232_QUALITY_PRIVACY",
        **_zero_credit_truth(),
    }
    result = _self_hashed(result_core, "result_identity_sha256")
    report_file = _decision("report-file")
    evidence_file = _decision("evidence-file")
    result_file = _decision("result-file")
    proof_core = {
        "schema_version": "12-6.d03-rada-current-data232-two-clean.v1",
        "execution_head_sha": target.PARENT_EXECUTION_HEAD,
        "parent_execution_head_sha": target.UPSTREAM_GLOBAL_DEDUP_HEAD,
        "full_selection_projection_sha256":
            target.EXPECTED_FULL_SELECTION_SHA256,
        "current_rada_slice_authority_sha256":
            target.EXPECTED_RADA_SLICE_SHA256,
        "postdedup_inventory_identity_sha256":
            inventory["inventory_identity_sha256"],
        "training_handoff_identity_sha256":
            handoff["handoff_identity_sha256"],
        "reserved_payload_binding_identity_sha256":
            result["reserved_payload_binding_identity_sha256"],
        "data232_report_sha256": report_id,
        "data232_execution_identity_sha256": evidence_id,
        "result_identity_sha256": result["result_identity_sha256"],
        "data232_report_file_sha256": report_file,
        "data232_execution_evidence_file_sha256": evidence_file,
        "result_file_sha256": result_file,
        "two_fresh_processes_byte_identical": True,
        **counts,
        **{
            key: value
            for key, value in _zero_credit_truth().items()
            if key != "model_architecture_or_hyperparameters_selected"
        },
    }
    proof = _self_hashed(proof_core, "proof_identity_sha256")
    return {
        "inventory": inventory,
        "handoff": handoff,
        "report": report,
        "evidence": evidence,
        "result": result,
        "proof": proof,
        "report_file": report_file,
        "evidence_file": evidence_file,
        "result_file": result_file,
    }


def _verify_parent_fixture(fixture: dict[str, object]) -> None:
    target.verify_parent(
        reserved_module=_ReservedStub,
        verify_report_fn=lambda report: None,
        training_records=[],
        inventory=fixture["inventory"],
        handoff=fixture["handoff"],
        report=fixture["report"],
        evidence=fixture["evidence"],
        result=fixture["result"],
        proof=fixture["proof"],
        expected_inventory_identity_sha256=fixture["inventory"][
            "inventory_identity_sha256"
        ],
        expected_handoff_identity_sha256=fixture["handoff"][
            "handoff_identity_sha256"
        ],
        expected_result_identity_sha256=fixture["result"][
            "result_identity_sha256"
        ],
        expected_proof_identity_sha256=fixture["proof"][
            "proof_identity_sha256"
        ],
        report_file_sha256=fixture["report_file"],
        evidence_file_sha256=fixture["evidence_file"],
        result_file_sha256=fixture["result_file"],
    )


def test_parent_cross_binding_accepts_consistent_exact_lineage() -> None:
    _verify_parent_fixture(_parent_fixture())


@pytest.mark.parametrize(
    ("document", "field", "value", "message"),
    [
        (
            "result",
            "parent_execution_head_sha",
            "0" * 40,
            "upstream global-dedup HEAD drift",
        ),
        (
            "result",
            "selection_validation_identity_sha256",
            "0" * 64,
            "reserved-evaluation identity drift",
        ),
        (
            "proof",
            "reserved_payload_binding_identity_sha256",
            "0" * 64,
            "corpus-lineage drift",
        ),
        (
            "proof",
            "training_records",
            4,
            "count projection drift",
        ),
    ],
)
def test_parent_cross_binding_rejects_resealed_lineage_drift(
    document: str,
    field: str,
    value: object,
    message: str,
) -> None:
    fixture = _parent_fixture()
    mutated = dict(fixture[document])
    identity_field = (
        "result_identity_sha256"
        if document == "result"
        else "proof_identity_sha256"
    )
    mutated.pop(identity_field)
    mutated[field] = value
    fixture[document] = _self_hashed(mutated, identity_field)

    if document == "result":
        proof = dict(fixture["proof"])
        proof.pop("proof_identity_sha256")
        proof["result_identity_sha256"] = fixture["result"][
            "result_identity_sha256"
        ]
        if field == "reserved_payload_binding_identity_sha256":
            proof["reserved_payload_binding_identity_sha256"] = value
        fixture["proof"] = _self_hashed(proof, "proof_identity_sha256")

    with pytest.raises(target.RadaPostData232Error, match=message):
        _verify_parent_fixture(fixture)



def test_text_free_durable_guard_accepts_hash_only_evidence() -> None:
    target.assert_text_free_durable(
        {
            "raw_training_text_persisted": False,
            "records": [
                {
                    "record_id": "x",
                    "payload_sha256": "0" * 64,
                    "payload_bytes": 3,
                }
            ],
        },
        label="synthetic",
    )


@pytest.mark.parametrize(
    "forbidden_key",
    sorted(target._DURABLE_FORBIDDEN_TEXT_KEYS),
)
def test_text_free_durable_guard_rejects_nested_payload_text(
    forbidden_key: str,
) -> None:
    with pytest.raises(
        target.RadaPostData232Error,
        match="raw-text-bearing durable key",
    ):
        target.assert_text_free_durable(
            {"outer": [{"safe": {forbidden_key: "secret"}}]},
            label="synthetic",
        )



def test_durable_bundle_commits_receipt_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels: list[str] = []

    def capture(path: Path, payload: bytes, *, label: str) -> None:
        labels.append(label)

    monkeypatch.setattr(target, "write_immutable_bytes", capture)
    target.commit_durable_bundle(
        evidence_path=tmp_path / "evidence.json",
        evidence_bytes=b"e",
        quality_path=tmp_path / "quality.json",
        quality_bytes=b"q",
        privacy_path=tmp_path / "privacy.json",
        privacy_bytes=b"p",
        survivor_inventory_path=tmp_path / "inventory.json",
        survivor_inventory_bytes=b"i",
    )

    assert labels == [
        "G05 authority",
        "G06 authority",
        "survivor inventory",
        "post-G05/G06 evidence",
    ]



def test_durable_bundle_rejects_output_path_alias(tmp_path: Path) -> None:
    shared = tmp_path / "shared.json"
    with pytest.raises(
        target.RadaPostData232Error,
        match="pairwise distinct",
    ):
        target.commit_durable_bundle(
            evidence_path=shared,
            evidence_bytes=b"e",
            quality_path=shared,
            quality_bytes=b"q",
            privacy_path=tmp_path / "privacy.json",
            privacy_bytes=b"p",
            survivor_inventory_path=tmp_path / "inventory.json",
            survivor_inventory_bytes=b"i",
        )



def test_project_import_parser_covers_supported_forms() -> None:
    source = """
from twelve_six.data.alpha import value
from twelve_six.data import beta, gamma
import twelve_six.data.delta
"""
    assert target._project_import_paths(source, label="synthetic.py") == {
        "src/twelve_six/data/alpha.py",
        "src/twelve_six/data/beta.py",
        "src/twelve_six/data/gamma.py",
        "src/twelve_six/data/delta.py",
    }


def test_project_import_parser_rejects_wildcard() -> None:
    with pytest.raises(
        target.RadaPostData232Error,
        match="wildcard project import is not auditable",
    ):
        target._project_import_paths(
            "from twelve_six.data import *\n",
            label="synthetic.py",
        )


def test_declared_dependency_import_closure_is_fully_pinned() -> None:
    allowed = set(target.EXPECTED_DEPENDENCY_BLOBS)
    carrier = "tools/run_d03_rada_current_postdata232_g05_g06_v1.py"
    for relative in sorted(allowed | {carrier}):
        source = (target.ROOT / relative).read_text(
            encoding="utf-8",
            errors="strict",
        )
        assert target._project_import_paths(
            source,
            label=relative,
        ) <= allowed

def _post_g06_survivor(record_id: str, payload: str) -> dict[str, str]:
    return {
        "record_id": record_id,
        "source_id": f"source:{record_id}",
        "family": "ua.rada.open-data.laws-texts",
        "modality": "uk",
        "normalized_payload": payload,
    }


def test_post_g06_payload_uniqueness_accepts_distinct_payloads_deterministically() -> None:
    records = [
        _post_g06_survivor("r-1", "перший текст"),
        _post_g06_survivor("r-2", "другий текст"),
    ]

    first = target.verify_post_g06_exact_payload_uniqueness(records)
    second = target.verify_post_g06_exact_payload_uniqueness(
        list(reversed(records))
    )

    assert first == second
    assert first["unique_payload_count"] == 2
    assert len(first["payload_set_identity_sha256"]) == 64


def test_post_g06_payload_uniqueness_rejects_redaction_collapse() -> None:
    transformed = "контакт <redacted>"
    records = [
        _post_g06_survivor("r-before-a", transformed),
        _post_g06_survivor("r-before-b", transformed),
    ]

    with pytest.raises(
        target.RadaPostData232Error,
        match="post-G06 exact payload collision after transformation",
    ):
        target.verify_post_g06_exact_payload_uniqueness(records)


def test_post_g06_payload_uniqueness_fails_closed_on_digest_collision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [
        _post_g06_survivor("r-hash-a", "payload A"),
        _post_g06_survivor("r-hash-b", "payload B"),
    ]
    monkeypatch.setattr(target, "sha256", lambda _raw: "0" * 64)

    with pytest.raises(
        target.RadaPostData232Error,
        match="post-G06 exact payload collision after transformation",
    ):
        target.verify_post_g06_exact_payload_uniqueness(records)

