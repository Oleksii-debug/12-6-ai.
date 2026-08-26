from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))
TOOL_PATH = TOOLS / "classify_d03_rada_trees_members.py"
CONFIG_PATH = ROOT / "configs/data/d03_rada_trees_member_classification_v1.json"
SPEC = importlib.util.spec_from_file_location("rada_trees_classify", TOOL_PATH)
assert SPEC and SPEC.loader
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)


def config() -> dict[str, object]:
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_committed_config_is_fail_closed() -> None:
    tool.validate_config(config())


def test_training_credit_mutation_rejected() -> None:
    value = copy.deepcopy(config())
    value["claim_boundary"]["training_authorized_bytes"] = 1
    with pytest.raises(tool.ClassificationError, match="training bytes"):
        tool.validate_config(value)


def test_parent_head_mutation_rejected() -> None:
    value = copy.deepcopy(config())
    value["parent"]["head_sha"] = "0" * 40
    with pytest.raises(tool.ClassificationError, match="parent head drift"):
        tool.validate_config(value)


def test_plain_transcript_candidate_keeps_text_private() -> None:
    policy = config()["classification_policy"]
    result = tool.classify_content(
        "plain/2024/session-001.txt",
        "ГОЛОВУЮЧИЙ. Розпочинаємо пленарне засідання Верховної Ради України.\n".encode(),
        policy,
    )
    assert result["class"] == "PLAIN_TEXT_CANDIDATE"
    assert result["encoding"] == "utf-8-sig"
    assert "text" not in result
    assert result["metrics"]["ukrainian_specific_letter_count"] > 0


def test_txt_with_conllu_shape_is_derived_hold() -> None:
    policy = config()["classification_policy"]
    payload = (
        "1\tМи\tми\tPRON\tPpers\tCase=Nom\t2\tnsubj\t_\t_\n"
        "2\tпрацюємо\tпрацювати\tVERB\tVmpip1p\t_\t0\troot\t_\t_\n"
        "3\t.\t.\tPUNCT\tU\t_\t2\tpunct\t_\t_\n"
    ).encode()
    result = tool.classify_content("plain/misnamed.txt", payload, policy)
    assert result["class"] == "DERIVED_UD_HOLD"


def test_known_conllu_suffix_is_always_derived_hold() -> None:
    result = tool.classify_content(
        "ud/session.conllu",
        b"opaque",
        config()["classification_policy"],
    )
    assert result["class"] == "DERIVED_UD_HOLD"


def test_tabular_annotation_txt_is_not_plain_candidate() -> None:
    payload = "слово\tlemma\ttag\nінше\tlemma\ttag\nтретє\tlemma\ttag\n".encode()
    result = tool.classify_content(
        "nlp_uk/session.txt",
        payload,
        config()["classification_policy"],
    )
    assert result["class"] == "TABULAR_ANNOTATION_HOLD"


def test_windows_1251_plain_text_is_explicitly_provenanced() -> None:
    payload = "Пленарне засідання Верховної Ради України.".encode("cp1251")
    result = tool.classify_content(
        "plain/legacy-1999.txt",
        payload,
        config()["classification_policy"],
    )
    assert result["class"] == "PLAIN_TEXT_CANDIDATE"
    assert result["encoding"] == "windows-1251"
    assert tool.year_hints("plain/legacy-1999.txt") == [1999]


def test_nul_payload_is_held() -> None:
    result = tool.classify_content(
        "plain/binary.txt",
        b"text\x00more",
        config()["classification_policy"],
    )
    assert result["class"] == "BINARY_OR_NUL_HOLD"


def test_markup_txt_is_held() -> None:
    result = tool.classify_content(
        "plain/not-plain.txt",
        b"<?xml version='1.0'?><doc>text</doc>",
        config()["classification_policy"],
    )
    assert result["class"] == "MARKUP_ANNOTATION_HOLD"


def test_report_verifier_rejects_training_promotion() -> None:
    report = {
        "schema_version": tool.REPORT_SCHEMA,
        "worker_id": "D03-RADA-TREES-MEMBER-CLASSIFICATION-20260826",
        "execution_profile": "LOCAL_FREE",
        "dataset": tool.DATASET,
        "dataset_head": tool.DATASET_HEAD,
        "parent": {},
        "config_sha256": "0" * 64,
        "classification": {
            "file_count": 0,
            "class_counts": {},
            "class_bytes": {},
            "plain_text_candidate_member_count": 0,
            "plain_text_candidate_bytes_before_exact_duplicate_collapse": 0,
            "plain_text_candidate_bytes_after_exact_duplicate_collapse": 0,
            "exact_duplicate_group_count": 0,
            "exact_duplicate_groups": [],
            "candidate_path_year_hint_counts": {},
            "members": [],
        },
        "interpretation": {},
        "claim_boundary": {
            "plain_text_member_classification_complete": True,
            "period_provenance_stratification_complete": False,
            "member_rights_terminal": False,
            "member_provenance_terminal": False,
            "language_quality_privacy_complete": False,
            "global_lineage_dedup_complete": False,
            "evaluation_decontamination_complete": False,
            "family_independence_terminal": False,
            "training_authorized_bytes": 1,
            "unique_causal_loss_positions_authorized": 0,
            "tokenizer_fit_authorized": False,
            "model_training_executed": False,
            "optimizer_updates": 0,
            "paid_compute_used": False,
            "research_corpus_v1_released": False,
            "safe_result": "INVALID_TEST_FIXTURE",
        },
        "raw_member_text_emitted": False,
    }
    report["report_sha256"] = tool.canonical_sha256(report)
    with pytest.raises(tool.ClassificationError):
        tool.verify_report(report)
