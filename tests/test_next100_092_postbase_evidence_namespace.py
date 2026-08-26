from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from twelve_six.postbase_evidence_firewall import (  # noqa: E402
    EvidenceEnvelope,
    NamespaceViolation,
    validate_audit_manifest,
    validate_post_base_envelope,
)

MANIFEST = (
    ROOT / "configs/post_base/next100_092_evidence_namespace_audit_v1.json"
)


class PostBaseEvidenceNamespaceFirewallTests(unittest.TestCase):
    def test_frozen_ten_component_audit_manifest_passes(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        validate_audit_manifest(manifest)

    def test_adapter_may_reference_base_provenance_without_promotion(self) -> None:
        validate_post_base_envelope(
            EvidenceEnvelope(
                component_id="model_adapter",
                artifact_kind="generation",
                artifact_path="evidence/post_base/adapter/generation.json",
                payload={
                    "base_evidence": {
                        "evidence_namespace": "base",
                        "checkpoint_id": "learned-10m",
                        "git_sha": "c" * 40,
                        "model_spec_sha256": "a" * 64,
                    },
                    "post_base_evidence": {
                        "evidence_namespace": "post_base",
                        "controller": "deliberation",
                        "stop_reason": "max_new_tokens",
                    },
                },
            )
        )

    def test_base_lineage_is_not_base_evidence_authority(self) -> None:
        validate_post_base_envelope(
            EvidenceEnvelope(
                component_id="tools",
                artifact_kind="tool_cycle",
                payload={
                    "lineage": "BASE",
                    "observations": [
                        {
                            "training_eligible": False,
                            "weight_update_eligible": False,
                        }
                    ],
                },
            )
        )

    def test_base_namespace_reference_string_for_separation_is_allowed(self) -> None:
        validate_post_base_envelope(
            EvidenceEnvelope(
                component_id="communication_eval",
                artifact_kind="evaluation_manifest",
                payload={
                    "separation": {
                        "base_evidence_namespace": "evidence/base",
                        "base_raw_lm_diagnostics": False,
                        "post_base_evidence_namespace": "evidence/post_base/eval354",
                        "training_eligible": False,
                    }
                },
            )
        )

    def test_hypothesis_internal_evidence_remains_post_base(self) -> None:
        validate_post_base_envelope(
            EvidenceEnvelope(
                component_id="hypothesis_search",
                artifact_kind="search_export",
                payload={
                    "schema": "12-6.postbase-hypothesis-search.v1",
                    "evidence": [
                        {
                            "id": "E001",
                            "kind": "support",
                            "source": "deterministic_fixture",
                        }
                    ],
                    "selected_hypothesis_id": "H002",
                },
            )
        )

    def test_rejects_envelope_targeting_base_namespace(self) -> None:
        with self.assertRaises(NamespaceViolation):
            validate_post_base_envelope(
                EvidenceEnvelope(
                    component_id="deliberation",
                    artifact_kind="behavior_result",
                    payload={"score": 1.0},
                    evidence_namespace="base",
                )
            )

    def test_rejects_post_base_artifact_under_base_path(self) -> None:
        with self.assertRaises(NamespaceViolation):
            validate_post_base_envelope(
                EvidenceEnvelope(
                    component_id="verifier",
                    artifact_kind="verifier_result",
                    artifact_path="evidence/base/postbase-verdict.json",
                    payload={"status": "PASS"},
                )
            )

    def test_rejects_payload_relabeling_as_base_scientific_evidence(self) -> None:
        with self.assertRaises(NamespaceViolation):
            validate_post_base_envelope(
                EvidenceEnvelope(
                    component_id="memory_rag",
                    artifact_kind="retrieval_result",
                    payload={
                        "evidence_namespace": "base",
                        "evidence": [{"memory_id": "M001"}],
                    },
                )
            )

    def test_rejects_canonical_base_training_flag(self) -> None:
        with self.assertRaises(NamespaceViolation):
            validate_post_base_envelope(
                EvidenceEnvelope(
                    component_id="communication_data",
                    artifact_kind="dataset_manifest",
                    payload={"canonical_base_training_eligible": True},
                )
            )

    def test_rejects_synthetic_teacher_record_promoted_to_base(self) -> None:
        with self.assertRaises(NamespaceViolation):
            validate_post_base_envelope(
                EvidenceEnvelope(
                    component_id="teacher_factory",
                    artifact_kind="synthetic_dataset_record",
                    payload={
                        "base_corpus_evidence": False,
                        "canonical_base_training_eligible": False,
                        "training_use": "CANONICAL_BASE_TRAINING",
                    },
                )
            )

    def test_rejects_post_base_result_smuggled_into_base_provenance(self) -> None:
        with self.assertRaises(NamespaceViolation):
            validate_post_base_envelope(
                EvidenceEnvelope(
                    component_id="model_adapter",
                    artifact_kind="generation",
                    payload={
                        "base_evidence": {
                            "evidence_namespace": "base",
                            "checkpoint_id": "learned-10m",
                            "score": 0.99,
                        }
                    },
                )
            )

    def test_audit_manifest_fails_if_one_component_is_promoted(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        mutated = copy.deepcopy(manifest)
        mutated["components"][5]["canonical_base_scientific_evidence"] = True
        with self.assertRaises(NamespaceViolation):
            validate_audit_manifest(mutated)


if __name__ == "__main__":
    unittest.main()
