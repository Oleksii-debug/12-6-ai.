from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path

from twelve_six.post_base.contract import TokenizerCompatibility
from twelve_six.post_base.data_contract import (
    CommunicationDataError,
    CommunicationSplit,
    SyntheticDataAuthority,
    _load_jsonl,
    _parse_record,
    _validate_provenance,
    _validate_split_isolation,
    require_exact_base_tokenizer,
    require_logical_tokenizer_compatibility,
    to_posttraining_records,
    validate_dataset,
)
from twelve_six.posttraining.contracts import Split


class CommunicationDataContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.dataset_root = cls.repo_root / "data" / "post_base" / "communication_v1"
        cls.manifest_path = cls.dataset_root / "manifest.json"

    def test_seed_dataset_validates_and_is_post_base_only(self) -> None:
        audit = validate_dataset(self.dataset_root, self.manifest_path)
        self.assertEqual(audit.record_counts, {"train": 12, "selection": 4, "final": 4})
        self.assertEqual(audit.foreign_model_records, 0)
        self.assertLessEqual(audit.max_sft_example_bytes_observed, 256)
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["classification"], "POSTBASE_COMMUNICATION_ONLY")
        self.assertFalse(manifest["base_corpus_evidence"])
        self.assertFalse(manifest["canonical_base_training_eligible"])
        self.assertFalse(manifest["training_authorized"])

    def test_selection_and_final_cannot_be_training_rows(self) -> None:
        selection = _load_jsonl(
            self.dataset_root / "selection.jsonl", CommunicationSplit.SELECTION
        )[0]
        final = _load_jsonl(self.dataset_root / "final.jsonl", CommunicationSplit.FINAL)[0]
        self.assertEqual(to_posttraining_records(selection)[0].split, Split.VALIDATION)
        self.assertEqual(to_posttraining_records(final)[0].split, Split.TEST)
        with self.assertRaises(CommunicationDataError):
            to_posttraining_records(selection, for_training=True)
        with self.assertRaises(CommunicationDataError):
            to_posttraining_records(final, for_training=True)

    def test_foreign_output_requires_explicit_later_authority(self) -> None:
        base = _load_jsonl(self.dataset_root / "train.jsonl", CommunicationSplit.TRAIN)[0]
        foreign = replace(
            base,
            source_id="teacher:future-example",
            rights="authority_bound",
            foreign_model_output=True,
            synthetic_authority_id="SYNTH-AUTH-001",
        )
        with self.assertRaises(CommunicationDataError):
            _validate_provenance(foreign, None)
        authority = SyntheticDataAuthority(
            authority_id="SYNTH-AUTH-001",
            authority_sha256="a" * 64,
            allowed_source_ids=("teacher:future-example",),
            owner_approved=True,
        )
        _validate_provenance(foreign, authority)

    def test_family_and_near_duplicate_leakage_fail_closed(self) -> None:
        train = _load_jsonl(self.dataset_root / "train.jsonl", CommunicationSplit.TRAIN)[0]
        selection = _load_jsonl(
            self.dataset_root / "selection.jsonl", CommunicationSplit.SELECTION
        )[0]
        with self.assertRaises(CommunicationDataError):
            _validate_split_isolation((train, replace(selection, family_id=train.family_id)), 0.85)
        with self.assertRaises(CommunicationDataError):
            _validate_split_isolation(
                (train, replace(selection, family_id="other-family", messages=train.messages)),
                0.85,
            )

    def test_dialogue_role_contract_is_strict(self) -> None:
        line = (self.dataset_root / "train.jsonl").read_text(encoding="utf-8").splitlines()[0]
        row = json.loads(line)
        row["messages"][0]["role"] = "system"
        with self.assertRaises(CommunicationDataError):
            _parse_record(row)

    def test_tokenizer_profile_and_exact_identity_are_separate_gates(self) -> None:
        expected = TokenizerCompatibility("s0-byte-v1", "a" * 64, "b" * 64, 256)
        same = TokenizerCompatibility("s0-byte-v1", "a" * 64, "b" * 64, 256)
        require_logical_tokenizer_compatibility(expected)
        require_exact_base_tokenizer(expected, same)
        drifted = TokenizerCompatibility("s0-byte-v1", "c" * 64, "b" * 64, 256)
        with self.assertRaises(ValueError):
            require_exact_base_tokenizer(expected, drifted)
        wrong = TokenizerCompatibility("other-tokenizer", "a" * 64, "b" * 64, 256)
        with self.assertRaises(CommunicationDataError):
            require_logical_tokenizer_compatibility(wrong)


if __name__ == "__main__":
    unittest.main()
