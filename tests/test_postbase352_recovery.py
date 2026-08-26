from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from twelve_six.post_base.data_contract import (
    CommunicationDataError,
    CommunicationSplit,
    SyntheticDataAuthority,
    _load_jsonl,
    _parse_record,
    _validate_provenance,
    _validate_split_isolation,
    to_posttraining_records,
    validate_dataset,
)
from twelve_six.posttraining.contracts import Split


class PostBase352RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.dataset_root = cls.repo_root / "data" / "post_base" / "communication_v1"
        cls.manifest_path = cls.dataset_root / "manifest.json"

    def test_project_authored_seed_has_ua_and_en_in_every_physical_split(self) -> None:
        expected_files = {
            CommunicationSplit.TRAIN: "train.jsonl",
            CommunicationSplit.SELECTION: "selection.jsonl",
            CommunicationSplit.FINAL: "final.jsonl",
        }
        for split, filename in expected_files.items():
            records = _load_jsonl(self.dataset_root / filename, split)
            self.assertTrue(records)
            self.assertTrue(all(not record.foreign_model_output for record in records))
            self.assertTrue(all(record.rights == "project_owned" for record in records))
            self.assertEqual({record.language for record in records}, {"en", "uk"})

    def test_final_is_operationally_ineligible_for_selection(self) -> None:
        train = _load_jsonl(
            self.dataset_root / "train.jsonl", CommunicationSplit.TRAIN
        )[0]
        selection = _load_jsonl(
            self.dataset_root / "selection.jsonl", CommunicationSplit.SELECTION
        )[0]
        final = _load_jsonl(
            self.dataset_root / "final.jsonl", CommunicationSplit.FINAL
        )[0]

        rows = to_posttraining_records(selection, for_selection=True)
        self.assertEqual(rows[0].split, Split.VALIDATION)
        with self.assertRaises(CommunicationDataError):
            to_posttraining_records(final, for_selection=True)
        with self.assertRaises(CommunicationDataError):
            to_posttraining_records(train, for_selection=True)
        with self.assertRaises(CommunicationDataError):
            to_posttraining_records(selection, for_training=True, for_selection=True)

    def test_near_duplicates_fail_closed_even_within_one_split(self) -> None:
        train = _load_jsonl(
            self.dataset_root / "train.jsonl", CommunicationSplit.TRAIN
        )[0]
        messages = list(train.messages)
        role, content = messages[-1]
        messages[-1] = (role, content + "!")
        near_duplicate = replace(
            train,
            record_id="comm-v1-train-near-duplicate",
            family_id="en-direct-cpu-near-duplicate",
            messages=tuple(messages),
        )
        with self.assertRaisesRegex(CommunicationDataError, "near duplicate"):
            _validate_split_isolation((train, near_duplicate), 0.85)

    def test_v1_split_file_names_cannot_be_aliased_or_renamed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "communication_v1"
            shutil.copytree(self.dataset_root, root)
            shutil.copy2(root / "final.jsonl", root / "final-copy.jsonl")
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["split_files"]["final"] = "final-copy.jsonl"
            manifest_path.write_text(
                json.dumps(
                    manifest,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(CommunicationDataError, "physically separate"):
                validate_dataset(root, manifest_path)

    def test_foreign_output_requires_authority_bound_rights(self) -> None:
        base = _load_jsonl(
            self.dataset_root / "train.jsonl", CommunicationSplit.TRAIN
        )[0]
        authority = SyntheticDataAuthority(
            authority_id="SYNTH-AUTH-001",
            authority_sha256="a" * 64,
            allowed_source_ids=("teacher:future-example",),
            owner_approved=True,
        )
        foreign = replace(
            base,
            source_id="teacher:future-example",
            rights="project_owned",
            foreign_model_output=True,
            synthetic_authority_id="SYNTH-AUTH-001",
        )
        with self.assertRaisesRegex(CommunicationDataError, "authority_bound"):
            _validate_provenance(foreign, authority)

    def test_synthetic_authority_source_ids_must_be_unique(self) -> None:
        with self.assertRaisesRegex(CommunicationDataError, "must be unique"):
            SyntheticDataAuthority(
                authority_id="SYNTH-AUTH-001",
                authority_sha256="a" * 64,
                allowed_source_ids=("teacher:one", "teacher:one"),
                owner_approved=True,
            )

    def test_hidden_reasoning_target_gate_cannot_be_disabled(self) -> None:
        row = json.loads(
            (self.dataset_root / "train.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()[0]
        )
        row["quality"]["no_hidden_reasoning"] = False
        with self.assertRaisesRegex(CommunicationDataError, "no-hidden-reasoning"):
            _parse_record(row)

    def test_manifest_still_denies_base_and_training_eligibility(self) -> None:
        audit = validate_dataset(self.dataset_root, self.manifest_path)
        self.assertEqual(audit.foreign_model_records, 0)
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        for field in (
            "base_corpus_evidence",
            "canonical_base_training_eligible",
            "training_authorized",
            "selection_for_training",
            "final_for_training",
            "final_for_selection",
        ):
            self.assertIs(manifest[field], False)
        self.assertIsNone(manifest["synthetic_data_authority"])


if __name__ == "__main__":
    unittest.main()
