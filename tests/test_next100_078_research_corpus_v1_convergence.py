from __future__ import annotations

import unittest

from twelve_six.data.research_corpus_v1_convergence import ConvergenceError, validate_convergence


def _source(source_id: str, family: str, modality: str, capacity: int) -> dict:
    return {
        "source_id": source_id,
        "source_family": family,
        "modality": modality,
        "declared_capacity_bytes": capacity,
    }


def _base_config() -> dict:
    return {
        "sources": [
            _source("se8", "en.standardebooks.manual", "en", 48002),
            _source("se9", "en.standardebooks.manual", "en", 36791),
            _source("rada", "ua.rada.open-data.laws-texts", "uk", 88565),
            _source("httpx", "github:encode/httpx", "code", 8161),
            _source("requests", "github:psf/requests", "code", 1542),
            _source(
                "wikisource",
                "ua.literature.lesia-ukrainka.na-krylah-pisen.1892-lviv",
                "uk",
                1479,
            ),
            _source("django1", "github:django/django", "code", 9594),
            _source("django2", "github:django/django", "code", 12506),
            _source("django3", "github:django/django", "code", 32056),
            _source("star1", "github:Kludex/starlette", "code", 2970),
            _source("star2", "github:Kludex/starlette", "code", 2304),
        ]
    }


def _workflow(name: str, run_id: int) -> dict:
    return {"name": name, "run_id": run_id, "conclusion": "success"}


def _manifest() -> dict:
    return {
        "schema_version": "12-6.next100-078-research-corpus-v1-authority-convergence.v1",
        "worker_id": "NEXT100-078-RESEARCH-CORPUS-V1-AUTHORITY-CONVERGENCE",
        "execution_class": "LOCAL_FREE",
        "training_executed": False,
        "compute_authorized": False,
        "base_vector": {
            "name": "NEXT100-065-CROSSSOURCE-DEDUP-V3",
            "head_sha": "efc278cec0e4773eb4ff405bf4b4d24ee63b5d13",
            "config_path": "configs/data/next100_065_cross_source_dedup_v3.json",
            "expected_capacity_bytes": 243970,
            "expected_family_counts": {"uk": 2, "en": 1, "code": 4},
        },
        "mixture_policy": {
            "uk": 0.45,
            "en": 0.35,
            "code": 0.20,
            "replay_allowed": False,
            "planning_target_source_bytes": 20000000,
        },
        "minimum_independent_families": {"uk": 2, "en": 2, "code": 2},
        "additive_authorities": [
            {
                "authority_id": "NEXT100-026",
                "pr_number": 449,
                "head_sha": "40950a950b60921fd856af2719e1ae2486d9e892",
                "dedicated_workflow": _workflow(
                    "NEXT100-026 KMu Source Rights Audit",
                    32997970539,
                ),
                "verdict": "ADMIT",
                "modality": "uk",
                "source_family": "ua.kmu.portal.secretariat-news",
                "declared_capacity_bytes": 9153,
                "authority_identity_kind": "source_manifest_identity_sha256",
                "authority_identity_sha256": (
                    "1f068e6cc5ce3fc4a51d8477acee31fab5a0178e15f49225b57de94c5178f7d9"
                ),
                "materialization_state": "NOT_COMPOSED_REMOTE_AUTHORITY",
            },
            {
                "authority_id": "NEXT100-038",
                "pr_number": 445,
                "head_sha": "902eccc0b3efff09a38dc89cda789180b6c6e754",
                "dedicated_workflow": _workflow(
                    "NEXT100-038 MDN Source Authority",
                    32998544359,
                ),
                "verdict": "ADMIT_PROSE_ONLY",
                "modality": "en",
                "source_family": "en.mdn.webdocs.prose",
                "declared_capacity_bytes": 6492,
                "authority_identity_kind": "authority_identity_sha256",
                "authority_identity_sha256": (
                    "0f5dbd5313f8196811e2a99f77eb8698c6bc69f69648d76a7e240ee9757ecc47"
                ),
                "normalized_sha256": (
                    "10855740b0ed5588d133f421318c637be99d9e9f4921675af9f6dc8a5663507b"
                ),
                "materialization_state": "NOT_COMPOSED_REMOTE_AUTHORITY",
            },
            {
                "authority_id": "NEXT100-034",
                "pr_number": 472,
                "head_sha": "b7491745b34ac8679baaf69cb96cd609dcbe0a16",
                "dedicated_workflow": _workflow(
                    "NEXT100-034 NIST authority",
                    32998703545,
                ),
                "verdict": "ADMIT",
                "modality": "en",
                "source_family": "en.usgov.nist.technical-series",
                "declared_capacity_bytes": 59358,
                "authority_identity_kind": "terminal_payload_sha256",
                "authority_identity_sha256": (
                    "3ffba0fcd08ab42e940b2db12ffafb6f7234ad0bae6f7fe523071497485b9d1c"
                ),
                "normalized_objects": [
                    {
                        "source_id": "NIST.SP.800-204",
                        "bytes": 19668,
                        "sha256": (
                            "570e8d75b6dc6aefee1f089818b46765c0dd1965e06947bcc2fff0169d22274e"
                        ),
                    },
                    {
                        "source_id": "NIST.SP.800-204C",
                        "bytes": 19736,
                        "sha256": (
                            "558da6a0886036a01a5139d635b1352b5cf5d74655d919c66a04e84f2d49c0fe"
                        ),
                    },
                    {
                        "source_id": "NIST.SP.800-215",
                        "bytes": 19954,
                        "sha256": (
                            "6c99c3b14ee3ea7fe915940e38c080dbf2a785f1abcee2fd73e7fd731424770d"
                        ),
                    },
                ],
                "materialization_state": "NOT_COMPOSED_REMOTE_AUTHORITY",
            },
        ],
    }


class ResearchCorpusV1ConvergenceTests(unittest.TestCase):
    def test_expected_authority_vector(self) -> None:
        report = validate_convergence(_manifest(), _base_config())
        self.assertEqual(report["decision"], "PASS_DIVERSITY_ONLY_BLOCK_EXACT_CORPUS")
        self.assertEqual(
            report["composed_authority_capacity_bytes"],
            {"uk": 99197, "en": 150643, "code": 69133},
        )
        self.assertEqual(report["composed_authority_total_bytes"], 318973)
        self.assertEqual(
            report["independent_family_counts"],
            {"uk": 3, "en": 3, "code": 4},
        )
        self.assertEqual(report["authority_diversity_gate"], "PASS")
        self.assertEqual(report["max_nonreplay_mixture_source_bytes"], 220437)
        self.assertEqual(report["limiting_modalities"], ["uk"])
        self.assertEqual(report["remaining_source_capacity_gap_total_bytes"], 19681027)
        self.assertFalse(report["training_authorized"])

    def test_authority_set_identity_is_deterministic(self) -> None:
        first = validate_convergence(_manifest(), _base_config())
        second = validate_convergence(_manifest(), _base_config())
        self.assertEqual(first["authority_set_identity_sha256"], second["authority_set_identity_sha256"])
        self.assertEqual(
            first["authority_set_identity_sha256"],
            "24831f5388303ee4dfaa1186269f3cd0f52989dc67e58ac546d9dd18a5faf3db",
        )

    def test_nonterminal_workflow_is_rejected(self) -> None:
        manifest = _manifest()
        manifest["additive_authorities"][0]["dedicated_workflow"]["conclusion"] = "queued"
        with self.assertRaisesRegex(ConvergenceError, "not terminal"):
            validate_convergence(manifest, _base_config())

    def test_duplicate_family_credit_is_rejected(self) -> None:
        manifest = _manifest()
        manifest["additive_authorities"][0]["source_family"] = "ua.rada.open-data.laws-texts"
        with self.assertRaisesRegex(ConvergenceError, "duplicate independent-family credit"):
            validate_convergence(manifest, _base_config())

    def test_base_capacity_drift_is_rejected(self) -> None:
        base = _base_config()
        base["sources"][0]["declared_capacity_bytes"] += 1
        with self.assertRaisesRegex(ConvergenceError, "base capacity drift"):
            validate_convergence(_manifest(), base)

    def test_nist_object_bytes_must_match_declared_capacity(self) -> None:
        manifest = _manifest()
        manifest["additive_authorities"][2]["normalized_objects"][0]["bytes"] -= 1
        with self.assertRaisesRegex(ConvergenceError, "normalized object bytes"):
            validate_convergence(manifest, _base_config())


if __name__ == "__main__":
    unittest.main()
