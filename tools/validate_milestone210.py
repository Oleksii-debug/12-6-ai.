from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M210 = ROOT / "evidence" / "milestone210" / "learned-base-ladder-v2.json"
EXPECTED_SOURCE = "5838cd16869dcfcf762368d8673eddf52d51b7e3"
EXPECTED_EVAL = "7189e6df053574beb686727c94e684cdbaf08a34ef33aa953eff7cdae0320113"
EXPECTED_CORPUS = "422f545da93526ceba2c0ff738e0b9deac65f35bfb31d87d2aab068cda091fc8"
EXPECTED_PARAMS = {"100k": 95568, "500k": 467808, "1m": 1037696}


def canonical_sha(obj: dict, field: str) -> str:
    copy = dict(obj)
    copy.pop(field, None)
    return hashlib.sha256(
        json.dumps(
            copy,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()


def validate_manifest(m: dict) -> None:
    assert canonical_sha(m, "manifest_sha256") == m["manifest_sha256"]
    assert m["terminal_producer"]["source_sha"] == EXPECTED_SOURCE
    assert m["terminal_producer"]["conclusion"] == "success"
    assert m["terminal_producer"]["workflow_run_id"] == 32937411703
    assert m["terminal_producer"]["artifact_id"] == 9595677772
    assert m["common"]["corpus"]["sha256"] == EXPECTED_CORPUS
    assert m["common"]["corpus"]["origin"] == "PROJECT_AUTHORED"
    assert m["common"]["evaluation_identity_sha256"] == EXPECTED_EVAL
    assert m["common"]["optimized_tokens"] == 948504

    for scale, params in EXPECTED_PARAMS.items():
        r = m["rungs"][scale]
        assert r["status"] == "ADMITTED_TERMINAL_LEARNED"
        assert r["parameters"] == params
        assert r["random_init"] is True
        assert [step for step, _ in r["selection_validation"]] == [
            0,
            250,
            500,
            750,
            1000,
        ]
        assert set(r["best"]) >= {
            "checkpoint_id",
            "step",
            "bpb",
            "ua",
            "en",
            "code",
        }
        assert set(r["final"]) >= {
            "checkpoint_id",
            "step",
            "bpb",
            "ua",
            "en",
            "code",
        }
        assert r["resume"] == {
            "fresh_process": True,
            "passed": True,
            "loaded_step": 500,
            "first_resumed_optimizer_step": 501,
        }
        assert r["fresh_verification"]["status"] == "PASS"
        for gate in (
            "checkpoint_load",
            "checkpoint_identity",
            "first_party_logits",
            "generation",
            "evaluation_non_mutation",
            "reproducibility_manifest_validation",
        ):
            assert r["fresh_verification"][gate] is True
        assert set(r["first_party_logits_sha256"]) == {"uk", "en", "code"}
        assert set(r["raw_base_generation"]) == {
            "random_init",
            "best_checkpoint",
            "final_checkpoint",
        }

    assert [x["scale"] for x in m["quality_ranking"]] == [
        "1m",
        "500k",
        "100k",
    ]
    assert m["excluded"]["3m"]["status"].startswith("NOT_ADMITTED")
    assert m["excluded"]["10m"]["status"].startswith("NOT_ADMITTED")
    assert (
        m["excluded"]["100m"]["status"]
        == "QUALIFICATION_MECHANICS_ONLY_NOT_LEARNED"
    )
    assert (
        m["external_real_boundary"]["learned_rungs_use_external_real_training_data"]
        is False
    )


def validate_terminal_m150(path: Path, m: dict) -> None:
    src = json.loads(path.read_text())
    assert canonical_sha(src, "report_sha256") == src["report_sha256"]
    assert (
        src["report_sha256"]
        == m["terminal_producer"]["ladder_report_identity_sha256"]
    )
    assert src["source"]["git_sha"] == EXPECTED_SOURCE
    assert (
        src["truth_model"]["evaluation_identity"]["identity_sha256"]
        == EXPECTED_EVAL
    )
    assert src["truth_model"]["corpus_identity_sha256"] == EXPECTED_CORPUS

    for scale in EXPECTED_PARAMS:
        r = m["rungs"][scale]
        s = src["scales"][scale]
        assert s["model"]["parameter_count"] == r["parameters"]
        assert s["model"]["spec_sha256"] == r["model_spec_sha256"]
        assert s["checkpoints"]["best_checkpoint_id"] == r["best"]["checkpoint_id"]
        assert (
            s["checkpoints"]["final_checkpoint_id"]
            == r["final"]["checkpoint_id"]
        )
        assert s["training"]["optimized_tokens"] == 948504
        assert s["fresh_verification"]["status"] == "PASS"


def validate(m150_report: Path | None = None) -> None:
    m = json.loads(M210.read_text())
    validate_manifest(m)
    if m150_report is not None:
        validate_terminal_m150(m150_report, m)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m150-report", type=Path)
    args = parser.parse_args()
    validate(args.m150_report)
    print("MILESTONE-210 terminal ladder validation: PASS")
