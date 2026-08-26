#!/usr/bin/env python3
"""Fail-closed immutable/live authority validator for NEXT100-065C."""
from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path
from typing import Any

CONFIG = Path("configs/data/next100_065c_cross_source_dedup_v5.json")
REPO = "Oleksii-debug/12-6-ai."

EXPECTED_ACCEPTED = [
    "604c7243b37696ca7173dbdf9dd2b5663f54075590e599b741243f882739e869",
    "65eff0ae0fbb474ee32bc02f12111f57775474d1b6375c3dd0894ec666d05267",
    "49771a2eedc6cae523e10e7fa04feac4e79713e324d7fe4d37577689ded1b279",
    "1998f1a712c71a4cdec3d6f9693b0de79451004fcc5bdc78149d7d7a180427af",
    "647bda8d594fec2c8be35b0505986f9f5e885d2fba8692838cb6f48494b8bc04",
    "68bb8862c2b25dc702412cc1470410f7a218498dbbcdfb52a4db3269bd973d26",
    "ab11186bb4c1fb0ddc849693cff0b4f230b162dc41d6908d6807f1f0f86ac6c2",
    "aab52bdb384c5503d34d524edb06cacdc0c7ea03385c5413bd1db29cb7f7f388",
    "bb78d8aad6fcd8ccf9408bd64a998aebedff14804aa79f78421cdd1fcdd91fef",
    "a19741280be37f9268367aa31e8c3eedc1876b839b9aca4d13299f2872153b55",
    "c9e8a1dc6709e6e50dadc9f22269f3ba149c2c4ba0f1b53ae32f0811acab79cb",
    "e67ce9871a098147df5caa26900518e61422ce789252a766ae046a6eb4fde742",
    "3b86d261ef94dd7b0deb0c577faaa41b9026f50cd18abee6c5eb84aa5aeb38ee",
    "a09756447fdbd535629939d1bcaf8db5f6fba4b23bdc9468e27625f67c11e470",
]

LIVE_AUTHORITIES = {
    449: ("40950a950b60921fd856af2719e1ae2486d9e892", 32997970539),
    462: ("d75edd497c7fb1054e86d892c9462f059c1f4aa9", 32998503672),
    472: ("b7491745b34ac8679baaf69cb96cd609dcbe0a16", 32998703545),
    445: ("902eccc0b3efff09a38dc89cda789180b6c6e754", 32998544359),
    467: ("5a6a495a24bce449334cbc5126d0114f61a9f57c", 32998356906),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"NEXT100-065C FAIL: {message}")


def load_config() -> dict[str, Any]:
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "config root must be object")
    return value


def validate_static(config: dict[str, Any]) -> None:
    require(config.get("schema_version") == "12-6.next100-065c-cross-source-dedup.v5", "schema drift")
    require(config.get("worker_id") == "NEXT100-065C-CROSSSOURCE-DEDUP-V5", "worker drift")
    require(config.get("local_free_only") is True, "LOCAL_FREE weakened")
    for key in ("model_training_executed", "tokenizer_fit_executed", "paid_compute_used", "final_test_payload_read"):
        require(config.get(key) is False, f"execution boundary weakened: {key}")

    base = config.get("base_v4", {})
    require(base.get("pr") == 576, "base PR drift")
    require(base.get("head_sha") == "5738bb8bac8fda058d5ae9c1361c4a0c3756f360", "base V4 head drift")
    require(base.get("source_object_count") == 21, "base object count drift")
    require(base.get("source_family_counts") == {"uk": 4, "en": 2, "code": 4}, "base family drift")
    require(base.get("source_capacity_bytes") == {"uk": 100856, "en": 144151, "code": 69133, "total": 314140}, "base capacity drift")

    mdn = config.get("mdn", {})
    mdn_expected = {
        "worker": "NEXT100-038-DATA-EN-MDN",
        "pr": 445,
        "head_sha": "902eccc0b3efff09a38dc89cda789180b6c6e754",
        "dedicated_workflow_run": 32998544359,
        "dedicated_workflow_conclusion": "success",
        "authority_identity_sha256": "0f5dbd5313f8196811e2a99f77eb8698c6bc69f69648d76a7e240ee9757ecc47",
        "raw_bytes": 11280,
        "raw_sha256": "8bde46ef0fc270baf85ca8fb55a5b5662b49ef7f6d1c948825ded9377e019638",
        "git_blob_sha1": "528fb9e09861897eca0661cb03178dd47afee5ef",
        "normalized_bytes": 6492,
        "normalized_sha256": "10855740b0ed5588d133f421318c637be99d9e9f4921675af9f6dc8a5663507b",
        "source_family": "en.mdn.webdocs.prose",
        "training": "ALLOWED_UNDER_LICENSE_TERMS",
        "evaluation": "NOT_SEPARATELY_ADMITTED",
    }
    for key, value in mdn_expected.items():
        require(mdn.get(key) == value, f"MDN binding drift: {key}")

    cp = config.get("cpython", {})
    cp_expected = {
        "worker": "NEXT100-037-DATA-EN-PYTHON-DOCS",
        "pr": 467,
        "head_sha": "5a6a495a24bce449334cbc5126d0114f61a9f57c",
        "dedicated_workflow_run": 32998356906,
        "dedicated_workflow_conclusion": "success",
        "authority_identity_sha256": "46a00dc70db690ae2b3c4495a75283e7e752bdccb1047d4318c2ebadfa392f0d",
        "raw_bytes": 19188,
        "raw_sha256": "cf1674daf9568abeb5fc22f62a991e17751fea4deb06f598362ce6e7de264808",
        "git_blob_sha1": "465c32d0b72431cc446aae7edeb6b829c657b243",
        "normalized_source_bytes": 17901,
        "normalized_source_sha256": "64a4ec4fd7574ba4c22e615a032b157e446b9c7f5a7917cb7f10fa214a05bd1a",
        "chunk_count": 16,
        "accepted_chunk_count": 14,
        "rejected_chunk_count": 2,
        "rejection_reasons": {"pii_phone": 2},
        "source_family": "python.cpython.documentation",
        "training": "ALLOWED_ACCEPTED_CHUNKS_ONLY",
        "evaluation": "NOT_SEPARATELY_ADMITTED",
    }
    for key, value in cp_expected.items():
        require(cp.get(key) == value, f"CPython binding drift: {key}")
    require(cp.get("accepted_normalized_sha256") == EXPECTED_ACCEPTED, "CPython accepted hash vector drift")
    require(cp.get("quality_policy") == {
        "min_chars": 60,
        "max_chars": 1600,
        "min_alpha_ratio": 0.35,
        "reject_control_characters": True,
        "reject_email": True,
        "reject_phone": True,
    }, "CPython quality policy drift")
    require(cp.get("chunking") == {"max_chars": 1200, "min_chars": 80}, "CPython chunking drift")

    expected = config.get("expected_vector", {})
    require(expected.get("source_object_count") == 23, "expected source count drift")
    require(expected.get("source_family_counts") == {"uk": 4, "en": 4, "code": 4}, "expected family vector drift")
    require(expected.get("fixed_capacity_without_cpython_accepted_chunks") == {"uk": 100856, "en": 150643, "code": 69133, "total": 320632}, "fixed capacity drift")
    require(expected.get("full_cpython_normalized_bytes_must_not_be_credited") == 17901, "CPython full-byte prohibition drift")

    boundary = config.get("claim_boundary", {})
    for key in (
        "canonical_registry_replaced", "corpus_materialized", "decontamination_pass_claimed",
        "balance_release_claimed", "postpack_unique_loss_ledger_complete", "tokenizer_fit_authorized",
        "training_authorized", "paid_compute_authorized", "research_corpus_v1_terminal",
    ):
        require(boundary.get(key) is False, f"truth boundary weakened: {key}")


def github_get(path: str) -> dict[str, Any]:
    token = os.environ.get("GITHUB_TOKEN")
    require(bool(token), "GITHUB_TOKEN is required for --github-live")
    request = urllib.request.Request(
        f"https://api.github.com/repos/{REPO}/{path}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "12-6-ai-next100-065c-authority-validator",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        value = json.loads(response.read().decode("utf-8"))
    require(isinstance(value, dict), f"GitHub response not object: {path}")
    return value


def validate_live() -> None:
    for pr, (head, run_id) in LIVE_AUTHORITIES.items():
        pr_data = github_get(f"pulls/{pr}")
        require(pr_data.get("head", {}).get("sha") == head, f"PR #{pr} head moved")
        run = github_get(f"actions/runs/{run_id}")
        require(run.get("head_sha") == head, f"PR #{pr} dedicated run head mismatch")
        require(run.get("status") == "completed", f"PR #{pr} dedicated run nonterminal")
        require(run.get("conclusion") == "success", f"PR #{pr} dedicated run not success")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--github-live", action="store_true")
    args = parser.parse_args()
    config = load_config()
    validate_static(config)
    if args.github_live:
        validate_live()
    print("NEXT100-065C AUTHORITY PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
