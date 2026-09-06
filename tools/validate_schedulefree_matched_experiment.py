"""Fail-closed validator for the D02 Schedule-Free matched-arm contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

EXPECTED_PARENT = "e4ff486fd90802fc123bebf60eed4e59196a98df"
EXPECTED_MODEL = "fbff24d561a2818453554d58ca23fc6ace3303b078f1935a8576c4565bd92441"
EXPECTED_INIT = "86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5"
EXPECTED_PACKAGE = "schedulefree"
EXPECTED_VERSION = "1.4.1"
EXPECTED_SOURCE = "facebookresearch/schedule_free"
EXPECTED_SOURCE_COMMIT = "70785b53e778d0e872c0bbb75ff4ee54ee10c291"
EXPECTED_LICENSE = "Apache-2.0"
EXPECTED_LRS = [0.00016, 0.00022, 0.00026]


class ContractError(ValueError):
    pass


def _identity_payload(document: dict[str, Any]) -> bytes:
    canonical = dict(document)
    canonical.pop("experiment_identity_sha256", None)
    return json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def experiment_identity(document: dict[str, Any]) -> str:
    return hashlib.sha256(_identity_payload(document)).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def validate(document: dict[str, Any]) -> str:
    _require(document.get("schema_version") == 1, "schema_version must be 1")
    _require(document.get("status") == "CANDIDATE_MECHANICS_ONLY", "candidate status mismatch")
    _require(document.get("swarm_protocol") == "SWARM-300-V2", "swarm protocol mismatch")
    _require(document.get("product_parent_sha") == EXPECTED_PARENT, "product parent mismatch")

    model = document.get("model_identity")
    _require(isinstance(model, dict), "model_identity missing")
    _require(model.get("model_spec_sha256") == EXPECTED_MODEL, "model spec mismatch")
    _require(model.get("init_spec_sha256") == EXPECTED_INIT, "init spec mismatch")
    _require(model.get("parameter_count") == 20613440, "parameter count mismatch")

    upstream = document.get("upstream_optimizer")
    _require(isinstance(upstream, dict), "upstream_optimizer missing")
    _require(upstream.get("package") == EXPECTED_PACKAGE, "package mismatch")
    _require(upstream.get("package_version") == EXPECTED_VERSION, "package version mismatch")
    _require(upstream.get("source_repo") == EXPECTED_SOURCE, "source repository mismatch")
    _require(upstream.get("source_commit") == EXPECTED_SOURCE_COMMIT, "source commit mismatch")
    _require(upstream.get("license") == EXPECTED_LICENSE, "license mismatch")
    _require(upstream.get("optimizer_class") == "AdamWScheduleFree", "optimizer class mismatch")
    _require(upstream.get("inner_momentum") == 0.0, "inner_momentum must be 0.0")
    _require(upstream.get("foreach") is False, "foreach must be false")

    control = document.get("adamw_control_provenance")
    _require(isinstance(control, dict), "AdamW control provenance missing")
    _require(control.get("pr") == 583, "AdamW control PR mismatch")
    _require(control.get("scoped_workflow_conclusion") == "success", "scoped control evidence not success")
    _require(control.get("shared_ci_conclusion") == "failure", "shared CI truth must remain failure")

    matched = document.get("matched_variables")
    _require(isinstance(matched, dict), "matched_variables missing")
    _require(matched.get("learning_rate_grid") == EXPECTED_LRS, "learning-rate grid mismatch")
    _require(matched.get("betas") == [0.9, 0.95], "betas mismatch")
    _require(matched.get("eps") == 1e-8, "eps mismatch")
    _require(matched.get("weight_decay") == 0.1, "weight decay mismatch")
    _require(matched.get("gradient_clip_norm") == 1.0, "clip norm mismatch")
    _require(matched.get("scheduler") == "constant", "scheduler must be constant")
    _require(matched.get("warmup_steps") == 0, "warmup must be zero")
    _require(matched.get("precision") == "fp32", "precision mismatch")
    _require(matched.get("seed") == 1337, "seed mismatch")
    _require(matched.get("sequence_length") == 256, "sequence length mismatch")
    _require(matched.get("micro_batch_size") == 1, "micro batch mismatch")
    _require(matched.get("gradient_accumulation_steps") == 1, "accumulation mismatch")
    _require(matched.get("optimizer_updates") == 32, "update count mismatch")
    expected_tokens = 32 * (256 - 1) * 1 * 1
    _require(matched.get("target_tokens_per_arm") == expected_tokens, "target-token budget mismatch")
    _require(
        matched.get("data_scope") == "synthetic_local_mechanical_fixture_only",
        "learned/unknown data scope is forbidden for this contract",
    )

    arms = document.get("arms")
    _require(isinstance(arms, list) and len(arms) == 2, "exactly two arms required")
    by_id = {arm.get("arm_id"): arm for arm in arms if isinstance(arm, dict)}
    _require(set(by_id) == {"adamw_control", "schedulefree_candidate"}, "arm identities mismatch")
    control_arm = by_id["adamw_control"]
    candidate_arm = by_id["schedulefree_candidate"]
    _require(control_arm.get("optimizer_kind") == "torch_adamw", "control optimizer mismatch")
    _require(candidate_arm.get("optimizer_kind") == "schedulefree_adamw", "candidate optimizer mismatch")
    _require(control_arm.get("learning_rate_grid") == EXPECTED_LRS, "control LR grid mismatch")
    _require(candidate_arm.get("learning_rate_grid") == EXPECTED_LRS, "candidate LR grid mismatch")
    _require(candidate_arm.get("package") == EXPECTED_PACKAGE, "candidate package mismatch")
    _require(candidate_arm.get("package_version") == EXPECTED_VERSION, "candidate version mismatch")
    _require(candidate_arm.get("inner_momentum") == 0.0, "candidate inner momentum mismatch")
    _require(candidate_arm.get("foreach") is False, "candidate foreach mismatch")

    forbidden = document.get("forbidden_claims")
    _require(isinstance(forbidden, dict) and forbidden, "forbidden_claims missing")
    _require(all(value is False for value in forbidden.values()), "forbidden scientific/compute claim set true")
    _require(document.get("scientific_conclusion") == "not_run_no_winner", "scientific conclusion overclaims")

    calculated = experiment_identity(document)
    _require(
        document.get("experiment_identity_sha256") == calculated,
        "experiment identity hash mismatch",
    )
    return calculated


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        nargs="?",
        default="configs/candidates/schedulefree_adamw_matched_v1.json",
    )
    args = parser.parse_args()
    path = Path(args.path)
    document = json.loads(path.read_text(encoding="utf-8"))
    identity = validate(document)
    print(json.dumps({"status": "PASS", "experiment_identity_sha256": identity}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
