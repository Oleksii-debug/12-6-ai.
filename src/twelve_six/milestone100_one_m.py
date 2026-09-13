"""~1M model selection for the MILESTONE-100 learned Base convergence run.

This module changes only model selection and the corresponding report validator.
All data, packing, Trainer, observability, checkpoint, evaluation and inference
execution remains owned by milestone100_first_learned and its incumbents.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from twelve_six import milestone100_first_learned as milestone
from twelve_six.checkpoint import hash_json
from twelve_six.model import InitSpec
from twelve_six.scaling_experiment import controlled_specs

EXPECTED_PARAMETERS = 1_037_696
RESEARCH41_HEAD = "9775a3432795dde9c96b3e84f6de143b2033a08c"
RESEARCH41_MODEL_FAMILY_BLOB = "04b5c3173f2af139e9228e422cc1245a533a6c5d"
_ORIGINAL_RUN_MANIFEST = milestone._run_manifest


def json_normalize(value: Any) -> Any:
    """Return the canonical JSON data-model representation of ``value``."""
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _normalized_run_manifest(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Preserve run-manifest identity while making fresh-process comparison stable."""
    value = _ORIGINAL_RUN_MANIFEST(*args, **kwargs)
    normalized = json_normalize(value)
    if not isinstance(normalized, dict):
        raise milestone.MilestoneError("MILESTONE-100 run manifest must normalize to an object")
    supplied = normalized.get("identity_sha256")
    unsigned = dict(normalized)
    unsigned.pop("identity_sha256", None)
    if not isinstance(supplied, str) or supplied != hash_json(unsigned):
        raise milestone.MilestoneError("run-manifest identity changed during JSON normalization")
    return normalized


def _one_m_model(repo: Path):
    del repo
    spec = controlled_specs()[-1]
    if spec.parameter_count() != EXPECTED_PARAMETERS:
        raise milestone.MilestoneError(
            f"RESEARCH41 ~1M geometry drift: {spec.parameter_count()} != {EXPECTED_PARAMETERS}"
        )
    if spec.vocab_size != 256 or spec.max_seq_len != 256:
        raise milestone.MilestoneError("RESEARCH41 ~1M byte/context contract drift")
    init = InitSpec()
    return spec, init, {
        "incumbent": "RESEARCH41 controlled_specs()[-1]",
        "incumbent_head_sha": RESEARCH41_HEAD,
        "incumbent_source_blob_sha": RESEARCH41_MODEL_FAMILY_BLOB,
        "parameter_count": EXPECTED_PARAMETERS,
        "selection_reason": (
            "strongest already-proven fixed-byte-tokenizer controlled geometry in the "
            "requested ~100K-1M small-model band; RESEARCH41 data/trajectory evidence is not reused"
        ),
        "geometry_changes_by_milestone": "NONE",
    }


def _validate_one_m(path: Path, expected_source_sha: str | None = None):
    r = milestone._read_json(path)
    supplied = r["report_sha256"]
    unsigned = dict(r)
    unsigned.pop("report_sha256")
    if supplied != hash_json(unsigned):
        raise milestone.MilestoneError("report self-hash mismatch")
    if r["schema"] != milestone.SCHEMA or r["authority"] != milestone.AUTHORITY:
        raise milestone.MilestoneError("report schema/authority mismatch")
    if expected_source_sha and r["source"]["git_sha"] != expected_source_sha:
        raise milestone.MilestoneError("report source mismatch")
    if r["model"]["parameter_count"] != EXPECTED_PARAMETERS:
        raise milestone.MilestoneError("~1M parameter gate failed")
    if r["model"]["runtime_parameter_count"] != EXPECTED_PARAMETERS:
        raise milestone.MilestoneError("~1M runtime parameter gate failed")
    if r["model"]["geometry_provenance"]["incumbent_head_sha"] != RESEARCH41_HEAD:
        raise milestone.MilestoneError("RESEARCH41 geometry provenance gate failed")
    if not r["training"]["train_loss_decreased"]:
        raise milestone.MilestoneError("train-loss gate failed")
    if not r["evaluation"]["heldout_bits_per_byte_decreased"]:
        raise milestone.MilestoneError("held-out BPB gate failed")
    if not r["evaluation"]["evaluation_non_mutation"]:
        raise milestone.MilestoneError("eval mutation gate failed")
    if not r["runtime"]["fresh_process_resume"]["passed"]:
        raise milestone.MilestoneError("resume gate failed")
    if r["truth_boundary"]["external_real_world_training_data_present"] is not False:
        raise milestone.MilestoneError("corpus truth boundary weakened")
    if r["success"]["overall_requested_milestone"] != "PARTIAL_FAIL_CLOSED":
        raise milestone.MilestoneError("full milestone must remain fail-closed on real corpus gate")


def install_override() -> None:
    milestone._model = _one_m_model
    milestone.validate = _validate_one_m
    milestone._run_manifest = _normalized_run_manifest


def main(argv=None) -> int:
    install_override()
    return milestone.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
