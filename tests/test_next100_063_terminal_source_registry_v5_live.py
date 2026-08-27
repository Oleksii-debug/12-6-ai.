from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/validate_next100_063_terminal_source_registry_v5_live.py"
spec = importlib.util.spec_from_file_location("next100_063_v5_live", TOOL)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _run() -> dict[str, object]:
    return {
        "id": module.EXPECTED_ATTRS_RUN,
        "name": module.EXPECTED_ATTRS_WORKFLOW,
        "path": module.EXPECTED_ATTRS_WORKFLOW_PATH,
        "event": "pull_request",
        "head_sha": module.EXPECTED_ATTRS_HEAD,
        "status": "completed",
        "conclusion": "success",
        "pull_requests": [
            {
                "number": module.EXPECTED_ATTRS_PR,
                "head": {"sha": module.EXPECTED_ATTRS_HEAD},
            }
        ],
    }


def _artifacts() -> dict[str, object]:
    return {
        "artifacts": [
            {
                "id": module.EXPECTED_ATTRS_ARTIFACT_ID,
                "name": module.EXPECTED_ATTRS_ARTIFACT_NAME,
                "digest": module.EXPECTED_ATTRS_ARTIFACT_DIGEST,
                "expired": False,
                "workflow_run": {
                    "id": module.EXPECTED_ATTRS_RUN,
                    "head_sha": module.EXPECTED_ATTRS_HEAD,
                },
            }
        ]
    }


def test_exact_attrs_execution_evidence_passes() -> None:
    module.validate_attrs_live(_run(), _artifacts())


def test_nonterminal_or_wrong_head_run_fails_closed() -> None:
    for field, value in (("status", "queued"), ("head_sha", "0" * 40)):
        run = copy.deepcopy(_run())
        run[field] = value
        with pytest.raises(module.LiveAuthorityError):
            module.validate_attrs_live(run, _artifacts())


def test_artifact_digest_or_expiry_fails_closed() -> None:
    for field, value in (("digest", "sha256:" + "0" * 64), ("expired", True)):
        artifacts = copy.deepcopy(_artifacts())
        artifacts["artifacts"][0][field] = value
        with pytest.raises(module.LiveAuthorityError):
            module.validate_attrs_live(_run(), artifacts)
