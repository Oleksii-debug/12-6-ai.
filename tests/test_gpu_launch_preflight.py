from __future__ import annotations

import json
from pathlib import Path

from twelve_six import load_stage_config
from twelve_six.training import gpu_launch_preflight as preflight

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/compute/gpu_launch_preflight.current.json"


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_current_scale_bindings_are_exact() -> None:
    manifest = _manifest()
    for name in ("10m", "100m"):
        scale = manifest["scales"][name]
        stage = load_stage_config(ROOT / scale["stage_config"])
        assert stage.model.parameter_count() == scale["parameter_count"]
        assert stage.model.identity_sha256() == scale["model_spec_sha256"]
        assert stage.init.identity_sha256() == scale["init_spec_sha256"]
        run = json.loads((ROOT / scale["run_config"]).read_text(encoding="utf-8"))
        assert run["stage_config"] == scale["stage_config"]


def test_current_manifest_cannot_authorize_campaign() -> None:
    manifest = _manifest()
    assert manifest["campaign_incumbent"]["status"] == "STALE_NOT_LAUNCH_AUTHORITY"
    for scale in manifest["scales"].values():
        assert scale["authorization"]["field"] == "COMPUTE_AUTHORIZED"
        assert scale["authorization"]["compute_authorized"] is False
        assert not scale["authorization"]["authorization_id"]
        assert scale["freeze"]["corpus"]["status"] != preflight.FROZEN
        assert scale["freeze"]["eval"]["status"] != preflight.FROZEN


def test_10m_memory_is_explicitly_lower_bound_only() -> None:
    scale = _manifest()["scales"]["10m"]
    assert scale["memory_estimate"]["estimated_bytes"] == 160010240
    assert scale["memory_estimate"]["estimate_complete"] is False
    assert "activations" in scale["memory_estimate"]["method"]


def test_no_cuda_is_retained_as_not_run_no_gpu(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(preflight.torch.cuda, "is_available", lambda: False)
    result = preflight._run_10m_smoke(
        ROOT,
        _manifest()["scales"]["10m"],
        tmp_path,
        skip_smoke=False,
    )
    assert result["status"] == "NOT_RUN_NO_GPU"
    assert result["tokens_per_second"] is None
    assert result["cuda_peak_allocated_bytes"] is None
    assert result["cuda_peak_reserved_bytes"] is None


def test_freeze_gate_requires_status_and_exact_identity() -> None:
    gates: list[preflight.Gate] = []
    blockers: list[str] = []
    scale = {
        "freeze": {
            "tokenizer": {"status": "FROZEN", "identity_sha256": "a" * 64},
            "corpus": {"status": "FROZEN", "identity_sha256": "b" * 64},
            "eval": {"status": "FROZEN", "identity_sha256": "c" * 64},
        }
    }
    preflight._check_freeze("x", scale, gates, blockers)
    assert not blockers
    assert all(gate.passed for gate in gates)


def test_runtime_lock_is_current_d08_identity() -> None:
    runtime = _manifest()["runtime_lock"]
    index = json.loads((ROOT / runtime["index_path"]).read_text(encoding="utf-8"))
    profile = index["profiles"][runtime["profile"]]
    assert index["index_sha256"] == runtime["purpose_index_sha256"]
    assert index["canonical_lock"]["file_sha256"] == runtime["canonical_lock_file_sha256"]
    assert index["canonical_lock"]["index_sha256"] == runtime["canonical_lock_index_sha256"]
    assert profile["profile_sha256"] == runtime["profile_sha256"]
    assert profile["sha256"] == runtime["resolved_sha256"]
