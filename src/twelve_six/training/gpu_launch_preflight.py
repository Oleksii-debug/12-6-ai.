"""Fail-closed GPU launch qualification for current 12-6 scale candidates."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from twelve_six.checkpoint import hash_json
from twelve_six.model import load_stage_config

SCHEMA = "12-6.gpu-launch-preflight.v1"
FROZEN = "FROZEN"


@dataclass(frozen=True, slots=True)
class Gate:
    name: str
    passed: bool
    detail: str


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )


def _git_sha(repo_root: Path) -> str | None:
    result = _git(repo_root, "rev-parse", "HEAD")
    return result.stdout.strip() if result.returncode == 0 else None


def _is_ancestor(repo_root: Path, sha: str) -> bool:
    return _git(repo_root, "merge-base", "--is-ancestor", sha, "HEAD").returncode == 0


def _hex64(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _add(gates: list[Gate], blockers: list[str], name: str, passed: bool, detail: str) -> None:
    gates.append(Gate(name=name, passed=passed, detail=detail))
    if not passed:
        blockers.append(name)


def _check_runtime_lock(
    repo_root: Path,
    manifest: dict[str, Any],
    gates: list[Gate],
    blockers: list[str],
) -> None:
    runtime = manifest.get("runtime_lock")
    if not isinstance(runtime, dict):
        _add(gates, blockers, "runtime.lock", False, "runtime_lock mapping missing")
        return
    path = repo_root / str(runtime.get("index_path", ""))
    if not path.is_file():
        _add(gates, blockers, "runtime.lock", False, f"missing runtime index {path}")
        return
    index = _load_json(path)
    profile = index.get("profiles", {}).get(runtime.get("profile"), {})
    canonical = index.get("canonical_lock", {})
    passed = (
        index.get("index_sha256") == runtime.get("purpose_index_sha256")
        and canonical.get("file_sha256") == runtime.get("canonical_lock_file_sha256")
        and canonical.get("index_sha256") == runtime.get("canonical_lock_index_sha256")
        and profile.get("profile_sha256") == runtime.get("profile_sha256")
        and profile.get("sha256") == runtime.get("resolved_sha256")
    )
    _add(
        gates,
        blockers,
        "runtime.lock",
        passed,
        f"profile={runtime.get('profile')}; exact D08 identities required",
    )


def _check_freeze(
    scale_name: str,
    scale: dict[str, Any],
    gates: list[Gate],
    blockers: list[str],
) -> None:
    freeze = scale.get("freeze")
    freeze = freeze if isinstance(freeze, dict) else {}
    for kind in ("tokenizer", "corpus", "eval"):
        entry = freeze.get(kind)
        passed = (
            isinstance(entry, dict)
            and entry.get("status") == FROZEN
            and _hex64(entry.get("identity_sha256"))
        )
        status = entry.get("status") if isinstance(entry, dict) else "MISSING"
        _add(
            gates,
            blockers,
            f"{scale_name}.{kind}_freeze",
            passed,
            f"status={status}; FROZEN plus exact identity required",
        )


def _native_bf16_supported() -> bool:
    try:
        return bool(torch.cuda.is_bf16_supported(including_emulation=False))
    except TypeError:
        return bool(torch.cuda.is_bf16_supported())


def _check_precision(
    repo_root: Path,
    source: dict[str, Any],
    scale_name: str,
    precision: str,
    gates: list[Gate],
    blockers: list[str],
) -> dict[str, Any]:
    incumbents = source.get("incumbents")
    incumbents = incumbents if isinstance(incumbents, dict) else {}
    incumbent = incumbents.get("precision")
    incumbent = incumbent if isinstance(incumbent, dict) else {}
    required_head = incumbent.get("head_sha")
    composed = (
        _hex64(required_head)
        and _is_ancestor(repo_root, required_head)
        and (repo_root / "src/twelve_six/training/precision.py").is_file()
    )
    _add(
        gates,
        blockers,
        f"{scale_name}.precision_incumbent_composed",
        composed,
        f"required D02 precision ancestry={required_head}; hardened runtime path must exist",
    )
    cuda = bool(torch.cuda.is_available())
    visible = int(torch.cuda.device_count()) if cuda else 0
    native_bf16: bool | None = None
    if not cuda:
        supported = False
        detail = "NOT_RUN_NO_GPU: CUDA precision support cannot be established"
    elif precision == "bf16":
        native_bf16 = _native_bf16_supported()
        supported = native_bf16
        detail = "native CUDA BF16 required; emulation/fallback is not accepted"
    elif precision in {"fp16", "fp32"}:
        supported = True
        detail = f"CUDA {precision}; hardened runtime semantics still required"
    else:
        supported = False
        detail = f"unsupported precision={precision!r}"
    _add(
        gates,
        blockers,
        f"{scale_name}.precision_hardware_support",
        supported,
        detail,
    )
    return {
        "requested": precision,
        "incumbent_head_sha": required_head,
        "incumbent_composed": composed,
        "cuda_available": cuda,
        "visible_cuda_devices": visible,
        "native_bf16_supported": native_bf16,
    }


def _check_memory(
    scale_name: str,
    scale: dict[str, Any],
    gates: list[Gate],
    blockers: list[str],
) -> dict[str, Any]:
    memory = scale.get("memory_estimate")
    memory = memory if isinstance(memory, dict) else {}
    passed = (
        isinstance(memory.get("estimated_bytes"), int)
        and memory["estimated_bytes"] > 0
        and bool(memory.get("method"))
        and memory.get("estimate_complete") is True
    )
    _add(
        gates,
        blockers,
        f"{scale_name}.memory_estimate",
        passed,
        "complete positive byte estimate required; CPU throughput is not a GPU cost proxy",
    )
    result = dict(memory)
    if torch.cuda.is_available() and passed:
        total = int(torch.cuda.get_device_properties(0).total_memory)
        required = int(memory["estimated_bytes"] * float(memory.get("required_headroom_factor", 1.25)))
        result["device_total_bytes"] = total
        result["required_with_headroom_bytes"] = required
        _add(
            gates,
            blockers,
            f"{scale_name}.memory_capacity",
            total >= required,
            f"device={total}; required_with_headroom={required}",
        )
    else:
        result["device_total_bytes"] = None
        result["required_with_headroom_bytes"] = None
    return result


def _check_checkpoint_recovery_auth(
    scale_name: str,
    scale: dict[str, Any],
    gates: list[Gate],
    blockers: list[str],
) -> None:
    checkpoint = scale.get("checkpoint")
    checkpoint_ok = (
        isinstance(checkpoint, dict)
        and isinstance(checkpoint.get("path"), str)
        and bool(checkpoint["path"].strip())
        and checkpoint.get("fresh_object_restore_required") is True
        and checkpoint.get("durability_status") == "APPROVED"
    )
    _add(
        gates,
        blockers,
        f"{scale_name}.checkpoint_path",
        checkpoint_ok,
        "approved durable checkpoint target plus fresh-object restore required",
    )

    recovery = scale.get("recovery_policy")
    recovery_ok = (
        isinstance(recovery, dict)
        and recovery.get("status") == "APPROVED"
        and recovery.get("restore_last_verified_checkpoint") is True
        and recovery.get("retry_same_in_memory_state_after_oom") is False
        and isinstance(recovery.get("retain_verified_generations"), int)
        and recovery["retain_verified_generations"] >= 2
    )
    _add(
        gates,
        blockers,
        f"{scale_name}.recovery_policy",
        recovery_ok,
        "approved LKG/fresh-process recovery with >=2 verified generations required",
    )

    authorization = scale.get("authorization")
    authorization_ok = (
        isinstance(authorization, dict)
        and authorization.get("field") == "COMPUTE_AUTHORIZED"
        and authorization.get("compute_authorized") is True
        and isinstance(authorization.get("authorization_id"), str)
        and bool(authorization["authorization_id"].strip())
    )
    _add(
        gates,
        blockers,
        f"{scale_name}.authorization",
        authorization_ok,
        "explicit COMPUTE_AUTHORIZED=true plus non-empty authorization_id required",
    )


def _run_10m_smoke(
    repo_root: Path,
    scale: dict[str, Any],
    output_root: Path,
    *,
    skip_smoke: bool,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {
            "status": "NOT_RUN_NO_GPU",
            "cuda_available": False,
            "visible_cuda_devices": 0,
            "tokens_per_second": None,
            "cuda_peak_allocated_bytes": None,
            "cuda_peak_reserved_bytes": None,
        }
    visible = int(torch.cuda.device_count())
    if skip_smoke:
        return {
            "status": "AVAILABLE_BUT_SKIPPED",
            "cuda_available": True,
            "visible_cuda_devices": visible,
            "tokens_per_second": None,
            "cuda_peak_allocated_bytes": None,
            "cuda_peak_reserved_bytes": None,
        }
    if visible != 1:
        return {
            "status": "NOT_RUN_VISIBLE_DEVICE_COUNT_NOT_ONE",
            "cuda_available": True,
            "visible_cuda_devices": visible,
            "tokens_per_second": None,
            "cuda_peak_allocated_bytes": None,
            "cuda_peak_reserved_bytes": None,
        }

    output_root.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(repo_root / "tools/run_single_gpu_pilot.py"),
        "--config",
        str(repo_root / str(scale["run_config"])),
        "--output-dir",
        str(output_root),
        "--device",
        "cuda:0",
    ]
    process = subprocess.run(
        command,
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if process.returncode != 0:
        return {
            "status": "FAILED",
            "returncode": process.returncode,
            "stdout_tail": process.stdout[-4000:],
            "stderr_tail": process.stderr[-4000:],
            "tokens_per_second": None,
            "cuda_peak_allocated_bytes": None,
            "cuda_peak_reserved_bytes": None,
        }
    try:
        summary = json.loads(process.stdout)
    except json.JSONDecodeError:
        return {
            "status": "FAILED_INVALID_SUMMARY",
            "stdout_tail": process.stdout[-4000:],
            "stderr_tail": process.stderr[-4000:],
            "tokens_per_second": None,
            "cuda_peak_allocated_bytes": None,
            "cuda_peak_reserved_bytes": None,
        }
    return {
        "status": "PASS",
        "device": summary.get("device"),
        "git_sha": summary.get("git_sha"),
        "model_spec_sha256": summary.get("model_spec_sha256"),
        "precision_runtime": summary.get("precision_runtime"),
        "tokens_per_second": summary.get("tokens_per_second"),
        "cuda_peak_allocated_bytes": summary.get("cuda_peak_allocated_bytes"),
        "cuda_peak_reserved_bytes": summary.get("cuda_peak_reserved_bytes"),
        "checkpoint_bytes": summary.get("checkpoint_bytes"),
    }


def evaluate_preflight(
    repo_root: Path,
    manifest_path: Path,
    *,
    output_root: Path,
    skip_smoke: bool = False,
) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    if manifest.get("schema_version") != SCHEMA:
        raise ValueError(f"expected schema_version={SCHEMA}")
    gates: list[Gate] = []
    blockers: list[str] = []

    source = manifest.get("source")
    source = source if isinstance(source, dict) else {}
    base_sha = source.get("required_product_base_sha")
    source_ok = isinstance(base_sha, str) and _hex64(base_sha) and _is_ancestor(repo_root, base_sha)
    _add(
        gates,
        blockers,
        "source.product_base_ancestry",
        source_ok,
        f"required_base={base_sha}; head={_git_sha(repo_root)}",
    )
    required_paths = source.get("required_paths")
    paths_ok = (
        isinstance(required_paths, list)
        and bool(required_paths)
        and all((repo_root / str(path)).is_file() for path in required_paths)
    )
    _add(
        gates,
        blockers,
        "source.incumbent_surfaces",
        paths_ok,
        "observability/checkpoint/single-GPU/runtime incumbent surfaces must exist",
    )
    _check_runtime_lock(repo_root, manifest, gates, blockers)

    scales = manifest.get("scales")
    if not isinstance(scales, dict):
        raise TypeError("scales mapping is required")
    campaign = manifest.get("campaign_incumbent")
    campaign_ok = (
        isinstance(campaign, dict)
        and campaign.get("status") == "BOUND_TO_CURRENT_PRODUCT"
        and campaign.get("10m_model_spec_sha256") == scales.get("10m", {}).get("model_spec_sha256")
        and campaign.get("100m_model_spec_sha256") == scales.get("100m", {}).get("model_spec_sha256")
    )
    _add(
        gates,
        blockers,
        "campaign.current_product_binding",
        campaign_ok,
        "campaign authority must be regenerated/rebound to exact current 10M/100M identities",
    )

    scale_reports: dict[str, Any] = {}
    for scale_name in ("10m", "100m"):
        scale = scales.get(scale_name)
        if not isinstance(scale, dict):
            _add(gates, blockers, f"{scale_name}.manifest", False, "scale entry missing")
            continue
        stage_path = repo_root / str(scale.get("stage_config", ""))
        try:
            stage = load_stage_config(stage_path)
            model_ok = (
                stage.model.identity_sha256() == scale.get("model_spec_sha256")
                and stage.model.parameter_count() == scale.get("parameter_count")
                and stage.init.identity_sha256() == scale.get("init_spec_sha256")
            )
            model_detail = (
                f"model={stage.model.identity_sha256()} params={stage.model.parameter_count()} "
                f"init={stage.init.identity_sha256()}"
            )
        except (OSError, TypeError, ValueError, KeyError) as exc:
            model_ok = False
            model_detail = f"{type(exc).__name__}: {exc}"
        _add(gates, blockers, f"{scale_name}.exact_model_spec", model_ok, model_detail)

        run_path = repo_root / str(scale.get("run_config", ""))
        run_payload = _load_json(run_path) if run_path.is_file() else {}
        actual_run_hash = hash_json(run_payload) if run_payload else None
        run_ok = (
            str(run_payload.get("stage_config")) == str(scale.get("stage_config"))
            and actual_run_hash == scale.get("run_manifest_sha256")
        )
        _add(
            gates,
            blockers,
            f"{scale_name}.run_manifest_binding",
            run_ok,
            f"run={scale.get('run_config')} sha256={actual_run_hash} stage={scale.get('stage_config')}",
        )
        run_campaign_status = scale.get("run_manifest_status")
        _add(
            gates,
            blockers,
            f"{scale_name}.run_manifest_campaign_status",
            run_campaign_status == "APPROVED_CAMPAIGN",
            f"status={run_campaign_status}; APPROVED_CAMPAIGN required",
        )

        _check_freeze(scale_name, scale, gates, blockers)
        precision = str(scale.get("precision", run_payload.get("precision", "fp32")))
        precision_report = _check_precision(
            repo_root,
            source,
            scale_name,
            precision,
            gates,
            blockers,
        )
        memory_report = _check_memory(scale_name, scale, gates, blockers)
        _check_checkpoint_recovery_auth(scale_name, scale, gates, blockers)
        scale_reports[scale_name] = {
            "stage_config": scale.get("stage_config"),
            "run_config": scale.get("run_config"),
            "run_manifest_sha256": actual_run_hash,
            "run_manifest_status": run_campaign_status,
            "parameter_count": scale.get("parameter_count"),
            "model_spec_sha256": scale.get("model_spec_sha256"),
            "init_spec_sha256": scale.get("init_spec_sha256"),
            "freeze": scale.get("freeze"),
            "precision": precision_report,
            "memory_estimate": memory_report,
            "checkpoint": scale.get("checkpoint"),
            "recovery_policy": scale.get("recovery_policy"),
            "authorization": scale.get("authorization"),
        }

    smoke = _run_10m_smoke(
        repo_root,
        scales["10m"],
        output_root,
        skip_smoke=skip_smoke,
    )
    _add(
        gates,
        blockers,
        "10m.cuda_smoke",
        smoke.get("status") == "PASS",
        str(smoke.get("status")),
    )
    return {
        "schema_version": SCHEMA,
        "status": "PASS_LAUNCH_READY" if not blockers else "BLOCKED",
        "launch_ready": not blockers,
        "git_sha": _git_sha(repo_root),
        "required_product_base_sha": base_sha,
        "runtime_lock": manifest.get("runtime_lock"),
        "campaign_incumbent": campaign,
        "scales": scale_reports,
        "smoke_10m": smoke,
        "gates": [asdict(gate) for gate in gates],
        "blockers": blockers,
        "cost_truth_boundary": (
            "Never derive euro cost from CPU throughput. Cost projection requires measured "
            "accelerator throughput on the bound geometry plus an explicit current rate."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail-closed 12-6 GPU launch preflight")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/compute/gpu_launch_preflight.current.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Skip an already-visible CUDA smoke. This tool never provisions compute.",
    )
    parser.add_argument("--assert-launch-ready", action="store_true")
    args = parser.parse_args()

    repo_root = Path.cwd()
    report = evaluate_preflight(
        repo_root,
        args.manifest,
        output_root=args.output.parent / "cuda-smoke",
        skip_smoke=args.skip_smoke,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True, indent=2))
    return 2 if args.assert_launch_ready and not report["launch_ready"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
