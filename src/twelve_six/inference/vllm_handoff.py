"""Fail-closed vLLM handoff preflight for verified 12-6 Base exports.

This module intentionally does not import vLLM or claim that the current S0
architecture is supported by vLLM. It binds a D07 parity report to the exact
D05 export bytes and reports whether the artifact is eligible for a future
out-of-tree vLLM model plugin handoff.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

EXPORT_ATTESTATION_NAME = "12-6-export.json"
PARITY_REQUEST_NAME = "12-6-parity-request.json"
CHECKPOINT_MANIFEST_COPY_NAME = "12-6-checkpoint-manifest.json"
CONFIG_NAME = "config.json"
WEIGHTS_NAME = "model.safetensors"

PARITY_SCHEMA = "12-6.inference-parity.v1"
PARITY_BINDING_SCHEMA = "12-6.vllm-parity-binding.v1"
PREFLIGHT_SCHEMA = "12-6.vllm-handoff-preflight.v1"
EXPORT_SCHEMA = "12-6.hf-style-export.v1"
PARITY_REQUEST_SCHEMA = "12-6.export-parity-request.v1"

REQUIRED_MODEL_ARCHITECTURE = "TwelveSixForCausalLM"
REQUIRED_MODEL_TYPE = "twelve_six"
VLLM_PLUGIN_GROUP = "vllm.general_plugins"
VLLM_REGISTRATION_API = "vllm.ModelRegistry.register_model"

_MAX_JSON_BYTES = 1_048_576


class VllmHandoffError(ValueError):
    """Raised when parity evidence cannot be bound safely to an export."""


@dataclass(frozen=True, slots=True)
class VllmHandoffPreflight:
    ready_for_plugin_implementation: bool
    checkpoint_id: str | None
    model_architecture: str
    model_type: str
    weights_sha256: str | None
    config_sha256: str | None
    parity_report_sha256: str | None
    plugin_group: str
    registration_api: str
    vllm_runtime_status: str
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["schema"] = PREFLIGHT_SCHEMA
        return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _json_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise VllmHandoffError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise VllmHandoffError(f"required JSON file is not a regular file: {path.name}")
    size = path.stat().st_size
    if size > _MAX_JSON_BYTES:
        raise VllmHandoffError(f"JSON file exceeds {_MAX_JSON_BYTES} bytes: {path.name}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_json_no_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VllmHandoffError(f"invalid UTF-8 JSON in {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise VllmHandoffError(f"JSON root must be an object: {path.name}")
    return value


def _regular_payload(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


def _export_identities(export_dir: Path) -> tuple[dict[str, Any], str, str]:
    if export_dir.is_symlink() or not export_dir.is_dir():
        raise VllmHandoffError("export directory must be a regular directory path")

    weights = export_dir / WEIGHTS_NAME
    config = export_dir / CONFIG_NAME
    source_manifest_path = export_dir / CHECKPOINT_MANIFEST_COPY_NAME
    if not _regular_payload(weights):
        raise VllmHandoffError(f"missing or non-regular {WEIGHTS_NAME}")
    if not _regular_payload(config):
        raise VllmHandoffError(f"missing or non-regular {CONFIG_NAME}")
    if not _regular_payload(source_manifest_path):
        raise VllmHandoffError(f"missing or non-regular {CHECKPOINT_MANIFEST_COPY_NAME}")

    attestation = _load_json_object(export_dir / EXPORT_ATTESTATION_NAME)
    if attestation.get("schema") != EXPORT_SCHEMA:
        raise VllmHandoffError("unsupported HF-style export attestation schema")

    weights_sha = _sha256_file(weights)
    config_sha = _sha256_file(config)
    if attestation.get("model_safetensors_sha256") != weights_sha:
        raise VllmHandoffError("exported model.safetensors hash does not match attestation")
    if attestation.get("config_sha256") != config_sha:
        raise VllmHandoffError("exported config.json hash does not match attestation")

    checkpoint_id = attestation.get("checkpoint_id")
    if not isinstance(checkpoint_id, str) or not checkpoint_id:
        raise VllmHandoffError("export attestation checkpoint_id is missing")

    source_manifest_sha = _sha256_file(source_manifest_path)
    if attestation.get("source_manifest_sha256") != source_manifest_sha:
        raise VllmHandoffError("copied source manifest hash does not match export attestation")
    source_manifest = _load_json_object(source_manifest_path)
    if source_manifest.get("checkpoint_id") != checkpoint_id:
        raise VllmHandoffError("copied source manifest checkpoint_id does not match export")

    request = _load_json_object(export_dir / PARITY_REQUEST_NAME)
    if request.get("schema") != PARITY_REQUEST_SCHEMA:
        raise VllmHandoffError("unsupported export parity-request schema")
    if request.get("checkpoint_id") != checkpoint_id:
        raise VllmHandoffError("parity request checkpoint_id does not match export attestation")
    if request.get("reference_weights_sha256") != weights_sha:
        raise VllmHandoffError("parity request reference weight hash does not match export")
    if request.get("candidate_weights_sha256") != weights_sha:
        raise VllmHandoffError("parity request candidate weight hash does not match export")
    if request.get("candidate_config_sha256") != config_sha:
        raise VllmHandoffError("parity request candidate config hash does not match export")

    required_checks = request.get("required_checks")
    if not isinstance(required_checks, list):
        raise VllmHandoffError("parity request required_checks must be a list")
    if not all(isinstance(item, str) for item in required_checks):
        raise VllmHandoffError("parity request required_checks entries must be strings")
    required = {
        "prompt_token_identity",
        "next_token_logit_parity",
        "greedy_generation_parity",
    }
    if not required.issubset(set(required_checks)):
        raise VllmHandoffError("parity request is missing required canonical comparison checks")

    return attestation, weights_sha, config_sha


def bind_parity_report(
    export_dir: str | Path,
    parity_report_path: str | Path,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Bind a passing D07 parity report to exact D05 export bytes.

    The current D07 parity schema is intentionally backend-neutral and does not
    carry artifact hashes. This binding envelope prevents a passing report for
    one candidate from being reused as evidence for a different export.
    """

    export = Path(export_dir)
    attestation, weights_sha, config_sha = _export_identities(export)
    report_path = Path(parity_report_path)
    report = _load_json_object(report_path)
    if report.get("schema") != PARITY_SCHEMA:
        raise VllmHandoffError("unsupported inference parity report schema")
    if report.get("passed") is not True:
        raise VllmHandoffError("cannot bind a parity report that did not pass")
    failures = report.get("failures")
    if failures != []:
        raise VllmHandoffError("passing parity report must contain an empty failures list")
    prompts_compared = report.get("prompts_compared")
    steps_compared = report.get("steps_compared")
    if (
        not isinstance(prompts_compared, int)
        or isinstance(prompts_compared, bool)
        or prompts_compared < 1
    ):
        raise VllmHandoffError("parity report must compare at least one prompt")
    if (
        not isinstance(steps_compared, int)
        or isinstance(steps_compared, bool)
        or steps_compared < 1
    ):
        raise VllmHandoffError("parity report must compare at least one logit step")

    output = Path(output_path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"parity binding already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": PARITY_BINDING_SCHEMA,
        "checkpoint_id": attestation["checkpoint_id"],
        "candidate_weights_sha256": weights_sha,
        "candidate_config_sha256": config_sha,
        "parity_report_sha256": _sha256_file(report_path),
        "parity_report_canonical_sha256": hashlib.sha256(
            _canonical_json_bytes(report)
        ).hexdigest(),
        "parity_report": report,
    }
    output.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output


def inspect_vllm_handoff(
    export_dir: str | Path,
    parity_binding_path: str | Path | None = None,
) -> VllmHandoffPreflight:
    """Return fail-closed readiness for a future out-of-tree vLLM model plugin."""

    export = Path(export_dir)
    blockers: list[str] = []
    checkpoint_id: str | None = None
    weights_sha: str | None = None
    config_sha: str | None = None
    parity_report_sha: str | None = None

    try:
        attestation, weights_sha, config_sha = _export_identities(export)
        checkpoint_id = str(attestation["checkpoint_id"])
    except (OSError, VllmHandoffError) as exc:
        blockers.append(f"export_integrity: {exc}")
        return VllmHandoffPreflight(
            ready_for_plugin_implementation=False,
            checkpoint_id=checkpoint_id,
            model_architecture=REQUIRED_MODEL_ARCHITECTURE,
            model_type=REQUIRED_MODEL_TYPE,
            weights_sha256=weights_sha,
            config_sha256=config_sha,
            parity_report_sha256=None,
            plugin_group=VLLM_PLUGIN_GROUP,
            registration_api=VLLM_REGISTRATION_API,
            vllm_runtime_status="NOT_TESTED",
            blockers=tuple(blockers),
        )

    try:
        config = _load_json_object(export / CONFIG_NAME)
    except (OSError, VllmHandoffError) as exc:
        blockers.append(f"config: {exc}")
        config = {}
    if config.get("model_type") != REQUIRED_MODEL_TYPE:
        blockers.append(f"config.model_type must equal {REQUIRED_MODEL_TYPE!r}")
    if config.get("architectures") != [REQUIRED_MODEL_ARCHITECTURE]:
        blockers.append(
            "config.architectures must contain exactly "
            f"[{REQUIRED_MODEL_ARCHITECTURE!r}]"
        )

    compatibility = attestation.get("compatibility")
    if not isinstance(compatibility, dict):
        blockers.append("export attestation compatibility object is missing")
    else:
        if compatibility.get("layout") != "HF_STYLE_SAFETENSORS_DIRECTORY":
            blockers.append("export layout is not the canonical HF-style SafeTensors layout")
        if compatibility.get("weights") != "EXACT_CANONICAL_BYTE_COPY":
            blockers.append("export weights are not attested as an exact canonical byte copy")
        if compatibility.get("transformers_architecture") not in {"PASS", "VERIFIED"}:
            blockers.append("Transformers architecture compatibility is not verified")
        if compatibility.get("runtime_logit_generation_parity") not in {"PASS", "VERIFIED"}:
            blockers.append("export attestation runtime parity is not verified")

    if parity_binding_path is None:
        blockers.append("artifact-bound D07 parity evidence is missing")
    else:
        try:
            binding_path = Path(parity_binding_path)
            binding = _load_json_object(binding_path)
            if binding.get("schema") != PARITY_BINDING_SCHEMA:
                blockers.append("unsupported vLLM parity-binding schema")
            if binding.get("checkpoint_id") != checkpoint_id:
                blockers.append("parity binding checkpoint_id does not match export")
            if binding.get("candidate_weights_sha256") != weights_sha:
                blockers.append("parity binding weight hash does not match export")
            if binding.get("candidate_config_sha256") != config_sha:
                blockers.append("parity binding config hash does not match export")
            report = binding.get("parity_report")
            if not isinstance(report, dict) or report.get("schema") != PARITY_SCHEMA:
                blockers.append("parity binding does not contain a supported D07 report")
            else:
                if report.get("passed") is not True or report.get("failures") != []:
                    blockers.append("bound D07 parity report is not a clean PASS")
                prompts_compared = report.get("prompts_compared")
                steps_compared = report.get("steps_compared")
                if (
                    not isinstance(prompts_compared, int)
                    or isinstance(prompts_compared, bool)
                    or prompts_compared < 1
                ):
                    blockers.append("bound D07 parity report compared no prompts")
                if (
                    not isinstance(steps_compared, int)
                    or isinstance(steps_compared, bool)
                    or steps_compared < 1
                ):
                    blockers.append("bound D07 parity report compared no logit steps")
                canonical_report_sha = hashlib.sha256(
                    _canonical_json_bytes(report)
                ).hexdigest()
                if binding.get("parity_report_canonical_sha256") != canonical_report_sha:
                    blockers.append("embedded parity report canonical hash mismatch")
            expected_report_sha = binding.get("parity_report_sha256")
            if not isinstance(expected_report_sha, str) or len(expected_report_sha) != 64:
                blockers.append("parity binding report file hash is malformed")
            else:
                parity_report_sha = expected_report_sha
        except (OSError, VllmHandoffError) as exc:
            blockers.append(f"parity_binding: {exc}")

    return VllmHandoffPreflight(
        ready_for_plugin_implementation=not blockers,
        checkpoint_id=checkpoint_id,
        model_architecture=REQUIRED_MODEL_ARCHITECTURE,
        model_type=REQUIRED_MODEL_TYPE,
        weights_sha256=weights_sha,
        config_sha256=config_sha,
        parity_report_sha256=parity_report_sha,
        plugin_group=VLLM_PLUGIN_GROUP,
        registration_api=VLLM_REGISTRATION_API,
        vllm_runtime_status="NOT_TESTED",
        blockers=tuple(blockers),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m twelve_six.inference.vllm_handoff",
        description="Bind parity evidence and preflight a 12-6 export for future vLLM integration.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    bind = subparsers.add_parser("bind-parity")
    bind.add_argument("--export-dir", type=Path, required=True)
    bind.add_argument("--parity-report", type=Path, required=True)
    bind.add_argument("--output", type=Path, required=True)
    bind.add_argument("--overwrite", action="store_true")

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--export-dir", type=Path, required=True)
    preflight.add_argument("--parity-binding", type=Path)
    preflight.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "bind-parity":
            output = bind_parity_report(
                args.export_dir,
                args.parity_report,
                args.output,
                overwrite=args.overwrite,
            )
            print(output)
            return 0

        report = inspect_vllm_handoff(args.export_dir, args.parity_binding)
        if args.json:
            print(json.dumps(report.to_dict(), sort_keys=True, separators=(",", ":")))
        else:
            print(
                "ready_for_plugin_implementation="
                f"{str(report.ready_for_plugin_implementation).lower()}"
            )
            print(f"checkpoint_id={report.checkpoint_id or '-'}")
            print(f"plugin_group={report.plugin_group}")
            print(f"model_architecture={report.model_architecture}")
            print(f"vllm_runtime_status={report.vllm_runtime_status}")
            for blocker in report.blockers:
                print(f"blocker={blocker}")
        return 0 if report.ready_for_plugin_implementation else 2
    except (FileExistsError, OSError, VllmHandoffError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
