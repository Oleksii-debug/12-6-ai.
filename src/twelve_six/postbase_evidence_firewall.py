from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Mapping

POST_BASE_NAMESPACE = "post_base"
BASE_NAMESPACE = "base"
POLICY_ID = "next100092.base-postbase-evidence-firewall.v1"

FORBIDDEN_TRUE_KEYS = frozenset(
    {
        "base_corpus_evidence",
        "canonical_base_training_eligible",
        "canonical_base_scientific_evidence",
        "base_training_evidence",
        "base_scientific_evidence",
        "base_evaluation_evidence",
        "base_raw_lm_diagnostics",
    }
)

NAMESPACE_KEYS = frozenset(
    {
        "evidence_namespace",
        "scientific_evidence_namespace",
        "training_evidence_namespace",
        "evaluation_evidence_namespace",
    }
)

CANONICAL_BASE_CLASSIFICATIONS = frozenset(
    {
        "BASE",
        "CANONICAL_BASE",
        "BASE_TRAINING",
        "BASE_TRAINING_EVIDENCE",
        "BASE_SCIENTIFIC_EVIDENCE",
        "CANONICAL_BASE_SCIENTIFIC_EVIDENCE",
    }
)

CANONICAL_BASE_PATH_PREFIXES = (
    "evidence/base",
    "artifacts/base",
    "data/base",
)

BASE_REFERENCE_ALLOWED_KEYS = {
    "base_evidence": frozenset(
        {
            "evidence_namespace",
            "checkpoint_id",
            "git_sha",
            "model_spec_sha256",
            "parameter_count",
            "vocab_size",
            "max_context_tokens",
            "tokenizer_version",
            "tokenizer_config_sha256",
            "tokenizer_vocab_sha256",
            "dataset_manifest_sha256",
            "run_manifest_sha256",
            "step",
            "tokens_seen",
            "device",
        }
    ),
    "base_checkpoint": frozenset(
        {
            "checkpoint_id",
            "sha256",
            "git_sha",
            "stage",
            "lineage",
        }
    ),
    "base_provenance": frozenset(
        {
            "checkpoint_id",
            "sha256",
            "git_sha",
            "model_spec_sha256",
            "parameter_count",
            "vocab_size",
            "max_context_tokens",
            "tokenizer_version",
            "tokenizer_config_sha256",
            "tokenizer_vocab_sha256",
            "dataset_manifest_sha256",
            "run_manifest_sha256",
            "manifest_sha256",
            "artifact_digest",
            "evidence_identity",
            "source_id",
            "step",
            "tokens_seen",
            "device",
            "stage",
            "lineage",
        }
    ),
}

REQUIRED_COMPONENTS = frozenset(
    {
        "model_adapter",
        "communication_data",
        "sft_runner",
        "communication_eval",
        "tools",
        "deliberation",
        "hypothesis_search",
        "verifier",
        "memory_rag",
        "teacher_factory",
    }
)

_SHA40 = re.compile(r"^[0-9a-f]{40}$")


class NamespaceViolation(ValueError):
    """Raised when post-Base evidence crosses into canonical Base authority."""


@dataclass(frozen=True, slots=True)
class EvidenceEnvelope:
    """Machine envelope required before post-Base output is published as evidence."""

    component_id: str
    artifact_kind: str
    payload: Mapping[str, Any]
    artifact_path: str | None = None
    origin_namespace: str = POST_BASE_NAMESPACE
    evidence_namespace: str = POST_BASE_NAMESPACE


def _normalized_path(path: str) -> str:
    raw = path.replace("\\", "/").strip("/")
    pure = PurePosixPath(raw)
    if not raw or any(part == ".." for part in pure.parts):
        raise NamespaceViolation(f"unsafe artifact path: {path!r}")
    return pure.as_posix()


def _is_base_path(path: str) -> bool:
    normalized = _normalized_path(path)
    return any(
        normalized == prefix or normalized.startswith(prefix + "/")
        for prefix in CANONICAL_BASE_PATH_PREFIXES
    )


def _validate_base_reference(container: str, value: Any) -> None:
    if not isinstance(value, Mapping):
        raise NamespaceViolation(f"{container} must be a provenance object")
    allowed = BASE_REFERENCE_ALLOWED_KEYS[container]
    keys = {str(key) for key in value}
    unexpected = sorted(keys - allowed)
    if unexpected:
        raise NamespaceViolation(
            f"{container} contains non-provenance fields: {', '.join(unexpected)}"
        )
    for key, item in value.items():
        if isinstance(item, (Mapping, list, tuple)):
            raise NamespaceViolation(
                f"{container}.{key} must remain a flat immutable provenance value"
            )
    if container == "base_evidence" and "evidence_namespace" in value:
        namespace = value["evidence_namespace"]
        if namespace != BASE_NAMESPACE:
            raise NamespaceViolation(
                "base_evidence.evidence_namespace must equal base when supplied"
            )


def _walk(value: Any, *, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            current = (*path, key)

            if key in BASE_REFERENCE_ALLOWED_KEYS:
                _validate_base_reference(key, item)
                continue

            if key in FORBIDDEN_TRUE_KEYS and item is True:
                raise NamespaceViolation(
                    f"{'.'.join(current)} must remain false for post-Base artifacts"
                )

            if key in NAMESPACE_KEYS and isinstance(item, str):
                normalized = item.strip().lower().replace("-", "_")
                if normalized == BASE_NAMESPACE:
                    raise NamespaceViolation(
                        f"{'.'.join(current)} cannot target canonical Base evidence"
                    )

            if key == "classification" and isinstance(item, str):
                if item.strip().upper() in CANONICAL_BASE_CLASSIFICATIONS:
                    raise NamespaceViolation(
                        f"{'.'.join(current)} relabels post-Base output as Base"
                    )

            if key == "training_use" and isinstance(item, str):
                upper = item.strip().upper()
                if upper in {
                    "BASE",
                    "BASE_TRAINING",
                    "CANONICAL_BASE",
                    "CANONICAL_BASE_TRAINING",
                }:
                    raise NamespaceViolation(
                        f"{'.'.join(current)} relabels post-Base output as Base training"
                    )

            _walk(item, path=current)
        return

    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _walk(item, path=(*path, str(index)))


def validate_post_base_envelope(envelope: EvidenceEnvelope) -> None:
    """Fail closed on any Base namespace/training/scientific-evidence promotion."""

    if envelope.origin_namespace != POST_BASE_NAMESPACE:
        raise NamespaceViolation(
            "post-Base gate requires origin_namespace=post_base"
        )
    if envelope.evidence_namespace != POST_BASE_NAMESPACE:
        raise NamespaceViolation(
            "post-Base behavior/evaluation/synthetic results must remain in "
            "post_base evidence"
        )
    if envelope.component_id not in REQUIRED_COMPONENTS:
        raise NamespaceViolation(
            f"unknown post-Base component: {envelope.component_id}"
        )
    if not envelope.artifact_kind.strip():
        raise NamespaceViolation("artifact_kind must be non-empty")
    if envelope.artifact_path is not None:
        if not isinstance(envelope.artifact_path, str):
            raise NamespaceViolation("artifact_path must be text when supplied")
        if _is_base_path(envelope.artifact_path):
            raise NamespaceViolation(
                "post-Base artifact cannot be written to canonical Base path: "
                f"{envelope.artifact_path}"
            )
    _walk(envelope.payload)


def validate_audit_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate the frozen ten-component NEXT100-092 audit inventory."""

    if manifest.get("schema") != "next100092.postbase-evidence-namespace-audit.v1":
        raise NamespaceViolation("unexpected NEXT100-092 audit schema")
    if manifest.get("worker_id") != (
        "NEXT100-092-POSTBASE-EVIDENCE-NAMESPACE-AUDIT"
    ):
        raise NamespaceViolation("unexpected worker_id")

    execution = manifest.get("execution")
    if not isinstance(execution, Mapping):
        raise NamespaceViolation("execution object is required")
    if execution.get("profile") != "LOCAL_FREE":
        raise NamespaceViolation("audit must remain LOCAL_FREE")
    for forbidden in (
        "external_llm_calls",
        "teacher_api_calls",
        "training_performed",
    ):
        if execution.get(forbidden) is not False:
            raise NamespaceViolation(
                f"execution.{forbidden} must be false"
            )

    policy = manifest.get("policy")
    if not isinstance(policy, Mapping) or policy.get("policy_id") != POLICY_ID:
        raise NamespaceViolation("audit policy identity mismatch")
    if policy.get("post_base_canonical_base_training_eligible") is not False:
        raise NamespaceViolation(
            "post-Base canonical Base training eligibility must be false"
        )
    if policy.get("post_base_canonical_base_scientific_evidence") is not False:
        raise NamespaceViolation(
            "post-Base canonical Base scientific evidence must be false"
        )

    components = manifest.get("components")
    if not isinstance(components, list):
        raise NamespaceViolation("components must be a list")
    ids = [
        entry.get("component_id")
        for entry in components
        if isinstance(entry, Mapping)
    ]
    if len(ids) != len(components) or set(ids) != REQUIRED_COMPONENTS:
        raise NamespaceViolation(
            "audit manifest must bind exactly the ten required components"
        )
    if len(ids) != len(set(ids)):
        raise NamespaceViolation("duplicate component_id")

    for entry in components:
        assert isinstance(entry, Mapping)
        component = str(entry["component_id"])
        if entry.get("authority_scope") != "POST_BASE_ONLY":
            raise NamespaceViolation(
                f"{component}: authority_scope must be POST_BASE_ONLY"
            )
        if entry.get("canonical_base_training_eligible") is not False:
            raise NamespaceViolation(
                f"{component}: Base training eligibility must be false"
            )
        if entry.get("canonical_base_scientific_evidence") is not False:
            raise NamespaceViolation(
                f"{component}: Base scientific evidence must be false"
            )
        head_sha = entry.get("head_sha")
        if not isinstance(head_sha, str) or not _SHA40.fullmatch(head_sha):
            raise NamespaceViolation(f"{component}: invalid head_sha")
        pr = entry.get("pr")
        if not isinstance(pr, int) or isinstance(pr, bool) or pr <= 0:
            raise NamespaceViolation(f"{component}: invalid PR")
        status = entry.get("native_namespace_status")
        if status not in {"EXPLICIT_FIREWALL", "CENTRAL_GATE_REQUIRED"}:
            raise NamespaceViolation(
                f"{component}: invalid native namespace status"
            )
        paths = entry.get("source_paths")
        if not isinstance(paths, list) or not paths:
            raise NamespaceViolation(
                f"{component}: source_paths must be non-empty"
            )
        for path in paths:
            if not isinstance(path, str) or not path.strip():
                raise NamespaceViolation(f"{component}: invalid source path")
            if _is_base_path(path):
                raise NamespaceViolation(
                    f"{component}: source path crosses Base namespace"
                )


def validate_artifact_dict(artifact: Mapping[str, Any]) -> None:
    """Validate one serialized EvidenceEnvelope-shaped JSON object."""

    payload = artifact.get("payload")
    if not isinstance(payload, Mapping):
        raise NamespaceViolation("artifact payload must be an object")
    artifact_path = artifact.get("artifact_path")
    if artifact_path is not None and not isinstance(artifact_path, str):
        raise NamespaceViolation("artifact_path must be text when supplied")
    validate_post_base_envelope(
        EvidenceEnvelope(
            component_id=str(artifact.get("component_id", "")),
            artifact_kind=str(artifact.get("artifact_kind", "")),
            payload=payload,
            artifact_path=artifact_path,
            origin_namespace=str(artifact.get("origin_namespace", "")),
            evidence_namespace=str(artifact.get("evidence_namespace", "")),
        )
    )
