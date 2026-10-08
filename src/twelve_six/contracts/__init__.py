"""Public v1 contract packaging; existing modules own identity and architecture truth."""
from twelve_six.artifact_identity import (
    ArtifactKind, ArtifactManifest, ArtifactRef, GenerationIdentityManifest,
    ParentBinding, bind_artifact, parse_generation_identity_manifest, verify_parent_bindings,
)
from twelve_six.system_architecture import (
    CognitiveCoreBinding, CognitiveCoreIdentity, CoreReplacementReceipt,
    ProductAssembly, replace_cognitive_core,
    InterfaceContract, RuntimeShellContract, SystemArchitectureManifest, TypedBoundary,
    canonical_runtime_shell_v1, canonical_system_architecture_v1,
)
from .baseline_v1 import (
    ContractPackageError, EvidenceRef, ErrorRecord, LifecycleObservation,
    baseline_manifest_v1, encode_v1, decode_v1,
)
from .evolution import ContractEvolutionError, propose_version_change, validate_version_change

__all__ = [
    "ArtifactKind", "ArtifactManifest", "ArtifactRef", "GenerationIdentityManifest",
    "ParentBinding", "bind_artifact", "parse_generation_identity_manifest",
    "verify_parent_bindings", "InterfaceContract", "RuntimeShellContract",
    "SystemArchitectureManifest", "TypedBoundary",
    "CognitiveCoreBinding", "CognitiveCoreIdentity", "CoreReplacementReceipt",
    "ProductAssembly", "replace_cognitive_core", "canonical_runtime_shell_v1",
    "canonical_system_architecture_v1", "ContractPackageError", "EvidenceRef",
    "ErrorRecord", "LifecycleObservation", "baseline_manifest_v1", "encode_v1",
    "decode_v1", "ContractEvolutionError", "propose_version_change",
    "validate_version_change",
]
