"""Scale-readiness contracts for distributed 12-6 AI training."""

from .checkpointing import (
    D05CheckpointRef,
    DistributedCheckpointEnvelope,
    DistributedShardRecord,
    ResumeDecision,
    ResumeMode,
    decide_resume,
)
from .contracts import HardwareProfile, ModelScaleSpec, ParallelPlan
from .cpu_probe import CpuProbeResult, run_cpu_gloo_probe
from .memory import MemoryEstimate, estimate_training_memory
from .profiling import TensorByteMeasurement, measure_model_parameter_bytes, measure_tensor_bytes
from .rank_layout import ExpertCoordinate, RankCoordinate, RankLayout
from .runtime import BackendAdoptionDecision, TorchMeshSpec, build_torch_mesh_spec, choose_backend
from .topology import TopologyReport, build_torchrun_command, shell_join, validate_topology

__all__ = [
    "BackendAdoptionDecision",
    "CpuProbeResult",
    "D05CheckpointRef",
    "DistributedCheckpointEnvelope",
    "DistributedShardRecord",
    "ExpertCoordinate",
    "HardwareProfile",
    "MemoryEstimate",
    "ModelScaleSpec",
    "ParallelPlan",
    "RankCoordinate",
    "RankLayout",
    "ResumeDecision",
    "ResumeMode",
    "TensorByteMeasurement",
    "TopologyReport",
    "TorchMeshSpec",
    "build_torch_mesh_spec",
    "build_torchrun_command",
    "choose_backend",
    "decide_resume",
    "estimate_training_memory",
    "measure_model_parameter_bytes",
    "measure_tensor_bytes",
    "run_cpu_gloo_probe",
    "shell_join",
    "validate_topology",
]
