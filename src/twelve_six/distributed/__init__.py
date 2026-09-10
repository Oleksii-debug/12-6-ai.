"""Scale-readiness contracts for distributed 12-6 AI training."""

from .contracts import HardwareProfile, ModelScaleSpec, ParallelPlan
from .memory import MemoryEstimate, estimate_training_memory
from .topology import TopologyReport, build_torchrun_command, shell_join, validate_topology

__all__ = [
    "HardwareProfile",
    "MemoryEstimate",
    "ModelScaleSpec",
    "ParallelPlan",
    "TopologyReport",
    "build_torchrun_command",
    "estimate_training_memory",
    "shell_join",
    "validate_topology",
]
