"""Scale-readiness contracts for distributed 12-6 AI training."""

from .checkpoint_layout import (
    D05LogicalBinding,
    DistributedCheckpointManifest,
    ResumePlan,
    ShardRecord,
    TopologySnapshot,
    bind_d05_manifest,
    build_distributed_checkpoint_manifest,
    plan_resume,
    rank_identity,
    verify_distributed_checkpoint_manifest,
    verify_shard_files,
)
from .contracts import HardwareProfile, ModelScaleSpec, ParallelPlan
from .memory import MemoryEstimate, estimate_training_memory
from .mesh import FakeProcessGroups, RankCoordinate, coordinate_for_rank, fake_process_groups
from .runtime import (
    ModuleStateMeasurement,
    TorchNativeCapabilities,
    TorchNativePlan,
    build_torch_native_plan,
    measure_module_state,
    torch_native_capabilities,
)
from .topology import TopologyReport, build_torchrun_command, shell_join, validate_topology

__all__ = [
    "D05LogicalBinding",
    "DistributedCheckpointManifest",
    "FakeProcessGroups",
    "HardwareProfile",
    "MemoryEstimate",
    "ModelScaleSpec",
    "ModuleStateMeasurement",
    "ParallelPlan",
    "RankCoordinate",
    "ResumePlan",
    "ShardRecord",
    "TopologyReport",
    "TopologySnapshot",
    "TorchNativeCapabilities",
    "TorchNativePlan",
    "bind_d05_manifest",
    "build_distributed_checkpoint_manifest",
    "build_torch_native_plan",
    "build_torchrun_command",
    "coordinate_for_rank",
    "estimate_training_memory",
    "fake_process_groups",
    "measure_module_state",
    "plan_resume",
    "rank_identity",
    "shell_join",
    "torch_native_capabilities",
    "validate_topology",
    "verify_distributed_checkpoint_manifest",
    "verify_shard_files",
]
