"""LOCAL_FREE fail-closed orchestration over accepted post-Base component surfaces."""

from .authority import AUTHORITIES, CapabilityAuthority, CapabilityGate, CapabilityUnavailableError
from .orchestrator import (
    LocalAssistantOrchestrator,
    OrchestrationResult,
    RunOptions,
    TRACE_SCHEMA,
    WORKER_ID,
    write_trace,
)

__all__ = [
    "AUTHORITIES",
    "CapabilityAuthority",
    "CapabilityGate",
    "CapabilityUnavailableError",
    "LocalAssistantOrchestrator",
    "OrchestrationResult",
    "RunOptions",
    "TRACE_SCHEMA",
    "WORKER_ID",
    "write_trace",
]
