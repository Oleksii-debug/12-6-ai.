from __future__ import annotations

from dataclasses import asdict, dataclass


class CapabilityUnavailableError(RuntimeError):
    """Raised when an orchestration capability lacks terminal authority."""


@dataclass(frozen=True, slots=True)
class CapabilityAuthority:
    name: str
    source_pr: int
    head_sha: str
    source_status: str
    terminal: bool
    accepted_source: bool
    reason: str
    workflow_run_id: int | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


# A component is enabled only when a later live authority explicitly recognizes
# its tested source head as terminal/converged. Source presence alone is never enough.
AUTHORITIES: dict[str, CapabilityAuthority] = {
    "model_adapter": CapabilityAuthority(
        name="model_adapter",
        source_pr=428,
        head_sha="e805bf715617999209aef88946cea01f3668f583",
        source_status="TERMINAL_LOCAL_FREE_PASS",
        terminal=True,
        accepted_source=True,
        reason="POSTBASE-351 exact-head LOCAL_FREE proof and generic CI are green.",
        workflow_run_id=32998080549,
    ),
    "deliberation": CapabilityAuthority(
        name="deliberation",
        source_pr=386,
        head_sha="486bd91ca03bed41750c638d702f557f320b780a",
        source_status="PASS_COMPONENT_CONVERGENCE",
        terminal=True,
        accepted_source=True,
        reason="POSTBASE-255 dedicated exact-head convergence authority is green.",
        workflow_run_id=32997278311,
    ),
    "verifier": CapabilityAuthority(
        name="verifier",
        source_pr=423,
        head_sha="7eac24e250c0853745208bab8ba9b2d3d104fbf5",
        source_status="PASS_COMPONENT_CONVERGENCE",
        terminal=True,
        accepted_source=True,
        reason=(
            "POSTBASE-357 independent convergence authority is green; production verifier blob is pinned."
        ),
        workflow_run_id=32983319052,
    ),
    "hypothesis_search": CapabilityAuthority(
        name="hypothesis_search",
        source_pr=422,
        head_sha="ea1d8fff0d3235660dffe7ba411e192df83f5e1d",
        source_status="TERMINAL_EXACT_HEAD_LOCAL_FREE_SUCCESS",
        terminal=True,
        accepted_source=True,
        reason=(
            "POSTBASE-256/356 dedicated and generic exact-head workflows are green; "
            "NEXT100-097 final live authority refresh explicitly recognizes this head as terminal."
        ),
        workflow_run_id=32983600700,
    ),
    "memory_rag": CapabilityAuthority(
        name="memory_rag",
        source_pr=436,
        head_sha="976adda1cfe981d7b6363d267854759bee802006",
        source_status="TERMINAL_SCOPED_EXACT_HEAD_LOCAL_FREE_SUCCESS",
        terminal=True,
        accepted_source=True,
        reason=(
            "POSTBASE-358 scoped exact-head LOCAL_FREE convergence workflow is green; "
            "NEXT100-097 final live authority refresh explicitly recognizes the component as terminal."
        ),
        workflow_run_id=32983793329,
    ),
    "mock_tools": CapabilityAuthority(
        name="mock_tools",
        source_pr=384,
        head_sha="2f675e48a3172911a6f98ab6d4c46162ff536128",
        source_status="NONTERMINAL_EXACT_HEAD_GATE_FAILED",
        terminal=False,
        accepted_source=True,
        reason=(
            "POSTBASE-254 source is retained byte-exact, but dedicated exact-head run 32961334473 failed; "
            "NEXT100-097 live refresh still found no terminal NEXT100-086 successor."
        ),
        workflow_run_id=32961334473,
    ),
}


class CapabilityGate:
    def __init__(self, authorities: dict[str, CapabilityAuthority] | None = None) -> None:
        self._authorities = dict(AUTHORITIES if authorities is None else authorities)

    def authority(self, name: str) -> CapabilityAuthority:
        try:
            return self._authorities[name]
        except KeyError as exc:
            raise CapabilityUnavailableError(f"unknown capability: {name}") from exc

    def require(self, name: str) -> CapabilityAuthority:
        authority = self.authority(name)
        if not authority.terminal:
            raise CapabilityUnavailableError(
                f"capability {name!r} is fail-closed: {authority.source_status}: {authority.reason}"
            )
        return authority

    def snapshot(self) -> dict[str, dict[str, object]]:
        return {name: self._authorities[name].to_dict() for name in sorted(self._authorities)}
