"""Version-pinned research compatibility facts for future post-training runtimes.

This module records an observed compatibility snapshot only. Importing it does not
install, import, launch, or authorize TRL, verl, vLLM, or any training runtime.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_VERSION_RE = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
_DATE_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")


@dataclass(frozen=True, order=True, slots=True)
class SemanticVersion:
    major: int
    minor: int
    patch: int

    def __post_init__(self) -> None:
        if min(self.major, self.minor, self.patch) < 0:
            raise ValueError("semantic version components must be non-negative")

    @classmethod
    def parse(cls, value: str) -> SemanticVersion:
        match = _VERSION_RE.fullmatch(value.strip())
        if match is None:
            raise ValueError(f"invalid semantic version: {value!r}")
        return cls(*(int(component) for component in match.groups()))

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


@dataclass(frozen=True, slots=True)
class RuntimeCompatibilitySnapshot:
    """Observed compatibility bounds, not an executable environment lock."""

    snapshot_id: str
    observed_on: str
    trl_version: SemanticVersion
    verl_version: SemanticVersion
    vllm_latest_version: SemanticVersion
    vllm_selected_version: SemanticVersion
    trl_vllm_min: SemanticVersion
    trl_vllm_max: SemanticVersion
    verl_vllm_min: SemanticVersion
    sources: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.snapshot_id.strip():
            raise ValueError("snapshot_id must be non-empty")
        if _DATE_RE.fullmatch(self.observed_on) is None:
            raise ValueError("observed_on must use YYYY-MM-DD")
        if self.trl_vllm_min > self.trl_vllm_max:
            raise ValueError("TRL vLLM minimum cannot exceed maximum")
        if not self.sources or any(not source.strip() for source in self.sources):
            raise ValueError("compatibility sources must be non-empty")

    def trl_supports_vllm(self, version: SemanticVersion) -> bool:
        return self.trl_vllm_min <= version <= self.trl_vllm_max

    def verl_supports_vllm(self, version: SemanticVersion) -> bool:
        return version >= self.verl_vllm_min

    def selected_is_jointly_supported(self) -> bool:
        return self.trl_supports_vllm(self.vllm_selected_version) and self.verl_supports_vllm(
            self.vllm_selected_version
        )

    def latest_is_jointly_supported(self) -> bool:
        return self.trl_supports_vllm(self.vllm_latest_version) and self.verl_supports_vllm(
            self.vllm_latest_version
        )

    def assert_selected_is_jointly_supported(self) -> None:
        if not self.selected_is_jointly_supported():
            raise RuntimeError("selected vLLM version is outside the observed TRL/verl intersection")


CURRENT_RUNTIME_COMPATIBILITY = RuntimeCompatibilitySnapshot(
    snapshot_id="posttraining-runtime-compat-2026-08-23",
    observed_on="2026-08-23",
    trl_version=SemanticVersion.parse("1.10.0"),
    verl_version=SemanticVersion.parse("0.9.0"),
    vllm_latest_version=SemanticVersion.parse("0.27.1"),
    vllm_selected_version=SemanticVersion.parse("0.26.0"),
    trl_vllm_min=SemanticVersion.parse("0.17.0"),
    trl_vllm_max=SemanticVersion.parse("0.26.0"),
    verl_vllm_min=SemanticVersion.parse("0.18.0"),
    sources=(
        "https://pypi.org/project/trl/",
        "https://huggingface.co/docs/trl/vllm_integration",
        "https://pypi.org/project/verl/0.9.0/",
        "https://verl.readthedocs.io/en/latest/start/install.html",
        "https://pypi.org/project/vllm/",
    ),
)
