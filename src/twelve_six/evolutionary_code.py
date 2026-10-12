"""Plan 6 Section 12: isolated code-evolution candidates and independent evidence.

This module stages inert UTF-8 change proposals outside the production tree.
It does not execute arbitrary candidate code, merge branches or authorize compute.
Trusted runners own actual test/security/benchmark/review evidence.
"""
from __future__ import annotations

import hmac
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

from .experience_replay import _mac
from .post_base_instruction import bounded_id, canonical_digest, sha_field

_GATES = ("tests", "security", "benchmark", "review")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SAFE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")


@dataclass(frozen=True, slots=True)
class CodeCandidate:
    candidate_id: str
    producer_id: str
    parent_git_sha: str
    branch: str
    path: str
    kind: str
    text: str
    objective_sha256: str
    schema_version: int = 1

    def identity(self) -> str:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported evolution schema")
        for value in (self.candidate_id, self.producer_id):
            bounded_id(value)
        if not _SAFE.fullmatch(self.candidate_id):
            raise ValueError("unsafe candidate identifier")
        if type(self.parent_git_sha) is not str or not _GIT_SHA.fullmatch(
            self.parent_git_sha
        ):
            raise ValueError("unbound git parent")
        if self.branch != "evolution/candidate/" + self.candidate_id:
            raise ValueError("candidate must use an isolated branch identity")
        if self.kind not in ("code", "config", "experiment"):
            raise ValueError("unrecognized change kind")
        if type(self.path) is not str:
            raise ValueError("invalid candidate path")
        parts = PurePosixPath(self.path)
        if (
            parts.is_absolute() or len(parts.parts) < 2 or ".." in parts.parts
            or "\\" in self.path
            or not all(_SAFE.fullmatch(p) for p in parts.parts[:-1])
            or parts.parts[0] != "proposals" or parts.suffix != ".txt"
            or not _SAFE.fullmatch(parts.stem)
        ):
            raise ValueError("candidate changes may only use inert proposal paths")
        if (
            type(self.text) is not str or not self.text
            or "\x00" in self.text or len(self.text.encode("utf-8")) > 16384
        ):
            raise ValueError("invalid/unbounded candidate content")
        sha_field(self.objective_sha256)
        return canonical_digest(asdict(self))


@dataclass(frozen=True, slots=True)
class EvolutionPolicy:
    parent_git_sha: str
    max_candidate_bytes: int = 16384
    schema_version: int = 1

    def validate(self) -> None:
        if (
            type(self.schema_version) is not int or self.schema_version != 1
            or type(self.parent_git_sha) is not str
            or not _GIT_SHA.fullmatch(self.parent_git_sha)
            or type(self.max_candidate_bytes) is not int
            or not 1 <= self.max_candidate_bytes <= 16384
        ):
            raise ValueError("invalid bounded evolution policy")


def _sandbox_path(candidate: CodeCandidate, *, sandbox_root: Path,
                  production_root: Path) -> Path:
    candidate.identity()
    if not isinstance(sandbox_root, Path) or not isinstance(production_root, Path):
        raise TypeError("explicit local directories required")
    if sandbox_root.is_symlink() or production_root.is_symlink():
        raise ValueError("untrusted sandbox/repository symlink")
    sandbox = sandbox_root.resolve()
    production = production_root.resolve()
    if sandbox == production or sandbox.is_relative_to(production):
        raise ValueError("candidate sandbox cannot enter production authority")
    if production.is_relative_to(sandbox):
        raise ValueError("candidate sandbox cannot contain production authority")
    relative = Path(candidate.candidate_id, *PurePosixPath(candidate.path).parts)
    current = sandbox
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise ValueError("symlink inside candidate sandbox")
    dest = (sandbox / relative).resolve()
    if not dest.is_relative_to(sandbox):
        raise ValueError("candidate sandbox escape")
    return dest


def stage_candidate(
    candidate: CodeCandidate, policy: EvolutionPolicy, *,
    sandbox_root: Path, production_root: Path,
) -> str:
    """Stage inert candidate bytes; never touch a live repository or run code."""
    policy.validate()
    root = candidate.identity()
    if candidate.parent_git_sha != policy.parent_git_sha:
        raise ValueError("stale candidate parent")
    blob = candidate.text.encode("utf-8")
    if len(blob) > policy.max_candidate_bytes:
        raise ValueError("candidate over policy budget")
    path = _sandbox_path(candidate, sandbox_root=sandbox_root,
                         production_root=production_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Re-evaluate after mkdir: reject replacement of any parent by a symlink.
    if _sandbox_path(candidate, sandbox_root=sandbox_root,
                     production_root=production_root) != path:
        raise ValueError("sandbox changed during publication")
    try:
        with path.open("xb") as output:
            output.write(blob)
    except FileExistsError:
        if path.is_symlink() or path.read_bytes() != blob:
            raise ValueError("changed/stale candidate after restart") from None
    if path.read_bytes() != blob:
        raise ValueError("candidate readback failed")
    return root


@dataclass(frozen=True, slots=True)
class GateEvidence:
    candidate_sha256: str
    gate: str
    verifier_id: str
    verifier_version_sha256: str
    evidence_sha256: str
    passed: bool
    signature: str
    schema_version: int = 1

    def payload(self) -> dict[str, object]:
        return {k: v for k, v in asdict(self).items() if k != "signature"}


@dataclass(frozen=True, slots=True)
class EvolutionDecision:
    candidate_sha256: str
    policy_sha256: str
    evidence_sha256: str
    negative_gates: tuple[str, ...]
    state: str
    receipt_sha256: str
    production_write_authorized: bool = False
    promotion_authorized: bool = False


def evaluate_evolution(
    candidate: CodeCandidate, policy: EvolutionPolicy,
    gates: tuple[GateEvidence, ...], *, trusted_verifiers: dict[str, str],
    trusted_keys: dict[str, bytes], trusted_roots: dict[str, frozenset[str]],
) -> EvolutionDecision:
    """Only independently checked gate receipts may yield external-review readiness."""
    policy.validate()
    candidate_sha = candidate.identity()
    if candidate.parent_git_sha != policy.parent_git_sha:
        raise ValueError("wrong version of candidate")
    if len(candidate.text.encode("utf-8")) > policy.max_candidate_bytes:
        raise ValueError("candidate exceeds evolution budget")
    if type(gates) is not tuple or len(gates) > len(_GATES):
        raise ValueError("invalid gate bundle")
    if (
        type(trusted_verifiers) is not dict or set(trusted_verifiers) != set(_GATES)
        or type(trusted_keys) is not dict or set(trusted_keys) != set(_GATES)
        or type(trusted_roots) is not dict or set(trusted_roots) != set(_GATES)
    ):
        raise ValueError("missing external gate authority")
    for verifier in trusted_verifiers.values():
        bounded_id(verifier)
    if len(set(trusted_verifiers.values())) != len(_GATES):
        raise ValueError("gate authorities must be independent")
    seen: set[str] = set()
    negatives: list[str] = []
    for item in gates:
        if type(item) is not GateEvidence or item.gate not in _GATES:
            raise ValueError("untyped or unknown gate evidence")
        if item.gate in seen:
            raise ValueError("duplicate gate evidence")
        seen.add(item.gate)
        if (
            type(item.schema_version) is not int or item.schema_version != 1
            or item.candidate_sha256 != candidate_sha
            or item.verifier_id != trusted_verifiers[item.gate]
            or item.verifier_id == candidate.producer_id
            or type(item.passed) is not bool
        ):
            raise ValueError("untrusted gate identity")
        for name in (item.verifier_version_sha256, item.evidence_sha256, item.signature):
            sha_field(name)
        roots = trusted_roots[item.gate]
        if type(roots) is not frozenset or not roots:
            raise ValueError("missing independent evidence roots")
        for root in roots:
            sha_field(root)
        if item.evidence_sha256 not in roots:
            raise ValueError("untrusted evidence root")
        if not hmac.compare_digest(item.signature, _mac(
            trusted_keys[item.gate], item.payload()
        )):
            raise ValueError("forged gate receipt")
        if not item.passed:
            negatives.append(item.gate + ":FAILED")
    negatives.extend(gate + ":MISSING" for gate in _GATES if gate not in seen)
    state = "EXTERNAL_REVIEW_READY" if not negatives else "REJECTED_CANDIDATE"
    gate_sha = canonical_digest([asdict(g) for g in sorted(gates, key=lambda g: g.gate)])
    policy_sha = canonical_digest(asdict(policy))
    receipt = canonical_digest({
        "candidate": candidate_sha, "policy": policy_sha,
        "gates": gate_sha, "negatives": negatives, "state": state,
    })
    return EvolutionDecision(candidate_sha, policy_sha, gate_sha,
                             tuple(negatives), state, receipt)


def verify_evolution_restart(
    candidate: CodeCandidate, policy: EvolutionPolicy, gates: tuple[GateEvidence, ...],
    result: EvolutionDecision, *, sandbox_root: Path, production_root: Path,
    trusted_verifiers: dict[str, str], trusted_keys: dict[str, bytes],
    trusted_roots: dict[str, frozenset[str]],
) -> bool:
    if (
        type(result) is not EvolutionDecision or result.production_write_authorized
        or result.promotion_authorized
    ):
        raise ValueError("self-authorized promotion or invalid decision")
    expected = evaluate_evolution(
        candidate, policy, gates, trusted_verifiers=trusted_verifiers,
        trusted_keys=trusted_keys, trusted_roots=trusted_roots,
    )
    if result != expected:
        raise ValueError("evolution receipt changed after restart")
    path = _sandbox_path(candidate, sandbox_root=sandbox_root,
                         production_root=production_root)
    if not path.is_file() or path.is_symlink():
        raise ValueError("candidate artifact missing")
    if path.read_bytes() != candidate.text.encode("utf-8"):
        raise ValueError("candidate artifact drift")
    return True


def publish_evolution_evidence(
    candidate: CodeCandidate, policy: EvolutionPolicy,
    gates: tuple[GateEvidence, ...], result: EvolutionDecision, *,
    sandbox_root: Path, production_root: Path,
    trusted_verifiers: dict[str, str], trusted_keys: dict[str, bytes],
    trusted_roots: dict[str, frozenset[str]],
) -> str:
    """Persist an immutable positive or negative decision in the isolated sandbox."""
    verified = verify_evolution_restart(
        candidate, policy, gates, result, sandbox_root=sandbox_root,
        production_root=production_root, trusted_verifiers=trusted_verifiers,
        trusted_keys=trusted_keys, trusted_roots=trusted_roots,
    )
    if not verified:
        raise ValueError("unqualified evolution decision")
    staged = _sandbox_path(candidate, sandbox_root=sandbox_root,
                           production_root=production_root)
    receipt = staged.parents[1] / "decision.json"
    if receipt.is_symlink():
        raise ValueError("candidate evidence symlink")
    data = json.dumps(asdict(result), sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")
    try:
        with receipt.open("xb") as output:
            output.write(data)
    except FileExistsError:
        if receipt.is_symlink() or receipt.read_bytes() != data:
            raise ValueError("changed immutable evolution decision") from None
    if receipt.read_bytes() != data:
        raise ValueError("evolution evidence readback failed")
    return result.receipt_sha256
