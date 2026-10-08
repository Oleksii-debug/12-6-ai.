"""Plan 6 Section 14: isolated, signed fixture champion lifecycle.

Only immutable identities/receipts change here. No model weights, production
champion pointers, training, paid compute, or Plan-9 promotion are written.
"""
from __future__ import annotations

import hmac
from dataclasses import asdict, dataclass

from .experience_replay import _mac
from .post_base_instruction import bounded_id, canonical_digest, sha_field


@dataclass(frozen=True, slots=True)
class Descendant:
    candidate_id: str
    producer_id: str
    parent_sha256: str
    descendant_sha256: str
    base_sha256: str
    artifact_sha256: str
    recipe_sha256: str
    compatibility_sha256: str
    schema_version: int = 1

    def identity(self) -> str:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported descendant version")
        bounded_id(self.candidate_id)
        bounded_id(self.producer_id)
        for name in ("parent_sha256", "descendant_sha256", "base_sha256",
                     "artifact_sha256", "recipe_sha256", "compatibility_sha256"):
            sha_field(getattr(self, name))
        if self.descendant_sha256 == self.parent_sha256:
            raise ValueError("candidate cannot alias parent champion")
        return canonical_digest(asdict(self))


@dataclass(frozen=True, slots=True)
class Evaluation:
    candidate_sha256: str
    parent_sha256: str
    base_sha256: str
    compatibility_sha256: str
    suite_sha256: str
    evidence_sha256: str
    verifier_id: str
    verifier_version_sha256: str
    qualified: bool
    compatible: bool
    signature: str
    schema_version: int = 1

    def payload(self) -> dict[str, object]:
        return {k: v for k, v in asdict(self).items() if k != "signature"}

    def identity(self) -> str:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported evaluation version")
        for name in ("candidate_sha256", "parent_sha256", "base_sha256",
                     "compatibility_sha256", "suite_sha256", "evidence_sha256",
                     "verifier_version_sha256", "signature"):
            sha_field(getattr(self, name))
        bounded_id(self.verifier_id)
        if type(self.qualified) is not bool or type(self.compatible) is not bool:
            raise ValueError("untyped evaluation result")
        return canonical_digest(asdict(self))


@dataclass(frozen=True, slots=True)
class Authorization:
    action: str
    candidate_sha256: str
    evaluation_sha256: str
    expected_state_sha256: str
    authorizer_id: str
    version_sha256: str
    signature: str

    def payload(self) -> dict[str, object]:
        return {k: v for k, v in asdict(self).items() if k != "signature"}

    def identity(self) -> str:
        if self.action not in ("PROMOTE", "ROLLBACK"):
            raise ValueError("invalid authorization action")
        bounded_id(self.authorizer_id)
        for name in ("candidate_sha256", "evaluation_sha256", "expected_state_sha256",
                     "version_sha256", "signature"):
            sha_field(getattr(self, name))
        return canonical_digest(asdict(self))


@dataclass(frozen=True, slots=True)
class LifecycleEvent:
    action: str
    candidate_id: str
    candidate_sha256: str
    evaluation_sha256: str
    authorization_sha256: str | None
    old_champion_sha256: str
    new_champion_sha256: str
    previous_sha256: str
    event_sha256: str

    def identity(self) -> str:
        return canonical_digest({k: v for k, v in asdict(self).items()
                                 if k != "event_sha256"})


@dataclass(frozen=True, slots=True)
class LifecycleState:
    base_sha256: str
    genesis_champion_sha256: str
    champion_sha256: str
    head_sha256: str
    events: tuple[LifecycleEvent, ...] = ()

    @classmethod
    def genesis(cls, base_sha256: str, champion_sha256: str) -> LifecycleState:
        sha_field(base_sha256)
        sha_field(champion_sha256)
        return cls(base_sha256, champion_sha256, champion_sha256,
                   canonical_digest({"lifecycle": 1, "base": base_sha256,
                                     "champion": champion_sha256}))

    def validate(self) -> None:
        sha_field(self.base_sha256)
        sha_field(self.genesis_champion_sha256)
        if type(self.events) is not tuple or len(self.events) > 64:
            raise ValueError("unbounded lifecycle history")
        head = self.genesis(self.base_sha256, self.genesis_champion_sha256).head_sha256
        champion = self.genesis_champion_sha256
        used: set[str] = set()
        accepted: list[LifecycleEvent] = []
        for event in self.events:
            if type(event) is not LifecycleEvent or event.action not in (
                "PROMOTED", "REJECTED", "ROLLED_BACK"
            ):
                raise ValueError("invalid lifecycle event")
            bounded_id(event.candidate_id)
            for root in (event.candidate_sha256, event.evaluation_sha256,
                         event.old_champion_sha256, event.new_champion_sha256,
                         event.previous_sha256, event.event_sha256):
                sha_field(root)
            if event.authorization_sha256 is not None:
                sha_field(event.authorization_sha256)
            if (event.previous_sha256 != head or event.old_champion_sha256 != champion
                    or event.identity() != event.event_sha256):
                raise ValueError("tampered lifecycle hash chain")
            if event.action == "REJECTED":
                if event.authorization_sha256 is not None or event.new_champion_sha256 != champion:
                    raise ValueError("rejection changed champion")
            elif event.action == "PROMOTED":
                if (event.candidate_id in used or event.authorization_sha256 is None
                        or event.new_champion_sha256 == champion):
                    raise ValueError("invalid champion promotion")
                accepted.append(event)
                used.add(event.candidate_id)
                champion = event.new_champion_sha256
            else:
                if (event.authorization_sha256 is None or not accepted
                        or accepted[-1].new_champion_sha256 != champion
                        or event.candidate_sha256 != accepted[-1].candidate_sha256
                        or event.evaluation_sha256 != accepted[-1].evaluation_sha256
                        or event.new_champion_sha256 != accepted[-1].old_champion_sha256):
                    raise ValueError("rollback must restore last promoted champion")
                champion = event.new_champion_sha256
                accepted.pop()
            head = event.event_sha256
        if self.head_sha256 != head or self.champion_sha256 != champion:
            raise ValueError("lifecycle state or champion drift")


def _append(state: LifecycleState, action: str, candidate: Descendant,
            evaluation: Evaluation, grant: Authorization | None,
            next_champion: str) -> LifecycleState:
    event = LifecycleEvent(action, candidate.candidate_id, candidate.identity(),
                           evaluation.identity(), grant.identity() if grant else None,
                           state.champion_sha256, next_champion, state.head_sha256, "")
    event = LifecycleEvent(event.action, event.candidate_id, event.candidate_sha256,
                           event.evaluation_sha256, event.authorization_sha256,
                           event.old_champion_sha256, event.new_champion_sha256,
                           event.previous_sha256, event.identity())
    result = LifecycleState(state.base_sha256, state.genesis_champion_sha256,
                            next_champion, event.event_sha256, state.events + (event,))
    result.validate()
    return result


def _verify_grant(grant: Authorization, action: str, candidate: Descendant,
                  evaluation: Evaluation, state: LifecycleState, *,
                  authorizer_id: str, authorizer_version_sha256: str,
                  authorizer_key: bytes) -> None:
    if type(grant) is not Authorization:
        raise ValueError("explicit signed authorization required")
    grant.identity()
    bounded_id(authorizer_id)
    sha_field(authorizer_version_sha256)
    if (grant.action != action or grant.candidate_sha256 != candidate.identity()
            or grant.evaluation_sha256 != evaluation.identity()
            or grant.expected_state_sha256 != state.head_sha256
            or grant.authorizer_id != authorizer_id
            or grant.authorizer_id in (candidate.producer_id, evaluation.verifier_id)
            or grant.version_sha256 != authorizer_version_sha256
            or not hmac.compare_digest(grant.signature, _mac(authorizer_key, grant.payload()))):
        raise ValueError("stale/self-signed/forged promotion authorization")


def decide_candidate(state: LifecycleState, candidate: Descendant,
                     evaluation: Evaluation, grant: Authorization | None = None, *,
                     trusted_state_sha256: str, verifier_id: str,
                     verifier_version_sha256: str, trusted_evidence_roots: frozenset[str],
                     verifier_key: bytes, authorizer_id: str,
                     authorizer_version_sha256: str, authorizer_key: bytes) -> LifecycleState:
    state.validate()
    if state.head_sha256 != trusted_state_sha256 or len(state.events) >= 64:
        raise ValueError("untrusted/stale lifecycle head")
    if type(candidate) is not Descendant or type(evaluation) is not Evaluation:
        raise ValueError("untyped lifecycle evidence")
    candidate.identity()
    evaluation.identity()
    bounded_id(verifier_id)
    sha_field(verifier_version_sha256)
    if (candidate.parent_sha256 != state.champion_sha256
            or candidate.base_sha256 != state.base_sha256
            or candidate.candidate_id in {e.candidate_id for e in state.events}
            or candidate.descendant_sha256 in {state.champion_sha256,
                                                state.genesis_champion_sha256}):
        raise ValueError("stale, reused or foreign descendant")
    if (type(trusted_evidence_roots) is not frozenset or not trusted_evidence_roots):
        raise ValueError("independent evaluation roots required")
    for root in trusted_evidence_roots:
        sha_field(root)
    if (evaluation.candidate_sha256 != candidate.identity()
            or evaluation.parent_sha256 != candidate.parent_sha256
            or evaluation.base_sha256 != candidate.base_sha256
            or evaluation.compatibility_sha256 != candidate.compatibility_sha256
            or evaluation.verifier_id != verifier_id
            or evaluation.verifier_id == candidate.producer_id
            or evaluation.verifier_version_sha256 != verifier_version_sha256
            or evaluation.suite_sha256 not in trusted_evidence_roots
            or evaluation.evidence_sha256 not in trusted_evidence_roots
            or not hmac.compare_digest(evaluation.signature,
                                       _mac(verifier_key, evaluation.payload()))):
        raise ValueError("untrusted, self-issued or mismatched evaluation")
    if grant is None:
        if evaluation.qualified and evaluation.compatible:
            raise ValueError("qualified candidate awaits explicit promotion authorization")
        return _append(state, "REJECTED", candidate, evaluation, None,
                       state.champion_sha256)
    if not evaluation.qualified or not evaluation.compatible:
        raise ValueError("failed evaluation/compatibility cannot be promoted")
    _verify_grant(grant, "PROMOTE", candidate, evaluation, state,
                  authorizer_id=authorizer_id,
                  authorizer_version_sha256=authorizer_version_sha256,
                  authorizer_key=authorizer_key)
    return _append(state, "PROMOTED", candidate, evaluation, grant,
                   candidate.descendant_sha256)


def rollback_promotion(state: LifecycleState, candidate: Descendant,
                       evaluation: Evaluation, grant: Authorization, *,
                       trusted_state_sha256: str, authorizer_id: str,
                       authorizer_version_sha256: str, authorizer_key: bytes) -> LifecycleState:
    state.validate()
    if state.head_sha256 != trusted_state_sha256 or not state.events or len(state.events) >= 64:
        raise ValueError("untrusted/stale rollback state")
    last = state.events[-1]
    if (last.action != "PROMOTED" or last.new_champion_sha256 != state.champion_sha256
            or last.candidate_sha256 != candidate.identity()
            or last.evaluation_sha256 != evaluation.identity()):
        raise ValueError("rollback cannot modify arbitrary champion history")
    _verify_grant(grant, "ROLLBACK", candidate, evaluation, state,
                  authorizer_id=authorizer_id,
                  authorizer_version_sha256=authorizer_version_sha256,
                  authorizer_key=authorizer_key)
    return _append(state, "ROLLED_BACK", candidate, evaluation, grant,
                   last.old_champion_sha256)
