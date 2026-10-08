"""Plan 6 Section 6: bounded teacher council over injected ModelGateway adapters.

Teachers are candidate generators, never independent verification or promotion authority.
The adapter fixture is replaceable by a versioned Plan-4 ModelGateway binding.
"""
from __future__ import annotations

import hmac
from dataclasses import asdict, dataclass
from typing import Protocol

from .post_base_agentic import _seal
from .post_base_instruction import bounded_id, canonical_digest, sha_field


@dataclass(frozen=True, slots=True)
class TeacherModel:
    gateway_id: str
    model_sha256: str
    provider: str
    capability: str
    policy_sha256: str
    max_output_chars: int
    price_microunits: int = 0
    schema_version: int = 1

    def validate(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported teacher schema")
        bounded_id(self.gateway_id)
        bounded_id(self.capability)
        for value in (self.model_sha256, self.policy_sha256):
            sha_field(value)
        if self.provider not in ("local", "server", "external"):
            raise ValueError("unknown provider")
        if (type(self.max_output_chars) is not int or not 1 <= self.max_output_chars <= 4096
                or type(self.price_microunits) is not int or not 0 <= self.price_microunits <= 10_000_000):
            raise ValueError("unbounded teacher cost/output")


@dataclass(frozen=True, slots=True)
class TeacherPrompt:
    task_id: str
    prompt_text: str
    prompt_sha256: str
    capability: str
    policy_sha256: str
    max_output_chars: int = 512
    max_total_microunits: int = 0
    allow_external: bool = False
    allow_paid: bool = False
    schema_version: int = 1

    def validate(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unknown prompt version")
        bounded_id(self.task_id)
        bounded_id(self.capability)
        sha_field(self.policy_sha256)
        if (type(self.prompt_text) is not str or not 1 <= len(self.prompt_text) <= 8192
                or self.prompt_sha256 != canonical_digest({"prompt": self.prompt_text})):
            raise ValueError("prompt identity substitution")
        if (type(self.max_output_chars) is not int or not 1 <= self.max_output_chars <= 4096
                or type(self.max_total_microunits) is not int
                or not 0 <= self.max_total_microunits <= 10_000_000
                or type(self.allow_external) is not bool or type(self.allow_paid) is not bool):
            raise ValueError("unbounded or untyped teacher budget")


@dataclass(frozen=True, slots=True)
class GatewayReply:
    model_sha256: str
    prompt_sha256: str
    policy_sha256: str
    text: str | None
    charged_microunits: int = 0


class ModelGateway(Protocol):
    def invoke(self, model: TeacherModel, prompt: TeacherPrompt) -> GatewayReply: ...


@dataclass(frozen=True, slots=True)
class TeacherCandidate:
    teacher: TeacherModel
    task_id: str
    prompt_sha256: str
    response_text: str | None
    response_sha256: str | None
    charged_microunits: int
    status: str

    def identity(self) -> str:
        return canonical_digest(asdict(self))


@dataclass(frozen=True, slots=True)
class TeacherBatch:
    prompt: TeacherPrompt
    candidates: tuple[TeacherCandidate, ...]
    charged_microunits: int
    identity_sha256: str


def gather_teacher_candidates(prompt: TeacherPrompt, members: tuple[TeacherModel, ...],
                              gateway: ModelGateway) -> TeacherBatch:
    """Preflight all identity/policy/cost gates BEFORE any provider call."""
    prompt.validate()
    if type(members) is not tuple or not 1 <= len(members) <= 8:
        raise ValueError("unbounded council size")
    if not callable(getattr(gateway, "invoke", None)):
        raise TypeError("typed ModelGateway adapter missing")
    ids: set[str] = set()
    for model in members:
        if type(model) is not TeacherModel:
            raise ValueError("invalid teacher")
        model.validate()
        if model.gateway_id in ids or model.capability != prompt.capability or model.policy_sha256 != prompt.policy_sha256:
            raise ValueError("duplicate teacher or capability/policy mismatch")
        if model.max_output_chars < prompt.max_output_chars:
            raise ValueError("insufficient teacher output capability")
        if model.provider == "external" and not prompt.allow_external:
            raise ValueError("external teacher not authorized")
        if model.price_microunits and not prompt.allow_paid:
            raise ValueError("paid teacher call not authorized")
        ids.add(model.gateway_id)
    if sum(m.price_microunits for m in members) > prompt.max_total_microunits:
        raise ValueError("teacher budget exceeded before invocation")
    out = []
    spent = 0
    for model in members:
        try:
            reply = gateway.invoke(model, prompt)
        except (TimeoutError, ConnectionError):
            # Failure is an auditable abstention, not fabricated teacher output.
            out.append(TeacherCandidate(model, prompt.task_id, prompt.prompt_sha256,
                                        None, None, 0, "GATEWAY_FAILURE"))
            continue
        if (type(reply) is not GatewayReply or reply.model_sha256 != model.model_sha256
                or reply.prompt_sha256 != prompt.prompt_sha256
                or reply.policy_sha256 != prompt.policy_sha256):
            raise ValueError("ModelGateway response identity mismatch")
        if (type(reply.charged_microunits) is not int
                or not 0 <= reply.charged_microunits <= model.price_microunits
                or spent + reply.charged_microunits > prompt.max_total_microunits):
            raise ValueError("invalid actual teacher cost")
        if reply.text is not None and (type(reply.text) is not str or not 1 <= len(reply.text) <= prompt.max_output_chars):
            raise ValueError("invalid/oversized teacher output")
        spent += reply.charged_microunits
        out.append(TeacherCandidate(
            model, prompt.task_id, prompt.prompt_sha256, reply.text,
            canonical_digest({"response": reply.text}) if reply.text is not None else None,
            reply.charged_microunits, "CANDIDATE_UNVERIFIED" if reply.text is not None else "ABSTAIN",
        ))
    candidates = tuple(out)
    root = canonical_digest({"prompt": asdict(prompt), "candidates": [asdict(x) for x in candidates],
                             "charged_microunits": spent})
    return TeacherBatch(prompt, candidates, spent, root)


@dataclass(frozen=True, slots=True)
class TeacherVerdict:
    candidate_sha256: str
    verifier_id: str
    verifier_version_sha256: str
    accepted: bool
    rights_ok: bool
    quality_ok: bool
    signature: str

    def payload(self) -> dict[str, object]:
        return {key: value for key, value in asdict(self).items() if key != "signature"}


def attest_teacher_candidate(candidate: TeacherCandidate, *, verifier_id: str,
                             verifier_version_sha256: str, verifier_key: bytes,
                             accepted: bool, rights_ok: bool, quality_ok: bool) -> TeacherVerdict:
    bounded_id(verifier_id)
    sha_field(verifier_version_sha256)
    if verifier_id == candidate.teacher.gateway_id or any(type(x) is not bool for x in (accepted, rights_ok, quality_ok)):
        raise ValueError("teacher cannot self-verify")
    fields = {
        "candidate_sha256": candidate.identity(),
        "verifier_id": verifier_id,
        "verifier_version_sha256": verifier_version_sha256,
        "accepted": accepted,
        "rights_ok": rights_ok,
        "quality_ok": quality_ok,
    }
    return TeacherVerdict(**fields, signature=_seal(fields, verifier_key))


@dataclass(frozen=True, slots=True)
class CouncilJudge:
    batch_sha256: str
    selected_candidate_sha256: str
    judge_id: str
    judge_version_sha256: str
    signature: str

    def payload(self) -> dict[str, str]:
        return {key: value for key, value in asdict(self).items() if key != "signature"}


def attest_council_judge(batch: TeacherBatch, selected: TeacherCandidate, *, judge_id: str,
                         judge_version_sha256: str, judge_key: bytes) -> CouncilJudge:
    bounded_id(judge_id)
    sha_field(judge_version_sha256)
    if judge_id in {x.teacher.gateway_id for x in batch.candidates}:
        raise ValueError("teacher cannot self-judge")
    fields = {
        "batch_sha256": batch.identity_sha256,
        "selected_candidate_sha256": selected.identity(),
        "judge_id": judge_id,
        "judge_version_sha256": judge_version_sha256,
    }
    return CouncilJudge(**fields, signature=_seal(fields, judge_key))


@dataclass(frozen=True, slots=True)
class CouncilDecision:
    state: str
    candidate_sha256: str | None
    disagreement: bool
    evidence_sha256: str
    promotion_authorized: bool = False


def decide_teacher_council(batch: TeacherBatch, evidence: tuple[TeacherVerdict, ...], *,
                           verifier_id: str, verifier_version_sha256: str,
                           verifier_key: bytes, judge: CouncilJudge | None = None,
                           trusted_judge_id: str | None = None,
                           trusted_judge_version_sha256: str | None = None,
                           judge_key: bytes | None = None) -> CouncilDecision:
    if type(batch) is not TeacherBatch or type(evidence) is not tuple:
        raise ValueError("invalid council verification inputs")
    if batch.identity_sha256 != canonical_digest({
        "prompt": asdict(batch.prompt), "candidates": [asdict(x) for x in batch.candidates],
        "charged_microunits": batch.charged_microunits,
    }):
        raise ValueError("teacher batch integrity mismatch")
    if len(evidence) != len(batch.candidates):
        raise ValueError("missing verifier verdict")
    eligible = []
    for candidate, verdict in zip(batch.candidates, evidence, strict=True):
        if (type(verdict) is not TeacherVerdict or verdict.candidate_sha256 != candidate.identity()
                or verdict.verifier_id != verifier_id
                or verdict.verifier_id == candidate.teacher.gateway_id
                or verdict.verifier_version_sha256 != verifier_version_sha256
                or any(type(x) is not bool for x in (verdict.accepted, verdict.rights_ok, verdict.quality_ok))):
            raise ValueError("foreign or self-issued teacher verdict")
        sha_field(verdict.signature)
        if not hmac.compare_digest(verdict.signature, _seal(verdict.payload(), verifier_key)):
            raise ValueError("forged teacher verifier receipt")
        if (candidate.status == "CANDIDATE_UNVERIFIED" and candidate.response_text is not None
                and verdict.accepted and verdict.rights_ok and verdict.quality_ok):
            eligible.append(candidate)
    unique = {x.response_sha256 for x in eligible}
    disagreement = len(unique) > 1
    chosen = None
    if len(unique) == 1:
        chosen = eligible[0]
    elif disagreement and judge is not None:
        if (type(judge) is not CouncilJudge or judge.batch_sha256 != batch.identity_sha256
                or judge.judge_id != trusted_judge_id
                or judge.judge_id in {x.teacher.gateway_id for x in batch.candidates}
                or judge.judge_version_sha256 != trusted_judge_version_sha256
                or judge.selected_candidate_sha256 not in {x.identity() for x in eligible}
                or judge_key is None):
            raise ValueError("untrusted independent judge")
        sha_field(judge.signature)
        if not hmac.compare_digest(judge.signature, _seal(judge.payload(), judge_key)):
            raise ValueError("forged judge receipt")
        chosen = next(x for x in eligible if x.identity() == judge.selected_candidate_sha256)
    elif judge is not None:
        raise ValueError("unexpected judge without disagreement")
    state = "ACCEPTED_CANDIDATE" if chosen is not None else ("DISAGREEMENT" if disagreement else "ABSTAIN")
    return CouncilDecision(state, chosen.identity() if chosen is not None else None,
                           disagreement, canonical_digest({"batch": batch.identity_sha256,
                           "verdicts": [asdict(x) for x in evidence],
                           "judge": asdict(judge) if judge is not None else None}))
