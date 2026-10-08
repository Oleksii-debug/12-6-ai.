"""Plan 6: independent research evidence. No training or campaign authority."""
from __future__ import annotations

import hashlib
import hmac
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

_SHA = re.compile(r"^[0-9a-f]{64}$")
STAGES = ("deterministic", "holdout", "independent", "regression", "replay")
_MAX_JSON = 65536


def _digest(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in items:
        if key in output:
            raise ValueError("duplicate research JSON key")
        output[key] = value
    return output


def _nonfinite(_: str) -> None:
    raise ValueError("non-finite research JSON number")


def _canonical(value: object) -> bytes:
    def validate(node: object, depth: int = 0) -> None:
        if depth > 20:
            raise ValueError("research JSON too deeply nested")
        if type(node) is dict:
            for key, item in node.items():
                if type(key) is not str:
                    raise ValueError("research JSON keys must be strings")
                validate(item, depth + 1)
        elif type(node) is list:
            for item in node:
                validate(item, depth + 1)
        elif type(node) not in (str, int, float, bool, type(None)):
            raise ValueError("unsupported research JSON value")
    validate(value)
    try:
        blob = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False,
            ensure_ascii=False,
        ).encode("utf-8")
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("invalid research JSON") from exc
    if len(blob) > _MAX_JSON:
        raise ValueError("research JSON too large")
    return blob


def _sha(name: str, value: object) -> str:
    if type(value) is not str or not _SHA.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _name(name: str, value: object) -> str:
    if type(value) is not str or not value or len(value) > 128:
        raise ValueError(f"{name} must be a bounded nonempty string")
    return value


@dataclass(frozen=True, slots=True)
class ResearchTrial:
    protocol_sha256: str
    model_sha256: str
    data_sha256: str
    artifact_sha256: str
    seed: int
    config_json: bytes
    producer_id: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported research trial schema")
        for key in ("protocol_sha256", "model_sha256", "data_sha256", "artifact_sha256"):
            _sha(key, getattr(self, key))
        if type(self.seed) is not int or not 0 <= self.seed < 2**64:
            raise ValueError("seed must be unsigned 64-bit integer")
        _name("producer_id", self.producer_id)
        if type(self.config_json) is not bytes or len(self.config_json) > _MAX_JSON:
            raise ValueError("invalid frozen config")
        try:
            data = json.loads(
                self.config_json.decode("utf-8"), object_pairs_hook=_pairs,
                parse_constant=_nonfinite,
            )
        except (UnicodeError, ValueError) as exc:
            raise ValueError("invalid research config") from exc
        if type(data) is not dict or _canonical(data) != self.config_json:
            raise ValueError("config must be a canonical JSON object")

    @classmethod
    def create(
        cls, *, protocol_sha256: str, model_sha256: str, data_sha256: str,
        artifact_sha256: str, seed: int, config: Mapping[str, Any], producer_id: str,
    ) -> ResearchTrial:
        if type(config) is not dict:
            raise ValueError("config must be an ordinary dict")
        return cls(
            protocol_sha256, model_sha256, data_sha256, artifact_sha256,
            seed, _canonical(config), producer_id,
        )

    def payload(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": self.schema_version, "protocol_sha256": self.protocol_sha256,
            "model_sha256": self.model_sha256, "data_sha256": self.data_sha256,
            "artifact_sha256": self.artifact_sha256, "seed": self.seed,
            "config_sha256": _digest(self.config_json), "producer_id": self.producer_id,
        }

    @property
    def identity_sha256(self) -> str:
        return _digest(_canonical(self.payload()))


@dataclass(frozen=True, slots=True)
class StageEvidence:
    stage: str
    trial_sha256: str
    verifier_id: str
    input_sha256: str
    expected_sha256: str
    observed_sha256: str
    signature: str

    def payload(self) -> dict[str, str]:
        return {
            "stage": self.stage, "trial_sha256": self.trial_sha256,
            "verifier_id": self.verifier_id, "input_sha256": self.input_sha256,
            "expected_sha256": self.expected_sha256, "observed_sha256": self.observed_sha256,
        }


def _key(value: bytes) -> bytes:
    if type(value) is not bytes or len(value) < 32:
        raise ValueError("trusted verifier key must be 32+ bytes")
    return value


def issue_verified_stage(
    trial: ResearchTrial, *, stage: str, verifier_id: str, verifier_key: bytes,
    sample: object, expected_output: object, evaluator: Callable[[int, object], object],
) -> StageEvidence:
    """Trusted harness owns secret and expected value; two executions must agree."""
    if type(trial) is not ResearchTrial:
        raise ValueError("invalid trial")
    trial.__post_init__()
    if type(stage) is not str or stage not in STAGES:
        raise ValueError("unknown research stage")
    _name("verifier_id", verifier_id)
    if verifier_id == trial.producer_id:
        raise ValueError("self-issued model claim cannot be evidence")
    key = _key(verifier_key)
    if not callable(evaluator):
        raise ValueError("missing evaluator")
    expected = _digest(_canonical(expected_output))
    observed = _digest(_canonical(evaluator(trial.seed, sample)))
    replay = _digest(_canonical(evaluator(trial.seed, sample)))
    if observed != expected or replay != observed:
        raise ValueError("research outcome mismatch or replay instability")
    payload = {
        "stage": stage, "trial_sha256": trial.identity_sha256,
        "verifier_id": verifier_id, "input_sha256": _digest(_canonical(sample)),
        "expected_sha256": expected, "observed_sha256": observed,
    }
    seal = hmac.new(key, _canonical(payload), hashlib.sha256).hexdigest()
    return StageEvidence(**payload, signature=seal)


def verify_research_bundle(
    trial: ResearchTrial, evidence: tuple[StageEvidence, ...], *,
    verifier_id: str, verifier_key: bytes,
) -> str:
    """Verify all exact stages independently and return a research receipt digest."""
    if type(trial) is not ResearchTrial or type(evidence) is not tuple:
        raise ValueError("invalid research bundle")
    trial.__post_init__()
    _name("verifier_id", verifier_id)
    key = _key(verifier_key)
    if verifier_id == trial.producer_id or len(evidence) != len(STAGES):
        raise ValueError("independent research pyramid incomplete")
    for stage, item in zip(STAGES, evidence, strict=True):
        if type(item) is not StageEvidence or item.stage != stage:
            raise ValueError("research evidence stage absent, duplicate or unordered")
        if item.trial_sha256 != trial.identity_sha256 or item.verifier_id != verifier_id:
            raise ValueError("foreign research evidence")
        for name in ("input_sha256", "expected_sha256", "observed_sha256", "signature"):
            _sha(name, getattr(item, name))
        if item.observed_sha256 != item.expected_sha256:
            raise ValueError("research stage failed")
        seal = hmac.new(key, _canonical(item.payload()), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(seal, item.signature):
            raise ValueError("untrusted or forged research evidence")
    return _digest(_canonical({
        "trial_sha256": trial.identity_sha256,
        "verifier_id": verifier_id,
        "stages": [item.payload() | {"signature": item.signature} for item in evidence],
    }))
