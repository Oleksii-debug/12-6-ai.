"""Plan 5 / Section 4: model-neutral, evidence-aware knowledge passport.

This is a reference projection, not a live observer or external-truth oracle.
Only trusted callers may supply observed evidence. Model or remembered text
cannot declare an observation; derivations never acquire observed authority.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Literal

SCHEMA = "12-6.agent-world.v1"
SubjectKind = Literal["entity", "capability", "environment"]
FactKind = Literal["observed", "derived", "unknown"]
OriginKind = Literal["owner", "tool", "external", "model", "memory"]


class WorldError(ValueError):
    """Invalid evidence, stale writer or untrusted observation promotion."""


def _id(value: object) -> bool:
    return isinstance(value, str) and 0 < len(value) <= 256 and bool(value.strip())


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class WorldFact:
    fact_id: str
    subject_id: str
    subject_kind: SubjectKind
    attribute: str
    state: FactKind
    value: str | None
    source_id: str | None
    evidence_id: str | None
    origin_kind: OriginKind | None
    recorded_at: int
    expires_at: int | None = None
    depends_on: tuple[str, ...] = ()

    def validate(self) -> None:
        if not all(_id(v) for v in (self.fact_id, self.subject_id, self.attribute)):
            raise WorldError("invalid fact identity")
        if self.subject_kind not in ("entity", "capability", "environment"):
            raise WorldError("invalid subject kind")
        if self.state not in ("observed", "derived", "unknown"):
            raise WorldError("invalid fact state")
        if type(self.recorded_at) is not int or self.recorded_at < 0:
            raise WorldError("invalid observation timestamp")
        if self.expires_at is not None and (
            type(self.expires_at) is not int or self.expires_at <= self.recorded_at
        ):
            raise WorldError("invalid fact expiry")
        if type(self.depends_on) is not tuple or len(self.depends_on) > 128 or any(
            not _id(x) for x in self.depends_on
        ) or len(set(self.depends_on)) != len(self.depends_on):
            raise WorldError("invalid dependency identities")
        if self.state == "unknown":
            if self.value is not None or self.source_id is not None or self.evidence_id is not None or self.origin_kind is not None or self.depends_on:
                raise WorldError("unknown facts cannot pretend to have evidence")
        else:
            if not isinstance(self.value, str) or not self.value.strip() or len(self.value.encode("utf-8")) > 16384:
                raise WorldError("known fact needs bounded value")
            if not _id(self.source_id) or not _id(self.evidence_id):
                raise WorldError("known fact needs source and evidence")
            if self.origin_kind not in ("owner", "tool", "external", "model", "memory"):
                raise WorldError("invalid fact origin")
            if self.state == "observed" and (self.origin_kind in ("model", "memory") or self.depends_on):
                raise WorldError("model/memory/derivation cannot assert observed fact")
            if self.state == "derived" and not self.depends_on:
                raise WorldError("derived fact needs dependency lineage")


@dataclass(frozen=True)
class WorldView:
    subject_id: str
    subject_kind: SubjectKind | None
    attribute: str
    value: str | None
    state: Literal["observed", "derived", "unknown", "stale"]
    fact_id: str | None
    source_refs: tuple[str, ...]
    evidence_refs: tuple[str, ...]


class WorldModel:
    """CAS-guarded event log and deterministic observed/derived/stale projection."""

    def __init__(self) -> None:
        self._events: list[WorldFact] = []
        self._latest: dict[tuple[str, str], str] = {}
        self._by_id: dict[str, WorldFact] = {}

    @property
    def revision(self) -> int:
        return len(self._events)

    def apply(self, event: WorldFact, *, expected_revision: int) -> int:
        if not isinstance(event, WorldFact):
            raise WorldError("invalid world event")
        event.validate()
        if type(expected_revision) is not int or expected_revision != self.revision:
            raise WorldError("stale world revision")
        if event.fact_id in self._by_id:
            raise WorldError("duplicate fact id")
        if any(prior.subject_id == event.subject_id and prior.subject_kind != event.subject_kind
               for prior in self._events):
            raise WorldError("subject kind mismatch")
        for dep in event.depends_on:
            earlier = self._by_id.get(dep)
            if earlier is None or earlier.recorded_at > event.recorded_at:
                raise WorldError("unknown or future dependency")
        old = self._latest.get((event.subject_id, event.attribute))
        if old is not None:
            previous = self._by_id[old]
            if previous.subject_kind != event.subject_kind or previous.recorded_at > event.recorded_at:
                raise WorldError("subject-kind or time regression")
        self._events.append(event)
        self._by_id[event.fact_id] = event
        self._latest[(event.subject_id, event.attribute)] = event.fact_id
        return self.revision

    def _stale(self, fact_id: str, now: int, seen: set[str]) -> bool:
        if fact_id in seen:
            raise WorldError("cyclic world evidence")
        fact = self._by_id[fact_id]
        if self._latest.get((fact.subject_id, fact.attribute)) != fact_id:
            return True
        if fact.expires_at is not None and now >= fact.expires_at:
            return True
        if fact.recorded_at > now:
            return True
        return any(self._stale(dep, now, seen | {fact_id}) for dep in fact.depends_on)

    def lookup(self, subject_id: str, attribute: str, *, now: int) -> WorldView:
        if not _id(subject_id) or not _id(attribute) or type(now) is not int or now < 0:
            raise WorldError("invalid world lookup")
        fact_id = self._latest.get((subject_id, attribute))
        if fact_id is None:
            return WorldView(subject_id, None, attribute, None, "unknown", None, (), ())
        fact = self._by_id[fact_id]
        stale = self._stale(fact_id, now, set())
        return WorldView(
            subject_id, fact.subject_kind, attribute,
            None if stale or fact.state == "unknown" else fact.value,
            "stale" if stale else fact.state,
            fact.fact_id,
            (fact.source_id,) if fact.source_id is not None else (),
            (fact.evidence_id,) if fact.evidence_id is not None else (),
        )

    def passport(self, subject_id: str, *, now: int) -> dict[str, object]:
        if not _id(subject_id) or type(now) is not int or now < 0:
            raise WorldError("invalid passport request")
        attributes = sorted(key[1] for key in self._latest if key[0] == subject_id)
        views = [self.lookup(subject_id, attr, now=now) for attr in attributes]
        kinds = {view.subject_kind for view in views}
        if len(kinds) > 1:
            raise WorldError("ambiguous subject kind")
        return {
            "schema": SCHEMA,
            "subject_id": subject_id,
            "subject_kind": next(iter(kinds), None),
            "revision": self.revision,
            "attributes": {view.attribute: {"state": view.state, "value": view.value,
                                           "fact_id": view.fact_id,
                                           "source_refs": list(view.source_refs),
                                           "evidence_refs": list(view.evidence_refs)} for view in views},
        }

    def export(self) -> str:
        events = [asdict(event) for event in self._events]
        body = {"schema": SCHEMA, "events": events}
        return _json({**body, "sha256": hashlib.sha256(_json(body).encode("utf-8")).hexdigest()})

    @classmethod
    def restore(cls, raw: str) -> WorldModel:
        try:
            def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
                out: dict[str, object] = {}
                for key, value in pairs:
                    if key in out:
                        raise WorldError("duplicate world snapshot field")
                    out[key] = value
                return out
            payload = json.loads(raw, object_pairs_hook=unique)
            if not isinstance(payload, dict) or set(payload) != {"schema", "events", "sha256"} or payload["schema"] != SCHEMA or not isinstance(payload["events"], list):
                raise WorldError("invalid world snapshot schema")
            body = {"schema": payload["schema"], "events": payload["events"]}
            if hashlib.sha256(_json(body).encode("utf-8")).hexdigest() != payload["sha256"] or _json(payload) != raw:
                raise WorldError("world snapshot integrity mismatch")
            result = cls()
            for item in payload["events"]:
                if not isinstance(item, dict) or set(item) != set(WorldFact.__dataclass_fields__):
                    raise WorldError("invalid stored world fact")
                if not isinstance(item["depends_on"], list):
                    raise WorldError("invalid stored dependencies")
                event = WorldFact(**{**item, "depends_on": tuple(item["depends_on"])})
                result.apply(event, expected_revision=result.revision)
            return result
        except (TypeError, KeyError, ValueError, json.JSONDecodeError) as exc:
            raise WorldError("cannot restore world snapshot") from exc
