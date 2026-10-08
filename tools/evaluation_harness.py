"""Plan 4 / Section 2: deterministic public-fixture evaluation harness.

Never load reserved/terminal datasets here. Those are evaluator-only inputs to
EvaluationVault. Candidate callbacks receive prompt + seed, never answer keys.
This is a trusted-local component contract, not an untrusted Python sandbox.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

SCHEMA = "12-6.evaluation-harness.v1"
METRICS = {"exact_match", "mean_absolute_error"}


class EvaluationError(ValueError):
    """A failed or unsupported evaluation; never interpreted as score zero."""


def canonical(value: object) -> bytes:
    try:
        return (json.dumps(value, sort_keys=True, ensure_ascii=False,
                           separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise EvaluationError("non-canonical evaluation payload") from exc


def sha(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def is_digest(value: object) -> bool:
    return (type(value) is str and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


@dataclass(frozen=True)
class FrozenSuite:
    suite_id: str
    dataset_version: str
    cases: tuple[tuple[str, str, str], ...]
    visibility: str = "public_fixture"

    def __post_init__(self) -> None:
        if self.visibility != "public_fixture":
            raise EvaluationError("reserved suites must use EvaluationVault")
        if (not isinstance(self.suite_id, str) or not self.suite_id.strip()
                or not isinstance(self.dataset_version, str)
                or not self.dataset_version.strip()):
            raise EvaluationError("suite identity missing")
        if type(self.cases) is not tuple or not 1 <= len(self.cases) <= 10000:
            raise EvaluationError("bounded immutable cases required")
        if any(type(row) is not tuple or len(row) != 3
               or any(type(value) is not str or not value or len(value) > 10000
                      for value in row) for row in self.cases):
            raise EvaluationError("invalid case")
        ids = [row[0] for row in self.cases]
        if len(ids) != len(set(ids)):
            raise EvaluationError("duplicate case ID")

    @property
    def identity(self) -> str:
        return sha({"schema": SCHEMA, "suite_id": self.suite_id,
                    "dataset_version": self.dataset_version,
                    "visibility": self.visibility, "cases": self.cases})


@dataclass(frozen=True)
class FrozenProtocol:
    suite_sha256: str
    seeds: tuple[int, ...]
    metric: str
    evaluator_version: str

    def __post_init__(self) -> None:
        if not is_digest(self.suite_sha256):
            raise EvaluationError("suite digest required")
        if self.metric not in METRICS:
            raise EvaluationError("unsupported metric")
        if (type(self.seeds) is not tuple or not 1 <= len(self.seeds) <= 32
                or any(type(seed) is not int or not 0 <= seed < 2**63
                       for seed in self.seeds) or len(set(self.seeds)) != len(self.seeds)):
            raise EvaluationError("invalid frozen seeds")
        if type(self.evaluator_version) is not str or not self.evaluator_version.strip():
            raise EvaluationError("evaluator version required")

    @property
    def identity(self) -> str:
        return sha({"schema": SCHEMA, "suite_sha256": self.suite_sha256,
                    "seeds": self.seeds, "metric": self.metric,
                    "evaluator_version": self.evaluator_version})


def _finite_number(value: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise EvaluationError("invalid numeric prediction/reference") from exc
    if not math.isfinite(number):
        raise EvaluationError("non-finite numeric prediction/reference")
    return number


def _confidence(values: list[float], metric: str) -> dict[str, object]:
    n = len(values)
    mean = math.fsum(values) / n
    if metric == "exact_match":
        # Descriptive Wilson interval; seed/case observations are not independent trials.
        z = 1.959963984540054
        denom = 1 + z*z/n
        center = (mean + z*z/(2*n)) / denom
        half = z * math.sqrt(mean*(1-mean)/n + z*z/(4*n*n)) / denom
        return {"method": "wilson_descriptive_95", "lower": max(0.0, center-half),
                "upper": min(1.0, center+half), "sampling_caveat": "case_seed_dependence"}
    if n < 2:
        return {"method": "insufficient_samples", "lower": None, "upper": None,
                "sampling_caveat": "at_least_two_observations_required"}
    variance = math.fsum((x-mean)**2 for x in values) / (n-1)
    half = 1.959963984540054 * math.sqrt(variance/n)
    return {"method": "normal_approx_descriptive_95", "lower": max(0.0, mean-half),
            "upper": mean+half, "sampling_caveat": "case_seed_dependence"}


def evaluate(
    suite: FrozenSuite, protocol: FrozenProtocol, *, candidate_sha256: str,
    predict: Callable[[str, int], str],
) -> dict[str, object]:
    """Run identical frozen cases/seeds, replay each candidate call, fail closed on drift."""
    if not isinstance(suite, FrozenSuite) or not isinstance(protocol, FrozenProtocol):
        raise EvaluationError("typed suite/protocol required")
    if suite.identity != protocol.suite_sha256 or not is_digest(candidate_sha256):
        raise EvaluationError("evaluation identity mismatch")
    if not callable(predict):
        raise EvaluationError("prediction adapter required")
    scores: list[float] = []
    # Bound both calls to each (case, seed) so hidden model nondeterminism is detected.
    for case_id, prompt, expected in suite.cases:
        del case_id
        for seed in protocol.seeds:
            try:
                first = predict(prompt, seed)
                replay = predict(prompt, seed)
            except Exception:
                raise EvaluationError("candidate execution failed") from None
            if (type(first) is not str or type(replay) is not str
                    or len(first) > 10000 or len(replay) > 10000):
                raise EvaluationError("unsupported candidate output")
            if first != replay:
                raise EvaluationError("nondeterministic candidate replay")
            if protocol.metric == "exact_match":
                scores.append(float(first == expected))
            elif protocol.metric == "mean_absolute_error":
                scores.append(abs(_finite_number(first) - _finite_number(expected)))
            else:
                raise EvaluationError("unsupported metric")
    score = math.fsum(scores) / len(scores)
    if not math.isfinite(score):
        raise EvaluationError("non-finite aggregate")
    report: dict[str, object] = {
        "schema_version": SCHEMA,
        "suite_sha256": suite.identity,
        "protocol_sha256": protocol.identity,
        "evaluator_version": protocol.evaluator_version,
        "candidate_sha256": candidate_sha256,
        "metric": protocol.metric,
        "direction": "higher" if protocol.metric == "exact_match" else "lower",
        "seeds": list(protocol.seeds),
        "case_count": len(suite.cases),
        "observation_count": len(scores),
        "score": score,
        "uncertainty": _confidence(scores, protocol.metric),
        "status": "PASS",
    }
    report["report_sha256"] = sha(report)
    return report


def verify_report(report: dict[str, object]) -> None:
    if (type(report) is not dict or set(report) != {
        "schema_version", "suite_sha256", "protocol_sha256", "evaluator_version",
        "candidate_sha256", "metric", "direction", "seeds", "case_count",
        "observation_count", "score", "uncertainty", "status", "report_sha256"
    } or report.get("schema_version") != SCHEMA or report.get("status") != "PASS"):
        raise EvaluationError("invalid report contract")
    if not all(is_digest(report.get(key)) for key in
               ("suite_sha256", "protocol_sha256", "candidate_sha256", "report_sha256")):
        raise EvaluationError("invalid report identity")
    if report["metric"] not in METRICS or report["direction"] != (
        "higher" if report["metric"] == "exact_match" else "lower"
    ):
        raise EvaluationError("invalid report metric")
    if (type(report["score"]) not in (int, float)
            or not math.isfinite(report["score"])):
        raise EvaluationError("invalid report score")
    if sha({k: v for k, v in report.items() if k != "report_sha256"}) != report["report_sha256"]:
        raise EvaluationError("report digest mismatch")


def compare(first: dict[str, object], second: dict[str, object]) -> dict[str, object]:
    """Same suite/protocol or fail, never silently compare different test conditions."""
    verify_report(first)
    verify_report(second)
    if any(first[key] != second[key] for key in
           ("suite_sha256", "protocol_sha256", "metric", "direction", "seeds",
            "observation_count", "case_count", "evaluator_version")):
        raise EvaluationError("incomparable frozen evaluation protocols")
    if first["candidate_sha256"] == second["candidate_sha256"]:
        raise EvaluationError("different candidate identities required")
    direction = 1 if first["direction"] == "higher" else -1
    delta = direction * (float(first["score"]) - float(second["score"]))
    return {"schema_version": SCHEMA, "first_candidate_sha256": first["candidate_sha256"],
            "second_candidate_sha256": second["candidate_sha256"],
            "protocol_sha256": first["protocol_sha256"],
            "preferred": "first" if delta > 0 else ("second" if delta < 0 else "tie"),
            "directional_delta": delta}


def publish_report(path: Path, report: dict[str, object]) -> None:
    """Create-only machine-readable report. A restarted writer must match exact bytes."""
    verify_report(report)
    payload = canonical(report)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        if Path(path).read_bytes() != payload:
            raise EvaluationError("immutable report conflict") from None
        return
    try:
        with os.fdopen(fd, "wb") as writer:
            writer.write(payload)
            writer.flush()
            os.fsync(writer.fileno())
    except Exception:
        Path(path).unlink(missing_ok=True)
        raise


def read_report(path: Path) -> dict[str, object]:
    try:
        def no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, value in pairs:
                if key in result:
                    raise EvaluationError("duplicate report field")
                result[key] = value
            return result
        report = json.loads(Path(path).read_bytes(), object_pairs_hook=no_duplicates,
                            parse_constant=lambda _: (_ for _ in ()).throw(
                                EvaluationError("nonfinite report field")))
        verify_report(report)
        return report
    except (OSError, ValueError, TypeError):
        raise EvaluationError("invalid or corrupted report") from None
