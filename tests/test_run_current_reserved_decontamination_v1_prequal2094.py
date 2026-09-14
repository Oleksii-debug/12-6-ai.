from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import tools.run_current_reserved_decontamination_v1 as runner

ROOT = Path(__file__).resolve().parents[1]


def _head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_clean_exact_checkout_matches_isolated_behavior_closure() -> None:
    head = _head()
    assert runner.require_exact_implementation(ROOT, head) == head


def test_matching_re_substitution_is_rejected_before_payload_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    head = _head()

    class FakeRe:
        UNICODE = runner.matching_impl.re.UNICODE
        compile = staticmethod(runner.matching_impl.re.compile)
        fullmatch = staticmethod(lambda *_args, **_kwargs: None)
        sub = staticmethod(runner.matching_impl.re.sub)

    monkeypatch.setattr(runner.matching_impl, "re", FakeRe)

    with pytest.raises(
        RuntimeError,
        match="isolated executable behavior closure drift",
    ):
        runner.require_exact_implementation(ROOT, head)


def test_matching_unicodedata_substitution_is_rejected_before_payload_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    head = _head()

    class FakeUnicodeData:
        normalize = staticmethod(lambda _form, value: value)

    monkeypatch.setattr(
        runner.matching_impl,
        "unicodedata",
        FakeUnicodeData,
    )

    with pytest.raises(RuntimeError):
        runner.require_exact_implementation(ROOT, head)


def test_matching_defaultdict_substitution_is_rejected_before_payload_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    head = _head()
    monkeypatch.setattr(runner.matching_impl, "defaultdict", dict)

    with pytest.raises(
        RuntimeError,
        match="isolated executable behavior closure drift",
    ):
        runner.require_exact_implementation(ROOT, head)
