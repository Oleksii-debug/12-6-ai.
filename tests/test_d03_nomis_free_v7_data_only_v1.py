from __future__ import annotations

import importlib.util
import socket
import urllib.error
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/run_d03_nomis_free_v7_data_only_v1.py"
SPEC = importlib.util.spec_from_file_location("swarm2065_v7_data_only", TOOL)
assert SPEC is not None and SPEC.loader is not None
bootstrap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bootstrap)


class _Response:
    def __init__(self, attempt: int, fail_attempts: int, payload: bytes) -> None:
        self.attempt = attempt
        self.fail_attempts = fail_attempts
        self.payload = payload
        self.headers: dict[str, str] = {}

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        return None

    def read(self, _limit: int) -> bytes:
        if self.attempt <= self.fail_attempts:
            raise TimeoutError("The read operation timed out")
        return self.payload


def test_read_timeout_retries_whole_exact_fetch_transaction() -> None:
    open_calls = 0
    sleeps: list[float] = []
    expected = b"exact-authority-bytes"

    def fake_urlopen(_request: object, *, timeout: int) -> _Response:
        nonlocal open_calls
        assert timeout == 30
        open_calls += 1
        return _Response(open_calls, 2, expected)

    def historical_fetch(url: str) -> bytes:
        assert url == "https://example.invalid/exact"
        with fake_urlopen(object(), timeout=30) as response:
            return response.read(2_000_001)

    wrapped = bootstrap.build_bounded_exact_fetch_retry(
        historical_fetch, sleep=sleeps.append
    )
    assert wrapped("https://example.invalid/exact") == expected
    assert open_calls == 3
    assert sleeps == list(bootstrap.RETRY_DELAYS_SECONDS)


def test_read_timeout_is_bounded_and_reraises_after_three_whole_calls() -> None:
    open_calls = 0
    sleeps: list[float] = []

    def fake_urlopen(_request: object, *, timeout: int) -> _Response:
        nonlocal open_calls
        assert timeout == 30
        open_calls += 1
        return _Response(open_calls, bootstrap.MAX_TRANSIENT_FETCH_ATTEMPTS, b"never")

    def historical_fetch(_url: str) -> bytes:
        with fake_urlopen(object(), timeout=30) as response:
            return response.read(2_000_001)

    wrapped = bootstrap.build_bounded_exact_fetch_retry(
        historical_fetch, sleep=sleeps.append
    )
    with pytest.raises(TimeoutError, match="read operation timed out"):
        wrapped("https://example.invalid/exact")

    assert open_calls == bootstrap.MAX_TRANSIENT_FETCH_ATTEMPTS
    assert sleeps == list(bootstrap.RETRY_DELAYS_SECONDS)


def test_urlerror_wrapping_timeout_retries_whole_fetch() -> None:
    calls = 0

    def flaky(_url: str) -> bytes:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise urllib.error.URLError(socket.timeout("temporary"))
        return b"exact-bytes"

    wrapped = bootstrap.build_bounded_exact_fetch_retry(
        flaky, sleep=lambda _seconds: None
    )
    assert wrapped("https://example.invalid/exact") == b"exact-bytes"
    assert calls == 2


def test_nontransient_urlerror_fails_without_retry() -> None:
    calls = 0

    def invalid(_url: str) -> bytes:
        nonlocal calls
        calls += 1
        raise urllib.error.URLError("certificate or DNS failure")

    wrapped = bootstrap.build_bounded_exact_fetch_retry(
        invalid, sleep=lambda _seconds: None
    )
    with pytest.raises(urllib.error.URLError):
        wrapped("https://example.invalid/exact")
    assert calls == 1


def test_authority_failure_is_never_retried_or_promoted() -> None:
    calls = 0

    class AuthorityMismatch(RuntimeError):
        pass

    def invalid(_url: str) -> bytes:
        nonlocal calls
        calls += 1
        raise AuthorityMismatch("raw SHA-256 mismatch")

    wrapped = bootstrap.build_bounded_exact_fetch_retry(
        invalid, sleep=lambda _seconds: None
    )
    with pytest.raises(AuthorityMismatch, match="SHA-256 mismatch"):
        wrapped("https://example.invalid/exact")
    assert calls == 1
