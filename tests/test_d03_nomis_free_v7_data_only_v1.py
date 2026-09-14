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


def test_transient_timeout_retries_same_call_and_returns_unmodified_value() -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    sleeps: list[float] = []
    expected = object()

    def flaky(*args, **kwargs):
        calls.append((args, kwargs))
        if len(calls) < 3:
            raise TimeoutError("temporary exact-source read timeout")
        return expected

    wrapped = bootstrap.build_bounded_urlopen_retry(flaky, sleep=sleeps.append)
    actual = wrapped("https://example.invalid/exact", timeout=30)

    assert actual is expected
    assert calls == [
        (("https://example.invalid/exact",), {"timeout": 30}),
        (("https://example.invalid/exact",), {"timeout": 30}),
        (("https://example.invalid/exact",), {"timeout": 30}),
    ]
    assert sleeps == list(bootstrap.RETRY_DELAYS_SECONDS)


def test_connection_reset_is_bounded_and_reraises_after_three_attempts() -> None:
    calls = 0
    sleeps: list[float] = []

    def always_reset(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise ConnectionResetError("reset")

    wrapped = bootstrap.build_bounded_urlopen_retry(always_reset, sleep=sleeps.append)
    with pytest.raises(ConnectionResetError, match="reset"):
        wrapped("https://example.invalid/exact")

    assert calls == bootstrap.MAX_TRANSIENT_FETCH_ATTEMPTS
    assert sleeps == list(bootstrap.RETRY_DELAYS_SECONDS)


def test_urlerror_wrapping_timeout_is_retryable() -> None:
    calls = 0

    def flaky(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise urllib.error.URLError(socket.timeout("temporary"))
        return b"exact-bytes"

    wrapped = bootstrap.build_bounded_urlopen_retry(flaky, sleep=lambda _seconds: None)
    assert wrapped("https://example.invalid/exact") == b"exact-bytes"
    assert calls == 2


def test_nontransient_urlerror_fails_without_retry() -> None:
    calls = 0

    def invalid(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise urllib.error.URLError("certificate or DNS failure")

    wrapped = bootstrap.build_bounded_urlopen_retry(invalid, sleep=lambda _seconds: None)
    with pytest.raises(urllib.error.URLError):
        wrapped("https://example.invalid/exact")
    assert calls == 1
