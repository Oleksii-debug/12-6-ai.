"""Regression coverage for post-merge AUDIT991 privacy findings."""

import pytest

from twelve_six.data.privacy_filter_v3 import detect


def _has_secret_assignment(text: str) -> bool:
    return any(finding.detector_id == "environment_secret_assignment" for finding in detect(text))


@pytest.mark.parametrize(
    "text",
    [
        "password=process.env['SERVICE_TOKEN\"]",
        'password=process.env["SERVICE_TOKEN\']',
        'password=os.getenv("SERVICE_TOKEN\')',
        "password=env('SERVICE_TOKEN\")",
    ],
)
def test_mismatched_placeholder_quotes_fail_closed(text: str) -> None:
    assert _has_secret_assignment(text)


@pytest.mark.parametrize(
    "text",
    [
        'password="${SERVICE_TOKEN}"junk',
        'password="%SERVICE_TOKEN%"junk',
        'password="{{SERVICE_TOKEN}}"junk',
        'password="process.env.SERVICE_TOKEN"junk',
        'password="os.getenv(\'SERVICE_TOKEN\')"junk',
    ],
)
def test_quoted_placeholder_prefix_with_trailing_garbage_fails_closed(text: str) -> None:
    assert _has_secret_assignment(text)


@pytest.mark.parametrize(
    "text",
    [
        'password="${SERVICE_TOKEN}"',
        "password=process.env['SERVICE_TOKEN']",
        'password=process.env["SERVICE_TOKEN"]',
        'password=os.getenv("SERVICE_TOKEN")',
        "password=env('SERVICE_TOKEN')",
    ],
)
def test_complete_matching_placeholders_remain_suppressed(text: str) -> None:
    assert not _has_secret_assignment(text)
