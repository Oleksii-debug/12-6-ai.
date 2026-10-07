"""Regression coverage for post-#1029 malformed sensitive-key quote fail-open."""

from twelve_six.data.privacy_filter_v3 import detect


def _has_secret_assignment(text: str) -> bool:
    return any(finding.detector_id == "environment_secret_assignment" for finding in detect(text))


def _fixture(*parts: str) -> str:
    return "".join(parts)


def test_malformed_sensitive_key_quotes_fail_closed() -> None:
    secret = _fixture("AbCd", "1234!", "fixture")
    cases = (
        '"password\': "' + secret + '"',
        "password': " + secret,
        "'password: " + secret,
        'password" = ' + secret,
        '"api_key\': "' + secret + '"',
        "'client_secret\": '" + secret + "'",
    )
    for text in cases:
        assert _has_secret_assignment(text)


def test_valid_sensitive_key_quote_forms_still_detect() -> None:
    secret = _fixture("AbCd", "1234!", "fixture")
    cases = (
        "password=" + secret,
        '"password": "' + secret + '"',
        "'client_secret': '" + secret + "'",
        '"api_key" = "' + secret + '"',
    )
    for text in cases:
        assert _has_secret_assignment(text)


def test_matching_placeholder_controls_remain_suppressed() -> None:
    cases = (
        '"password": "${SERVICE_TOKEN}"',
        "'client_secret': '$CLIENT_SECRET'",
        '"api_key" = "%API_KEY%"',
        'password=process.env["SERVICE_TOKEN"]',
        "password=env('SERVICE_TOKEN')",
    )
    for text in cases:
        assert not _has_secret_assignment(text)


def test_non_sensitive_malformed_key_is_not_widened() -> None:
    secret = _fixture("AbCd", "1234!", "fixture")
    assert not _has_secret_assignment('"username\': "' + secret + '"')
