import pytest

from twelve_six.license_text import (
    LicensePhraseError,
    normalize_license_prose,
    require_license_phrases,
)


def test_normalize_license_prose_folds_common_line_wrapping() -> None:
    text = "Permission is hereby granted\r\n\tto deal\nin the Software without restriction"
    assert normalize_license_prose(text) == (
        "Permission is hereby granted to deal in the Software without restriction"
    )


def test_require_license_phrases_accepts_wrapped_mit_grant() -> None:
    license_text = (
        "Permission is hereby granted, free of charge, to any person obtaining a copy\n"
        "of this software and associated documentation files (the \"Software\"), to deal\n"
        "in the Software without restriction, including without limitation the rights\n"
        "to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\n"
        "copies of the Software.\n"
    )
    require_license_phrases(
        license_text,
        (
            "deal in the Software without restriction",
            "use, copy, modify, merge, publish, distribute, sublicense, and/or sell",
        ),
    )


def test_phrase_whitespace_is_normalized_too() -> None:
    require_license_phrases(
        "alpha beta gamma",
        ("alpha\n\tbeta",),
    )


def test_matching_remains_case_sensitive() -> None:
    with pytest.raises(LicensePhraseError):
        require_license_phrases("THE SOFTWARE IS PROVIDED AS IS", ("the software",))


def test_matching_remains_punctuation_sensitive() -> None:
    with pytest.raises(LicensePhraseError):
        require_license_phrases("use, copy, modify", ("use copy modify",))


def test_semantically_missing_phrase_still_fails() -> None:
    with pytest.raises(LicensePhraseError):
        require_license_phrases(
            "Permission is hereby granted to use and copy the Software.",
            ("deal in the Software without restriction",),
        )


def test_empty_phrase_is_rejected() -> None:
    with pytest.raises(ValueError):
        require_license_phrases("some license", (" \n\t ",))


def test_non_string_inputs_fail_closed() -> None:
    with pytest.raises(TypeError):
        normalize_license_prose(b"license")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        require_license_phrases("license", (123,))  # type: ignore[arg-type]
