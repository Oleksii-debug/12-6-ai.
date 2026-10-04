from twelve_six.data.wikisource_pd_contract import normalize_rendered_text


def test_internal_nbsp_is_typographic_space_not_stanza_boundary() -> None:
    assert normalize_rendered_text("Український\u00a0текст\n") == "Український текст\n"


def test_leading_nbsp_keeps_stanza_boundary_semantics() -> None:
    source = "Перший український рядок\n\u00a0Другий український рядок\n"
    assert normalize_rendered_text(source) == (
        "Перший український рядок\n\nДругий український рядок\n"
    )
