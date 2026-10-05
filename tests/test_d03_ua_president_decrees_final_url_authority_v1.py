from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_producer():
    path = ROOT / "tools" / "materialize_d03_ua_president_decrees_v1.py"
    spec = importlib.util.spec_from_file_location(
        "d03_president_decrees_final_url_authority",
        path,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


producer = load_producer()


class QueueFetcher:
    def __init__(self, results):
        self.results = list(results)

    def fetch(self, url: str):
        assert self.results
        result = self.results.pop(0)
        assert result.requested_url == url
        return result


def fetch_result(requested_url: str, final_url: str, body: bytes) -> object:
    return producer.FetchResult(
        requested_url=requested_url,
        final_url=final_url,
        content_type="text/html",
        body=body,
    )


def decree_html() -> bytes:
    body = "Український нормативний текст про державну політику та розвиток. " * 60
    return f"""<html><body>
<h1>УКАЗ ПРЕЗИДЕНТА УКРАЇНИ №873/2026</h1>
<div>Про Концепцію розвитку державної політики</div>
<article><p>{body}</p></article>
<footer><h3>Новини</h3></footer>
</body></html>""".encode()


@pytest.mark.parametrize("drift_index", [0, 1])
def test_document_probe_rejects_same_origin_final_url_drift(drift_index: int) -> None:
    requested = "https://www.president.gov.ua/documents/8732026-61461"
    redirected = "https://www.president.gov.ua/news/not-the-requested-decree"
    body = decree_html()
    results = [
        fetch_result(requested, requested, body),
        fetch_result(requested, requested, body),
    ]
    results[drift_index] = fetch_result(requested, redirected, body)

    with pytest.raises(RuntimeError, match="document fetch final URL drift"):
        producer.probe_document(QueueFetcher(results), requested)


@pytest.mark.parametrize("drift_index", [0, 1])
def test_catalogue_probe_rejects_same_origin_final_url_drift(
    monkeypatch: pytest.MonkeyPatch,
    drift_index: int,
) -> None:
    requested = producer.CATALOG
    redirected = "https://www.president.gov.ua/news/not-the-requested-catalogue"
    body = b"<html><body></body></html>"
    results = [
        fetch_result(requested, requested, body),
        fetch_result(requested, requested, body),
    ]
    results[drift_index] = fetch_result(requested, redirected, body)

    class FakeFetcher:
        def __init__(self, delay_seconds: float) -> None:
            assert delay_seconds >= 0
            self.queue = QueueFetcher(results)

        def fetch(self, url: str):
            return self.queue.fetch(url)

    monkeypatch.setattr(producer, "Fetcher", FakeFetcher)
    with pytest.raises(RuntimeError, match="catalogue fetch final URL drift"):
        producer.run(max_pages=1, max_documents=1, delay_seconds=1.0)
