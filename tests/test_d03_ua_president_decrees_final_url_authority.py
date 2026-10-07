from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_tool(filename: str, module_name: str):
    path = ROOT / "tools" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


producer = load_tool(
    "materialize_d03_ua_president_decrees_v1.py",
    "d03_president_decrees_final_url_authority",
)

DOCUMENT_URL = "https://www.president.gov.ua/documents/8732026-61461"
SAME_ORIGIN_DRIFT_URL = "https://www.president.gov.ua/news/authority-drift"


class SequenceFetcher:
    def __init__(self, results):
        self._results = iter(results)

    def fetch(self, url: str):
        result = next(self._results)
        assert result.requested_url == url
        return result


def fetch_pair(requested_url: str, drift_index: int):
    final_urls = [requested_url, requested_url]
    final_urls[drift_index] = SAME_ORIGIN_DRIFT_URL
    return [
        producer.FetchResult(
            requested_url=requested_url,
            final_url=final_url,
            content_type="text/html",
            body=b"",
        )
        for final_url in final_urls
    ]


@pytest.mark.parametrize(
    ("drift_index", "label"),
    [
        (0, "first document fetch"),
        (1, "second document fetch"),
    ],
)
def test_probe_document_rejects_same_origin_final_url_drift_before_parse(
    monkeypatch, drift_index, label
):
    fetcher = SequenceFetcher(fetch_pair(DOCUMENT_URL, drift_index))

    def unexpected_parse(_data):
        pytest.fail("document parser reached after final-URL authority drift")

    monkeypatch.setattr(producer, "extract_document", unexpected_parse)

    with pytest.raises(RuntimeError, match=rf"^{label} final URL drift:"):
        producer.probe_document(fetcher, DOCUMENT_URL)


@pytest.mark.parametrize(
    ("drift_index", "label"),
    [
        (0, "first catalogue fetch"),
        (1, "second catalogue fetch"),
    ],
)
def test_run_rejects_same_origin_catalogue_final_url_drift_before_discovery(
    monkeypatch, drift_index, label
):
    class DriftFetcher:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            self._delegate = SequenceFetcher(fetch_pair(producer.CATALOG, drift_index))

        def fetch(self, url: str):
            return self._delegate.fetch(url)

    def unexpected_discovery(*_args, **_kwargs):
        pytest.fail("catalogue discovery reached after final-URL authority drift")

    monkeypatch.setattr(producer, "Fetcher", DriftFetcher)
    monkeypatch.setattr(producer, "discover_document_urls", unexpected_discovery)

    with pytest.raises(RuntimeError, match=rf"^{label} final URL drift:"):
        producer.run(max_pages=1, max_documents=1, delay_seconds=0.0)
