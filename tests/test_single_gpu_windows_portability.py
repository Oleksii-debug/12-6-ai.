from __future__ import annotations

from types import SimpleNamespace

from twelve_six.training import single_gpu


def test_missing_resource_module_disables_rss_without_blocking_training(monkeypatch) -> None:
    monkeypatch.setattr(single_gpu, "_resource", None)
    assert single_gpu._process_rss_bytes() is None


def test_unavailable_resource_metric_is_nonfatal(monkeypatch) -> None:
    failing_resource = SimpleNamespace(
        RUSAGE_SELF=0,
        getrusage=lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("unsupported")),
    )
    monkeypatch.setattr(single_gpu, "_resource", failing_resource)
    assert single_gpu._process_rss_bytes() is None
