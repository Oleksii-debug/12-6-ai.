#!/usr/bin/env python3
"""Stdlib-only synthetic validation for DATA-539 G05 granularity repair."""

from __future__ import annotations

import importlib.machinery
import sys
import types
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"


def _install_namespace(name: str, path: Path) -> None:
    module = types.ModuleType(name)
    module.__package__ = name
    module.__path__ = [str(path)]  # type: ignore[attr-defined]
    spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    spec.submodule_search_locations = [str(path)]
    module.__spec__ = spec
    sys.modules[name] = module


if "twelve_six" not in sys.modules:
    _install_namespace("twelve_six", _SRC_ROOT / "twelve_six")
if "twelve_six.data" not in sys.modules:
    _install_namespace("twelve_six.data", _SRC_ROOT / "twelve_six" / "data")

from twelve_six.data.document_quality import assess_document as assess_incumbent
from twelve_six.data.document_quality_v2 import assess_document, diversity_window_evidence


def _alpha_word(index: int) -> str:
    first = chr(ord("a") + (index // 26) % 26)
    second = chr(ord("a") + index % 26)
    return f"lexeme{first}{second}"


def _cyclic_text(unique_words: int, total_tokens: int) -> str:
    vocabulary = [_alpha_word(index) for index in range(unique_words)]
    return " ".join(vocabulary[index % unique_words] for index in range(total_tokens))


def main() -> int:
    rich = _cyclic_text(80, 2048)
    old_rich = assess_incumbent("rich", rich, "en")
    if old_rich.reasons != ("low_token_diversity",):
        raise RuntimeError(f"expected incumbent granularity hazard, got {old_rich.reasons}")
    evidence = diversity_window_evidence(rich, "en")
    if not evidence.used_windowed_decision or evidence.low_diversity_windows != 0:
        raise RuntimeError(f"unexpected rich-window evidence: {evidence}")
    new_rich = assess_document("rich", rich, "en")
    if not new_rich.accepted or new_rich.reasons:
        raise RuntimeError(f"granularity repair did not admit rich document: {new_rich}")

    repetitive = _cyclic_text(16, 2048)
    repetitive_evidence = diversity_window_evidence(repetitive, "en")
    if repetitive_evidence.low_diversity_fraction != 1.0:
        raise RuntimeError(f"unexpected repetitive evidence: {repetitive_evidence}")
    new_repetitive = assess_document("repetitive", repetitive, "en")
    if new_repetitive.accepted or "low_token_diversity" not in new_repetitive.reasons:
        raise RuntimeError("systematically repetitive document escaped quality rejection")

    short = _cyclic_text(70, 120)
    if assess_document("short", short, "en") != assess_incumbent("short", short, "en"):
        raise RuntimeError("short-document semantics changed")

    code = "def add(left, right):\n    return left + right\n"
    if assess_document("code", code, "code") != assess_incumbent("code", code, "code"):
        raise RuntimeError("code-mode semantics changed")

    print("DATA-539 PASS: granularity repair is bounded and deterministic")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
