"""S15 full exact-size 32K byte BPE source fit, deterministic and nonrelease."""
from __future__ import annotations

import json
from collections import Counter
from itertools import pairwise
from pathlib import Path

import pytest

from tools import plan2_s15_real_32k_bpe_v1 as real

ROOT = Path(__file__).resolve().parents[1]


def _slow_reference_merges(rows: list[dict[str, str]], limit: int) -> list[list[int]]:
    sequences = [
        list(x["text"].encode("utf-8"))
        for x in sorted(rows, key=lambda x: x["record_id"])
    ]
    merges = []
    for _ in range(limit):
        counts = Counter(p for row in sequences for p in pairwise(row))
        if not counts:
            break
        mx = max(counts.values())
        pair = min(p for p, count in counts.items() if count == mx)
        merges.append(list(pair))
        new_id = 260 + len(merges) - 1
        updated = []
        for row in sequences:
            output = []
            i = 0
            while i < len(row):
                if i + 1 < len(row) and tuple(row[i:i+2]) == pair:
                    output.append(new_id)
                    i += 2
                else:
                    output.append(row[i])
                    i += 1
            updated.append(output)
        sequences = updated
    return merges


def test_incremental_merge_order_matches_incumbent_reference() -> None:
    rows = [
        {"record_id": "src:r0001", "text": "abc abc ABC abc!"},
        {"record_id": "src:r0000", "text": "abc abc abc ABC!"},
        {"record_id": "other:r0000", "text": "українська abc abc abc"},
    ]
    goal = 280
    merges, inventory = real.fit_incremental(rows, vocab_size=goal)
    assert merges == _slow_reference_merges(rows, goal - 260)
    assert len(inventory) == 64
    assert real.fit_incremental(list(reversed(rows)), vocab_size=goal) == (
        merges, inventory
    )
    model = real.ByteBPE32k(merges, required_vocab_size=goal)
    assert model.vocab_size == goal
    assert model.identity.to_dict()["vocab_size"] == goal
    with pytest.raises(TypeError):
        model.rank[(97, 98)] = 5
    with pytest.raises(AttributeError):
        model.tokens.append(b"forged")
    with pytest.raises(AttributeError):
        model.merges.append((97, 98))
    for row in rows:
        assert model.decode(model.encode(row["text"])) == row["text"]
        assert model.decode(model.encode(
            row["text"], add_bos=True, add_eos=True
        )) == row["text"]
    assert model.encode("😀 українська") == model.encode("😀 українська")


@pytest.mark.parametrize("text", [
    "", "hello world", "Українська мова", "😀♟️ шахи\n", "A " * 37,
])
def test_stable_32k_algorithm_core_byte_fallback_and_utf8(text: str) -> None:
    rows = [{"record_id": "short:r000", "text": "aaabbbbccccabc abc abcd xyz xyz"},
            {"record_id": "long:r000", "text": "abc abc abc xyz xyz xyz abc"}]
    merges, _ = real.fit_incremental(rows, vocab_size=272)
    fitted = real.ByteBPE32k(merges, required_vocab_size=272)
    assert fitted.decode(fitted.encode(text)) == text
    assert fitted.decode(fitted.encode(text, add_bos=True, add_eos=True)) == text
    assert max(fitted.encode(text, add_bos=True, add_eos=True)) < fitted.vocab_size


@pytest.mark.parametrize("forge", [
    "extra_token", "special_pair", "duplicate_pair", "vocab_target",
])
def test_unsafe_vocab_identity_or_special_token_denied(forge: str) -> None:
    merges = [[97, 98]]
    vocab = 261
    if forge == "extra_token":
        vocab = 262
    elif forge == "special_pair":
        merges = [[256, 97]]
    elif forge == "duplicate_pair":
        merges = [[97, 98], [97, 98]]
        vocab = 262
    else:
        vocab = 32768
    with pytest.raises(real.Real32kBpeDenied):
        real.ByteBPE32k(merges, required_vocab_size=vocab)


def test_real_32k_source_bound_fit_requires_train_only(tmp_path: Path) -> None:
    result = real.stage(ROOT, tmp_path / "model")
    assert result["decision"] == (
        "SOURCE_BOUND_32768_BYTE_BPE_PREPRODUCTION_NOT_RELEASE"
    )
    assert result["actual_vocab_size"] == real.VOCAB == 32768
    assert result["fitted_merge_count"] == real.VOCAB - real.SPECIALS
    assert result["tokenizer_identity"]["vocab_size"] == real.VOCAB
    assert result["heldout_payloads_fitted"] is False
    assert result["final_test_outcomes_read"] is False
    assert result["production_train_source_admitted"] is False
    assert result["training_corpus_authorized"] is False
    assert result["tokenizer_fit_authorized"] is False
    assert result["production_release_authorized"] is False
    assert result["terminal_done"] is False
    assert json.loads((tmp_path / "model" / real.OUTPUT).read_bytes()) == result
    assert len(result["manifest_sha256"]) == 64


def test_32k_input_source_promotion_blocks_unverified_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    partition = {
        "source_families_total": 3, "train_document_count": 1,
        "heldout_document_count": 1, "heldout_plaintext_materialized": False,
        "training_corpus_authorized": True, "tokenizer_fit_authorized": False,
        "physical_s9_admitted": False,
    }
    monkeypatch.setattr(real.training, "build", lambda _: (partition, {}))
    with pytest.raises(real.Real32kBpeDenied, match="rights"):
        real.inspect(tmp_path)


def test_32k_symlink_output_denied_before_any_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "real"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(real, "inspect", lambda _: pytest.fail("unsafe corpus read"))
    with pytest.raises(real.Real32kBpeDenied, match="symlink"):
        real.stage(tmp_path, link)
