"""Plan2 S15 deterministic exact-size 32768 byte-BPE PREPRODUCTION CANDIDATE.

Uses training-only real S10 source text, never holdout text; no pretrained
model, external LLM, GPU, paid compute or tokenizer replacement. Keeps the
incumbent byte fallback and IDs PAD=256/BOS=257/EOS=258/UNK=259, while
providing an incremental deterministic trainer for the *future* 32K target.
This is not S9 source admission or any permission to train a model.
"""
from __future__ import annotations

import argparse
import heapq
import json
from collections import defaultdict
from pathlib import Path
from types import MappingProxyType
from typing import Any

from tools import plan2_public_domain_books_v1 as books
from tools import plan2_s15_physical_tokenizer_candidate_v1 as segmenter
from tools import plan2_s15_three_family_train_materialization_v1 as training
from tools import plan2_tokenizer_fit_freeze_v1 as frozen
from tools.plan2_physical_materialization_v1 import _atomic_write, _read_destination
from twelve_six.tokenization.base import TokenizerIdentity

SCHEMA = "12-6.plan2-s15-real-32k-byte-bpe-preproduction-v1"
OUTPUT = "real-32k-byte-bpe-preproduction.json"
VOCAB = 32768
SPECIALS = 260
MAX_TOKEN_BYTES = 2048
VERSION = "12-6-byte-bpe-32k-preproduction-v1"


class Real32kBpeDenied(ValueError):
    """Unverifiable training-only BPE fit or target vocabulary."""


def need(ok: bool, reason: str) -> None:
    if not ok:
        raise Real32kBpeDenied(reason)


class ByteBPE32k:
    """Frozen ordered BPE rules with fast exact-rank encoding and byte fallback."""

    version = VERSION
    normalization = "none"
    encoding = "utf-8"
    pad_id = 256
    bos_id = 257
    eos_id = 258
    unk_id = 259

    def __init__(self, merges: list[list[int]],
                 *, required_vocab_size: int = VOCAB) -> None:
        need(type(merges) is list
             and type(required_vocab_size) is int
             and SPECIALS <= required_vocab_size <= VOCAB
             and len(merges) == required_vocab_size - SPECIALS,
             "frozen BPE target vocabulary count mismatch")
        self.tokens: list[bytes] = [bytes([b]) for b in range(256)] + [b""] * 4
        self.merges: list[tuple[int, int]] = []
        seen_blobs = set(self.tokens[:256])
        seen_pairs: set[tuple[int, int]] = set()
        for pair in merges:
            need(type(pair) is list and len(pair) == 2
                 and all(type(x) is int and 0 <= x < len(self.tokens)
                         and x not in (256, 257, 258, 259) for x in pair),
                 "BPE merge pair includes future ID or special token")
            left, right = pair
            need((left, right) not in seen_pairs,
                 "duplicate frozen BPE merge pair")
            token_bytes = self.tokens[left] + self.tokens[right]
            need(1 <= len(token_bytes) <= MAX_TOKEN_BYTES
                 and token_bytes not in seen_blobs,
                 "duplicate/oversized frozen BPE vocabulary bytes")
            self.tokens.append(token_bytes)
            self.merges.append((left, right))
            seen_blobs.add(token_bytes)
            seen_pairs.add((left, right))
        self.tokens = tuple(self.tokens)
        self.merges = tuple(self.merges)
        self.vocab_size = len(self.tokens)
        self.rank = MappingProxyType({pair: i for i, pair in enumerate(self.merges)})
        self.special_tokens = MappingProxyType(dict(frozen.SPECIAL))
        self._config = {
            "version": self.version,
            "normalization": self.normalization,
            "encoding": self.encoding,
            "special_tokens": dict(self.special_tokens),
            "byte_fallback": True,
            "max_token_bytes": MAX_TOKEN_BYTES,
            "merges": self.merges,
        }

        self._identity = TokenizerIdentity(
            version=self.version,
            config_sha256=frozen.sha(self._config),
            vocab_sha256=frozen.sha([value.hex() for value in self.tokens]),
            vocab_size=self.vocab_size,
            normalization=self.normalization,
            encoding=self.encoding,
            special_tokens=frozen.SPECIAL,
        )

    @property
    def identity(self) -> TokenizerIdentity:
        return self._identity

    def encode(self, text: str, *, add_bos: bool = False,
               add_eos: bool = False) -> list[int]:
        need(type(text) is str and type(add_bos) is bool
             and type(add_eos) is bool,
             "invalid real 32K BPE input")
        try:
            data = list(text.encode("utf-8", "strict"))
        except UnicodeError as exc:
            raise Real32kBpeDenied("invalid input Unicode") from exc
        if not data:
            return ([self.bos_id] if add_bos else []) + (
                [self.eos_id] if add_eos else []
            )
        n = len(data)
        prev = [i - 1 for i in range(n)]
        nxt = [i + 1 for i in range(n)]
        nxt[-1] = -1
        alive = [True] * n
        heap: list[tuple[int, int]] = []

        def enter(i: int) -> None:
            if i < 0 or nxt[i] < 0:
                return
            rank = self.rank.get((data[i], data[nxt[i]]))
            if rank is not None:
                heapq.heappush(heap, (rank, i))

        for index in range(n - 1):
            enter(index)
        while heap:
            rank, i = heapq.heappop(heap)
            if not alive[i]:
                continue
            j = nxt[i]
            if j < 0 or not alive[j] or self.rank.get(
                (data[i], data[j])
            ) != rank:
                continue
            before, after = prev[i], nxt[j]
            data[i] = SPECIALS + rank
            alive[j] = False
            nxt[i] = after
            if after >= 0:
                prev[after] = i
            enter(before)
            enter(i)
        ids = [data[i] for i in range(n) if alive[i]]
        return ([self.bos_id] if add_bos else []) + ids + (
            [self.eos_id] if add_eos else []
        )

    def decode(self, ids: list[int], *,
               skip_special_tokens: bool = True, errors: str = "strict") -> str:
        need(errors == "strict" and type(skip_special_tokens) is bool,
             "unsupported BPE decoder options")
        raw = bytearray()
        for item in ids:
            need(type(item) is int and 0 <= item < len(self.tokens),
                 "token is outside 32K vocabulary")
            if item in self.special_tokens.values():
                need(skip_special_tokens, "special tokens must be filtered")
                continue
            raw.extend(self.tokens[item])
        try:
            return raw.decode("utf-8", "strict")
        except UnicodeError as exc:
            raise Real32kBpeDenied("invalid token byte sequence") from exc


def fit_incremental(rows: list[dict[str, str]], *,
                    vocab_size: int = VOCAB) -> tuple[list[list[int]], str]:
    """Exact greedy pair-frequency tie-breaking, updating only touched pairs."""
    need(type(rows) is list and bool(rows)
         and type(vocab_size) is int
         and SPECIALS <= vocab_size <= VOCAB,
         "invalid bounded real-source fit request")
    ids: set[str] = set()
    values: list[int] = []
    previous: list[int] = []
    following: list[int] = []
    living: list[bool] = []
    input_inventory = []
    for row in sorted(rows, key=lambda r: r.get("record_id", "")):
        need(type(row) is dict and set(row) == {"record_id", "text"},
             "malformed training document")
        rid, text = row["record_id"], row["text"]
        need(type(rid) is str and bool(rid) and rid not in ids
             and type(text) is str and bool(text),
             "reused/empty physical training record")
        ids.add(rid)
        raw = text.encode("utf-8", "strict")
        need(0 < len(raw) <= 65536, "fit input too large or empty")
        start = len(values)
        for i, byte in enumerate(raw):
            values.append(byte)
            previous.append(start + i - 1 if i else -1)
            following.append(start + i + 1 if i + 1 < len(raw) else -1)
            living.append(True)
        input_inventory.append({"record_id": rid, "sha256": books.sha(raw)})
    need(bool(values) and len(rows) <= 1000,
         "physical fit record count invalid")
    tokens = [bytes([i]) for i in range(256)] + [b""] * 4
    seen_values = set(tokens[:256])
    pairs: dict[tuple[int, int], set[int]] = defaultdict(set)
    versions: dict[tuple[int, int], int] = defaultdict(int)
    heap: list[tuple[int, tuple[int, int], int]] = []
    prohibited: set[tuple[int, int]] = set()

    def eligible(pair: tuple[int, int]) -> bool:
        return (pair not in prohibited
                and len(tokens[pair[0]]) + len(tokens[pair[1]])
                    <= MAX_TOKEN_BYTES)

    def enqueue(pair: tuple[int, int]) -> None:
        if pairs[pair] and eligible(pair):
            heapq.heappush(
                heap, (-len(pairs[pair]), pair, versions[pair]),
            )

    def add(index: int) -> None:
        nxt = following[index]
        if nxt < 0:
            return
        pair = (values[index], values[nxt])
        pairs[pair].add(index)
        versions[pair] += 1
        enqueue(pair)

    def remove(index: int) -> None:
        nxt = following[index]
        if nxt < 0:
            return
        pair = (values[index], values[nxt])
        pairs[pair].remove(index)
        versions[pair] += 1
        enqueue(pair)

    for i in range(len(values)):
        if following[i] >= 0:
            add(i)
    merges: list[list[int]] = []
    while len(merges) < vocab_size - SPECIALS:
        while heap:
            minus_count, pair, version = heapq.heappop(heap)
            if (version == versions[pair]
                    and minus_count == -len(pairs[pair])
                    and pairs[pair]
                    and eligible(pair)):
                break
        else:
            raise Real32kBpeDenied(
                "actual training corpus lacks enough distinct source-bound "
                "merge pairs to fit full 32768-vocabulary without fabricated tokens"
            )
        token = tokens[pair[0]] + tokens[pair[1]]
        if token in seen_values:
            prohibited.add(pair)
            continue
        new_id = SPECIALS + len(merges)
        tokens.append(token)
        seen_values.add(token)
        merges.append(list(pair))
        for i in sorted(tuple(pairs[pair])):
            j = following[i]
            if (not living[i] or j < 0 or not living[j]
                    or (values[i], values[j]) != pair):
                continue
            left, right = previous[i], following[j]
            if left >= 0:
                remove(left)
            remove(i)
            if right >= 0:
                remove(j)
            values[i] = new_id
            living[j] = False
            following[i] = right
            if right >= 0:
                previous[right] = i
            if left >= 0:
                add(left)
            if right >= 0:
                add(i)
        # Bound lazy-deletion metadata without changing deterministic ordering.
        if len(heap) > max(20000, 12 * len(versions)):
            heap = [
                (-len(pairs[p]), p, versions[p])
                for p in versions if pairs[p] and eligible(p)
            ]
            heapq.heapify(heap)
    need(len(merges) == vocab_size - SPECIALS
         and len(tokens) == vocab_size
         and len(seen_values) == vocab_size - 4,
         "trained vocabulary has holes, duplicated bytes or missing IDs")
    return merges, books.sha(books.canonical(input_inventory))


def inspect(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    partition, physical = training.build(root)
    need(partition["source_families_total"] == 3
         and partition["train_document_count"] > 0
         and partition["heldout_document_count"] > 0
         and partition["heldout_plaintext_materialized"] is False
         and partition["training_corpus_authorized"] is False
         and partition["tokenizer_fit_authorized"] is False
         and partition["physical_s9_admitted"] is False,
         "physical training/holdout rights not fully audited")
    rows = []
    for member in sorted(partition["train_members"], key=lambda m: m["path"]):
        raw = physical[member["path"]]
        need(books.sha(raw) == member["normalized_sha256"]
             and len(raw) == member["normalized_bytes"],
             "source changed between train split and byte-BPE fit")
        rows.extend(segmenter.segments(
            member["record_id_sha256"], raw.decode("utf-8", "strict"),
        ))
    merges, input_sha = fit_incremental(rows, vocab_size=VOCAB)
    model = ByteBPE32k(merges)
    need(model.vocab_size == VOCAB
         and all(model.decode(model.encode(x["text"])) == x["text"]
                 for x in rows),
         "32K real source-byte BPE failed exact full training roundtrip")
    core = {
        "schema_version": SCHEMA,
        "decision": "SOURCE_BOUND_32768_BYTE_BPE_PREPRODUCTION_NOT_RELEASE",
        "source_train_partition_sha256": partition["manifest_sha256"],
        "source_s10_split_sha256": partition["s10_split_manifest_sha256"],
        "source_train_utf8_bytes": partition["physical_train_bytes"],
        "source_train_document_count": partition["train_document_count"],
        "heldout_document_count": partition["heldout_document_count"],
        "source_train_record_count": len(rows),
        "fit_input_inventory_sha256": input_sha,
        "fitted_merge_count": len(merges),
        "fitted_merges": merges,
        "target_vocab_size": VOCAB,
        "actual_vocab_size": model.vocab_size,
        "tokenizer_identity": model.identity.to_dict(),
        "max_token_bytes": MAX_TOKEN_BYTES,
        "byte_fallback": True,
        "physical_source_fit_was_preproduction_candidate": True,
        "heldout_payloads_fitted": False,
        "final_test_outcomes_read": False,
        "checkpoint_weight_reuse_authorized": False,
        "production_train_source_admitted": False,
        "production_release_authorized": False,
        "training_corpus_authorized": False,
        "tokenizer_fit_authorized": False,
        "terminal_done": False,
    }
    return {**core, "manifest_sha256": books.sha(books.canonical(core))}


def stage(root: Path, destination: Path) -> dict[str, Any]:
    need(not any(x.is_symlink() for x in (destination, *destination.parents)),
         "symlink 32K source-bound tokenizer output")
    report = inspect(root)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / OUTPUT
    raw = books.canonical(report)
    if target.exists() or target.is_symlink():
        need(target.is_file() and not target.is_symlink()
             and _read_destination(target) == raw,
             "immutable 32768 BPE candidate changed")
    else:
        _atomic_write(destination, target, raw)
    need(_read_destination(target) == raw,
         "32K BPE output readback changed")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    value = stage(args.root, args.out_dir)
    print(json.dumps({
        "decision": value["decision"],
        "manifest_sha256": value["manifest_sha256"],
        "actual_vocab_size": value["actual_vocab_size"],
        "terminal_done": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
