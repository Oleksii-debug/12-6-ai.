"""Plan2 S12: frozen byte-BPE engineering fixture with a fail-closed corpus gate.

This is NOT authorization to fit the physical S9 corpus, train a model, or reuse a
checkpoint. Reuse S10 whole-document train partition and S11 architecture only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

from tools import plan2_tokenizer_candidates_v1 as architecture
from tools.plan2_physical_materialization_v1 import _atomic_write, _read_destination, _read_source
from twelve_six.tokenization.base import TokenizerIdentity

SCHEMA = "12-6.plan2-frozen-tokenizer-fixture.v1"
POLICY_PATH = "configs/data/plan2_tokenizer_fit_policy_v1.json"
POLICY_BLOB = "db6e26ef1f34cb60efb23b106e54020cab2fe3fe"
SPECIAL = MappingProxyType({"pad": 256, "bos": 257, "eos": 258, "unk": 259})
HEX = re.compile(r"[0-9a-f]{64}\Z")


class FitDenied(ValueError):
    """Missing trusted fit authority or incompatible frozen tokenizer."""


def _need(value: bool, why: str) -> None:
    if not value:
        raise FitDenied(why)


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def sha(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def git_blob(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def _policy(root: Path) -> dict[str, Any]:
    raw = _read_source(root, POLICY_PATH)
    _need(git_blob(raw) == POLICY_BLOB, "unapproved fit policy")
    value = json.loads(raw.decode("utf-8"))
    _need(type(value) is dict and set(value) == {
        "schema_version", "purpose", "target_vocab_size", "max_fixture_merges",
    } and value["schema_version"] == "12-6.plan2-tokenizer-fit-policy.v1"
          and value["purpose"] == "LOCAL_FREE_SYNTHETIC_NO_RELEASE"
          and type(value["target_vocab_size"]) is int
          and value["target_vocab_size"] == 32768
          and type(value["max_fixture_merges"]) is int
          and 1 <= value["max_fixture_merges"] <= 128,
          "invalid fit policy")
    return value


def _replace(ids: list[int], pair: tuple[int, int], replacement: int) -> list[int]:
    result: list[int] = []
    index = 0
    while index < len(ids):
        if index + 1 < len(ids) and (ids[index], ids[index + 1]) == pair:
            result.append(replacement)
            index += 2
        else:
            result.append(ids[index])
            index += 1
    return result


class FrozenBPE:
    """Byte fallback + fixed special IDs + deterministic ordered merge table."""

    version = "12-6-byte-bpe-fixture-v1"
    normalization = "none"
    encoding = "utf-8"
    special_tokens = SPECIAL
    pad_id, bos_id, eos_id, unk_id = 256, 257, 258, 259

    def __init__(self, merges: Sequence[Sequence[int]]) -> None:
        _need(type(merges) is list and len(merges) <= 128, "merge count invalid")
        self.tokens: list[bytes] = [bytes([n]) for n in range(256)] + [b""] * 4
        self.merges: list[list[int]] = []
        for step in merges:
            _need(type(step) is list and len(step) == 2 and all(
                type(n) is int and 0 <= n < len(self.tokens) and n not in SPECIAL.values()
                for n in step), "invalid or future merge ID")
            a, b = step
            _need([a, b] not in self.merges, "duplicate merge pair")
            token = self.tokens[a] + self.tokens[b]
            _need(0 < len(token) <= 2048, "invalid merge payload")
            self.merges.append([a, b])
            self.tokens.append(token)
        self.vocab_size = len(self.tokens)
        self.tokens = tuple(self.tokens)
        self.merges = tuple(tuple(pair) for pair in self.merges)
        self._config = {
            "version": self.version, "normalization": self.normalization,
            "encoding": self.encoding, "special_tokens": dict(SPECIAL),
            "merges": self.merges, "byte_fallback": True,
        }

    @property
    def identity(self) -> TokenizerIdentity:
        return TokenizerIdentity(
            version=self.version, config_sha256=sha(self._config),
            vocab_sha256=sha([b.hex() for b in self.tokens]),
            vocab_size=self.vocab_size, normalization=self.normalization,
            encoding=self.encoding, special_tokens=SPECIAL,
        )

    def encode(self, text: str, *, add_bos: bool = False,
               add_eos: bool = False) -> list[int]:
        _need(type(text) is str and type(add_bos) is bool
              and type(add_eos) is bool, "invalid encode input")
        try:
            ids = list(text.encode("utf-8", "strict"))
        except UnicodeError as exc:
            raise FitDenied("invalid unicode") from exc
        for index, pair in enumerate(self.merges):
            ids = _replace(ids, (pair[0], pair[1]), 260 + index)
        return ([self.bos_id] if add_bos else []) + ids + ([self.eos_id] if add_eos else [])

    def decode(self, ids: Sequence[int], *, skip_special_tokens: bool = True,
               errors: str = "strict") -> str:
        _need(errors == "strict" and type(skip_special_tokens) is bool,
              "decoder must be strict")
        raw = bytearray()
        for item in ids:
            _need(type(item) is int and 0 <= item < self.vocab_size,
                  "out-of-vocabulary token ID")
            if item in SPECIAL.values():
                _need(skip_special_tokens, "special-token decoding must be explicit")
                continue
            raw.extend(self.tokens[item])
        try:
            return raw.decode("utf-8", "strict")
        except UnicodeError as exc:
            raise FitDenied("invalid token byte sequence") from exc


def _fit(records: list[dict[str, str]], cap: int) -> tuple[list[list[int]], str]:
    _need(type(records) is list and 1 <= len(records) <= 1000,
          "bounded train records only")
    seen = set()
    sequences = []
    audit = []
    for item in sorted(records, key=lambda x: x.get("record_id", "")):
        _need(type(item) is dict and set(item) == {"record_id", "text"},
              "invalid fit record")
        rid, text = item["record_id"], item["text"]
        _need(type(rid) is str and rid and rid not in seen and type(text) is str,
              "invalid/duplicate fit identity")
        seen.add(rid)
        try:
            raw = text.encode("utf-8", "strict")
        except UnicodeError as exc:
            raise FitDenied("invalid fit Unicode") from exc
        _need(bool(raw) and len(raw) <= 65536, "empty/oversized fit record")
        sequences.append(list(raw))
        audit.append({"record_id": rid, "sha256": hashlib.sha256(raw).hexdigest()})
    merges: list[list[int]] = []
    for _ in range(cap):
        counts: Counter[tuple[int, int]] = Counter()
        for seq in sequences:
            counts.update(zip(seq, seq[1:]))
        if not counts or max(counts.values()) < 2:
            break
        max_count = max(counts.values())
        pair = min(pair for pair, number in counts.items() if number == max_count)
        merges.append(list(pair))
        new_id = 259 + len(merges)
        sequences = [_replace(seq, pair, new_id) for seq in sequences]
    return merges, sha(audit)


def freeze_fixture(root: Path) -> dict[str, Any]:
    from tools import plan2_cluster_split_v1 as split

    policy = _policy(root)
    proposal = architecture.inspect(_read_source(root, architecture.POLICY_PATH),
                                    _read_source(root, architecture.FIXTURE_PATH))
    _need(proposal["recommended_architecture_for_future_fit"] == architecture.IDS[1]
          and proposal["tokenizer_fit_authorized"] is False,
          "untrusted S11 proposal")
    s10_policy = split._policy(_read_source(root, split.POLICY_PATH))
    s9, rows = split._synthetic_rows()
    split_manifest = split.build_cluster_split(s9, rows, s10_policy, fixture=True)
    _need(split_manifest["purpose"] == "LOCAL_FREE_SYNTHETIC_COMPONENT"
          and split_manifest["cluster_leakage_count"] == 0
          and split_manifest["tokenizer_fit_authorized"] is False,
          "cannot treat physical/untrusted corpus as fit authority")
    train_ids = set(split_manifest["train_record_ids"])
    train = [{"record_id": x["record_id"], "text": x["text"]}
             for x in rows if x["record_id"] in train_ids]
    merges, train_sha = _fit(train, policy["max_fixture_merges"])
    model = FrozenBPE(merges)
    for row in rows:
        raw = row["text"]
        _need(model.decode(model.encode(raw)) == raw, "noninvertible frozen tokenizer")
    identity = model.identity.to_dict()
    core = {
        "schema_version": SCHEMA,
        "mode": "LOCAL_FREE_SYNTHETIC_FIT_ONLY",
        "s10_split_manifest_sha256": split_manifest["split_manifest_sha256"],
        "s11_architecture_manifest_sha256": proposal["manifest_sha256"],
        "fit_policy_git_blob": POLICY_BLOB,
        "train_records_sha256": train_sha,
        "train_record_count": len(train),
        "fit_merges": merges,
        "tokenizer_identity": identity,
        "target_vocab_size_future": policy["target_vocab_size"],
        "model_spec_vocab_size_fixture": model.vocab_size,
        "frozen": True,
        "training_corpus_authorized": False,
        "physical_tokenizer_fit_authorized": False,
        "production_release_authorized": False,
        "checkpoint_weight_reuse_authorized": False,
        "paid_compute_used": False,
    }
    return {**core, "manifest_sha256": sha(core)}


def verify_frozen(value: Mapping[str, Any]) -> FrozenBPE:
    _need(type(value) is dict and value.get("schema_version") == SCHEMA,
          "invalid tokenizer manifest schema")
    core = dict(value)
    fingerprint = core.pop("manifest_sha256", None)
    _need(type(fingerprint) is str and HEX.fullmatch(fingerprint) is not None
          and sha(core) == fingerprint, "manifest drift")
    _need(core.get("frozen") is True
          and core.get("training_corpus_authorized") is False
          and core.get("physical_tokenizer_fit_authorized") is False
          and core.get("production_release_authorized") is False,
          "silent tokenizer authority mutation")
    fitted = FrozenBPE(core.get("fit_merges"))
    _need(fitted.identity.to_dict() == core.get("tokenizer_identity")
          and fitted.vocab_size == core.get("model_spec_vocab_size_fixture"),
          "tokenizer vocab or config tamper")
    return fitted


def stage(root: Path, out_dir: Path) -> dict[str, Any]:
    _need(not any(x.is_symlink() for x in (out_dir, *out_dir.parents)),
          "symlink output directory")
    result = freeze_fixture(root)
    verify_frozen(result)
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir / "frozen-tokenizer-fixture.json"
    raw = canonical(result)
    if destination.exists() or destination.is_symlink():
        _need(not destination.is_symlink()
              and _read_destination(destination) == raw,
              "immutable frozen manifest mismatch")
    else:
        _atomic_write(out_dir, destination, raw)
    _need(_read_destination(destination) == raw, "frozen readback mismatch")
    return result


def main() -> None:
    cli = argparse.ArgumentParser()
    cli.add_argument("--root", type=Path, default=Path("."))
    cli.add_argument("--out-dir", type=Path, required=True)
    obj = cli.parse_args()
    result = stage(obj.root, obj.out_dir)
    print(json.dumps({"manifest_sha256": result["manifest_sha256"],
                      "vocab_size": result["tokenizer_identity"]["vocab_size"],
                      "production_release_authorized": False}, sort_keys=True))


if __name__ == "__main__":
    main()
