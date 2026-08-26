"""Future BPE/Unigram experiment contracts; canonical S0 byte tokenization is unchanged."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Protocol

from .base import TokenizerIdentity

TOKENIZER_TRAINING_SCHEMA = "12-6.tokenizer-training-manifest.v1"
TOKENIZER_ARTIFACT_SCHEMA = "12-6.tokenizer-artifact-identity.v1"
HF_EXPERIMENT_VERSION = "future-hf-tokenizer-v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[.+-][0-9A-Za-z.-]+)?$")
_ALGORITHMS = frozenset({"bpe", "unigram"})
_UNK = "<unk>"


class TokenizerExperimentError(ValueError):
    """Fail-closed experiment contract error."""


class TokenizerExperimentDependencyError(RuntimeError):
    """Exact optional tokenizer runtime is unavailable."""


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_sha256(value: str, field: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be str")
    if _SHA256_RE.fullmatch(value) is None:
        raise TokenizerExperimentError(f"{field} must be lowercase SHA-256")


@dataclass(frozen=True)
class CorpusFileIdentity:
    path: str
    sha256: str
    byte_count: int

    def __post_init__(self) -> None:
        candidate = PurePosixPath(self.path)
        if not self.path or candidate.is_absolute() or ".." in candidate.parts or "\\" in self.path:
            raise TokenizerExperimentError("corpus path must be a safe relative POSIX path")
        _require_sha256(self.sha256, "corpus sha256")
        if self.byte_count < 0:
            raise TokenizerExperimentError("corpus byte_count must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return {"path": self.path, "sha256": self.sha256, "byte_count": self.byte_count}


@dataclass(frozen=True)
class TokenizerTrainingManifest:
    """Content-addressed, non-promoting training plan for a learned tokenizer."""

    experiment_id: str
    algorithm: str
    tokenizers_version: str
    dataset_id: str
    dataset_manifest_sha256: str
    corpus_files: tuple[CorpusFileIdentity, ...]
    vocab_size: int
    min_frequency: int | None = None
    normalization: str = "none"
    pre_tokenizer: str = "byte-level"
    decoder: str = "byte-level"
    special_tokens: tuple[str, ...] = (_UNK,)

    def __post_init__(self) -> None:
        if not self.experiment_id or any(char.isspace() for char in self.experiment_id):
            raise TokenizerExperimentError("experiment_id must be whitespace-free")
        if self.algorithm not in _ALGORITHMS:
            raise TokenizerExperimentError("algorithm must be 'bpe' or 'unigram'")
        if _VERSION_RE.fullmatch(self.tokenizers_version) is None:
            raise TokenizerExperimentError("tokenizers_version must be exact")
        if not self.dataset_id or not self.corpus_files:
            raise TokenizerExperimentError("dataset_id and corpus_files must be non-empty")
        _require_sha256(self.dataset_manifest_sha256, "dataset_manifest_sha256")
        paths = [item.path for item in self.corpus_files]
        if len(paths) != len(set(paths)):
            raise TokenizerExperimentError("corpus file paths must be unique")
        if self.normalization != "none":
            raise TokenizerExperimentError("D03-normalized text must remain unnormalized here")
        if (self.pre_tokenizer, self.decoder) != ("byte-level", "byte-level"):
            raise TokenizerExperimentError("experiment harness requires ByteLevel I/O")
        if not self.special_tokens or self.special_tokens[0] != _UNK:
            raise TokenizerExperimentError("first special token must be '<unk>'")
        if len(self.special_tokens) != len(set(self.special_tokens)):
            raise TokenizerExperimentError("special_tokens must be unique")
        if self.vocab_size < 256 + len(self.special_tokens):
            raise TokenizerExperimentError("vocab_size cannot cover byte alphabet plus specials")
        if self.algorithm == "bpe" and (self.min_frequency is None or self.min_frequency < 1):
            raise TokenizerExperimentError("BPE min_frequency must be positive")
        if self.algorithm == "unigram" and self.min_frequency is not None:
            raise TokenizerExperimentError("Unigram manifest does not define BPE min_frequency")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": TOKENIZER_TRAINING_SCHEMA,
            "experiment_id": self.experiment_id,
            "algorithm": self.algorithm,
            "library": "tokenizers",
            "library_version": self.tokenizers_version,
            "dataset_id": self.dataset_id,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "corpus_files": [
                item.to_dict() for item in sorted(self.corpus_files, key=lambda item: item.path)
            ],
            "vocab_size": self.vocab_size,
            "min_frequency": self.min_frequency,
            "normalization": self.normalization,
            "pre_tokenizer": self.pre_tokenizer,
            "decoder": self.decoder,
            "special_tokens": list(self.special_tokens),
            "input_order_policy": "corpus_path_lexicographic_then_record_order",
            "canonical_s0_unchanged": True,
            "promotion_allowed": False,
        }

    @property
    def sha256(self) -> str:
        return _sha256_text(_canonical_json(self.to_dict()))


class _VocabularyProvider(Protocol):
    def get_vocab(self) -> Mapping[str, int]: ...


def ordered_vocab_sha256(tokenizer: _VocabularyProvider) -> str:
    """Hash complete token->ID semantics; ID permutation changes the hash."""
    vocab = dict(tokenizer.get_vocab())
    if not vocab:
        raise TokenizerExperimentError("vocabulary must not be empty")
    if any(not isinstance(token, str) for token in vocab):
        raise TypeError("vocabulary tokens must be strings")
    ids = list(vocab.values())
    if any(not isinstance(token_id, int) for token_id in ids):
        raise TypeError("vocabulary IDs must be integers")
    if len(ids) != len(set(ids)):
        raise TokenizerExperimentError("vocabulary IDs must be unique")
    if sorted(ids) != list(range(len(vocab))):
        raise TokenizerExperimentError("vocabulary IDs must be exactly contiguous 0..N-1")
    entries = [
        {"id": token_id, "token": token}
        for token, token_id in sorted(vocab.items(), key=lambda item: item[1])
    ]
    return _sha256_text(
        _canonical_json({"schema": "12-6.ordered-token-vocabulary.v1", "entries": entries})
    )


@dataclass(frozen=True)
class TokenizerArtifactIdentity:
    algorithm: str
    tokenizers_version: str
    training_manifest_sha256: str
    tokenizer_json_sha256: str
    vocab_sha256: str
    vocab_size: int
    special_tokens: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if self.algorithm not in _ALGORITHMS:
            raise TokenizerExperimentError("unsupported artifact algorithm")
        if _VERSION_RE.fullmatch(self.tokenizers_version) is None:
            raise TokenizerExperimentError("tokenizers_version must be exact")
        for name in ("training_manifest_sha256", "tokenizer_json_sha256", "vocab_sha256"):
            _require_sha256(getattr(self, name), name)
        if self.vocab_size <= 0:
            raise TokenizerExperimentError("vocab_size must be positive")
        ids = [token_id for _, token_id in self.special_tokens]
        if len(ids) != len(set(ids)) or any(not 0 <= token_id < self.vocab_size for token_id in ids):
            raise TokenizerExperimentError("special token IDs must be unique and in vocabulary")

    @property
    def config_sha256(self) -> str:
        payload = {
            "schema": TOKENIZER_ARTIFACT_SCHEMA,
            "algorithm": self.algorithm,
            "library": "tokenizers",
            "library_version": self.tokenizers_version,
            "training_manifest_sha256": self.training_manifest_sha256,
            "tokenizer_json_sha256": self.tokenizer_json_sha256,
            "vocab_sha256": self.vocab_sha256,
            "vocab_size": self.vocab_size,
            "special_tokens": dict(self.special_tokens),
            "normalization": "none",
            "pre_tokenizer": "byte-level",
            "decoder": "byte-level",
            "canonical": False,
        }
        return _sha256_text(_canonical_json(payload))


@dataclass(frozen=True)
class TokenizerProbe:
    name: str
    language: str
    category: str
    text: str

    def __post_init__(self) -> None:
        if not self.name or not self.language or not self.category:
            raise TokenizerExperimentError("probe metadata must be non-empty")
        if not isinstance(self.text, str):
            raise TypeError("probe text must be str")


@dataclass(frozen=True)
class TokenizerProbeResult:
    name: str
    language: str
    category: str
    codepoints: int
    utf8_bytes: int
    tokens: int
    fertility_tokens_per_codepoint: float
    tokens_per_utf8_byte: float
    round_trip_exact: bool
    unknown_tokens: int


class _ProbeTokenizer(Protocol):
    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]: ...

    def decode(
        self,
        token_ids: Iterable[int],
        *,
        skip_special_tokens: bool = True,
        errors: str = "strict",
    ) -> str: ...


def measure_probe(
    tokenizer: _ProbeTokenizer,
    probe: TokenizerProbe,
    *,
    unknown_token_id: int | None = None,
) -> TokenizerProbeResult:
    token_ids = tokenizer.encode(probe.text)
    decoded = tokenizer.decode(token_ids, skip_special_tokens=False, errors="strict")
    codepoints = len(probe.text)
    utf8_bytes = len(probe.text.encode("utf-8"))
    tokens = len(token_ids)
    unknowns = (
        sum(token_id == unknown_token_id for token_id in token_ids)
        if unknown_token_id is not None
        else 0
    )
    return TokenizerProbeResult(
        probe.name,
        probe.language,
        probe.category,
        codepoints,
        utf8_bytes,
        tokens,
        tokens / codepoints if codepoints else 0.0,
        tokens / utf8_bytes if utf8_bytes else 0.0,
        decoded == probe.text,
        unknowns,
    )


def summarize_by_language(
    results: Sequence[TokenizerProbeResult],
) -> dict[str, dict[str, int | float | bool]]:
    groups: dict[str, list[TokenizerProbeResult]] = {}
    for result in results:
        groups.setdefault(result.language, []).append(result)
    summary: dict[str, dict[str, int | float | bool]] = {}
    for language, group in sorted(groups.items()):
        codepoints = sum(item.codepoints for item in group)
        utf8_bytes = sum(item.utf8_bytes for item in group)
        tokens = sum(item.tokens for item in group)
        summary[language] = {
            "probes": len(group),
            "codepoints": codepoints,
            "utf8_bytes": utf8_bytes,
            "tokens": tokens,
            "fertility_tokens_per_codepoint": tokens / codepoints if codepoints else 0.0,
            "tokens_per_utf8_byte": tokens / utf8_bytes if utf8_bytes else 0.0,
            "round_trip_exact": all(item.round_trip_exact for item in group),
            "unknown_tokens": sum(item.unknown_tokens for item in group),
        }
    return summary


@dataclass(frozen=True)
class VocabularyParameterCost:
    vocab_size: int
    d_model: int
    tied_lm_head: bool
    embedding_parameters: int
    lm_head_parameters: int
    total_vocabulary_parameters: int


def vocabulary_parameter_cost(
    *,
    vocab_size: int,
    d_model: int,
    tied_lm_head: bool,
) -> VocabularyParameterCost:
    if vocab_size <= 0 or d_model <= 0:
        raise TokenizerExperimentError("vocab_size and d_model must be positive")
    embedding = vocab_size * d_model
    head = 0 if tied_lm_head else embedding
    return VocabularyParameterCost(
        vocab_size,
        d_model,
        tied_lm_head,
        embedding,
        head,
        embedding + head,
    )


class HFTokenizerAdapter:
    """TokenizerProtocol-compatible wrapper for one non-canonical experiment."""

    pad_id = None
    bos_id = None
    eos_id = None
    normalization = "none"
    encoding = "utf-8"

    def __init__(
        self,
        tokenizer: Any,
        manifest: TokenizerTrainingManifest,
        artifact: TokenizerArtifactIdentity,
    ) -> None:
        self._tokenizer = tokenizer
        self.artifact_identity = artifact
        self.vocab_size = artifact.vocab_size
        self.version = f"{HF_EXPERIMENT_VERSION}:{manifest.algorithm}"
        specials = dict(artifact.special_tokens)
        self.unk_id = specials[_UNK]
        self.special_tokens = MappingProxyType(specials)

    @property
    def identity(self) -> TokenizerIdentity:
        return TokenizerIdentity(
            self.version,
            self.artifact_identity.config_sha256,
            self.artifact_identity.vocab_sha256,
            self.vocab_size,
            self.normalization,
            self.encoding,
            self.special_tokens,
        )

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be str")
        if add_bos or add_eos:
            raise TokenizerExperimentError("experimental harness has no BOS/EOS semantics")
        return list(self._tokenizer.encode(text, add_special_tokens=False).ids)

    def decode(
        self,
        token_ids: Iterable[int],
        *,
        skip_special_tokens: bool = True,
        errors: str = "strict",
    ) -> str:
        if errors != "strict":
            raise TokenizerExperimentError("experimental harness only supports strict decoding")
        ids = list(token_ids)
        if any(not isinstance(token_id, int) for token_id in ids):
            raise TypeError("token IDs must be integers")
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)


def _load_exact_hf_tokenizers(expected_version: str) -> Any:
    try:
        module = importlib.import_module("tokenizers")
        actual = importlib.metadata.version("tokenizers")
    except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
        raise TokenizerExperimentDependencyError(
            "Hugging Face tokenizers is an optional experiment runtime and is not installed"
        ) from exc
    if actual != expected_version:
        raise TokenizerExperimentDependencyError(
            f"tokenizers runtime {actual!r} does not match manifest {expected_version!r}"
        )
    return module


def train_hf_tokenizer(
    manifest: TokenizerTrainingManifest,
    texts: Iterable[str],
) -> HFTokenizerAdapter:
    """Train local ByteLevel BPE/Unigram using an exact HF Tokenizers runtime."""
    tokenizers = _load_exact_hf_tokenizers(manifest.tokenizers_version)
    ordered_texts = tuple(texts)
    if not ordered_texts:
        raise TokenizerExperimentError("tokenizer training requires at least one text")
    if any(not isinstance(text, str) for text in ordered_texts):
        raise TypeError("tokenizer training texts must be strings")

    byte_alphabet = tokenizers.pre_tokenizers.ByteLevel.alphabet()
    if manifest.algorithm == "bpe":
        model = tokenizers.models.BPE(unk_token=_UNK)
        trainer = tokenizers.trainers.BpeTrainer(
            vocab_size=manifest.vocab_size,
            min_frequency=manifest.min_frequency,
            special_tokens=list(manifest.special_tokens),
            initial_alphabet=byte_alphabet,
            show_progress=False,
        )
    else:
        model = tokenizers.models.Unigram()
        trainer = tokenizers.trainers.UnigramTrainer(
            vocab_size=manifest.vocab_size,
            unk_token=_UNK,
            special_tokens=list(manifest.special_tokens),
            initial_alphabet=byte_alphabet,
            show_progress=False,
        )

    runtime = tokenizers.Tokenizer(model)
    runtime.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
    runtime.decoder = tokenizers.decoders.ByteLevel()
    runtime.train_from_iterator(ordered_texts, trainer=trainer, length=len(ordered_texts))

    special_tokens: list[tuple[str, int]] = []
    for token in manifest.special_tokens:
        token_id = runtime.token_to_id(token)
        if token_id is None:
            raise TokenizerExperimentError(f"trained tokenizer lost special token {token!r}")
        special_tokens.append((token, token_id))
    artifact = TokenizerArtifactIdentity(
        manifest.algorithm,
        manifest.tokenizers_version,
        manifest.sha256,
        _sha256_text(runtime.to_str()),
        ordered_vocab_sha256(runtime),
        runtime.get_vocab_size(with_added_tokens=True),
        tuple(special_tokens),
    )
    adapter = HFTokenizerAdapter(runtime, manifest, artifact)
    for text in ordered_texts:
        if adapter.decode(adapter.encode(text), skip_special_tokens=False) != text:
            raise TokenizerExperimentError("trained tokenizer failed strict round trip")
    return adapter
