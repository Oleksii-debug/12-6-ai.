"""Future tokenizer experiment harnesses and controlled measurement contracts.

Canonical S0 remains ``s0-byte-v1``. Everything in this module is explicitly
experimental and must be bound to a versioned training manifest before its
token IDs can be used by a checkpoint.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .base import TokenizerIdentity

EXPERIMENT_MANIFEST_SCHEMA = "12-6.tokenizer-training-manifest.v1"
EXPERIMENT_ARTIFACT_SCHEMA = "12-6.tokenizer-artifact.v1"
_ALLOWED_STAGES = frozenset({"S1", "S2", "S3", "S4"})
_ALLOWED_BACKENDS = {"bpe": "tokenizers", "unigram": "sentencepiece"}


def _canonical_json(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: str, *, field: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")


def corpus_sha256(texts: Iterable[str]) -> str:
    """Hash an ordered training corpus without ambiguous concatenation."""
    digest = hashlib.sha256()
    count = 0
    for text in texts:
        if not isinstance(text, str):
            raise TypeError("training documents must be strings")
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        count += 1
    if count == 0:
        raise ValueError("training corpus must not be empty")
    return digest.hexdigest()


def ordered_vocab_sha256(vocab: Mapping[str, int]) -> str:
    """Fingerprint the complete token->ID mapping in exact ID order."""
    if not vocab:
        raise ValueError("vocabulary must not be empty")
    ids = list(vocab.values())
    if any(not isinstance(token_id, int) for token_id in ids):
        raise TypeError("vocabulary IDs must be integers")
    if len(set(ids)) != len(ids):
        raise ValueError("vocabulary IDs must be unique")
    if sorted(ids) != list(range(len(ids))):
        raise ValueError("vocabulary IDs must be dense 0..N-1")
    by_id = sorted(vocab.items(), key=lambda item: item[1])
    payload = {
        "schema_version": 1,
        "entries": [{"id": token_id, "token": token} for token, token_id in by_id],
    }
    return _sha256_bytes(_canonical_json(payload).encode("utf-8"))


@dataclass(frozen=True)
class TokenizerTrainingManifest:
    """Versioned identity for one tokenizer-training execution."""

    experiment_id: str
    stage: str
    algorithm: str
    backend_library: str
    backend_version: str
    requested_vocab_size: int
    training_corpus_sha256: str
    training_document_count: int
    normalization: str = "none"
    byte_fallback: bool = True
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.experiment_id:
            raise ValueError("experiment_id must be non-empty")
        if self.stage not in _ALLOWED_STAGES:
            raise ValueError("tokenizer experiments are currently restricted to S1-S4")
        expected_library = _ALLOWED_BACKENDS.get(self.algorithm)
        if expected_library is None:
            raise ValueError("algorithm must be 'bpe' or 'unigram'")
        if self.backend_library != expected_library:
            raise ValueError(
                f"{self.algorithm} experiments must use maintained library {expected_library!r}"
            )
        if not self.backend_version:
            raise ValueError("backend_version must be recorded")
        if self.requested_vocab_size < 256:
            raise ValueError("requested_vocab_size must be at least 256")
        _require_sha256(self.training_corpus_sha256, field="training_corpus_sha256")
        if self.training_document_count <= 0:
            raise ValueError("training_document_count must be positive")
        if self.normalization != "none":
            raise ValueError("experiment harness currently requires normalization='none'")
        if not isinstance(self.seed, int):
            raise TypeError("seed must be int")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": EXPERIMENT_MANIFEST_SCHEMA,
            "experiment_id": self.experiment_id,
            "stage": self.stage,
            "algorithm": self.algorithm,
            "backend_library": self.backend_library,
            "backend_version": self.backend_version,
            "requested_vocab_size": self.requested_vocab_size,
            "training_corpus_sha256": self.training_corpus_sha256,
            "training_document_count": self.training_document_count,
            "normalization": self.normalization,
            "byte_fallback": self.byte_fallback,
            "seed": self.seed,
            "canonical_s0_unchanged": True,
            "promotion_allowed": False,
        }

    @property
    def identity_sha256(self) -> str:
        return _sha256_bytes(_canonical_json(self.to_dict()).encode("utf-8"))


@dataclass(frozen=True)
class ExperimentalTokenizerArtifact:
    """Recorded token-ID semantics for an executed experiment."""

    manifest_sha256: str
    algorithm: str
    backend_library: str
    backend_version: str
    model_sha256: str
    vocab_sha256: str
    vocab_size: int
    tokenizer_version: str

    def __post_init__(self) -> None:
        for field, value in (
            ("manifest_sha256", self.manifest_sha256),
            ("model_sha256", self.model_sha256),
            ("vocab_sha256", self.vocab_sha256),
        ):
            _require_sha256(value, field=field)
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": EXPERIMENT_ARTIFACT_SCHEMA,
            "manifest_sha256": self.manifest_sha256,
            "algorithm": self.algorithm,
            "backend_library": self.backend_library,
            "backend_version": self.backend_version,
            "model_sha256": self.model_sha256,
            "vocab_sha256": self.vocab_sha256,
            "vocab_size": self.vocab_size,
            "tokenizer_version": self.tokenizer_version,
            "canonical_s0_unchanged": True,
            "promotion_allowed": False,
        }

    @property
    def identity_sha256(self) -> str:
        return _sha256_bytes(_canonical_json(self.to_dict()).encode("utf-8"))


@dataclass(frozen=True)
class Measurement:
    category: str
    documents: int
    code_points: int
    utf8_bytes: int
    tokens: int
    fertility_tokens_per_code_point: float
    bytes_per_token: float
    round_trip_rate: float

    def to_dict(self) -> dict[str, object]:
        return {
            "category": self.category,
            "documents": self.documents,
            "code_points": self.code_points,
            "utf8_bytes": self.utf8_bytes,
            "tokens": self.tokens,
            "fertility_tokens_per_code_point": self.fertility_tokens_per_code_point,
            "bytes_per_token": self.bytes_per_token,
            "round_trip_rate": self.round_trip_rate,
        }


CONTROLLED_SAMPLES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "en": (
            "The quick brown fox jumps over the lazy dog.",
            "Token accounting must be exact across every packed sequence.",
        ),
        "uk": (
            "Українська мова має точно зберігати кожен символ.",
            "Київ, Львів і Харків — міста України.",
        ),
        "code": (
            'def greet(name: str) -> str:\n    return f"Привіт, {name}!"',
            "values = [x * x for x in range(16) if x % 2 == 0]",
        ),
        "unicode": (
            "naïve café — 你好 — العربية — 😀 — 𝛑",
            "é ≠ é; tabs\tand\nnewlines stay exact.",
        ),
    }
)


def measure_tokenizer(
    tokenizer: Any,
    samples: Mapping[str, Sequence[str]] = CONTROLLED_SAMPLES,
) -> tuple[Measurement, ...]:
    """Measure fertility and exact round-trip integrity on controlled text."""
    results: list[Measurement] = []
    for category, texts in samples.items():
        if not texts:
            raise ValueError(f"sample category {category!r} must not be empty")
        code_points = 0
        utf8_bytes = 0
        tokens = 0
        exact_round_trips = 0
        for text in texts:
            encoded = list(tokenizer.encode(text))
            decoded = tokenizer.decode(encoded)
            code_points += len(text)
            utf8_bytes += len(text.encode("utf-8"))
            tokens += len(encoded)
            exact_round_trips += int(decoded == text)
        results.append(
            Measurement(
                category=category,
                documents=len(texts),
                code_points=code_points,
                utf8_bytes=utf8_bytes,
                tokens=tokens,
                fertility_tokens_per_code_point=tokens / code_points,
                bytes_per_token=utf8_bytes / tokens if tokens else 0.0,
                round_trip_rate=exact_round_trips / len(texts),
            )
        )
    return tuple(results)


def vocabulary_parameter_cost(
    vocab_size: int,
    d_model: int,
    *,
    tied_embeddings: bool = True,
    lm_head_bias: bool = False,
) -> int:
    """Return exact embedding/output-head parameters attributable to vocabulary."""
    if vocab_size <= 0 or d_model <= 0:
        raise ValueError("vocab_size and d_model must be positive")
    cost = vocab_size * d_model
    if not tied_embeddings:
        cost += vocab_size * d_model
    if lm_head_bias:
        cost += vocab_size
    return cost


class _HuggingFaceTokenizerAdapter:
    pad_id = None
    bos_id = None
    eos_id = None
    version = "exp-hf-bytelevel-bpe-v1"

    def __init__(self, backend: Any, *, manifest: TokenizerTrainingManifest) -> None:
        self._backend = backend
        raw_json = backend.to_str()
        canonical_model = _canonical_json(json.loads(raw_json)).encode("utf-8")
        self._model_sha256 = _sha256_bytes(canonical_model)
        vocab = backend.get_vocab()
        self.vocab_size = backend.get_vocab_size()
        self._identity = TokenizerIdentity(
            version=self.version,
            config_sha256=_sha256_bytes(
                _canonical_json(
                    {
                        "manifest_sha256": manifest.identity_sha256,
                        "model_sha256": self._model_sha256,
                    }
                ).encode("utf-8")
            ),
            vocab_sha256=ordered_vocab_sha256(vocab),
            vocab_size=self.vocab_size,
            normalization="none",
            encoding="utf-8",
            special_tokens=MappingProxyType({}),
        )

    @property
    def identity(self) -> TokenizerIdentity:
        return self._identity

    @property
    def model_sha256(self) -> str:
        return self._model_sha256

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        if add_bos or add_eos:
            raise ValueError("experimental BPE harness defines no BOS/EOS IDs")
        return list(self._backend.encode(text).ids)

    def decode(
        self,
        token_ids: Iterable[int],
        *,
        skip_special_tokens: bool = True,
        errors: str = "strict",
    ) -> str:
        del errors
        return str(self._backend.decode(list(token_ids), skip_special_tokens=skip_special_tokens))


class _SentencePieceAdapter:
    pad_id = None
    bos_id = None
    eos_id = None
    version = "exp-sentencepiece-unigram-v1"

    def __init__(
        self,
        processor: Any,
        model_bytes: bytes,
        *,
        manifest: TokenizerTrainingManifest,
    ) -> None:
        self._processor = processor
        self._model_sha256 = _sha256_bytes(model_bytes)
        self.vocab_size = int(processor.get_piece_size())
        vocab = {str(processor.id_to_piece(index)): index for index in range(self.vocab_size)}
        unk_id = int(processor.unk_id())
        special_tokens = (
            MappingProxyType({"unk": unk_id}) if unk_id >= 0 else MappingProxyType({})
        )
        self._identity = TokenizerIdentity(
            version=self.version,
            config_sha256=_sha256_bytes(
                _canonical_json(
                    {
                        "manifest_sha256": manifest.identity_sha256,
                        "model_sha256": self._model_sha256,
                    }
                ).encode("utf-8")
            ),
            vocab_sha256=ordered_vocab_sha256(vocab),
            vocab_size=self.vocab_size,
            normalization="none",
            encoding="utf-8",
            special_tokens=special_tokens,
        )

    @property
    def identity(self) -> TokenizerIdentity:
        return self._identity

    @property
    def model_sha256(self) -> str:
        return self._model_sha256

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        if add_bos or add_eos:
            raise ValueError("experimental Unigram harness defines no BOS/EOS IDs")
        return list(self._processor.encode(text, out_type=int))

    def decode(
        self,
        token_ids: Iterable[int],
        *,
        skip_special_tokens: bool = True,
        errors: str = "strict",
    ) -> str:
        del skip_special_tokens, errors
        return str(self._processor.decode(list(token_ids)))


def _backend_version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            f"optional tokenizer experiment backend {distribution!r} is not installed; "
            "canonical S0 does not require it"
        ) from exc


def train_huggingface_bpe(
    texts: Iterable[str],
    *,
    stage: str,
    experiment_id: str,
    vocab_size: int,
    seed: int = 0,
) -> tuple[_HuggingFaceTokenizerAdapter, TokenizerTrainingManifest]:
    """Train a ByteLevel BPE experiment with Hugging Face Tokenizers."""
    documents = tuple(texts)
    if vocab_size < 256:
        raise ValueError("ByteLevel BPE vocab_size must be at least 256")
    version = _backend_version("tokenizers")
    manifest = TokenizerTrainingManifest(
        experiment_id=experiment_id,
        stage=stage,
        algorithm="bpe",
        backend_library="tokenizers",
        backend_version=version,
        requested_vocab_size=vocab_size,
        training_corpus_sha256=corpus_sha256(documents),
        training_document_count=len(documents),
        normalization="none",
        byte_fallback=True,
        seed=seed,
    )
    tokenizers = importlib.import_module("tokenizers")
    models = importlib.import_module("tokenizers.models")
    pre_tokenizers = importlib.import_module("tokenizers.pre_tokenizers")
    decoders = importlib.import_module("tokenizers.decoders")
    trainers = importlib.import_module("tokenizers.trainers")

    tokenizer = tokenizers.Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=1,
        show_progress=False,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(documents, trainer=trainer, length=len(documents))
    return _HuggingFaceTokenizerAdapter(tokenizer, manifest=manifest), manifest


def train_sentencepiece_unigram(
    texts: Iterable[str],
    *,
    stage: str,
    experiment_id: str,
    vocab_size: int,
    seed: int = 0,
) -> tuple[_SentencePieceAdapter, TokenizerTrainingManifest]:
    """Train a byte-fallback Unigram experiment with SentencePiece."""
    documents = tuple(texts)
    if vocab_size < 300:
        raise ValueError("SentencePiece byte-fallback Unigram requires vocab_size >= 300")
    version = _backend_version("sentencepiece")
    manifest = TokenizerTrainingManifest(
        experiment_id=experiment_id,
        stage=stage,
        algorithm="unigram",
        backend_library="sentencepiece",
        backend_version=version,
        requested_vocab_size=vocab_size,
        training_corpus_sha256=corpus_sha256(documents),
        training_document_count=len(documents),
        normalization="none",
        byte_fallback=True,
        seed=seed,
    )
    sentencepiece = importlib.import_module("sentencepiece")
    with tempfile.TemporaryDirectory(prefix="twelve-six-sp-") as directory:
        root = Path(directory)
        corpus_path = root / "corpus.txt"
        corpus_path.write_text("\n".join(documents) + "\n", encoding="utf-8")
        model_prefix = root / "tokenizer"
        sentencepiece.SentencePieceTrainer.Train(
            input=str(corpus_path),
            model_prefix=str(model_prefix),
            model_type="unigram",
            vocab_size=vocab_size,
            character_coverage=1.0,
            byte_fallback=True,
            normalization_rule_name="identity",
            add_dummy_prefix=False,
            remove_extra_whitespaces=False,
            split_by_whitespace=False,
            bos_id=-1,
            eos_id=-1,
            pad_id=-1,
            unk_id=0,
            hard_vocab_limit=False,
            shuffle_input_sentence=False,
            num_threads=1,
        )
        model_path = model_prefix.with_suffix(".model")
        model_bytes = model_path.read_bytes()
        processor = sentencepiece.SentencePieceProcessor(model_file=str(model_path))
    return _SentencePieceAdapter(processor, model_bytes, manifest=manifest), manifest


def artifact_record(
    tokenizer: Any,
    manifest: TokenizerTrainingManifest,
) -> ExperimentalTokenizerArtifact:
    """Bind trained model bytes, vocabulary mapping and training manifest."""
    return ExperimentalTokenizerArtifact(
        manifest_sha256=manifest.identity_sha256,
        algorithm=manifest.algorithm,
        backend_library=manifest.backend_library,
        backend_version=manifest.backend_version,
        model_sha256=str(tokenizer.model_sha256),
        vocab_sha256=tokenizer.identity.vocab_sha256,
        vocab_size=int(tokenizer.vocab_size),
        tokenizer_version=str(tokenizer.version),
    )
