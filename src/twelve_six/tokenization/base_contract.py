"""Versioned research Base token contract for the next 10M baseline.

TOK-189 deliberately adds no token algorithm and no chat/instruction semantics.
It hardens the accepted byte/no-EOS behavior into one machine-readable contract.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from twelve_six.checkpoint import CheckpointCompatibilityError
from twelve_six.packing import PACKING_CONFIG_HASH, PACKING_VERSION
from twelve_six.tokenization.byte import (
    BYTE_TOKENIZER_HASH,
    BYTE_TOKENIZER_VERSION,
    BYTE_VOCAB_HASH,
    ByteTokenizer,
    canonical_vocab_json,
)

CONTRACT_ID = "base-byte-noeos-v1"
CONTRACT_SCHEMA = "12-6.research-base-token-contract.v1"
CONTRACT_RELATIVE_PATH = Path("configs/token_contracts/base_byte_noeos_v1.research.json")


class BaseTokenContractError(RuntimeError):
    """Raised when the research Base token contract drifts or is incompatible."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def contract_identity(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("identity_sha256", None)
    return hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()


def load_research_base_token_contract(path: str | Path | None = None) -> dict[str, Any]:
    target = Path(path) if path is not None else _repo_root() / CONTRACT_RELATIVE_PATH
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise BaseTokenContractError("token contract must be a JSON object")
    if payload.get("schema") != CONTRACT_SCHEMA or payload.get("contract_id") != CONTRACT_ID:
        raise BaseTokenContractError("unexpected token contract schema/id")
    expected = payload.get("identity_sha256")
    actual = contract_identity(payload)
    if expected != actual:
        raise BaseTokenContractError(
            f"token contract identity mismatch: expected={expected!r} actual={actual!r}"
        )
    return payload


def hf_transformers_token_mapping(
    contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = (
        load_research_base_token_contract() if contract is None else dict(contract)
    )
    mapping = dict(payload["hf_transformers_export"])
    expected = {
        "vocab_size": 256,
        "bos_token_id": None,
        "eos_token_id": None,
        "pad_token_id": None,
        "unk_token_id": None,
        "decoder_start_token_id": None,
        "forced_bos_token_id": None,
        "forced_eos_token_id": None,
        "added_tokens": [],
        "special_tokens_map": {},
        "transformers_architecture_compatibility": "NOT_CLAIMED_BY_TOKEN_CONTRACT",
    }
    if mapping != expected:
        raise BaseTokenContractError("HF/Transformers token mapping drift")
    return mapping


def assert_runtime_contract(
    tokenizer: ByteTokenizer | None = None,
    contract: Mapping[str, Any] | None = None,
) -> None:
    payload = (
        load_research_base_token_contract() if contract is None else dict(contract)
    )
    tok = ByteTokenizer() if tokenizer is None else tokenizer

    if tok.version != BYTE_TOKENIZER_VERSION:
        raise BaseTokenContractError("tokenizer version drift")
    if tok.identity.config_sha256 != BYTE_TOKENIZER_HASH:
        raise BaseTokenContractError("tokenizer config identity drift")
    if tok.identity.vocab_sha256 != BYTE_VOCAB_HASH:
        raise BaseTokenContractError("tokenizer vocabulary identity drift")
    if tok.vocab_size != 256 or tok.special_tokens:
        raise BaseTokenContractError("ordinary vocabulary or special-token semantics drift")
    if tok.bos_id is not None or tok.eos_id is not None or tok.pad_id is not None:
        raise BaseTokenContractError("Base contract forbids semantic BOS/EOS/PAD IDs")

    vocab = json.loads(canonical_vocab_json())
    entries = vocab["entries"]
    expected_entries = [
        {"id": token_id, "token_hex": f"{token_id:02x}"} for token_id in range(256)
    ]
    if entries != expected_entries:
        raise BaseTokenContractError("byte token IDs are not stable 0..255 identities")

    ordinary = payload["ordinary_vocabulary"]
    if ordinary["size"] != 256 or ordinary["id_range"] != [0, 255]:
        raise BaseTokenContractError("contract ordinary vocabulary drift")
    if payload["family"]["tokenizer_config_sha256"] != BYTE_TOKENIZER_HASH:
        raise BaseTokenContractError("contract/tokenizer config hash mismatch")
    if payload["family"]["tokenizer_vocab_sha256"] != BYTE_VOCAB_HASH:
        raise BaseTokenContractError("contract/tokenizer vocabulary hash mismatch")
    if payload["packing"]["version"] != PACKING_VERSION:
        raise BaseTokenContractError("contract packing version mismatch")
    if payload["packing"]["config_sha256"] != PACKING_CONFIG_HASH:
        raise BaseTokenContractError("contract packing hash mismatch")
    if payload["document_boundary"]["append_eos"]:
        raise BaseTokenContractError("EOS was not accepted for this contract")
    if payload["document_boundary"]["cross_document_packing"]:
        raise BaseTokenContractError("no-EOS Base contract requires document isolation")
    hf_transformers_token_mapping(payload)


def assert_checkpoint_compatible(
    manifest: Mapping[str, Any],
    *,
    require_training_resume_binding: bool = True,
) -> None:
    """Validate token/packing lineage before any caller mutates runtime state."""

    payload = load_research_base_token_contract()
    identity = manifest.get("identity")
    if not isinstance(identity, Mapping):
        raise CheckpointCompatibilityError("checkpoint identity is missing")

    required = payload["checkpoint_compatibility"]
    mismatches: dict[str, dict[str, Any]] = {}
    checks = {
        "tokenizer_hash": required["required_tokenizer_config_sha256"],
        "tokenizer_vocab_hash": required["required_tokenizer_vocab_sha256"],
    }
    for field, expected in checks.items():
        actual = identity.get(field)
        if actual != expected:
            mismatches[field] = {"expected": expected, "actual": actual}

    model_spec = identity.get("model_spec")
    actual_vocab = model_spec.get("vocab_size") if isinstance(model_spec, Mapping) else None
    if actual_vocab != required["required_model_vocab_size"]:
        mismatches["model_spec.vocab_size"] = {
            "expected": required["required_model_vocab_size"],
            "actual": actual_vocab,
        }

    if require_training_resume_binding:
        training_config = identity.get("training_config")
        data = training_config.get("data") if isinstance(training_config, Mapping) else None
        packing_version = data.get("packing_version") if isinstance(data, Mapping) else None
        if packing_version != required["required_packing_version_for_training_resume"]:
            mismatches["training_config.data.packing_version"] = {
                "expected": required["required_packing_version_for_training_resume"],
                "actual": packing_version,
            }

    if mismatches:
        raise CheckpointCompatibilityError(
            f"Base token contract checkpoint mismatch: {mismatches}"
        )


def deterministic_artifact_proof(path: str | Path | None = None) -> dict[str, Any]:
    payload_a = load_research_base_token_contract(path)
    payload_b = load_research_base_token_contract(path)
    bytes_a = canonical_json_bytes(payload_a)
    bytes_b = canonical_json_bytes(payload_b)
    if bytes_a != bytes_b:
        raise BaseTokenContractError("contract canonical serialization is not deterministic")
    return {
        "contract_id": CONTRACT_ID,
        "identity_sha256": payload_a["identity_sha256"],
        "canonical_full_sha256": hashlib.sha256(bytes_a).hexdigest(),
        "byte_identical_reloads": True,
    }
