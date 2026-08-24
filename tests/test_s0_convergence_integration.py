from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from twelve_six.data import build_dataset
from twelve_six.inference import GenerationConfig, generate
from twelve_six.integration import (
    CIEvidence,
    CandidateStatus,
    ComponentDisposition,
    ComponentRef,
    S0TorchInferenceBackend,
    StageCandidateManifest,
)
from twelve_six.model import TwelveSixDecoder, load_stage_config
from twelve_six.tokenization import BYTE_TOKENIZER_HASH, ByteTokenizer
from twelve_six.training import Trainer, TrainerConfig

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = ROOT / "configs/releases/s0_convergence_20260824.experimental.json"
DATASET_MANIFEST_SHA256 = "b085a7ab56510575a11a80824fcff3a95a17f237d46d1be820e59d1289f220c2"
DATASET_IDENTITY_SHA256 = "bab60119d49e93303c972b77900fcb5553817f754cbc5d9a58019228cfa0ca89"


def _load_first_jsonl(path: Path) -> dict[str, object]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            return json.loads(line)
    raise AssertionError(f"no records in {path}")


def test_s0_accepted_contracts_execute_model_data_tokenizer_train_and_inference(tmp_path: Path) -> None:
    stage = load_stage_config(ROOT / "configs/stages/s0_10k.json")
    tokenizer = ByteTokenizer()

    assert stage.canonical_base == "random_init"
    assert stage.expected_parameters == 10_140
    assert stage.model.vocab_size == tokenizer.vocab_size == 256
    assert tokenizer.identity.config_sha256 == BYTE_TOKENIZER_HASH

    rebuilt_dir = tmp_path / "rebuilt-s0"
    manifest = build_dataset(
        ROOT / "data/s0/source_registry.json",
        ROOT / "data/s0/contamination_registry.json",
        rebuilt_dir,
    )
    assert manifest["dataset_identity_sha256"] == DATASET_IDENTITY_SHA256
    committed_manifest_bytes = (ROOT / "data/s0/packaged/manifest.json").read_bytes()
    assert __import__("hashlib").sha256(committed_manifest_bytes).hexdigest() == DATASET_MANIFEST_SHA256
    assert (rebuilt_dir / "manifest.json").read_bytes() == committed_manifest_bytes

    record = _load_first_jsonl(rebuilt_dir / "train.jsonl")
    text = str(record["text"])
    token_ids = tokenizer.encode(text)[: min(stage.model.max_seq_len, 64)]
    assert len(token_ids) >= 2

    torch.manual_seed(20260824)
    model = TwelveSixDecoder(stage.model, stage.init)
    before = model.token_embedding.weight.detach().clone()
    batch_ids = torch.tensor([token_ids], dtype=torch.long)
    trainer = Trainer(
        model,
        TrainerConfig(
            learning_rate=1e-2,
            max_steps=1,
            seed=20260824,
            precision="fp32",
            deterministic_algorithms=True,
        ),
        device="cpu",
    )
    metrics = trainer.train_microbatch({"input_ids": batch_ids, "labels": batch_ids})

    assert metrics.optimizer_stepped is True
    assert math.isfinite(metrics.loss)
    assert metrics.optimizer_step == 1
    assert not torch.equal(before, model.token_embedding.weight.detach())

    backend = S0TorchInferenceBackend(model, tokenizer)
    result = generate(
        backend,
        "12-6",
        GenerationConfig(max_new_tokens=2, sample=False, seed=20260824),
    )
    assert len(result.generated_token_ids) == 2
    assert all(0 <= token_id < tokenizer.vocab_size for token_id in result.generated_token_ids)
    assert result.stop_reason == "max_new_tokens"


def test_s0_evidence_fails_closed_while_checkpoint_and_eval_heads_are_red() -> None:
    payload = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    components = []
    for row in payload["components"]:
        ci_evidence = CIEvidence(
            run_id=row["ci_run_id"],
            head_sha=row["source_sha"],
            conclusion=row["ci_conclusion"],
            evidence_ref=f"github-actions:{row['ci_run_id']}",
        )
        components.append(
            ComponentRef(
                lane=row["lane"],
                source_sha=row["source_sha"],
                disposition=ComponentDisposition(row["disposition"]),
                component_kind=row["component_kind"],
                pr_number=row["pr_number"],
                ci_evidence=ci_evidence,
                contains_behavioral_weights=row.get("contains_behavioral_weights"),
                contains_foreign_pretrained_weights=row.get(
                    "contains_foreign_pretrained_weights"
                ),
                notes=row.get("hold_reason", ""),
            )
        )

    convergence = StageCandidateManifest.compose(
        stage=payload["stage"],
        integration_anchor_sha=payload["integration_anchor_sha"],
        status=CandidateStatus(payload["status"]),
        base_lineage=payload["base_lineage"],
        components=components,
    )

    assert convergence.accepted_lanes() == frozenset(
        {"D01", "D02", "D03", "D04", "D07", "D08"}
    )
    assert convergence.missing_required_lanes() == ("D05", "D06")
    assert convergence.ready_for_candidate() is False
    assert payload["audits"] == {
        "AUDIT-A": "CHANGES_REQUIRED",
        "AUDIT-B": "CHANGES_REQUIRED",
    }
