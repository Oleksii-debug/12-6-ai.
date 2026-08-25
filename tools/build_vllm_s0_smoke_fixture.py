from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from twelve_six.checkpoint import CheckpointIdentity, export_hf_directory, save_checkpoint
from twelve_six.inference.transformers_llama import llama_config_dict
from twelve_six.inference.vllm_native_llama import (
    materialize_vllm_llama_directory,
    verify_vllm_llama_directory,
)
from twelve_six.model import TwelveSixDecoder, load_stage_config
from twelve_six.tokenization import ByteTokenizer

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260825


def build_fixture(output_root: Path) -> dict[str, object]:
    if output_root.exists():
        raise FileExistsError(f"fixture output already exists: {output_root}")
    output_root.mkdir(parents=True)

    stage = load_stage_config(ROOT / "configs/stages/s0_10k.json")
    tokenizer = ByteTokenizer()
    torch.manual_seed(SEED)
    model = TwelveSixDecoder(stage.model, stage.init).eval()

    checkpoint = output_root / "checkpoint"
    identity = CheckpointIdentity(
        git_sha="a" * 40,
        model_spec=stage.model.to_dict(),
        parameter_count=stage.expected_parameters,
        tokenizer_hash=tokenizer.identity.config_sha256,
        tokenizer_vocab_hash=tokenizer.identity.vocab_sha256,
        dataset_manifest_hash="b" * 64,
        run_manifest_hash="c" * 64,
        training_config={
            "data": {"tokenizer_version": tokenizer.identity.version},
            "training": {"context_length": stage.model.max_seq_len},
        },
        seed=SEED,
        precision="fp32",
        step=0,
        tokens_seen=0,
        optimizer={"name": "none"},
        scheduler=None,
    )
    checkpoint_manifest = save_checkpoint(checkpoint, model=model, identity=identity)

    export = export_hf_directory(
        checkpoint,
        output_root / "hf-export",
        hf_config=llama_config_dict(stage.model),
    )
    model_dir = materialize_vllm_llama_directory(export, output_root / "vllm-model")
    provenance = verify_vllm_llama_directory(model_dir)

    evidence = {
        "schema": "12-6.vllm-s0-smoke-fixture.v1",
        "seed": SEED,
        "checkpoint_id": checkpoint_manifest["checkpoint_id"],
        "model_spec_sha256": stage.model.identity_sha256(),
        "parameter_count": stage.expected_parameters,
        "tokenizer_config_sha256": tokenizer.identity.config_sha256,
        "tokenizer_vocab_sha256": tokenizer.identity.vocab_sha256,
        "checkpoint": str(checkpoint),
        "hf_export": str(export),
        "vllm_model": str(model_dir),
        "vllm_materialization": provenance,
        "pretrained_weights_used": False,
    }
    (output_root / "fixture.json").write_text(
        json.dumps(evidence, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a canonical random-init S0 export for the vLLM CPU import/config probe."
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    evidence = build_fixture(args.output_root)
    print(json.dumps(evidence, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
