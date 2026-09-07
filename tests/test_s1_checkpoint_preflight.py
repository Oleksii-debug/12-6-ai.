from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from twelve_six.checkpoint import hash_json
from twelve_six.checkpoint.s1_preflight import (
    AUTHORITY,
    FIXTURE_SCOPE,
    REPOSITORY,
    SCHEMA,
    collect_s1_checkpoint_preflight,
    validate_s1_checkpoint_preflight,
)

ROOT = Path(__file__).resolve().parents[1]

_FRESH_PROCESS_WORKER = textwrap.dedent(
    r"""
    import hashlib
    import json
    import os
    import random
    import sys
    from collections.abc import Mapping
    from dataclasses import asdict, is_dataclass
    from pathlib import Path

    import torch

    from twelve_six.checkpoint import hash_json, load_trainer_checkpoint, sha256_file
    from twelve_six.checkpoint.s1_preflight import _batch_for_step, _load_texts, _train_range
    from twelve_six.model import TwelveSixDecoder, load_stage_config
    from twelve_six.packing import PACKING_CONFIG_HASH, PACKING_VERSION
    from twelve_six.tokenization import ByteTokenizer
    from twelve_six.training import Trainer, TrainerConfig


    def _digest(value):
        hasher = hashlib.sha256()

        def visit(item):
            if is_dataclass(item) and not isinstance(item, type):
                visit(asdict(item))
                return
            if isinstance(item, torch.Tensor):
                tensor = item.detach().cpu().contiguous()
                hasher.update(b"tensor:")
                hasher.update(str(tensor.dtype).encode("ascii"))
                hasher.update(json.dumps(list(tensor.shape)).encode("ascii"))
                hasher.update(tensor.numpy().tobytes())
                return
            if isinstance(item, Mapping):
                hasher.update(b"mapping{")
                for key in sorted(item, key=repr):
                    visit(key)
                    visit(item[key])
                hasher.update(b"}")
                return
            if isinstance(item, tuple):
                hasher.update(b"tuple[")
                for value in item:
                    visit(value)
                hasher.update(b"]")
                return
            if isinstance(item, list):
                hasher.update(b"list[")
                for value in item:
                    visit(value)
                hasher.update(b"]")
                return
            if item is None or isinstance(item, (bool, int, float, str)):
                hasher.update(type(item).__name__.encode("ascii"))
                hasher.update(json.dumps(item, sort_keys=True).encode("utf-8"))
                return
            raise TypeError(f"unsupported digest value: {type(item)!r}")

        visit(value)
        return hasher.hexdigest()


    mode = sys.argv[1]
    root = Path(sys.argv[2]).resolve()
    checkpoint = Path(sys.argv[3]).resolve()
    candidate_sha = sys.argv[4]
    total_steps = int(sys.argv[5])
    split_step = int(sys.argv[6])
    seed = int(sys.argv[7])

    stage = load_stage_config(root / "configs/stages/s1_100k.json")
    tokenizer = ByteTokenizer()
    train_path = root / "data/s0/packaged/train.jsonl"
    dataset_manifest_path = root / "data/s0/packaged/manifest.json"
    environment_lock_path = root / "requirements/locks/index.json"
    texts = _load_texts(train_path)
    train_sha256 = sha256_file(train_path)
    dataset_sha256 = sha256_file(dataset_manifest_path)
    environment_sha256 = sha256_file(environment_lock_path)
    trainer_config = TrainerConfig(
        learning_rate=1e-3,
        weight_decay=0.0,
        max_steps=total_steps,
        scheduler="constant",
        gradient_accumulation_steps=1,
        gradient_clip_norm=1.0,
        precision="fp32",
        seed=seed,
        deterministic_algorithms=True,
    )

    random.seed(seed)
    torch.manual_seed(seed)
    model = TwelveSixDecoder(stage.model, stage.init)
    trainer = Trainer(model, trainer_config, device="cpu")

    if mode == "baseline":
        start_step = 0
    elif mode == "resume":
        manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
        identity = manifest["identity"]
        random.seed(seed + 991)
        torch.manual_seed(seed + 991)
        load_trainer_checkpoint(
            checkpoint,
            model=model,
            trainer=trainer,
            restore_rng=True,
            expected_git_sha=candidate_sha,
            expected_model_spec_hash=hash_json(stage.model.to_dict()),
            expected_init_spec_hash=hash_json(stage.init.to_dict()),
            expected_tokenizer_hash=tokenizer.identity.config_sha256,
            expected_tokenizer_vocab_hash=tokenizer.identity.vocab_sha256,
            expected_dataset_manifest_hash=dataset_sha256,
            expected_split_identity=f"controlled-s0-train:{train_sha256}",
            expected_packing_hash=PACKING_CONFIG_HASH,
            expected_packing_version=PACKING_VERSION,
            expected_run_manifest_hash=identity["run_manifest_hash"],
            expected_training_config_hash=identity["training_config_hash"],
            expected_environment_lock_hash=environment_sha256,
            expected_seed=seed,
            expected_step=split_step,
            expected_tokens_seen=identity["tokens_seen"],
        )
        if trainer.optimizer_step != split_step:
            raise RuntimeError("fresh process restored the wrong optimizer step")
        start_step = split_step
    else:
        raise ValueError(f"unknown probe mode: {mode}")

    next_batch = _batch_for_step(
        texts,
        tokenizer,
        step=split_step,
        max_seq_len=stage.model.max_seq_len,
    )
    next_exposure_sha256 = hashlib.sha256(
        next_batch["input_ids"].contiguous().numpy().tobytes()
    ).hexdigest()
    _train_range(
        trainer,
        texts,
        tokenizer,
        start_step=start_step,
        end_step=total_steps,
        max_seq_len=stage.model.max_seq_len,
    )
    payload = {
        "mode": mode,
        "pid": os.getpid(),
        "optimizer_step": trainer.optimizer_step,
        "tokens_seen": trainer.tokens_seen,
        "next_exposure_step": split_step,
        "next_exposure_sha256": next_exposure_sha256,
        "model_sha256": _digest(model.state_dict()),
        "trainer_sha256": _digest(trainer.state_dict()),
        "python_rng_sha256": _digest(random.getstate()),
        "torch_rng_sha256": _digest(torch.get_rng_state()),
    }
    print(json.dumps(payload, sort_keys=True))
    """
)


def _head() -> str:
    value = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    assert len(value) in {40, 64}
    return value


def _rehash(payload: dict[str, Any]) -> None:
    material = dict(payload)
    material.pop("evidence_sha256", None)
    payload["evidence_sha256"] = hash_json(material)


def _valid_minimal() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "repository": REPOSITORY,
        "candidate_sha": "a" * 40,
        "s1_architecture_status": "ENGINEERING_CANDIDATE_NOT_FROZEN",
        "s1_tokenizer_selected": False,
        "s1_data_selected": False,
        "fixture_scope": FIXTURE_SCOPE,
        "canonical_binding": {
            "accepted": False,
            "rejected_as_expected": True,
            "reason": "ModelSpec/tokenizer vocab mismatch: model=512, tokenizer=256",
        },
        "checkpoint": {"save_verified": True, "pickle": False},
        "resume": {"model_state_exact": True, "trainer_state_exact": True},
        "constraints": {
            "paid_compute": False,
            "promotion_claimed": False,
            "s1_quality_claimed": False,
        },
    }
    _rehash(payload)
    return payload


def _run_fresh_process_probe(
    *,
    mode: str,
    checkpoint: Path,
    candidate_sha: str,
    total_steps: int,
    split_step: int,
    seed: int,
) -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _FRESH_PROCESS_WORKER,
            mode,
            str(ROOT),
            str(checkpoint),
            candidate_sha,
            str(total_steps),
            str(split_step),
            str(seed),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_real_s1_checkpoint_preflight_is_exact_but_noncanonical(tmp_path: Path) -> None:
    head = _head()
    evidence = collect_s1_checkpoint_preflight(
        ROOT,
        head,
        tmp_path / "s1-checkpoint-preflight",
        total_steps=2,
        split_step=1,
        seed=20260825,
    )

    assert evidence["candidate_sha"] == head
    assert evidence["authority"] == AUTHORITY
    assert evidence["model"]["parameter_count"] == 107_856
    assert evidence["model"]["model_vocab_size"] == 512
    assert evidence["fixture"]["tokenizer_vocab_size"] == 256
    assert evidence["s1_tokenizer_selected"] is False
    assert evidence["s1_data_selected"] is False
    assert evidence["canonical_binding"]["accepted"] is False
    assert evidence["canonical_binding"]["rejected_as_expected"] is True
    assert "ModelSpec/tokenizer vocab mismatch" in evidence["canonical_binding"]["reason"]
    assert evidence["checkpoint"]["save_verified"] is True
    assert evidence["checkpoint"]["pickle"] is False
    assert evidence["checkpoint"]["format"] == "12-6-checkpoint"
    assert evidence["checkpoint"]["format_version"] == 1
    assert evidence["resume"]["model_state_exact"] is True
    assert evidence["resume"]["trainer_state_exact"] is True
    assert evidence["resume"]["baseline_tokens_seen"] == evidence["resume"]["resumed_tokens_seen"]
    assert (tmp_path / "s1-checkpoint-preflight/checkpoint/manifest.json").is_file()
    assert (tmp_path / "s1-checkpoint-preflight/s1-checkpoint-preflight.json").is_file()
    validate_s1_checkpoint_preflight(evidence, expected_candidate_sha=head)


def test_fresh_process_resume_matches_uninterrupted_state_and_next_exposure(
    tmp_path: Path,
) -> None:
    head = _head()
    total_steps = 3
    split_step = 1
    seed = 20260825
    output = tmp_path / "fresh-process-proof"
    collect_s1_checkpoint_preflight(
        ROOT,
        head,
        output,
        total_steps=total_steps,
        split_step=split_step,
        seed=seed,
    )
    checkpoint = output / "checkpoint"

    baseline = _run_fresh_process_probe(
        mode="baseline",
        checkpoint=checkpoint,
        candidate_sha=head,
        total_steps=total_steps,
        split_step=split_step,
        seed=seed,
    )
    resumed = _run_fresh_process_probe(
        mode="resume",
        checkpoint=checkpoint,
        candidate_sha=head,
        total_steps=total_steps,
        split_step=split_step,
        seed=seed,
    )

    assert baseline["pid"] != os.getpid()
    assert resumed["pid"] != os.getpid()
    assert baseline["optimizer_step"] == resumed["optimizer_step"] == total_steps
    assert baseline["tokens_seen"] == resumed["tokens_seen"]
    assert baseline["next_exposure_step"] == resumed["next_exposure_step"] == split_step
    assert baseline["next_exposure_sha256"] == resumed["next_exposure_sha256"]
    assert baseline["model_sha256"] == resumed["model_sha256"]
    assert baseline["trainer_sha256"] == resumed["trainer_sha256"]
    assert baseline["python_rng_sha256"] == resumed["python_rng_sha256"]
    assert baseline["torch_rng_sha256"] == resumed["torch_rng_sha256"]


def test_validator_rejects_premature_s1_tokenizer_claim() -> None:
    payload = copy.deepcopy(_valid_minimal())
    payload["s1_tokenizer_selected"] = True
    _rehash(payload)

    with pytest.raises(ValueError, match="must not select an S1 tokenizer"):
        validate_s1_checkpoint_preflight(payload, expected_candidate_sha="a" * 40)


def test_validator_rejects_quality_or_promotion_overclaim() -> None:
    payload = copy.deepcopy(_valid_minimal())
    payload["constraints"]["s1_quality_claimed"] = True
    _rehash(payload)
    with pytest.raises(ValueError, match="cannot claim S1 quality"):
        validate_s1_checkpoint_preflight(payload, expected_candidate_sha="a" * 40)

    payload = copy.deepcopy(_valid_minimal())
    payload["constraints"]["promotion_claimed"] = True
    _rehash(payload)
    with pytest.raises(ValueError, match="cannot grant promotion"):
        validate_s1_checkpoint_preflight(payload, expected_candidate_sha="a" * 40)


def test_validator_rejects_stale_candidate_and_tamper() -> None:
    payload = _valid_minimal()
    with pytest.raises(ValueError, match="candidate SHA is stale"):
        validate_s1_checkpoint_preflight(payload, expected_candidate_sha="0" * 40)

    payload = copy.deepcopy(payload)
    payload["resume"]["model_state_exact"] = False
    with pytest.raises(ValueError, match="interrupted/resumed preflight is not exact"):
        validate_s1_checkpoint_preflight(payload, expected_candidate_sha="a" * 40)
