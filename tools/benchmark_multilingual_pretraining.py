"""LOCAL_FREE multilingual tokenizer, mixture, packing, and S2 mechanics probe."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from twelve_six.data.external_sources import (
    PROJECT_RIGHTS_POLICY_REF,
    RIGHTS_APPROVED,
    USE_ALLOWED,
    USE_DENIED,
    EligibilityResolver,
    ExternalSourceSpec,
    RightsDecision,
    RightsEvidenceRef,
    SnapshotSpec,
    UsePermissions,
    build_external_source_registry,
)
from twelve_six.data.multilingual_pretraining import (
    PretrainingRecord,
    admit_for_pretraining,
    build_token_budget_mixture,
    corpus_requirements,
    default_token_budget_strata,
    replay_schedule,
    tokenizer_cost,
)
from twelve_six.model import ModelSpec, TwelveSixDecoder, count_trainable_parameters
from twelve_six.packing import PACKING_CONFIG_HASH, TextRecord, iter_packed_examples
from twelve_six.tokenization import BYTE_TOKENIZER_HASH, BYTE_VOCAB_HASH, ByteTokenizer
from twelve_six.tokenization.experiments import (
    CorpusFileIdentity,
    TokenizerProbe,
    TokenizerTrainingManifest,
    measure_probe,
    summarize_by_language,
    train_hf_tokenizer,
)

TOKENIZERS_VERSION = "0.23.1"
TRAIN = (
    ("uk-1", "Українська мова має відмінки, дієвідмінювання і словотвір. Ці дані потрібні для базового передтренування моделі.", "uk", "natural"),
    ("uk-2", "Дослідники працюють із текстами різних жанрів, щоб модель бачила слова у називному, родовому, давальному, знахідному та орудному відмінках.", "uk", "natural"),
    ("uk-3", "Київ, Львів і Ужгород мають різні мовні контексти; ґрунтовний корпус повинен містити літери ґ, ї, є, і та природні апострофи.", "uk", "natural"),
    ("en-1", "The training corpus contains English prose with varied syntax and vocabulary so the base model learns next-token statistics rather than instructions.", "en", "natural"),
    ("en-2", "These records test deterministic data selection, source provenance, deduplication, and restart behavior for a universal language model.", "en", "natural"),
    ("en-3", "Data quality includes valid encoding, stable normalization, explicit source rights, and strict separation from held-out evaluation material.", "en", "natural"),
    ("code-1", "def stable_hash(value: str) -> str:\n    return hashlib.sha256(value.encode('utf-8')).hexdigest()\n", None, "code"),
    ("code-2", "class Counter:\n    def __init__(self):\n        self.value = 0\n    def increment(self):\n        self.value += 1\n        return self.value\n", None, "code"),
    ("code-3", "SELECT source_id, COUNT(*) FROM records\nWHERE split = 'train'\nGROUP BY source_id ORDER BY source_id;\n", None, "code"),
)
PROBES = (
    TokenizerProbe("uk-cases", "uk", "morphology-heldout", "книга книги книзі книгу книгою; учень учня учневі учнем"),
    TokenizerProbe("uk-verbs", "uk", "morphology-heldout", "працювати працюю працюєш працює працюємо працюють; прочитати прочитають"),
    TokenizerProbe("uk-orthography", "uk", "orthography-heldout", "п'ять, об'єкт, м'який, під'їзд, ґанок, їжак, Європа, Україна"),
    TokenizerProbe("en", "en", "language-heldout", "The multilingual base model compares token fertility on unseen English."),
    TokenizerProbe("code", "code", "code-heldout", "for index, item in enumerate(records):\n    assert item.split == 'train'\n"),
    TokenizerProbe("unicode", "multi", "unicode-heldout", "Україна — Kyiv — naïve café — λ = 3.14 — 😀"),
)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _mechanics_source(modality: str) -> ExternalSourceSpec:
    source_id = f"data10::{modality}"
    source_version = "v1"
    fixture = Path("data/synthetic/data10/uk-en-code-train.txt")
    payload = fixture.read_bytes()
    evidence = (
        RightsEvidenceRef(
            evidence_id=f"{modality}-project-authorship-fixture",
            evidence_kind="project_authorship",
            uri=f"file:///mechanics/data10/{modality}/project-authorship.txt",
            sha256=sha(f"DATA10 mechanics project-authorship fixture:{modality}"),
            captured_at="2026-08-25T12:00:00Z",
            source_id=source_id,
            source_version=source_version,
        ),
        RightsEvidenceRef(
            evidence_id=f"{modality}-policy-decision-fixture",
            evidence_kind="policy_decision",
            uri=f"file:///mechanics/data10/{modality}/policy-decision.json",
            sha256=sha(f"DATA10 mechanics policy fixture:{modality}"),
            captured_at="2026-08-25T12:00:00Z",
            source_id=source_id,
            source_version=source_version,
        ),
    )
    return ExternalSourceSpec(
        source_id=source_id,
        source_version=source_version,
        provider="12-6-project-mechanics-fixture",
        source_url=f"https://example.invalid/data10-mechanics/{modality}",
        source_kind="text" if modality == "natural" else "code",
        purpose="pretraining",
        synthetic=True,
        benchmark_material=False,
        held_out=False,
        snapshot=SnapshotSpec(
            uri="file:///data/synthetic/data10/uk-en-code-train.txt",
            sha256=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
            retrieved_at="2026-08-25T12:00:00Z",
            upstream_version=source_version,
            retrieval_method="repository_fixture",
        ),
        rights=RightsDecision(
            status=RIGHTS_APPROVED,
            license_id="PROJECT-AUTHORED-MECHANICS-FIXTURE",
            terms_url="https://example.invalid/data10-mechanics/terms",
            allows_model_training=True,
            allows_derivatives=True,
            allows_redistribution=False,
            policy_ref=PROJECT_RIGHTS_POLICY_REF,
            reviewed_at="2026-08-25T12:00:00Z",
            reviewer_ref="role://data10-mechanics-test-fixture",
            uses=UsePermissions(USE_ALLOWED, USE_ALLOWED, USE_ALLOWED, USE_ALLOWED, USE_DENIED),
            evidence_refs=evidence,
        ),
    )


def admitted_train() -> tuple:
    sources = {name: _mechanics_source(name) for name in ("natural", "code")}
    resolver = EligibilityResolver(build_external_source_registry(sources.values()))
    records = tuple(
        admit_for_pretraining(
            PretrainingRecord(
                record_id=record_id,
                source_id=sources[modality].source_id,
                source_version=sources[modality].source_version,
                source_manifest_sha256=sources[modality].source_manifest_sha256,
                split="train",
                source_purpose="pretraining",
                modality=modality,
                text=text,
                language_hint=hint,
                project_authored_synthetic=True,
            ),
            eligibility_resolver=resolver,
        )
        for record_id, text, hint, modality in TRAIN
    )
    code = next(record for record in records if record.record_id == "code-1")
    if "\n    return" not in code.normalized_text:
        raise RuntimeError("code normalization corrupted indentation")
    return records


def training_manifest(algorithm: str, texts: tuple[str, ...]) -> TokenizerTrainingManifest:
    joined = "\n".join(texts) + "\n"
    snapshot = Path("data/synthetic/data10/uk-en-code-train.txt")
    if snapshot.read_text(encoding="utf-8") != joined:
        raise RuntimeError("manifested synthetic tokenizer corpus drifted")
    corpus_sha = sha(joined)
    dataset_sha = sha(json.dumps({"records": [row[0] for row in TRAIN], "corpus_sha256": corpus_sha}, sort_keys=True, separators=(",", ":")))
    return TokenizerTrainingManifest(
        experiment_id=f"data10-{algorithm}-512-v1",
        algorithm=algorithm,
        tokenizers_version=TOKENIZERS_VERSION,
        dataset_id="data10-project-authored-uk-en-code-v1",
        dataset_manifest_sha256=dataset_sha,
        corpus_files=(CorpusFileIdentity(snapshot.as_posix(), corpus_sha, len(joined.encode())),),
        vocab_size=512,
        min_frequency=2 if algorithm == "bpe" else None,
    )


def probe_metrics(tokenizer, unknown_token_id: int | None = None) -> dict[str, object]:
    results = tuple(measure_probe(tokenizer, probe, unknown_token_id=unknown_token_id) for probe in PROBES)
    return {
        "summary": summarize_by_language(results),
        "probes": [asdict(result) for result in results],
        "tokens": sum(result.tokens for result in results),
        "utf8_bytes": sum(result.utf8_bytes for result in results),
        "strict_roundtrip_all": all(result.round_trip_exact for result in results),
        "unknown_tokens": sum(result.unknown_tokens for result in results),
    }


def tokenizer_comparison(records: tuple) -> dict[str, object]:
    texts = tuple(record.normalized_text for record in records)
    byte = probe_metrics(ByteTokenizer())
    algorithms: dict[str, object] = {}
    for algorithm in ("bpe", "unigram"):
        manifest = training_manifest(algorithm, texts)
        first, second = train_hf_tokenizer(manifest, texts), train_hf_tokenizer(manifest, texts)
        metrics = probe_metrics(first, first.unk_id)
        byte_tokens, observed = int(byte["tokens"]), int(metrics["tokens"])
        costs = {
            str(d_model): asdict(tokenizer_cost(
                name=algorithm,
                vocab_size=first.vocab_size,
                observed_tokens=observed,
                byte_baseline_tokens=byte_tokens,
                d_model=d_model,
            ))
            for d_model in (128, 320, 768)
        }
        algorithms[algorithm] = {
            "requested_vocab_size": 512,
            "actual_vocab_size": first.vocab_size,
            "training_manifest_sha256": manifest.sha256,
            "artifact_config_sha256": first.artifact_identity.config_sha256,
            "repeatable_artifact_identity": first.artifact_identity.config_sha256 == second.artifact_identity.config_sha256,
            "heldout": metrics,
            "repeat_heldout": probe_metrics(second, second.unk_id),
            "token_reduction_vs_bytes": 1.0 - observed / byte_tokens,
            "tied_vocab_parameter_cost": costs,
        }
    return {"same_train_corpus_for_algorithms": True, "validation_or_probe_used_for_training": False, "byte": byte, "algorithms": algorithms}


def mixture_probe(records: tuple) -> dict[str, object]:
    manifests = {
        name: sha(json.dumps(sorted(record.record_id for record in records if record.language == name), separators=(",", ":")))
        for name in ("uk", "en", "code")
    }
    plan = build_token_budget_mixture(
        default_token_budget_strata(manifests),
        tokenizer_config_sha256=BYTE_TOKENIZER_HASH,
        tokenizer_vocab_sha256=BYTE_VOCAB_HASH,
        packing_config_sha256=PACKING_CONFIG_HASH,
        seed=126,
        num_shards=32,
    )
    full_counts, full_cursor = replay_schedule(plan, samples=10_000)
    left_counts, cursor = replay_schedule(plan, samples=4_321)
    right_counts, resumed = replay_schedule(plan, samples=5_679, cursor=cursor)
    if left_counts + right_counts != full_counts or resumed != full_cursor:
        raise RuntimeError("restart replay diverged")
    return {
        "plan_sha256": plan.sha256,
        "sample_count": 10_000,
        "source_counts": dict(sorted(full_counts.items())),
        "restart_replay_exact": True,
        "restart_cursor_sha256": resumed.sha256,
    }


def s2_probe(records: tuple) -> dict[str, object]:
    packed = list(iter_packed_examples(tuple(TextRecord(r.record_id, r.normalized_text, "train") for r in records), ByteTokenizer(), expected_split="train"))
    stage = json.loads(Path("configs/stages/s2_1m.json").read_text(encoding="utf-8"))
    spec = ModelSpec.from_dict(stage["model"])
    torch.manual_seed(126)
    model = TwelveSixDecoder(spec)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    example = packed[0]
    input_ids = torch.tensor([example.input_ids], dtype=torch.long)
    labels = torch.tensor([example.labels], dtype=torch.long)
    before = model.token_embedding.weight.detach().clone()
    logits = model(input_ids).logits
    loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, spec.vocab_size), labels[:, 1:].reshape(-1), ignore_index=-100)
    loss.backward()
    optimizer.step()
    changed = not torch.equal(before, model.token_embedding.weight.detach())
    if not torch.isfinite(loss) or not changed:
        raise RuntimeError("S2 forward/backward/update probe failed")
    return {
        "stage": stage["stage"], "parameters": count_trainable_parameters(model),
        "vocab_size": spec.vocab_size, "packed_sequences": len(packed),
        "loss": float(loss.detach()), "optimizer_step_changed_embedding": changed,
        "foreign_pretrained_weights_used": False, "paid_compute_used": False,
    }


def run(source_sha: str) -> dict[str, object]:
    records = admitted_train()
    payload: dict[str, object] = {
        "schema": "12-6.data10-multilingual-evidence.v1",
        "source_sha": source_sha,
        "authority": "PROJECT_AUTHORED_SYNTHETIC_MECHANICS_NOT_CANONICAL_TRAINING_CORPUS",
        "base_pretraining_only": True,
        "instruction_tuning": False,
        "external_sources_training_approved": 0,
        "canonical_registry_training_sources": 0,
        "mechanics_fixture_registry_only": True,
        "canonical_s0_tokenizer_unchanged": True,
        "admitted_records": len(records),
        "language_counts": {name: sum(record.language == name for record in records) for name in ("uk", "en", "code")},
        "tokenizers": tokenizer_comparison(records),
        "mixture": mixture_probe(records),
        "s2_probe": s2_probe(records),
        "corpus_requirements": corpus_requirements(),
        "decision": {"tokenizer_frozen": False, "reason": "representative rights-approved multilingual corpus evidence is required"},
    }
    payload["evidence_sha256"] = sha(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    evidence = run(args.source_sha)
    args.output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"evidence_sha256": evidence["evidence_sha256"], "language_counts": evidence["language_counts"], "mixture": evidence["mixture"], "s2_probe": evidence["s2_probe"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
