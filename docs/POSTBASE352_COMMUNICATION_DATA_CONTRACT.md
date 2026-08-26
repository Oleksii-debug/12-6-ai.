# POSTBASE-352 communication data contract v1

Worker: `POSTBASE-352-COMMUNICATION-DATA-CONTRACT`

Status: contract and seed materialization only. No training is authorized or executed.

## Scope

This worker defines the first concrete user/assistant communication dataset contract for a future small post-Base assistant. It stacks on POSTBASE-253's immutable Base-consumption boundary and D09's generic post-training records rather than creating another training framework.

The committed 20-record seed is deliberately small. It proves the schema, provenance, split firewall, formatting and validation mechanics. It is not a claim that 20 examples are enough to train a useful assistant or that any behavioral capability has been learned.

## Canonical Base firewall

All communication material lives under `data/post_base/communication_v1/` and is classified `POSTBASE_COMMUNICATION_ONLY`.

The manifest hard-codes all of the following to false:

- `base_corpus_evidence`;
- `canonical_base_training_eligible`;
- `training_authorized`;
- `selection_for_training`;
- `final_for_training`;
- `final_for_selection`.

No file in this worker is admitted to canonical Base pretraining, Base evidence, tokenizer fitting or Base checkpoint construction. The worker modifies no Base dataset, ModelSpec, checkpoint, optimizer or trainer.

## Record contract

Each canonical UTF-8 JSONL row contains exactly:

- stable `record_id` and `family_id`;
- immutable split: `train`, `selection` or `final`;
- language and skill labels;
- an alternating dialogue that starts with `user`, ends with `assistant`, and permits only those two roles;
- provenance;
- quality-gate results.

Text must be Unicode NFC, LF-only, free of NUL and forbidden control characters. Every row carries a SHA-256 over the canonical message payload.

The post-training adapter emits ordinary `PROMPT_COMPLETION` records. Selection maps to D09 `VALIDATION`; final maps to D09 `TEST`. `for_training=True` fails closed for both.

## Seed split

The v1 seed has:

- train: 12 dialogues;
- selection: 4 dialogues;
- final: 4 dialogues.

English and Ukrainian occur in every split. Train covers direct answers, transformation, summarization, structured response, clarification, exact reasoning and multi-turn context carryover.

`family_id` may occur in exactly one split. Normalized exact duplicates are rejected. Cross-split near duplicates are rejected at the frozen character-5-gram Jaccard threshold of `0.85`. This is a leakage gate, not a universal semantic-contamination detector; a larger future corpus should add stronger scalable decontamination while preserving the same fail-closed split law.

Selection is for post-Base model/checkpoint/configuration choice only. Final is reserved for terminal post-Base reporting and is prohibited from both training and selection.

## Provenance and foreign-model gate

Every committed seed row is project-authored with:

- `source_id = project:postbase352-manual-v1`;
- `rights = project_owned`;
- `foreign_model_output = false`;
- `synthetic_authority_id = null`.

The manifest therefore binds `foreign_model_records = 0` and `synthetic_data_authority = null`.

A future foreign-model or teacher-generated row is rejected unless a separate, explicit later `SyntheticDataAuthority` is supplied. That authority must be owner-approved for `post_base_communication_data`, carry its own SHA-256 identity, name the allowed source IDs, and exactly match each admitted foreign row's authority ID. This class is only a fail-closed interface for future authorization; POSTBASE-352 itself creates no synthetic-data authority and calls no external model.

## Quality gates

Every admitted row must pass all of these checks:

- answer independently marked verified;
- relevance review;
- language review;
- PII review;
- secret review;
- copyright review;
- no hidden chain-of-thought or private reasoning stored as a target;
- canonical JSON and content-hash verification;
- split-family, exact-duplicate and near-duplicate isolation;
- required language and train-skill coverage;
- tokenizer/context representability.

The initial examples are short on purpose: each formatted supervised example must be at most 256 UTF-8 bytes under v1.

## Tokenizer compatibility

The logical v1 profile is the existing `s0-byte-v1` tokenizer family with 256 byte values. Dialogue formatting uses ordinary text prefixes (`User:` and `Assistant:`), adds no special tokens and installs no chat template into canonical Base.

This worker intentionally does not invent canonical Base tokenizer config or vocabulary hashes. Before any future training, the dataset's logical profile must be paired with the exact retained Base `TokenizerCompatibility` from POSTBASE-253 and `require_exact_base_tokenizer()` must pass. A different tokenizer ID, vocabulary size, config SHA-256 or vocabulary SHA-256 fails closed.

## Immutable seed identities

- train JSONL SHA-256: `ddafe61ce3255dd30d207ec1ee811efa59a2da37288368a9bbc3fa0602cb2ba7`;
- selection JSONL SHA-256: `e36f7c560c44fd2812935b5382dd628fabbd7af0e79c080c295d1332de13309f`;
- final JSONL SHA-256: `f50262994089f276fb7d3f4c644180d0854273b51d7f7920aca1d2048031d039`;
- manifest SHA-256: `51a927c40b4274f8b8f992b8dd83b4dbddac1e925a45834832a79ee6be18d3d6`.

The manifest binds the three split hashes and record counts; any byte drift fails validation.

## Verification and truth boundary

Focused contract validation before publication used a contract-compatible local stub of the already published POSTBASE-253/D09 interfaces: 6 tests passed. Repository CI remains the authority for the actual stacked branch after publication.

No optimizer update, gradient, SFT, preference optimization, RL, checkpoint mutation, Base-data admission, foreign model inference, network data generation or paid compute occurs in this worker.

Execution profile: `LOCAL_FREE`.
