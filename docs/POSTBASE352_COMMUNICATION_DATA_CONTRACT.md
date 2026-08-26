# POSTBASE-352 communication data contract v1

Workers: `POSTBASE-352-COMMUNICATION-DATA-CONTRACT`, recovered by `NEXT100-012-POSTBASE352-RECOVERY`.

Status: post-Base communication-data contract and seed materialization only. No training is authorized or executed.

## Authority consumed

This contract is stacked directly on terminal POSTBASE-253 head `f6463424b5f53152fce6e6053b705f94e03f9f06`. It reuses POSTBASE-253 `TokenizerCompatibility` and keeps the canonical Base/post-Base evidence and artifact boundaries intact.

The newer POSTBASE-351 adapter head was checked during convergence. It was not treated as terminal authority because its exact-head GitHub workflows had no terminal PASS at the time of recovery. The communication-data contract therefore does not invent or silently bind an unverified adapter dependency. A later terminal adapter may consume this dataset only through the preserved POSTBASE-253 tokenizer/Base boundary.

## Scope and seed

The committed seed has 20 project-authored user/assistant dialogues under `data/post_base/communication_v1/`:

- train: 12;
- selection: 4;
- final: 4.

English (`en`) and Ukrainian (`uk`) are required in every split. Train covers direct answers, transformation, summarization, structured response, clarification, exact reasoning and multi-turn context carryover.

The seed proves schema, provenance, split separation, formatting and validation mechanics. It is not evidence that 20 examples are sufficient to train a useful assistant.

## Canonical Base firewall

The manifest is classified `POSTBASE_COMMUNICATION_ONLY` and hard-binds all of the following to `false`:

- `base_corpus_evidence`;
- `canonical_base_training_eligible`;
- `training_authorized`;
- `selection_for_training`;
- `final_for_training`;
- `final_for_selection`.

No communication row is eligible for canonical Base pretraining, tokenizer fitting, Base scientific evidence or Base checkpoint construction. POSTBASE-352 modifies no Base weights, optimizer state, ModelSpec or checkpoint.

## Physical split firewall

V1 requires three distinct canonical files with fixed names:

- `train.jsonl`;
- `selection.jsonl`;
- `final.jsonl`.

The manifest must bind exactly those files, all three SHA-256 identities and all three record counts. Renaming, aliasing, path escape, symlink substitution, byte drift or row/file split mismatch fails closed.

`to_posttraining_records(..., for_training=True)` admits train only. `to_posttraining_records(..., for_selection=True)` admits selection only. Training and selection use are mutually exclusive. Final rows map to D09 `TEST` and are operationally ineligible for both training and selection.

## Record and quality contract

Each canonical UTF-8 JSONL row contains exactly:

- stable `record_id` and `family_id`;
- split, language and skill labels;
- an alternating dialogue that starts with `user`, ends with `assistant`, and permits only those two roles;
- provenance;
- quality-gate results.

Text must be Unicode NFC, LF-only, free of NUL and forbidden control characters. Every row carries SHA-256 over the canonical message payload.

Every admitted row must pass:

- answer verification;
- relevance review;
- language review;
- PII review;
- secret review;
- copyright review;
- `no_hidden_reasoning = true`, meaning no hidden chain-of-thought/private-reasoning target is stored;
- canonical JSON and content-hash verification;
- split/family isolation;
- duplicate and near-duplicate rejection;
- required language and train-skill coverage;
- tokenizer/context representability.

## Duplicate and leakage gate

`record_id` must be unique. A `family_id` may exist in only one split. Normalized exact duplicates are rejected globally.

Recovery strengthens the v1 near-duplicate rule: character-5-gram Jaccard at the frozen threshold `0.85` is rejected globally, including within the same split. Cross-split near duplicates therefore remain rejected, while same-split near-copy amplification can no longer bypass the contract.

This deterministic gate is not claimed to detect all semantic contamination; future larger corpora may add stronger scalable decontamination without weakening these v1 laws.

## Provenance and foreign-model fail-close

Every committed seed row is project-authored with:

- `source_id = project:postbase352-manual-v1`;
- `rights = project_owned`;
- `foreign_model_output = false`;
- `synthetic_authority_id = null`.

The seed manifest therefore binds `foreign_model_records = 0` and `synthetic_data_authority = null`.

A future foreign-model/teacher-generated row is rejected unless a separate later `SyntheticDataAuthority` is supplied. The authority must be owner-approved specifically for `post_base_communication_data`, carry a valid SHA-256 identity, name unique allowed source IDs and exactly match the row authority ID/source. A foreign row must also carry `rights = authority_bound`.

POSTBASE-352 creates no synthetic-data authority, calls no external model and does not admit foreign-model output merely because a row claims to be synthetic.

## Tokenizer compatibility

The logical v1 tokenizer profile remains `s0-byte-v1`, vocabulary size 256, UTF-8 byte encoding. Dialogue formatting uses ordinary `User:` / `Assistant:` text prefixes, adds no special tokens and installs no chat template into canonical Base.

Before any later training authority exists, a consumer must bind the exact retained Base `TokenizerCompatibility` from POSTBASE-253. `require_exact_base_tokenizer()` rejects tokenizer ID, vocabulary size, config SHA-256 or vocabulary SHA-256 drift. POSTBASE-352 invents no Base tokenizer hashes.

The v1 formatted example ceiling remains 256 UTF-8 bytes.

## Immutable seed identities

Recovery does not modify the 20 seed rows or manifest, so their identities remain:

- train JSONL SHA-256: `ddafe61ce3255dd30d207ec1ee811efa59a2da37288368a9bbc3fa0602cb2ba7`;
- selection JSONL SHA-256: `e36f7c560c44fd2812935b5382dd628fabbd7af0e79c080c295d1332de13309f`;
- final JSONL SHA-256: `f50262994089f276fb7d3f4c644180d0854273b51d7f7920aca1d2048031d039`;
- manifest SHA-256: `51a927c40b4274f8b8f992b8dd83b4dbddac1e925a45834832a79ee6be18d3d6`.

## Verification and truth boundary

The original PR CI run `32984343041` ended as GitHub Actions `startup_failure` before a runner executed test steps; that result is not treated as a contract failure or as a PASS. Recovery keeps the normal `ubuntu-latest` CI contract unchanged and adds deterministic tests for the strengthened gates rather than weakening checks to work around runner startup.

Terminal status requires the repository CI on the recovered exact head to execute Ruff and pytest successfully.

No optimizer update, gradient, SFT, preference optimization, RL, checkpoint mutation, Base-data admission, foreign model inference, network data generation or paid compute occurs here.

Execution profile: `LOCAL_FREE`.
