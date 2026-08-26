# DATA-300 Corpus V0.3 Frozen Build Contract

Worker: `DATA-300-CORPUS-V03-FROZEN-BUILD-CONTRACT`

Contract identity: `2a7348a2f6e737dde2b8471bf0ce680d7f803033043981bfce233166e18b87e5`

## Truth boundary

This change freezes the executable **build contract**, not the corpus. The corpus state remains `NOT_BUILT_NOT_FROZEN_NOT_TERMINAL` until Wave 3 produces the required artifacts, two independent clean builds are byte-identical, every hard gate passes, and a separate freeze review explicitly accepts the resulting corpus identity.

A valid DATA-300 contract therefore never means `CORPUS_FROZEN`, `TERMINAL_CORPUS`, or `PRODUCTION_READY`.

LOCAL_FREE only. No paid compute is authorized.

## Exact terminal source inventory at this cutoff

The Wave-3 intake is exact and fail-closed. No source may be silently added, removed, replaced, or expanded.

| Registry source ID | Family | Modality | Exact identity | Training rights |
|---|---|---|---|---|
| `external-real:en.standardebooks.manual.8-typography` | `en.standardebooks.manual` | EN text | raw SHA-256 `21582c7f0e4ad39f2b0ed97bbc2c082d275e898b7a63c28e6d9badb8ee0f7860`; normalized SHA-256 `154fb4034929714087e75150d678bf65049ddac32e79dcdf97162c8972c2be83`; git `d1143a9b459b5e6f9cdda93a7c1e04676bff4f6b` | ALLOWED |
| `external-real:en.standardebooks.manual.9-metadata` | `en.standardebooks.manual` | EN text | raw SHA-256 `7ac53dfb4bf6f73f178560e09f33160d0250c69fb679802f3254dc0eb4c9f509`; normalized SHA-256 `94eb2f529922d125b3bd40691778886f4d5d80b128b925d0274fb3d94646ec5a`; git `d1143a9b459b5e6f9cdda93a7c1e04676bff4f6b` | ALLOWED |
| `external-real:ua.rada.open-data.laws-texts.d23314` | `ua.rada.open-data.laws-texts` | UK text | raw SHA-256 `36eae31c3b0676ea7c02236fa05bd695c240c9a8eade5febc00457b8103ee1a4`; normalized SHA-256 `72c301db0b2539f3f7a73c9c15e2e425700a6b758a1114f1a861e2d60c704c50` | ALLOWED |
| `external-real:code.encode.httpx._content` | `github:encode/httpx` | Python code | commit `b5addb64f0161ff6bfe94c124ef76f6a1fba5254`; path `httpx/_content.py`; Git blob `6f479a0885f723b7395843d41164a87041820776`; BSD-3-Clause license blob `ab79d16a3f4c6c894c028d1f7431811e8711b42b` | ALLOWED |
| `external-real:code.psf.requests._internal_utils` | `github:psf/requests` | Python code | commit `5460f467b02e49471c0fd6cfc9ca0adab6351f98`; path `src/requests/_internal_utils.py`; Git blob `0466a7d347db4ed34a37db51b75fc8e80bc06055`; Apache-2.0 license blob `67db8588217f266eb561f75fae738656325deac9` | ALLOWED |

The current DATA-228 head `46a70c990dab6ff72bb84ddb54cff1156b491b40` is not admitted. Its dedicated immutable-source-probe run `32957120454` failed. The Kubernetes and CPython candidates from that branch therefore remain candidates, not Wave-3 inventory.

## Exact component authorities

### Source and rights

- DATA-229 head `90bc0b7f8b696ec35202532b13edf6ab29a662fe`; dedicated run `32957147036` PASS.
- DATA-229 registry identity `1357a343eb4ea973950d8991913109cbea53fe4fa891f0be9745ab497eb59486`.
- DATA-213 source authority `5ae74eb162641712ff97a326572bec5bfe2b0607`; workflow `32950910634`; artifact `9600107886`.
- Canonical DATA-24 registry identity `82abd7dca04947d72a6d07d8228025c58373d17018fa8dc3a7bca30f7a2714c2`.
- DATA-227 head `8ebdb2e132ed7bae5245e9d4c140752640ab9885`; dedicated run `32956209865` PASS; rights-policy Git blob `0ce5223a1cade10031899bf27348a1a65121d4c6`.
- Public accessibility, an SPDX label, or repository visibility alone never grants training admission. Purpose-specific evidence remains mandatory.

### Quality and privacy

DATA-214 retained immutable evidence binds:

- DATA-32 producer commit `b1c9449ca839ed10c872f444010f74fd225acae1`; retained quality-report Git blob `ed11a4d605d57fbb7102ab58da7614a4c476cf85`.
- DATA-33 producer commit `290b82fd0f7d1cc3a1840deae4378b9c500f1c15`; retained privacy-report Git blob `d670568437c555dc6e8f1228a6205007dbc1ffb8`.

These are policy/evidence anchors, not stale coverage authority for the new five-object inventory. Wave 3 must rerun the retained quality and privacy/secret gates on every exact candidate record.

### Deduplication and reserved decontamination

DATA-31 head `6b1c7f3418357e0ea1cfc6ab5ceda6a740dc5921`, dedicated run `32893344618` PASS, binds:

- benchmark-registry identity `10f7454f77eb2dc3871eeafa5055b1969eab42954eb8e19e61565f217c67df31`;
- reference-bundle identity `1ea12613c4bd2528d30bd9c9139a77bd972f72e9ea72829e66e2a617bfeda0d9`;
- report identity `1ababadb8c652ca50a88cf41635fe03928730e083b5c7e669cd52a6250259373`.

Wave 3 must rerun exact and near deduplication plus the reserved scan against its exact training inventory. `semantic_universal_cleanliness_claimed=false` remains mandatory.

The newer DATA-232 line is not promoted as terminal completed decontamination evidence: its current dedicated workflow failed, and its earlier blocker state did not execute a DATA-230 scan.

### Reservation and split authority

EVAL-233 head `b5512b4648cb09dd052b08884dc53f291e1ce935`, dedicated run `32957254139` PASS, binds authority Git blob `2008570890819f32c356677e1e250707d339b53a` and evidence identity `37473834df31c69faf39f5c1152e9fe1f7d4aeb1487fcf7489059e8ec444d4a7`.

The existing final test remains exactly 16 immutable UA/EN records under RECOVER-174 authority identity `c7211b3e1e6a4f22463d0e6174f0d6162c2452585704efad5564a35de8de609f` and seed blob `4bfbfbf29fa9538cabda6068efd3a1fd036a9479`. Those records are not tokenizer-fit, hyperparameter-selection, or checkpoint-selection data.

EVAL-233 currently has zero immutable selection-validation records. That is a hard Wave-3 blocker, not permission to split or reuse final-test bytes. A new nonempty immutable selection-validation authority must be bound before tokenizer/model selection.

DATA-227 code objects remain training-authorized but not separately evaluation-authorized or reserved. They cannot be silently moved into final-test/selection sets.

### Balance/diversity and no-repetition boundary

EVAL-237 head `af1075168009dcf4ed53cff20d7e08538c1968c3`, dedicated run `32955916650` PASS, is consumed as a fail-closed diversity/exposure boundary. It explicitly rejects padding as data and rejects repeating examples/loss tokens merely to fill a matched budget.

Its stricter leave-one-family-out identifiability thresholds are experiment requirements, not automatically a Corpus V0.3 balancing winner. DATA-105's older 35% cap/sqrt alternatives are not promoted because its dedicated run failed.

Wave 3 must publish deterministic mass tables by source, source family, language/modality and origin while preserving the exact intake inventory unless an earlier hard gate excludes material. It may not manufacture diversity or balance by copying documents, sampling with replacement, or replaying loss positions.

### Unique-loss authority

LEARN-217 head `c02c8aa38e691521ae2ab6a4ff3ea1d643efd6ef`, dedicated run `32952787070` PASS, is the terminal execution anchor for no-replay optimized-token accounting. The strongest observed no-replay exposure is 2,000,060 actual non-ignored causal targets.

DATA-25's 20,000,775 source-token ceiling is not relabelled as an exact unique-loss count. Wave 3 must publish a complete source-position ledger proving that every optimized training target is used at most once. Padding is never data.

## Split contract

`train` may fit the tokenizer and update model parameters. It may not serve as checkpoint/hyperparameter selection data or final-test reporting data.

`selection-validation` must be immutable, prebound and nonempty before tokenizer/model selection. It may select hyperparameters/checkpoints but may not fit the tokenizer or update model parameters.

`final-test` is immutable and reserved. It may not be read before selection is locked and may not fit the tokenizer, tune hyperparameters, select checkpoints, or update model parameters.

No content hash may occur in more than one split. A deduplication cluster may not straddle splits.

## Artificial repetition is forbidden

Wave 3 fails if any of the following occurs:

- physical document replication to increase volume or diversity;
- with-replacement sampling to manufacture balance;
- repeated source/record/target loss positions;
- corpus recycling merely to hit a target budget;
- counting padding or ignored labels as useful data.

## Two independent clean builds

Wave 3 requires two independent clean build roots. They may not share mutable build cache/state. Every relative path, file size and SHA-256 in the complete artifact tree must be identical between the two builds.

Build identities must not depend on wall-clock time, host name, absolute workspace path, random UUIDs, filesystem iteration order, or network response order.

A byte-identical pair does not itself freeze the corpus. It only permits the result to advance to a separate freeze review.

## Required Wave-3 artifact structure

```text
<build-root>/
  authority/contract-lock.json
  source/source-inventory.json
  source/rights-evidence.json
  quality/quality-report.json
  privacy/privacy-report.json
  dedup/exact-dedup.json
  dedup/near-dedup-clusters.jsonl
  decontamination/reserved-scan.json
  balance/balance-report.json
  splits/train/manifest.json
  splits/selection-validation/manifest.json
  splits/final-test/manifest.json
  unique-loss/train-ledger.jsonl
  unique-loss/summary.json
  shards/train/manifest.json
  release/gate-report.json
  release/release-manifest.json
```

No listed file is optional for release evaluation.

## Hard pass/fail gates

| Gate | PASS requirement |
|---|---|
| G01 CONTRACT IDENTITY | Exact DATA-300 self-identity and all component locks match. |
| G02 SOURCE INVENTORY | Exact five-object intake; no unbound additions/removals; failed DATA-228 remains excluded. |
| G03 RIGHTS | Every training object has purpose-specific training authorization. Public access is insufficient. |
| G04 QUALITY | Retained DATA-32 policy reruns on every Wave-3 candidate record and passes. |
| G05 PRIVACY | Retained DATA-33 privacy/secret policy reruns on every Wave-3 candidate record and passes. |
| G06 DEDUP | Exact + near deduplication and deterministic cluster handling pass. |
| G07 RESERVED DECONTAM | Reserved scan actually executes; evaluation-connected training components are excluded; no universal-cleanliness overclaim. |
| G08 BALANCE / DIVERSITY | Deterministic mass audit covers exact intake; no duplicate materialization, with-replacement sampling, or repeated loss positions. |
| G09 SELECTION VALIDATION | A distinct nonempty immutable selection-validation authority exists before any fit/selection. |
| G10 FINAL TEST ISOLATION | Exact final-test authority stays immutable and unread until selection lock. |
| G11 UNIQUE LOSS | Complete source-position ledger has zero repeated optimized targets; padding/replay are zero. |
| G12 TWO CLEAN BUILDS | Both independently valid complete build trees are byte-identical. |
| G13 RELEASE TRUTH | No contract/build result calls the corpus frozen, terminal, or production-ready before the separate freeze review. |

Every gate is hard. Any failure keeps the candidate unreleased.

## Executable validator

Validate only the frozen contract:

```text
python tools/validate_data300_corpus_v03_build_contract.py validate-contract
```

Validate one Wave-3 build root:

```text
python tools/validate_data300_corpus_v03_build_contract.py validate-build /path/to/build-a
```

Validate two clean builds and full-tree byte identity:

```text
python tools/validate_data300_corpus_v03_build_contract.py compare-builds /path/to/build-a /path/to/build-b
```

Even a successful two-build comparison returns `CANDIDATE_READY_FOR_SEPARATE_FREEZE_REVIEW` with `corpus_frozen=false` and `terminal_corpus_claimed=false`.
