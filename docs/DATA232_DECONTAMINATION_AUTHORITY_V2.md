# DATA-232 Decontamination Authority V2

Worker: `DATA-232-DECONTAMINATION-AUTHORITY-V2`.

## Decision at this source cutoff

`BLOCKED_MISSING_DATA230`.

No terminal DATA-230 corpus identity or exact DATA-230 training inventory was published when this worker executed. DATA-183 and DATA-25 are not substituted. The committed blocker report is therefore the strongest honest immutable result: it binds the reserved evaluation authorities and the matching contract, but leaves `training_corpus_identity` and `selection_validation_identity` null.

No model training is executed by this worker.

## Reserved evaluation authority binding

The V2 registry contains metadata and identities only; no final-test outcomes are imported.

- DATA-31/D06 incumbent decontamination registry: `10f7454f77eb2dc3871eeafa5055b1969eab42954eb8e19e61565f217c67df31`, source `6b1c7f3418357e0ea1cfc6ab5ceda6a740dc5921`.
- RECOVER-174/EVAL-131 partial real UA+EN heldout authority: `c7211b3e1e6a4f22463d0e6174f0d6162c2452585704efad5564a35de8de609f`, source `976c101b20ad2e31b0b3e2dda2beed8e7b03c2f3`. Its code modality remains blocked; DATA-232 does not invent code evaluation rights.
- EVAL-132 Ukrainian raw-Base dataset: `ca8d9c9d97c854127e0209871e8929f19e06ec91f3f19902bc8fda33481691ff`, source `82e837c1053d6a77c4d1eb86cfaa7dfb521e2e63`.
- EVAL-133 English raw-LM suite: `f9e713ff336e6189f7aa0ddbb21303431ab2041b6700ed38243eaf65865805cb`, source `d10cdac1f6e2ff04196dbe39b0fb0095c4c6be6f`.
- EVAL-134 code diagnostic: `df18192f6190cc5d8be9492103a15097daaaf31afdd1cd45b2f4c21af5721105`, source `74fee51945c83ebdf39e171a894741964ba51b6d`.
- EVAL-136 memorization authority is retained hash-only as auxiliary evidence: artifact digest `bba24085b45f1c73f7f4735b7cbef9994d4c1a6f1d585921641a2a271375b665`, source `fc4b3a1ed39216ee8e4cc938283ece2bd44f4d68`. It is not reclassified as an ordinary final-test dataset because nonzero canary exposure is experiment-local.

DATA-232 freezes EVAL-131/132/133/134 as no-training final diagnostic authorities without reading their outcomes. EVAL-136 and DATA-31 are auxiliary reserved authorities. The composite final-test identity and the composite all-authorities identity are derived deterministically from exact authority metadata.

## Strengthened matching contract

`data232-deterministic-overlap-cluster-v2` is stdlib-only and deterministic.

It checks:

1. raw UTF-8 SHA-256 equality;
2. contamination-normalized SHA-256 equality;
3. token-shingle near matches;
4. high-containment document fragments, including header/footer wrapping;
5. cross-source-family mirrors;
6. code fork/copy overlap using a separate code skeleton that removes comments, replaces strings/numbers, and canonicalizes non-keyword identifiers before shingling.

Contamination normalization uses Unicode NFKC, removes BOM/soft-hyphen/zero-width format characters, canonicalizes line endings, and collapses text whitespace. Natural language matching case-folds. Code matching retains a separate structure-aware path.

Default thresholds are preregistered in `configs/data/data232_decontamination_v2.json`; the scanner rejects unknown threshold keys. No threshold is tuned from evaluation outcomes.

## Cluster and family exclusion

A graph is built over training and evaluation records. Exact, normalized, near, fragment, mirror, and code-copy evidence creates deterministic edges. Every training node in a connected component containing an evaluation node is excluded. This makes mirror-chain leakage fail closed even when the training record that directly overlaps evaluation is not the only copy.

For contaminated cross-source mirror/copy evidence, the default policy additionally quarantines the implicated training source family. This is intentionally conservative. The report records only source-family hashes and record-ID hashes.

## Outcome isolation and hash-only evidence

Evaluation-authority metadata is recursively rejected if a key contains outcome-bearing terms such as `score`, `result`, `metric`, `loss`, `bpb`, `accuracy`, `perplexity`, `margin`, or `outcome`. This prevents final-test results from being imported into the data-selection process.

Public reports never contain `text`, `source_text`, `content`, `prefix`, `continuation`, or `canary_text`. Match evidence contains only record-ID hashes, raw/normalized hashes, match type, deterministic score, and cross-family flags. Reports are self-hashed and immutable: an existing report may be reproduced byte-for-byte but not overwritten with different bytes.

## Adversarial fixtures

`tests/fixtures/data232_decontamination_adversarial_v1.json` covers:

- Unicode full-width/NFKC equivalence;
- NBSP, zero-width and CRLF normalization leakage;
- reserved text wrapped by publisher header/footer;
- a near duplicate with multiple token substitutions;
- code copied with renamed identifiers, changed comments and changed numeric literals.

Focused tests additionally cover raw exact overlap, mirror-chain propagation, family quarantine, final-test outcome-key rejection, hash-only report enforcement and immutable-write refusal.

## Unblock contract

A successor may emit `PASS_CLEAN` or `PASS_WITH_EXCLUSIONS` only after all of these exist:

- terminal DATA-230 corpus identity;
- exact DATA-230 training record inventory with source/source-family/modality metadata;
- immutable DATA-230 selection-validation identity;
- the same reserved authority registry or a deliberately versioned successor;
- a full scan under the committed V2 matching algorithm and thresholds.

The full report must then bind `training_corpus_identity`, `selection_validation_identity`, `final_test_identity`, algorithm/version, thresholds, exclusions and hash-only evidence. No model architecture or hyperparameter decision may consume final-test outcomes.
