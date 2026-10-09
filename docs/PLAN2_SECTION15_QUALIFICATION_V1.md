# Plan 2 / Section 15 — terminal data qualification contract

Status: **OPEN — component audit implemented but real dataset release NOT authorized**.
Plan authority: Google Drive "2. Другий план", Sections 15.1–15.3, and
`MULTI_PLAN_CLOSURE_STATE.md` (GitHub overrides the Drive migration snapshot).

## Reused implementation

The single candidate lineage is the existing Plan-2 S1–S14 source registry,
rights catalog, physical materializer, normalization, G06 privacy, exact
and complete-link near dedup, DATA-232 reserved evaluation firewall, corpus
mixture, cluster-safe split, fitted frozen tokenizer fixture, packer and
ordered exposure ledger. No second ingestion, tokenizer, trainer, dedup,
split, checkpoint or permission authority is created.

`tools/plan2_terminal_qualification_v1.py` runs two independently
materialized LOCAL_FREE builds in fresh directories and demands byte-identical
inventories. It re-verifies the S5 privacy -> S6 exact -> S7 near ->
S8 eval firewall -> S9 mixture chain using the canonical staged manifests
and SHA-256 content identities. It refuses noncanonical, missing, changed,
or symlinked outputs, source-count drift, unexpected split failures,
rights escalation, leaked raw text / final-test access, synthetic-as-production
tokenizer/shard/exposure promotion, incompatible manifests, and immutable
readback tampering. It emits only hash/metadata evidence (not training text).

Current source is the pinned DATA324 Ukrainian Kubernetes documentation,
a single document family. Its Plan-2 rights grant admits
`allowed_uses=["source_candidate"]`, not training or release.
Physical three-way family-safe train/validation/test split correctly refuses
the one-family candidate. S12 tokenizer and S13/S14 packing/exposure results
are expressly **synthetic component fixtures**, not production artifacts.

Other D03 evidence on main may contain additional candidate families,
but is not a lawful Plan-2 release. Candidate source admission, partial
balance, or independent component success is not final training authority.

## Real public-domain source expansion (2026-10-08)

This branch now physically carries **1,265,481 UTF-8 bytes** from three
original public-domain books (Austen, Carroll, Shelley), each pinned to its
exact GITenberg source Git blob and commit in
`configs/data/plan2_public_domain_books_v1.json`. Project Gutenberg
trademark and license envelopes were stripped before storing the bodies.

**One source family, three document clusters:** These three works belong to
the ONE existing canonical
`en.project-gutenberg.public-domain-books` source family under the
accepted NEXT100-107 seal. Individual authors/works have different
`document_family` identities and upstream revisions. They are NOT three
independent source-family capacity credits.

`tools/plan2_public_domain_books_v1.py` reuses the canonical S1 inventory,
S2 source rights/materialization receipts, incumbent G06 privacy authority,
and DATA-232 contamination matching against the pinned synthetic S8 reserve.
It refuses source/rights/snapshot drift, unauthorized training release,
source symlinks, inconsistent source-family assignments, broken privacy
receipt roots, contamination report tampering and changed publications.
The two clean S15 builds reverify the full real-book snapshots and hash-only
evidence. New tests and steps are in
`tests/test_plan2_public_domain_books_v1.py` and the existing Plan-2 CI.

**The new books are NOT a terminal dataset or admitted P9 training input.**
Source-level legal eligibility and G06/DATA232 candidate audits are not
real reserved final-test custody, complete S3–S9 multi-source admission,
balanced train/validation/test splits, frozen physical tokenizer,
production packed shards, ordered exposures or Plan9 launch approval.
The older physical S3/S9 cohort remains one separately pinned Ukrainian
Kubernetes source with allowed use `source_candidate` only. No source
family was invented or silently promoted into that authority.

## Real-book document-cluster split mechanics probe (not release)

`tools/plan2_s15_physical_book_split_probe_v1.py` now consumes the three
physically pinned public-domain books and their incumbent S1/S2/G06/DATA232
candidate audit. It reconstructs exactly six non-overlapping physical
line-span records, two per original book, and verifies full byte reconstruction.
The **existing** S10 deterministic whole-document cluster split is exercised
with the existing pinned S10 policy: no record of a source document may cross
train/validation/test, and receipt partition IDs must exactly match assignments.
The output is immutable, hash-only and independently replayable; negatives cover
snapshot drift, a false extra family, forged training permission, mismatched
partition assignments, tampered output, and symlink destinations.

The S10 interface requires an S9-shaped input. This isolated probe
deliberately labels that compatibility input `fixture_only=true`; it is **NOT**
a physically accepted S9 receipt, and the resulting S10 split is a mechanism
test, not an authorized train/validation/test production partition. Both real
book snapshots and this probe are now included in the Section-15 two-clean-build
member SHA-256 inventory. **Three books remain exactly one canonical Gutenberg
source family**, not three independently credited families. The physical
Kubernetes S9 candidate remains one separate non-training-authorized family.
No true multi-family S3→S9 admission, release-grade decontamination, production
tokenizer, real shards/exposure release, or Plan-9 binding is claimed.

## Qualification commands

From a complete checkout of the exact PR head with project requirements:

```sh
PYTHONPATH=src:. pytest -q tests/test_plan2_terminal_qualification_v1.py
ruff check tools/plan2_terminal_qualification_v1.py tests/test_plan2_terminal_qualification_v1.py
PYTHONPATH=src:. python tools/plan2_terminal_qualification_v1.py \
  --root . --out-dir /tmp/plan2-s15-audit
PYTHONPATH=src:. python tools/plan2_terminal_qualification_v1.py \
  --root . --out-dir /tmp/plan2-s15-audit
```

The same-directory repeat must preserve all bytes and hash identities.
Run once more with a fresh output directory and compare reports.
The GitHub "Plan 2 Data Component Qualification" workflow includes prior
S1–S14 scoped gates and the S15 audit. A queued or cancelled run is **NOT PASS**.

## Exact terminal acceptance — no promotion by assertion

Only mark Plan-2 Section 15 DONE after all are *actually* true:

1. An accepted multi-family corpus has independently verified source identity,
   explicit training/release rights, member provenance and removal path.
2. Physical rights, privacy, global dedup, evaluation decontamination,
   mixture and family-safe split pass on the **same exact** candidate, without
   seeding test/validation records into tokenizer fit or model training.
3. The tokenizer is fitted/frozen on this accepted train partition, with
   immutable vocab, IDs, normalization and compatibility proof (not S12 fixture).
4. Physical deterministic shards, source/target mapping and stable ordered
   exposures exist; exact identities match the frozen corpus and tokenizer.
5. At least two clean rebuilds produce identical physical dataset/tokenizer/
   shard/exposure hashes; negative and restart/recovery tests qualify the
   release, not merely a staging fixture.
6. The release manifest provides immutable, read-back-verified dataset,
   tokenizer, shard and exposure identities for Plan 9. This is a **data
   handoff only**; Plan 9 must still independently authorize any pretraining.
7. A canonical integrated SHA and evidence are read back from GitHub, then
   both GitHub closure state and Drive plan are updated to terminal DONE.

Until then, the report must keep
`decision=COMPONENT_AUDIT_ONLY_NOT_TERMINAL`,
`terminal_done=false` and `production_release_authorized=false`.
No model training, final-test reading, paid GPU run, foreign pretrained weights,
source-rights promotion or PLAN-9 authority is conveyed.

There is no Section 16 in the second plan; after terminal Section 15 the
numbered plan closes. Any next-plan assignment must be explicit.


## 2026-10-09 S15 real-source holdout and physical-tokenizer candidate — NOT terminal

The continued S15 canonical PR now executes **two exact Git HEAD source
checkouts**, runs the existing S15 qualification independently in both, and
checks matching canonical SHA-256 inventories. It rejects dirty tracked HEAD,
unsafe tar members, symlinks, altered receipts, and a forged release decision.

The new \`plan2_s15_physical_heldout_decontam_v1.py\` stage uses the three
immutable public-domain **document** identities and the existing S10 whole-book
split probe. It binds an actual one-document training / one-document validation /
one-document final-test partition to original source SHA hashes, runs the
incumbent DATA-232 contamination matcher on the **physical book text**, and
records a *hash-only*, immutable report. Only the document assignments and
DATA-232 source scan are evaluated; no final-test **outcomes** are read. This
is not independent external benchmark custody and is NOT a physical S9 admission.

The new \`plan2_s15_physical_tokenizer_candidate_v1.py\` fits the **incumbent
frozen byte-BPE algorithm** to exact physical training-document text, after a
clean real heldout scan. A deterministic lossless UTF-8/line partition uses a
separate reserved S15 training-segment record-ID range rather than reusing
S10 span record IDs. It records exact fitted merges, training record hashes,
an immutable tokenizer identity and complete byte roundtrips. This is a
**real-data candidate**, not a frozen production tokenizer: the incumbent S12
cap is only 128 merges and its policy expressly says
\`LOCAL_FREE_SYNTHETIC_NO_RELEASE\`. No vocabulary-32768 claim is made, and
no synthetic fixture is promoted. S15 clean builds include these receipts.

The CI extension includes scoped lint, real data checks, forged-authority,
corruption, symlink, Unicode and restart negatives, plus clean Git-root archive
rebuild. Exact-head CI and accepted-main integration/readback must still pass
before this component evidence may be accepted.

**Terminal S15 remains OPEN** until a genuinely admitted full physical S1–S9
production corpus (with both rights and reserve custody), production-scale
frozen tokenizer, deterministic production shards and exposures, two clean
production rebuilds, and a Plan9-compatible signed release handoff are
established. Neither this PR nor a green component CI alone constitutes DONE.


## Physical real-text S13/S14 candidates (not production release)

Section 15 now separately reconstructs physical S13 token targets from the
verified **actual training book**. The versioned
`tools/plan2_s15_physical_packing_candidate_v1.py` reuses the accepted S12
FrozenBPE identity and incumbent S13 packer's BOS/EOS, token labels, source
boundaries and loss/attention masks. Source text is segmented losslessly under
S12's bounded fit policy; its S15 segment IDs occupy a separate numeric domain
from S10's two original per-book spans. Every physical shard carries a
deterministic canonical JSON SHA-256, a block count and a parent tokenizer/split
identity. A changed or additional shard, symlink or manifest is rejected at
immutable readback. A versioned **64-block per shard** grouping is explicitly
recorded; it does not impersonate the original four-block S13 fixture layout.

`tools/plan2_s15_physical_exposure_candidate_v1.py` uses the incumbent
S14 target/exposure identity and causal hash-chain formulas to generate
ordered, uniquely identified **real physical next-token exposures**. Each
exposure shard binds the exact parent shard hash; missing targets, offset
discontinuities, mask drift, changed parent shards, duplicate target/exposure
IDs and unexpected output members fail closed. The bound physical target count
and chain head are compared against S13 under both S15 clean rebuilds.

**These are physical LOCAL_FREE CANDIDATE artifacts, not released training
inputs.** Neither S9 `training_corpus_authorized` nor the S12
`physical_tokenizer_fit_authorized` grant is present. The incumbent tokenizer
is a bounded 128-merge engineering candidate, **not** a production-frozen
32K-vocabulary tokenizer. No Plan9 launch or optimizer training is permitted
from these artifacts. Current exact-head hosted qualification, production
rights/mix/holdout acceptance, Plan3/4 token compatibility, and final Plan9
release authority remain terminal gates. Do not change S15 or Plan2 to DONE
until all of them have verifiable accepted-main evidence.


## Additional physical provenance and mirrored-book controls

A whole-document cross-book mirror check now applies the incumbent S7
versioned near-duplicate matcher even between the two physical reserved
evaluation roles. Identical spans of the **same original book** are not falsely
credited as independent documents. A detected different-book mirror is recorded
as text-free hashed evidence and prevents physical token-fit qualification;
the original books remain candidate-only. The existing DATA-232 hash-only
training/evaluation report remains an independent required check.

The physical S13 candidate manifest now carries the complete byte-span offset
and SHA-256 map from every generated `r50000000+` S15 training segment back to
the original pinned book bytes. Segments are contiguous, lossless and non-
overlapping, and the total recorded bytes must equal the original checked
snapshot. Those exact physical bytes are still NOT a production dataset merely
because they produce loss targets and shards.


## Published physical S13/S14 replay witness

The scoped S15 code now produces a separate independent
\`tools/plan2_s15_physical_readback_v1.py\` witness after the real shard and
exposure publications. Unlike building S13/S14 from the source text again, the
witness reads the **already published physical files** and checks exact shard
SHA-256, byte counts, immutable manifest hashes, every canonical S13 block
mapping, token offsets, loss masks, all S14 target IDs/exposure IDs, and the
cross-shard ordered causal hash chain. No missing, duplicate or extra physical
artifact may pass. The witness refuses re-signed incorrect token identities
and unauthorized capability flags; negative tests cover corruption, orphan
files and symlinks.

This witness is bound to the S15 double-clean-build audit hash, and scoped
Actions lint/pytest must verify it at the same frozen PR head. It is still a
**non-authorizing real physical candidate** because the upstream S9 material is
not a production-admitted training corpus, the fitted BPE uses the incumbent
fixture-capped merge policy rather than a frozen target-vocab tokenizer, and
Plan9 has no signed production data handoff. Final PASS requires actual
exact-head CI execution plus original-Dataset train/release rights, independent
final-test custody and full production compatibility, not just these
reproducibility properties.


## Actual 3-family source intake and whole-source split (2026-10-09)

The D03 historical artifacts have been independently retrieved and checked:
10 original Ukrainian PHP Manual XML objects and 2 original Ukrainian Rust Book
Markdown objects reproduce all historical normalized object hashes (30,510 +
18,165 = **48,675** UTF-8 bytes). The exact **12 upstream Git blobs** and 3
upstream license blobs, attribution notices, preserved source revisions and
historical D03 source-admission evidence are now present in this Plan2-owned
branch. PHP docs use CC-BY-3.0-or-later with attribution; Rust Book translation
retains MIT+Apache-2.0 notices. Historical rights are source-level and have
**zero canonical Plan2 training-corpus credit**, as stated in the frozen D03
authority. Do not report this as a Plan2 release.

`tools/plan2_s15_d03_physical_sources_v1.py` replays normalization byte for
byte from these committed original sources and the frozen metadata, checks
source/license/attribution/provenance seals, runs the incumbent G06 and DATA-232
LOCAL_FREE reserved-fixture audit and produces a signed nonrelease receipt.

`tools/plan2_s15_three_family_physical_v1.py` unites **three actual physical
source families**, i.e. 3 Gutenberg public-domain original books (one upstream
family) plus 10 PHP pages (one upstream family) plus 2 Rust Book pages (one
upstream family): **15 real documents, 1,314,156 normalized UTF-8 bytes**.
It enforces global physical exact hashes, cross-family same-language S7 near
mirror inspection, combined G06 privacy, DATA-232 reserve-fixture checks and
candidate-only source provenance. Cross-language semantic dedup is explicitly
NOT established, and the fixture is not real benchmark custody.

`tools/plan2_s15_three_family_split_probe_v1.py` binds all 15 original
document IDs to an incumbent S10 **fixture-shaped, non-authoritative** S9
wrapper and proves whole-original-source (not individual paragraph) cluster
assignment. The exact S10 policy creates 5 source clusters, not 15 independent
source families. Train/validation/test family coverage and leakage are recorded
explicitly. This mechanism does NOT supply a verified S3-S9 production mix,
externally sealed test custody, fitted production tokenizer or a Plan9 binding.

Every part is currently **candidate-only and no-release**; final Plan2 S15
remains OPEN pending actual accepted S1–S9 full lineage, final eval custody,
production-sized frozen 32K tokenizer, compatible production shards/exposure
stream, two fully clean production rebuilds and Plan9 handoff. Exact current
head hosted CI remains unverified when the jobs are queued; queued/cancelled is
never a PASS and a component PASS is not terminal DONE.


## Physical UA normalized publication and exact-head lint repair

The candidate D03 authority now also emits **12 standalone immutable normalized
UTF-8 files** with source-record IDs, upstream normalized SHA-256, source
snapshot lineage and total physical bytes (**48,675**), instead of merely
proving normalization transiently in memory. Restart/no-clobber, extra-file,
symlink and tamper checks refuse unsafe publication; two independent clean
artifact trees are compared and the resulting source receipt is bound to S15.

Hosted scoped Plan2 run #37958202998 at old head `cd93891e` actually executed
through the pinned books, S10 mechanics, actual real-book tokenizer fit and
restart, then **FAILED** at the new Git checkout script's Ruff UP022
(`stdout=PIPE, stderr=PIPE` instead of `capture_output=True`). The
incumbent frozen physical tokenizer candidate passed its 8 scoped tests at
that historical head. The specific source lint error was repaired on this
canonical branch in commit `e415fc24b21225c1d93b5cb3c2a8408c4a924a20`
without weakening any release gate. No green CI is asserted for the current
head until its own complete hosted job finishes; historical failures and
cancelled/superseded workflow heads are not treated as PASS.


## 2026-10-09 physical train-only partition of three-source-family candidate

The same canonical S10 whole-source split of fifteen physical documents
(Gutenberg English books, PHP Ukrainian docs, Rust Book Ukrainian docs)
now drives `tools/plan2_s15_three_family_train_materialization_v1.py`.
It writes only SHA-256-bound training UTF-8 bytes, with immutable restart,
clean-rebuild comparison and full source-byte accounting. Validation and
final-test records are represented by **hash-only member inventories**;
no heldout plaintext is staged or passed to a tokenizer fitter by this
new component. The parent S10 split manifest and split-mechanics probe
identities are separately asserted by the S15 two-build qualification.

The three actual canonical source families and source-level training rights
have NOT yet been accepted by the inherited Plan2 S3-S9 physical pipeline.
The physical train partition is explicitly `training_corpus_authorized=false`,
`tokenizer_fit_authorized=false`, and `production_release_authorized=false`.
A source-level candidate fit, even on real bytes, is not a production 32K
vocabulary or Plan9 training permission. Exact-head CI remains an independent
required test, not presumed from any historical success.

## Real S13/S14 physical three-family target lane — nonrelease

The S10-bound train-only physical source partition can now be packed by the existing S13 deterministic packer with original source-level provenance; no validation or final-test bytes are included. The same S14 builder signs stable ordered target/exposure IDs across all physical shard boundaries. The independent physical reader replays published blocks, next-token targets, source byte maps and causal hash chain. This does not freeze production vocabulary 32768, admit the source families through S3–S9, or authorize Plan9 optimization. A dedicated scoped Actions job must pass at the identical SHA before this lane can be integrated into the terminal two-clean-source-build audit.
