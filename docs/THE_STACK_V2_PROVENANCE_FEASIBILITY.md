# The Stack v2 provenance feasibility

Status: **FEASIBILITY ONLY — NOT TRAINING AUTHORIZED**  
Swarm claim: `D03|THE-STACK-V2|REDTEAM-AUDIT|TERMS-PROVENANCE-FEASIBILITY`  
Control: #723; research parent: #720.

## Decision

The Stack v2 is useful as a code-corpus candidate, but it is not a dataset that 12-6 may admit wholesale. The current project state is **blocked for content ingestion**. No bulk-access agreement is recorded in the project policy and no source repository has been admitted by this package.

This package is intentionally narrower than an ingestion implementation. It freezes the upstream identity researched on 2026-08-27, encodes the access and rights boundaries, and provides a fail-closed per-source feasibility validator. A positive validator result is only `SOURCE_FEASIBLE_FOR_D03_ADMISSION_REVIEW`; it never means `TRAINING_AUTHORIZED`.

## Confirmed upstream facts

Primary sources were read directly rather than inferred from an aggregator.

- Hugging Face dataset identity: `bigcode/the-stack-v2`.
- Observed immutable repository revision: `e565caa3a78c2423bd374333a472b049eb090e47`; API `lastModified` was `2026-08-03T13:24:25Z` at the research cutoff.
- The Hugging Face repository is gated and labels the dataset license as `other`.
- The dataset terms state that bulk download requires an agreement with **Software Heritage and Inria**, using `datasets@softwareheritage.org` for access information.
- The dataset card states that file contents are stored in the Software Heritage S3 bucket; its example path is `s3://softwareheritage/content/{blob_id}`.
- Provenance fields exposed by the dataset include `repo_name`, `snapshot_id`, `revision_id`, `blob_id`, `content_id`, `directory_id`, `path`, branch and crawl/revision timestamps.
- License signals include ScanCode-derived `detected_licenses` / `license_type` and GH Archive `gha_license_id`. They are evidence signals, not a project legal determination.
- The card states that The Stack v2 gathers repositories under different original licenses and that use must follow those original licenses, including attribution obligations where applicable.
- The card says validated removal requests are enacted by dataset updates and users agree to update to the most recent usable version. Therefore a stale revision is not silently acceptable after upstream changes.
- Software Heritage API terms prohibit using the pointwise public API for massive extraction. Software Heritage also disclaims a guarantee that its computed license information is correct and says users remain responsible for determining applicable source rights.
- Software Heritage terms separately constrain privacy/personal-data use. Public availability does not erase privacy, security, license or attribution obligations.

## Confirmed versus inferred

Confirmed in primary sources:

1. The exact observed Hugging Face revision and gate state.
2. The bulk-access agreement requirement and named Software Heritage/Inria boundary.
3. The SWH S3 content path pattern.
4. Per-record provenance/license-signal fields.
5. Original source licenses remain controlling.
6. Removal-refresh, privacy and anti-mass-extraction obligations exist.

Project inference/policy, deliberately conservative:

1. Dataset-level `license: other`, dataset access acceptance, Software Heritage terms, ScanCode inference or GH Archive license metadata cannot by themselves grant 12-6 source-level training rights.
2. A source must receive a separate project rights review bound to its immutable provenance before it can even proceed to D03 admission review.
3. Evaluation-firewall and decontamination evidence are required independently of license/access because rights correctness does not prove benchmark cleanliness.
4. A current bulk-access agreement must be durably recorded before any SWH S3 bulk-content path can become feasible. The canonical policy currently stores `null`, so real content records fail closed.

These inferences implement #720's explicit hard boundary: no dataset becomes training-authorized from a package/dataset-card license alone.

## Machine contract

Canonical policy: `configs/research/the_stack_v2_provenance_feasibility_v1.json`.

Every source-content record must bind:

- exact The Stack v2 revision and removal-sync revision;
- repository, SWH snapshot/revision, blob/content and path provenance;
- a source-specific SPDX-style license identifier;
- durable **human/project source-license review evidence**, not a dataset/access/detector shortcut;
- the project Software Heritage/Inria bulk-access agreement reference and acknowledged access terms;
- passed privacy review;
- bound evaluation firewall;
- passed decontamination review.

Unknown, missing, stale, ambiguous or self-authorized fields block. `training_authorized=true` on an input record is itself rejected.

The canonical policy deliberately contains no project bulk-access agreement reference, so it currently evaluates all real SWH S3 content candidates as blocked. Tests use a synthetic agreement reference only to prove that later source-level evidence cannot escalate into training authority.

## Validator

Policy-only check:

```bash
PYTHONPATH=src python tools/validate_the_stack_v2_provenance.py
```

Optional per-record evaluation:

```bash
PYTHONPATH=src python tools/validate_the_stack_v2_provenance.py --records records.json
```

The records file is a JSON array. Exit `0` means the policy is valid and, when records were supplied, none were blocked. Exit `2` means one or more supplied records failed closed. Even exit `0` never authorizes training.

## Adversarial contract

Focused tests reject at least these bypass classes:

- missing repository/SWH provenance;
- mutable or stale dataset/removal revision;
- no/unknown source license;
- package-level dataset license as source authority;
- Software Heritage access terms as source-license authority;
- detector/GH Archive license signal as source-license authority;
- no project bulk-access agreement;
- access agreement mismatch;
- privacy/evaluation-firewall/decontamination gaps;
- source records that self-declare training authorization;
- policy edits that turn on bulk or canonical training authorization or weaken source-license control.

## Not performed

- No gated dataset access was accepted on behalf of the owner.
- No Software Heritage/Inria agreement was created or asserted.
- No dataset files or source contents were downloaded.
- No source repository was legally approved or admitted to a corpus.
- No deduplication, decontamination, PII scan, tokenizer fitting or training ran on The Stack v2 content.
- No paid compute was used.
- No legal advice is claimed.

## Next safe action

The next owner/legal-data action is to decide whether to pursue a Software Heritage/Inria bulk-access agreement. If an agreement is obtained, record only its durable project evidence reference, then review a small bounded set of repositories source-by-source. Each record still requires independent source-license, privacy, evaluation-firewall and decontamination evidence before separate D03 admission. Any newer usable The Stack v2 revision requires a fresh provenance/removal-sync review rather than silent drift.
