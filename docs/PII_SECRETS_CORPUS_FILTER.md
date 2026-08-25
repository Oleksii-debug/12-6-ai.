# D03 PII and Secrets Corpus Filter

## Incumbent audit

DATA-33 extends the existing D03 policy seam rather than creating a parallel eligibility gate. `RecordPolicyMetadata` already contains a `pii` `PolicyHookEvidence` slot, accepts `NOT_RUN`, `PASS`, `REJECT`, and `REVIEW_REQUIRED`, and `assert_passed()` rejects every non-`PASS` hook. Source-rights decisions remain separate from this privacy/security result.

`PrivacyRecordResult.policy_hook_evidence()` therefore emits the incumbent `hook_id="pii"` evidence. A redacted or clean document emits `PASS`; generic suspicious secret assignments emit `REVIEW_REQUIRED`; high-confidence credential/private-key findings emit `REJECT`.

## Explicit policy

The retained machine-readable policy is `configs/data/pii_secrets_policy_v1.json`. Its SHA-256 identity is included in every record result and aggregate scan manifest.

Actions are deliberately conservative:

- `REDACT`: email addresses, phone-like strings after shape validation, valid US SSN forms, Luhn-valid payment-card numbers, and mod-97-valid IBANs.
- `EXCLUDE`: private-key headers, AWS access/secret keys, GitHub tokens, Slack tokens, Stripe live secret keys, Google API keys, bearer JWTs, and Azure storage account keys.
- `QUARANTINE`: plausible generic password or secret/token assignments that are not recognized as a stronger vendor-specific credential.
- `ALLOW`: no detector fired under this bounded policy.

Common placeholder/example values are suppressed for the generic assignment detector. Strong structured identifiers use validators where practical to reduce false positives.

## Evidence safety

Matched values and text previews are never persisted in privacy evidence. Per-record evidence contains only record/source/version identity, modality, action/status, detector counts, redaction count, input/output SHA-256 values, and policy SHA-256. Excluded or quarantined documents do not receive sanitized text output. A structural guard rejects value-bearing fields such as `text`, `match_value`, or previews if they are later added to manifests.

Large-corpus summaries do not retain every record row. They retain aggregate counts plus a deterministic `records_evidence_sha256` over sanitized per-record evidence, preserving a compact audit identity without putting source text or matched values in the report.

The existing CI already runs Gitleaks with artifact upload disabled. Adversarial credential fixtures are assembled at runtime so tests do not introduce secret-shaped literal credentials into Git history.

## Current corpus V0.1 scan

Primary retained evidence: `reports/data33/pii_secrets_scan_corpus_v01_20260825.json`.

The scan is bound to current DATA-25 corpus identity `422f545da93526ceba2c0ff738e0b9deac65f35bfb31d87d2aab068cda091fc8` and privacy policy identity `8c905e3b8f81391c3f928f375bca8fe6d1b5d38b41dec5c081577e2c5ce58526`.

The deterministic DATA-25 builder was reproduced before scanning. Reproduced counts matched its retained manifest exactly: 46,207 final documents, including 33,992 natural-text records and 12,215 code records; split counts were 43,238 train and 2,969 validation. The privacy pass observed 46,207 `ALLOW`, 0 `REDACT`, 0 `QUARANTINE`, 0 `EXCLUDE`, with zero findings for every retained detector in both modalities. The sanitized per-record evidence digest is `7c8658fcecd34f62ef18e565228576afb11b925ac43c5052084b9c61a24ef227`.

This current V0.1 corpus is project-authored synthetic data because its source registry has zero external training-eligible sources. The scan therefore is a real pass over the current versioned corpus, but it is not evidence about external-source privacy prevalence or representativeness.

`tools/scan_corpus_v01_privacy.py` rebuilds V0.1, verifies the rebuilt corpus identity and every shard hash against the retained DATA-25 manifest, scans all materialized records, emits a compact report, and exits nonzero if any record remains non-training-eligible after the privacy pass.

## S0 regression scan

Additional retained evidence: `reports/d03/pii_secrets_scan_s0_project_authored_20260825.json`.

It is bound to S0 input SHA-256 `72d6eca189a9cab43ae15d9daeba4da1c0d023310f6ca54f27cda9dff08c22bf` and S0 source-registry SHA-256 `fd87553dd1260e8f96560a50d226c766b954a1384bf76ad0d9907d9e228d6a11`. All 12 natural-text S0 records were `ALLOW`; code counts are zero because S0 contains no code records.

## Reproducibility and limitations

`tests/test_privacy_filter.py` covers redaction, exclusion, quarantine, validators, evidence sanitization, policy identity, D03 policy-hook composition, and synthetic adversarial credentials. It also reconstructs the retained S0 scan. `tests/test_privacy_reporting.py` verifies compact aggregate reporting and binds the retained V0.1 privacy report to the current corpus manifest and its modality/source document counts.

Known residual limitations include obfuscated or novel credential formats, names and postal addresses, context-dependent identifiers, secrets split across fields or records, encoded/encrypted payloads, and semantic PII not recognizable by the bounded patterns. Absence of detections must never be promoted to a universal privacy-clean claim; zero detections means only that no configured high-confidence detector fired.
