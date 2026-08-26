# DATA-297 External-Real Privacy Audit

DATA-297 audits the complete terminal training-eligible external-real inventory at its fixed cutoff with the existing DATA-33 privacy authority. It does not create a second privacy framework and does not add or tune detectors before measurement.

## Authority boundary

Privacy authority is DATA-33 source `290b82fd0f7d1cc3a1840deae4378b9c500f1c15`, policy `12-6.pii-secrets-policy.v1`, identity `8c905e3b8f81391c3f928f375bca8fe6d1b5d38b41dec5c081577e2c5ce58526`.

The terminal source inventory composes DATA-229 text evidence with the later terminal DATA-227 code authority. DATA-228 remains nonterminal for this purpose and is explicitly excluded rather than promoted by inference.

Training-eligible input capacity at the audit boundary is 183,061 UTF-8 bytes across five source objects and four independent source families:

- `en.standardebooks.manual`: 84,793 bytes across two English text objects.
- `ua.rada.open-data.laws-texts`: 88,565 bytes across one Ukrainian text object.
- `github:encode/httpx`: 8,161 bytes across one Python code object.
- `github:psf/requests`: 1,542 bytes across one Python code object.

## What is measured

The exact DATA-33 pass is applied to every admitted object. The sanitized report records per-source and per-family input bytes, retained training bytes, dropped bytes, action counts and detector counts. REDACT remains train-eligible under DATA-33; QUARANTINE and EXCLUDE contribute zero retained training bytes.

A separate labeled challenge panel is assembled only at runtime. It covers configured PII and credential classes plus private user paths and code-oriented patterns such as environment assignments, authentication headers, credential-bearing connection URIs and package/service token shapes. Benign near-miss fixtures measure false positives. The audit records TP/FN/TN/FP counts and rates without retaining fixture payloads.

The fixture panel is an audit of bounded detector coverage, not an estimate of real-world secret prevalence. A false negative identifies a pattern class not caught by the incumbent authority; it does not authorize a post-hoc policy change inside this audit.

## Public evidence safety

No matched value, text preview, raw source text, sanitized text or fixture payload is written to DATA-297 public evidence. Reports retain only public source/family identifiers, hashes, detector IDs, actions, statuses, counts and byte accounting. DATA-33's structural evidence guard is applied to the DATA-297 report as well.

The workflow downloads exact terminal source artifacts by immutable artifact IDs and digests, performs the scan in the job workspace, removes materialized source bytes, and uploads only the sanitized DATA-297 JSON reports and input authority configuration.

## Reproducibility

Run the fixture-only audit with:

`PYTHONPATH=src python -m twelve_six.data297_privacy_external_audit fixtures`

The full external-real audit is executed by `.github/workflows/data297-privacy-external-audit.yml`, which binds the exact pull-request head and DATA-33 ancestry before reading the terminal DATA-213/DATA-227 artifacts.

Execution is LOCAL_FREE only. DATA-297 makes no claim of universal PII/secret detection, no claim that an ALLOW result proves absence of private material, and no model-result claim.
