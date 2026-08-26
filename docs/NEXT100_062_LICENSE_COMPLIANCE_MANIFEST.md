# NEXT100-062 License Compliance Manifest

Worker: `NEXT100-062-LICENSE-COMPLIANCE-MANIFEST`

This authority records license-compliance obligations for every terminal training-admitted source visible at the sealed authority vector. It does not create, broaden, or relax source rights.

## Terminal source set

The bound DATA-287/DATA-293 baseline contributes five exact objects: Rada laws text, two Standard Ebooks manual objects, HTTPX `_content.py`, and Requests `_internal_utils.py`. The final concurrency scan also found NEXT100-022 Ukrainian Wikisource at an exact-head dedicated terminal-success workflow, so its bounded page snapshot is included as the sixth source.

Queued, RETEST, rejected, or otherwise nonterminal concurrent source candidates are not promoted merely because their license appears compatible.

## Compliance decisions

- Rada: redistribution requires source attribution/link. A deterministic attribution sidecar is committed.
- Standard Ebooks selected RST files: CC0-1.0; no attribution, NOTICE, or ShareAlike obligation is imposed on the selected objects. The `build-manual.py` GPL exception remains out of scope.
- HTTPX: BSD-3-Clause copyright notice, conditions, disclaimer, and non-endorsement boundary are retained in an exact license sidecar.
- Requests: Apache-2.0 license delivery, change notices for modified files, applicable source-form notices, and the exact upstream NOTICE are retained. The upstream NOTICE is `Requests / Copyright 2019 Kenneth Reitz`.
- Ukrainian Wikisource bounded 1892 page: the underlying edition is public domain, but the terminal authority conservatively preserves Wikimedia contributor-layer attribution plus CC BY-SA 4.0 and/or GFDL obligations as applicable. The exact reuse-license path and ShareAlike fulfillment are not yet materialized, so redistribution is blocked rather than guessed.

The Wikisource redistribution blocker does not revoke its terminal model-training permission. It is a packaging/compliance blocker only. Its separate corpus-selection near-match blocker is also preserved.

## Purpose firewall

This manifest preserves model-training and redistribution decisions from the terminal source authorities. It never infers evaluation, tokenizer fitting, selection-validation, final-test, or another project purpose from an upstream license or a training verdict. Those purposes require separate authority.

## Fail-closed execution

`python tools/validate_next100_062_license_compliance_manifest.py validate-manifest` validates the exact source set, baseline registry identity, source/license identities, terminal authority bindings, required fields, retained evidence, and Git-blob identities of every required sidecar.

`python tools/validate_next100_062_license_compliance_manifest.py check-training` verifies that the manifest preserves every terminal training verdict.

`python tools/validate_next100_062_license_compliance_manifest.py check-redistribution` intentionally fails until all redistribution obligations are resolved. At this authority vector it fails on the unresolved Wikisource ShareAlike/GFDL reuse path.

The dedicated workflow is stdlib-only and `LOCAL_FREE`. No model training or paid compute is performed.
