# DATA-287 External Snapshot Registry V2

`DATA-287-EXTERNAL-SNAPSHOT-REGISTRY-V2` supersedes the DATA-229 inventory at the task cutoff `2026-08-26T12:00:55Z`. It is a registry convergence layer only; it does not acquire new data or relax upstream rights decisions.

## Terminal admission boundary

The registry consumes only terminal Wave-1 admission evidence:

- DATA-213: terminal success. Carries the three DATA-229 real text snapshots: one Ukrainian Verkhovna Rada object and two English Standard Ebooks objects.
- DATA-227: terminal success at `8ebdb2e132ed7bae5245e9d4c140752640ab9885`, workflow `32956209865`. Carries exactly two D03-admitted Python objects from independent `encode/httpx` and `psf/requests` families.
- DATA-228: terminal failure at `46a70c990dab6ff72bb84ddb54cff1156b491b40`, workflow `32957120454`. Its Kubernetes/Python-documentation candidates are excluded because immutable admission evidence was not retained.
- DATA-23 `itsdangerous` and `pluggy` pilot objects remain historical rights-blocked evidence. DATA-227 explicitly did not reinterpret them.

No failed or review-only candidate is promoted by this registry.

## Canonical inventory and bytes

The V2 inventory contains five snapshots in four independent source families. Family-level accounting treats the two Standard Ebooks objects as one independent family and rejects mirror/fork admissions.

| Group | Snapshots | Unique raw bytes | Unique normalized bytes |
| --- | ---: | ---: | ---: |
| Ukrainian (`uk`, reporting alias UA) | 1 | 332,400 | 88,565 |
| English (`en`) | 2 | 106,111 | 84,793 |
| Python code | 2 | 9,703 | 9,703 |
| Text modality | 3 | 438,511 | 173,358 |
| Code modality | 2 | 9,703 | 9,703 |
| Total | 5 | 448,214 | 183,061 |

Independent family totals:

- `ua.rada.open-data.laws-texts`: 332,400 raw / 88,565 normalized bytes.
- `en.standardebooks.manual`: 106,111 raw / 84,793 normalized bytes across two exact objects.
- `github:encode/httpx`: 8,161 raw / 8,161 normalized bytes.
- `github:psf/requests`: 1,542 raw / 1,542 normalized bytes.

The DATA-227 code normalization is strict UTF-8 identity preservation, so its raw and normalized identities are byte-identical.

## Purpose-specific rights

Every admitted source keeps independent decisions for `model_training`, `evaluation`, and `redistribution`.

- Model training: `ALLOWED` for all five snapshots under their terminal source authorities.
- Redistribution: `ALLOWED` for all five snapshots, with source-specific license/notice conditions retained.
- Evaluation: `NOT_SEPARATELY_ADMITTED` for all five snapshots. Evaluation permission is not inferred from training permission, public availability, or redistribution permission.

A future evaluation authority must publish a separate admission/reservation decision; DATA-287 does not widen rights.

## Determinism and fail-closed rules

`tools/build_external_snapshot_registry_v2.py` builds the registry twice from the same committed inputs and requires byte-identical canonical JSON before either writing or verifying the artifact. Verification also requires the committed `data/registry/external_snapshots.v2.json` bytes to equal the independent rebuild.

The builder fails closed on DATA-229 text identity/version/family/license drift, non-terminal-success producers, DATA-228 consumption, incomplete purpose rights, widened evaluation rights, mirrors/forks, malformed hashes, and non-identity-preserving DATA-227 code normalization.

## Truth boundary

This is `LOCAL_FREE` registry work. It does not claim representative corpus coverage, universal benchmark cleanliness, evaluation authorization, production readiness, model capability, or corpus freeze. It only establishes the exact terminal rights-approved external snapshot inventory available at the stated cutoff.
