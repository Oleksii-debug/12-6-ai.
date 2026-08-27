# Hydra configuration qualification V1

Status: **CANDIDATE_NOT_ADOPTED**

Owner/claim: `SWARM-749`, issue `#749`  
Parent research authority: issue `#720`  
Live project base used by this package:
`5020afd671a3885c1b738c8b4eafe7525f630546`

## Decision

Hydra is being evaluated only as an optional experiment-configuration convenience layer.
It is **not** canonical lineage authority, checkpoint authority, run authority, compute
authorization, or stage-promotion authority.

The project continues to treat exact Git SHA plus D05 run/checkpoint manifests as the
authoritative lineage. A Hydra output directory, working-directory layout, generated
`.hydra` metadata, job identifier, or runtime state must never replace those project
records.

This package proves the project-owned identity/provenance contract mechanics without
installing or executing Hydra. It therefore does not claim Hydra runtime parity, Hydra
adoption, model-training evidence, or any performance benefit.

## Exact upstream identity checked

The project research registry on the package base identifies component `HYDRA` as
`P1_CONFIG_SYSTEM_CANDIDATE`.

Independent upstream check performed for this qualification:

- repository: `https://github.com/hydra-ecosystem/hydra`
- release: `v1.3.5`
- release commit: `51647a2183512bc4e5556842c494e7efdbd75375`
- license at that commit: MIT
- published release timestamp: `2026-08-05T18:32:10Z`

The machine contract binds the project registry blob
`d80a60357c56eacac135f948b8a72556bb849e5a`. If that registry changes, this evidence
does not silently follow it; a successor qualification must deliberately rebind and
retest.

## Why the contract is fail-closed

Configuration frameworks can create reproducibility gaps when defaults order, command
line overrides, config-group selections, environment interpolation, generated output
directories, or runtime mutations are treated as incidental. For 12-6 those are
identity-bearing inputs when they affect a run.

The contract therefore requires:

1. every composed default to have a stable path and SHA-256;
2. every identity-affecting override to appear in one ordered ledger;
3. only explicit CLI, config-group, or explicit sweep overrides;
4. runtime environment interpolation to be absent from the identity path;
5. the fully resolved configuration to be exported and SHA-256 bound;
6. a clean rebuild to reproduce the same resolved-config hash;
7. a portable canonical-JSON export independent of Hydra runtime directories;
8. exact Git and project authority binding;
9. no secret-bearing metadata in the reproducibility record; and
10. explicit preservation of Git/D05 authority.

A failure of any gate rejects the evidence. There is no "best effort" promotion state.

## Identity model

Canonical JSON means UTF-8 JSON with recursively sorted object keys, no insignificant
whitespace, no NaN/Infinity, and JSON-native scalar/list/object values.

The resolved configuration receives its own SHA-256. The portable export binds:

- exact base Git SHA;
- ordered defaults trace with per-entry SHA-256;
- ordered override ledger; and
- the resolved configuration.

The experiment identity then binds:

- qualification contract SHA-256;
- project base Git SHA;
- project registry blob SHA;
- pinned Hydra upstream commit; and
- portable export SHA-256.

Consequently an explicit override that changes the resolved configuration produces a
different experiment identity. Hidden overrides are rejected rather than folded into a
new identity after the fact.

## Project-owned fixture

`configs/research/hydra_config_qualification_v1.json` contains a small synthetic config
fixture. It is configuration mechanics evidence only. It is not training data and does
not authorize a model run.

The fixture demonstrates a deterministic defaults trace, one explicit CLI override,
resolved-config hashing, clean rebuild equality, and portable export hashing.

## Validator and deterministic evidence

Validate the static contract:

```bash
python tools/validate_hydra_config_qualification.py
```

Build evidence from the embedded project-owned fixture:

```bash
python tools/validate_hydra_config_qualification.py \
  --use-contract-fixture \
  --output reports/d05/hydra_config_qualification_v1_evidence.json
```

The evidence intentionally contains no wall-clock timestamp, hostname, process ID, or
temporary path. Identical input must therefore produce byte-identical semantic evidence.
The output has a self-hash over the evidence core.

Run focused and adversarial tests:

```bash
pytest -q tests/test_hydra_config_qualification.py
```

Run lint on the owned Python surfaces:

```bash
ruff check tools/validate_hydra_config_qualification.py \
  tests/test_hydra_config_qualification.py
```

## Adversarial coverage

The tests reject, among other cases:

- making Hydra canonical lineage authority;
- letting Hydra gate stage promotion;
- mutable/drifted upstream release identity;
- malformed or drifted registry binding;
- paid-compute or model-training authority escalation;
- hidden/implicit overrides;
- environment-derived identity state;
- secret-bearing evidence;
- unknown or duplicate override sources;
- missing default hashes;
- resolved-config hash mismatch;
- clean-rebuild mismatch;
- portable-export mismatch;
- wrong project base Git SHA; and
- any truth-boundary claim that Hydra actually executed when it did not.

The tests also prove deterministic evidence and that a declared override changing
`training.steps` changes the experiment identity.

## What remains before `PARITY_PROVEN`

This package is not sufficient for `PARITY_PROVEN` or `ADOPTED`. A successor must run
Hydra itself in an exact dependency-locked environment and demonstrate, on project-owned
real run configs, that:

- Hydra composition resolves to the same intended project config as the first-party
  canonical representation;
- clean independent processes reproduce the exact resolved-config identity;
- supported multirun/sweep use cannot introduce unrecorded identity-bearing state;
- path/output-directory behavior cannot leak into run identity;
- portable export/rollback can reproduce a run without Hydra-specific state;
- dependency/SBOM and license/notice requirements are exact; and
- any operational benefit is measured without changing model/data/training semantics.

Only then can an integration owner decide whether Hydra should advance beyond
`CANDIDATE`. Adoption still does not give Hydra authority over Git/checkpoint manifests,
compute authorization, evaluation firewalls, or stage promotion.

## Truth boundary

No Hydra installation or runtime execution was performed by this package. No model was
trained. No optimizer, tokenizer, corpus, checkpoint, benchmark/final-test payload, or
canonical Base weight was changed. No GPU was provisioned and no paid compute was used.
Canonical Base remains random-init/pretraining-only under existing project authority.
