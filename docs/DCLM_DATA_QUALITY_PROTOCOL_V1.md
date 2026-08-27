# DCLM data-quality protocol V1

## Scope

This package independently qualifies a **standardized-comparison methodology** for 12-6 data-quality/filter/mixture research. It is research-only and fail-closed. It does not admit a corpus, authorize training, alter canonical Base weights, access final-test payloads, or authorize paid compute.

Canonical project bindings at claim time:

- `SWARM-300-V2` control, live `main` `5020afd671a3885c1b738c8b4eafe7525f630546`.
- `configs/research/open_source_reuse_registry_v2.json`, Git blob `d80a60357c56eacac135f948b8a72556bb849e5a`, registry ID `OPEN-SOURCE-REUSE-REGISTRY-V2`.
- Registry component `DCLM`, decision `P0_DATA_QUALITY_EXPERIMENT_PROTOCOL`, allowed surface: filtering experiments, mixture experiments, standardized comparison.
- Upstream `mlfoundations/dclm` `main` at `361714bdd60bb9b7f4b2d8354cebbf0dec0c329e`, repository metadata license `MIT`; README blob `a70356ff43ad2e579b241eca3a5816dac2207f33`.

## Confirmed source facts

At the exact upstream identity above, the DCLM README describes DataComp-LM as a framework for building/training language models with standardized data-processing, training, and evaluation machinery. Its submission workflow distinguishes filtering a data pool or mixing participant-provided data, then training with standardized code/scale-specific hyperparameters and evaluating the resulting model. The README also warns that corrected CORE/EXTENDED baseline calculations make pre-September-2025 and newer aggregate values not directly comparable without version labeling.

These are upstream facts only. They are **not** evidence that a DCLM dataset, model, score, or dependency is suitable or authorized for canonical 12-6 use.

## Project inference implemented here

For 12-6, the useful transferable idea is controlled comparison rather than imported results. `DCLM-DATA-QUALITY-PROTOCOL-V1` therefore requires:

1. at least two preregistered arms;
2. immutable SHA-256 identities for each arm configuration and the shared input snapshot;
3. the same comparison budget and budget unit across arms;
4. one preregistered metric name and direction;
5. explicit PASS gates for rights, provenance, privacy, and contamination;
6. a fail-closed ambiguous-tie rule;
7. an explicit non-training calibration purpose;
8. a recommendation state limited to `CANDIDATE_RECOMMENDATION_ONLY`.

The validator rejects automatic `ADOPTED` state, training authorization, non-finite metrics, mutable/unhashed inputs, unequal budgets, duplicate arms, metric-direction drift, missing/extra hard gates, and failed hard gates.

## Synthetic calibration evidence

`evidence/research/dclm_data_quality_synthetic_v1.json` is original project-owned synthetic evidence. The scores and hashes are calibration fixtures only. They do not represent corpus quality, model quality, benchmark performance, or executed upstream DCLM experiments.

The deterministic example report is `evidence/research/dclm_data_quality_synthetic_report_v1.json`. A winner in that report means only that the validator/comparator can apply the preregistered rule to a controlled fixture. It has no training-authority or adoption effect.

## Operator use

Validate and emit a report:

```text
python tools/validate_dclm_data_quality_protocol_v1.py \
  --protocol configs/research/dclm_data_quality_protocol_v1.json \
  --evidence evidence/research/dclm_data_quality_synthetic_v1.json \
  --output /tmp/dclm-report.json
```

Run focused/adversarial tests:

```text
python -m unittest tests/test_dclm_data_quality_protocol_v1.py
python -m py_compile tools/validate_dclm_data_quality_protocol_v1.py tests/test_dclm_data_quality_protocol_v1.py
```

Before substituting real project evidence, operators must bind an exact source/corpus snapshot, exact arm configurations, a calibration-only evaluation identity that is not training material, and a comparable budget. Rights/provenance/privacy/contamination gates must already be independently justified; this tool does not manufacture those facts.

## Future experiment proposals

The next scientific step is a **separate** preregistered experiment using current terminal Research Corpus authority and a project-approved held-out calibration split. Candidate filter/mixture arms should keep model/training controls fixed and report raw per-arm evidence plus exact identities. Learned-model quality claims require actual executed training/evaluation evidence under the relevant training and evaluation owners; this package alone cannot supply them.

If upstream DCLM, the project reuse registry, SWARM protocol, or comparison metrics change, create a new protocol version and rebind exact immutable identities rather than silently mutating V1 semantics.
