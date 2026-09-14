# R01 200M Feasibility Packet V1

This package implements the executable evidence envelope required by issue #548
after terminal learned-20M qualification and before any ~200M authorization
request. It is a consumer of the canonical R01 accelerated-scaling router, not a
second scaling policy.

## Gate

`build_200m_feasibility_packet()` first calls the canonical
`validate_roadmap()` and `assess_roadmap()` functions. A packet can be prepared
only when the roadmap proves terminal learned-20M and routes exactly to
`PREPARE_200M_FEASIBILITY_PACKET`.

The checked-in roadmap currently remains before that transition. Therefore this
package does not make current learned-20M or ~200M work ready by itself.

## Evidence envelope

The packet binds all canonical `REQUIRED_200M_FEASIBILITY` dimensions, the full
canonical learned-20M measurement set, candidate architecture/parameter facts,
the exact roadmap snapshot, and exact learned-20M terminal/audit authority.

Evidence references are closed-world terminal references containing repository,
Git SHA, evidence SHA-256, workflow run ID, successful conclusion, and terminal
state. In addition:

- the measurement authority SHA-256 must equal the canonical hash of the exact
  measurement record;
- the candidate-architecture evidence SHA-256 must equal the canonical hash of
  the exact candidate record;
- every feasibility dimension has its own externally rooted evidence identity;
- numeric measurements reject bool aliases, NaN/Inf, and invalid non-positive
  values where positivity is required.

The packet carries a self-hash for corruption detection, but the self-hash is
not sufficient authority.

## Independent expectations

A verifier must receive expected identities from an independent retained source.
Do **not** derive the expected identities from the packet being verified and then
treat that as independent validation.

`expected_external_identities(packet)` is a transport helper for a trusted
builder/operator to persist those identities separately at build time. The
validator requires the retained packet hash, roadmap-snapshot hash, source Git
SHA, measurement hash, and every requirement-evidence hash. A coherently edited
and resealed packet therefore fails against stale retained expectations.

## CLI

Build:

```text
PYTHONPATH=src python tools/build_200m_feasibility_packet.py build \
  --roadmap configs/research/r01_accelerated_scaling_roadmap_v2.json \
  --input /path/to/build-input.json \
  --output /path/to/feasibility-200m.json \
  --external-identities /path/to/feasibility-200m.expected.json
```

The build input must contain exactly:

```text
source_git_sha
candidate
measurements_20m
measurement_authority
requirement_evidence
decision
```

Verify later against independently retained identities:

```text
PYTHONPATH=src python tools/build_200m_feasibility_packet.py verify \
  --roadmap /path/to/exact-roadmap-snapshot.json \
  --packet /path/to/feasibility-200m.json \
  --expected-identities /trusted/path/feasibility-200m.expected.json
```

The verify command emits one machine-readable JSON line and returns `0` only for
an error-free packet. Duplicate JSON keys and non-finite JSON constants fail
closed at the CLI boundary.

## Authority boundary

A prepared or validated packet is engineering/scientific feasibility evidence.
It does not freeze ModelSpec, authorize tokenizer fit, select or promote a
backend, promote a stage, authorize compute, authorize paid compute, authorize
material training, execute optimizer updates/training, create learned weights,
read final-test outcomes, introduce foreign pretrained weights, or authorize an
external LLM/API for data or intelligence.

A later canonical R01 roadmap transition still requires independently rooted
feasibility authority and any explicit training/compute authorization required
by the project controls. No paid compute is authorized by this package.
