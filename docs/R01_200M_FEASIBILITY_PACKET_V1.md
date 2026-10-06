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

`retained_identities_for_built_packet(packet)` is a transport helper only for a
trusted builder/operator to persist those identities separately immediately after
building the packet. It must never be called on a packet supplied for verification. The
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

The build command requires `--external-identities`. A successful CLI build
therefore always has an independently retained identity-map destination; stdout
alone is not treated as sufficient retention. The identity map is published
before the packet, so an identity-publication failure cannot expose a new packet.

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
closed at the CLI boundary. Invalid packet hashes are never echoed into the
machine-readable report, and roadmap contract failures do not echo
attacker-controlled roadmap identifiers.

The retained-identity helper validates the trusted built packet before deriving
the identity map, rejects scalar type aliases in authority fields, and is not a
verification primitive. Programmatic build/verify paths accept only exact built-in
JSON trees: plain dictionaries with plain string keys, plain lists, JSON scalar
builtins, and null. Caller-defined container/scalar subclasses are rejected before
semantic access, hashing, or defensive copying, so custom iteration/get/deepcopy
callbacks cannot change authority state inside the trusted boundary. Cyclic,
shared-container/aliased, and non-canonical programmatic inputs fail closed
without becoming packet authority.

Build and verify also reject canonical path collisions between their authority
inputs and outputs. Authority inputs must name regular files directly: symbolic
links are rejected before open, and the reader requests no-follow semantics where
the platform exposes them. Output publication renders before touching the
destination, rejects non-regular existing destinations, fsyncs a same-directory
temporary regular file, and uses atomic replacement. A publication failure
therefore does not truncate the previous destination.

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
