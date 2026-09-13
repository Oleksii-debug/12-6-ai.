# AUDIT-754 — MODEL-341 learned-20M launch-authority red-team

Authority: SWARM-754, control #723, parent critical path #548, canonical gate issue #654 and merged PR #714.

This is an independent audit package. It does not modify the Product readiness evaluator, launch packet, model, data, tokenizer, checkpoint, evaluation, optimizer, compute policy or workflows. It performs no training and uses no paid compute.

## Exact source binding

- live-main audit base: `5020afd671a3885c1b738c8b4eafe7525f630546`
- PR #714 final head: `abd9a3771e30a17ed9a956430b4d5ea1f8df8521`
- PR #714 merge commit: `f13e657832953b59049aa6fcbfae4b7a3c684272`
- readiness source blob: `9baaa2c201f4f28d3776908cf9939bf7f22eeab5`
- launch packet blob: `753c906ef053b997f4518ad825688dc03037ea73`
- upstream readiness tests blob: `6313af878a5cda42e0d4340e6b8abf253e139a3b`
- MODEL-341 mechanics: `e4ff486fd90802fc123bebf60eed4e59196a98df`
- ModelSpec SHA-256: `fbff24d561a2818453554d58ca23fc6ace3303b078f1935a8576c4565bd92441`
- parameters: `20,613,440`

## Verdict

`CHANGES_REQUIRED`.

The current checked-in packet remains blocked, and the phase separation is directionally strong. However, the machine gate does not yet satisfy issue #654's fail-closed requirement for stale/self-asserted evidence, and numeric/type adversarial cases can reach false authorization or crash instead of returning blockers.

### AUDIT754-001 — HIGH — self-asserted scientific evidence can satisfy the gate

The evaluator validates scientific authority objects structurally: repository string, 40-hex Git SHA, 64-hex evidence hash, `terminal=true`, and, for workflow-backed objects, a positive run ID plus the string `success`. It does not require those corpus/tokenizer/ledger/checkpoint/evaluation/recipe/pilot/scale/audit references to be present in an independently verified live-evidence set.

The red-team harness therefore fills every scientific prerequisite with synthetic exact-looking identifiers and then supplies only the two out-of-packet compute/training decision references. On the audited implementation this reaches `material_training_authorized=true`.

The documentation correctly says structural evidence is not a substitute for live reconciliation, but that requirement is not machine-enforced by the evaluator that returns the authorization boolean. A future launcher can therefore misuse a syntactically valid packet as launch authority.

Required repair: consume an independently verified authority set/resolver for every scientific reference, binding role, exact Git SHA, evidence digest and terminal workflow result. Syntactically valid nonexistent/stale/self-authored references must keep all affected phases false.

### AUDIT754-002 — HIGH — NaN estimated maximum cost bypass

`maximum_cost_usd=float('nan')` is accepted as a numeric value. In Python, comparisons with NaN are false, so neither `maximum_cost <= 0` nor `authorized_limit < maximum_cost` blocks. With otherwise passing evidence and verified decision refs, material training can be reported authorized without a finite estimate.

Required repair: require `math.isfinite(value) and value > 0` before any cost participates in readiness or authorization.

### AUDIT754-003 — HIGH — NaN authorized limit bypass

An authorized `maximum_cost_usd=float('nan')` also passes the numeric type check, and `nan < finite_estimate` is false. The result can therefore report material-training authorization with no finite authorized ceiling.

Required repair: require a finite positive authorized limit and then require `authorized_limit >= finite_estimate`.

### AUDIT754-004 — MEDIUM — boolean `seed_count` passes

`True` is an `int` subtype in Python. The current `seed_count < 1` check therefore accepts `seed_count=True` as a valid seed plan.

Required repair: use the same explicit positive-integer predicate that already rejects booleans for other integer fields.

### AUDIT754-005 — MEDIUM — malformed `seed_count` crashes

A value such as `seed_count="2"` reaches a direct `< 1` comparison and raises `TypeError`. A launch-readiness gate should return a deterministic blocker for malformed input rather than crash.

Required repair: type-check first and return a stable blocker for booleans, strings, floats, null, zero and negatives.

## Positive evidence

The current committed packet is deliberately blocked. Unique post-pack loss positions are separated from total exposure, replay is capped, learned 3M/10M evidence is required before a material-compute request, bounded-pilot evidence is separate, and compute/training decision references must be externally verified and distinct. These properties should be preserved during repair.

## Reproduction

From a checkout containing the audited gate:

```bash
PYTHONPATH=src python tools/audit754_model341_launch_authority.py
```

The command emits deterministic JSON with source bindings, adversarial cases, findings, truth boundaries and `report_sha256`. The audit command returns success when the audit itself completes; its `verdict` field carries the Product verdict.

## Not tested / truth boundary

- Local cloning of GitHub was unavailable in the auditor sandbox because DNS resolution for github.com failed; connected GitHub remained readable/writable and is the authority used for exact source binding.
- Exact canonical integration is intended to run in the repository's existing shared CI after the audit branch is published. Queued/running CI is not PASS.
- No GPU/CUDA evidence, training quality, learned-20M capability, external-real data quality, release promotion or paid-compute authorization is claimed.
