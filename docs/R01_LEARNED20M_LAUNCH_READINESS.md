# R01 learned-20M launch readiness

Issue: #654.

This package converts the learned-20M critical path into one fail-closed integration gate. It does not implement DATA, tokenizer, checkpoint, evaluation, optimizer, audit or compute authorities; it consumes their terminal evidence only after those lanes produce it.

The evaluator deliberately exposes three different decisions:

1. `ready_for_local_free_pilot` means the exact model/code/data/tokenizer/loss-ledger/checkpoint/evaluation/training-recipe prerequisites are bound strongly enough for a bounded no-material-cost pilot.
2. `ready_for_compute_authorization_request` additionally requires terminal bounded-pilot evidence, a positive cost envelope and an independent audit. This is permission to ask for material compute, not permission to spend it.
3. `material_training_authorized` additionally requires a durable explicit `COMPUTE_AUTHORIZED` authority whose maximum budget covers the estimated maximum cost.

No lower phase implies a higher phase.

## Exact incumbent model authority

The gate binds the current MODEL-341 mechanics control exactly: branch `model341/20m-candidate-a-20260826`, SHA `e4ff486fd90802fc123bebf60eed4e59196a98df`, ModelSpec SHA-256 `fbff24d561a2818453554d58ca23fc6ace3303b078f1935a8576c4565bd92441`, and 20,613,440 random-init parameters.

It also binds the merged R01 campaign config by Git blob SHA-1 `c50154db609d41eceb2ffc97912360df567bcc04` so a future campaign-policy replacement cannot silently inherit this launch packet.

## Evidence contract

Terminal scientific authorities are machine-addressable GitHub evidence references with exact repository, Git SHA, evidence SHA-256 and terminal state. Gates backed by Actions additionally require a positive workflow run ID and `workflow_conclusion=success`. The evaluator never treats queued, running, cancelled or failed workflows as PASS.

The current checked-in packet is intentionally blocked. Null identities and zero unique loss positions are the truthful state until successor lanes populate exact terminal evidence.

A structural evidence reference is not a substitute for live GitHub reconciliation. The integration worker that fills a future launch packet must verify each referenced head/run/artifact against GitHub before publication. This module prevents missing or contradictory evidence from becoming launch authority; it does not fabricate network verification.

## Data-budget boundary

The gate does not convert source bytes, raw tokenizer tokens or replayed positions into training capacity. It consumes a separate terminal `data_budget_authority` and requires `data_budget_status=QUALIFIED` plus a positive exact post-pack unique causal-loss-position ledger. The active R01 scaling-data-budget lane remains the owner of the numerical sufficiency policy.

## Compute boundary

The package executes no training and no paid compute. It distinguishes cost estimation from authorization and verifies that the explicit authorized ceiling is at least the estimated maximum cost. A general owner message such as `continue` cannot satisfy this field.

## Usage

From a repository checkout:

```bash
PYTHONPATH=src python tools/assess_r01_learned20m_launch_readiness.py
```

Exit code `0` is reserved for a packet that reaches `material_training_authorized=true`. A correctly blocked current packet exits `1` and prints exact blockers as JSON.
