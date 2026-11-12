# Plan 8 Qualification Fabric 1.0.0

The [release descriptor](../configs/plan8/qualification_fabric_release_v1.json)
pins the accepted component Git blobs and SHA-256 digests for SIL, AI QA
repair, signed physical evidence, scenario and chaos execution, capability
evidence, lineage and integration. The component source is the accepted
Section 10 commit `65986e78989d13430dbc32590919d014ad0a9372`. The Section 11
acceptance record binds the executed qualification to the exact Section 11
candidate and accepted Git tree.

From a full Git checkout with the historical merge in its ancestry, reopen the
descriptor and the archived Linux evidence with the pinned release digest.

```python
from twelve_six.qualification_fabric_release import verify_qualification_fabric_release

release = verify_qualification_fabric_release(
    ".",
    "configs/plan8/qualification_fabric_release_v1.json",
    expected_sha256="de7110efb59f95af2386ca24f2e913dd2098ec11c3a5e05949c7af58b588d726",
)
```

Readback checks the component source tree, ancestry and the component blobs
at the checkout's committed `HEAD`. It also checks the candidate and accepted
trees of the historical Linux run, the archived ZIP and all three members.
Release acceptance uses the [Section 10 integration verifier](../src/twelve_six/integration_qualification.py)
and the trusted main reference.

## Reproduce the local campaigns

Run the local campaigns from a clean checkout with the project test
dependencies installed. Choose a fresh temporary directory with `--basetemp`
to retain the actual local fixture receipts and SQLite lineage for inspection.

```bash
python -m pytest -q tests/test_plan8_qualification_fabric_release.py \
  tests/test_plan8_chaos_harness.py::ChaosHarnessTests::test_complete_versioned_q0_campaign_with_real_local_effects
```

The fixture campaign executes 17 SIL vectors in an isolated Git repository,
captures a real failed pytest subprocess, derives a defect from the evidence
and applies the repair on a separate branch. It reruns the component and
adversarial regressions and all SIL vectors, reopens the retained lineage by
digest and verifies an actual fixture squash against the predicted tree. The
promotion decision is `BLOCK` because the local SIL contract fixture supplies
no admitted capability evidence. `local-campaign.json` records the run's
checkout SHA and tree and the digests of the failed and passing evidence,
lineage snapshot and signed physical simulation. Raw local receipts stay in
the temporary directory.

The signed proxy exercise reopens the packet, log and artifact with the
retained public keys. Its verdict is `SIMULATION_PASS` and the real-host gate
rejects that verdict. The separate Q0 chaos campaign exercises eight bounded
recovery cases with real local process, filesystem and socket effects.

## Historical Linux SIL

The original [synthetic archive](evidence/plan8/section3-sil-11762209803.zip)
comes from the [Ubuntu 24.04 qualification job](https://github.com/Oleksii-debug/12-6-ai./actions/runs/38285068328/job/114933484308)
for candidate `d8f6846b4c1ae9cf5095f48c09ac0cd0e2310108`. Its accepted merge
`94cb029f60aaaaffda9df288da6d3e7883cd0a98` directly contains that candidate and
has the same complete tree `b026b85c568afc4575a43594cbf6bbea8d112377`. The run
passed 17 vectors across 15 journeys on the archived CPython 3.11.16
environment. Its evidence applies to that historical SHA. Current release
qualification covers the repository and local fixture campaigns. Actual model
training belongs to Plan 9, and whole-product HIL and NVDA belong to Plan 10.
