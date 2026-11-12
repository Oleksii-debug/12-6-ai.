# Plan 3 training runtime release

The [release descriptor](../configs/plan3/training_runtime_release_v1.json)
pins the incumbent model, recipe, numerical objective, trainer, controller,
checkpoint, exposure, portable packet and launcher sources. It binds their
Plan 9 data handoff to the accepted local Plan 2 S18 authority.

From a source checkout, verify the descriptor and its source digests. The
verifier normalizes CRLF to LF before hashing checkout files.

```bash
PYTHONPATH=src python -m twelve_six.training.runtime_release --root .
```

Use the same command with the path to the accepted local physical release
to reopen it through Plan 2's verified reader.

```bash
PYTHONPATH=src python -m twelve_six.training.runtime_release --root . \
  --physical-release "$RELEASE"
```

The handoff probe checks the exact digests of the aligned batch tensors from
the verified checkout. The batch has shape `[2, 32]` with 45 active positions.
The released Trainer objective must match the analytic FP64 reference for
logits with an elevated score at each active target.
It executes zero optimizer steps. Plan 9 owns campaign configuration and
authorization. The seven full-profile physical measurements in Section 10A
remain open.

The component qualification reruns the accepted numerical, checkpoint,
fresh-process interruption, exposure, portable packet, resource and launcher
suites against the exact release candidate. The launcher fixture migrates
through in-process, spawned and loopback transports and compares the final
checkpoint and exposure cursor with uninterrupted execution. The accepted
Git tree and executed evidence belong to the release qualification record.
