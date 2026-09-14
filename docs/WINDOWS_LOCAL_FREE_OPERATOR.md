# Windows LOCAL_FREE operator preflight

This is the operator-facing Windows 11 preflight for the canonical portable
LOCAL_FREE lineage. It is **not** a second trainer, a launch authorization, or
training evidence.

The supported owner path is CPU-first. AMD or other GPU acceleration is
optional and must be separately runtime-qualified; no GPU is required here.

## Verify

From a checked-out repository with the project Python environment available:

```text
python -m twelve_six.windows_operator_preflight verify --target 20m
```

For machine-readable output:

```text
python -m twelve_six.windows_operator_preflight --json verify --target 20m
```

The verifier requires Windows 11+, a visible CPU path, and conservative local
operator headroom. The RAM/disk/CPU floors are admission policy only. A PASS
does not claim that full learned-20M training will fit or finish at a particular
speed; measured resource-envelope evidence remains separate authority.

A 100M check is deliberately qualification-only:

```text
python -m twelve_six.windows_operator_preflight verify --target 100m
```

It cannot authorize or guarantee 100M training, runtime, or speed.

## Status

```text
python -m twelve_six.windows_operator_preflight status --target 20m
```

Output is line-oriented, color-free, and suitable for keyboard/NVDA use. JSON
mode is available with the global `--json` flag.

## Safe stop request

Request a checkpoint-safe stop without editing trainer/checkpoint internals:

```text
python -m twelve_six.windows_operator_preflight request-stop --target 20m
```

The command creates `.twelve-six-local/STOP_REQUEST.json` exactly once. The
marker is content-authenticated, binds the exact operator-profile and portable
packet bytes, is idempotent for the same binding, and fails closed on corruption,
symlinks, or binding drift.

The marker is only a **request** for the canonical trainer to consume. Its
presence does not mean that a checkpoint was written, that resume was verified,
that training executed, or that any optimized target was exposed.

## Exit codes

- `0`: preflight pass / 100M qualification-only result / valid stop request
- `2`: machine preflight blocked
- `3`: malformed contract, unsafe path, corrupt marker, or other fail-closed error

The profile preserves `LOCAL_FREE`, random-init/no-foreign-pretrained boundaries,
zero learned-target optimizer updates, no paid compute, and no final-test access.
