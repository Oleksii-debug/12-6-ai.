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

The verifier first applies the canonical portable-run packet contract and then
checks the local machine. A blocked launch template can therefore be inspected
without pretending that launch readiness exists, while forbidden authority
drift, embedded secrets, final-test access, unsafe checkpoint policy, foreign
pretrained/aligned weights, or paid-compute drift fail closed.

Windows 11 admission uses native Windows version facts from the Python standard
library, not the compatibility-facing `platform.release()` label. The current
operator floor is NT 10.0 build 22000 or newer. Thus Windows 11 may correctly
pass even when Python displays release `10`, while Windows 10 build 19045 is
blocked.

The verifier also requires a visible CPU path and conservative local operator
headroom. The RAM/disk/CPU floors are admission policy only. A PASS does not
claim that full learned-20M training will fit or finish at a particular speed;
measured resource-envelope evidence remains separate authority.

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

Trust-bearing profile, portable-packet, and stop-marker JSON is decoded
fail-closed: duplicate object keys and non-finite numbers such as NaN or
Infinity are rejected rather than normalized or silently overwritten.

## Safe stop request

Request a checkpoint-safe stop without editing trainer/checkpoint internals:

```text
python -m twelve_six.windows_operator_preflight request-stop --target 20m
```

The command creates `.twelve-six-local/STOP_REQUEST.json` exactly once. The
marker is content-authenticated, binds the exact operator-profile and portable
packet bytes, is idempotent for the same binding, and fails closed on corruption,
symlinks, duplicate JSON keys, non-finite JSON values, or binding drift.

The marker is only a **request** for the canonical trainer to consume. Its
presence does not mean that a checkpoint was written, that resume was verified,
that training executed, or that any optimized target was exposed.

## Exit codes

- `0`: preflight pass / 100M qualification-only result / valid stop request
- `2`: machine preflight blocked
- `3`: malformed contract, unsafe path, corrupt marker, or other fail-closed error

The profile preserves `LOCAL_FREE`, random-init/no-foreign-pretrained boundaries,
zero learned-target optimizer updates, no paid compute, and no final-test access.

This machine/operator leaf deliberately makes **no global claim** that the
training corpus is clean of external-LLM/API-derived material. Corpus provenance
and any decontamination/rebuild authority belong to upstream data lineage; this
preflight neither widens nor contradicts them.
