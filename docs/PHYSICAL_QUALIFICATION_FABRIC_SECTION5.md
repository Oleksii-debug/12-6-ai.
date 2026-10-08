# Plan 8 / Section 4 — Physical Qualification Fabric / device-server agent

This is the Plan 8 Section 4 component engineering candidate, converged from historical
monolithic Section 5 PR #3083 onto the accepted Plan 8 Section 3 main lineage. This text is
not final Plan 10 device/NVDA/product qualification evidence.

## Signed exact packet

A qualification authority supplies a canonical
\`12-6.signed-physical-qualification-packet.v1\` envelope. The signed bytes bind:

- exact target Git SHA;
- exact SHA-256 of the qualification-agent source;
- REAL_HOST versus SIMULATION execution mode;
- allowed Windows/Linux host families;
- a bounded validity window;
- canonical, no-shell pytest actions with per-action timeout/output limits;
- exact host Python executable identity in host inventory/evidence;
- the real resource kinds each action requires;
- exact artifact paths whose bytes must be hashed.

The core intentionally does not contain a self-signing key, fallback signature acceptance or a
test key. It accepts only the ED25519 algorithm identifier and requires an injected trust-store
verifier to validate the detached signature before any action runs. A production OS/service
Ed25519 trust adapter remains a required integration gate.

## Host and runtime evidence

The agent inventories the actual local OS, machine, Python runtime, logical CPU count, available
RAM/disk and PyTorch/CUDA device state. Resource observations are explicit:

- REAL_PROBED;
- NOT_PRESENT;
- NOT_PROBED;
- SIMULATED.

CPU/RAM/disk and CUDA GPU can be proven by the v1 local probes. NETWORK/MODEL/PROVIDER can only
become REAL_PROBED through a bounded external adapter evidence object whose adapter id and exact
bytes pass a trusted resource-specific verifier during execution and again during independent
evidence verification. Without that authoritative adapter/verifier pair the resource remains
NOT_PROBED, so a packet that requires it cannot receive physical PASS.

SIMULATION is a first-class evidence mode. Even if every test vector passes, simulation receives
SIMULATION_PASS and the independent verifier rejects it when real physical PASS is required.

## Bounded execution and exact-tree binding

Only exact Git-index-tracked regular \`tests/...\` pytest files are accepted. The default runner
rejects untracked lookalikes, symlinks and repository escapes before launching pytest. Shell command
strings, parent traversal and arbitrary executables are outside the v1 packet language. Evidence
binds the exact host \`sys.executable\` from host inventory rather than a generic \`python\` alias.
Action count, timeout, captured stdout/stderr and artifact count/bytes are capped. Packet/evidence JSON,
JSONL logs and artifact hashing enforce those byte ceilings while reading; oversized sparse or hostile files
are rejected after at most the configured bound plus one byte instead of being loaded fully first.

The agent probes the exact Git SHA and tracked/index cleanliness before the run and before/after
every action. A dirty checkout, changed HEAD, output overflow, non-zero test exit, missing required
real resource or artifact substitution prevents physical PASS.

The default pytest runner also strips inherited `PYTHON*` and `PYTEST*` host overrides before
launch, then pins `PYTHONHASHSEED=0`, `PYTHONNOUSERSITE=1` and
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`. This prevents host environment variables such as
`PYTEST_ADDOPTS=--collect-only`, `PYTEST_PLUGINS` or `PYTHONPATH` from silently changing the
signed action semantics.

The current subprocess implementation enforces timeout and the signed stdout/stderr capture cap
*during* execution: two bounded drainers retain at most `max_output_bytes + 1` bytes per stream.
Each action is launched in an isolated POSIX session or Windows process group; timeout/output
termination kills the POSIX process group or uses the absolute System32 `taskkill.exe /T /F`
path on Windows before any direct-process fallback. A hardened device-service transport with
OS-level memory/disk quotas is still required before final Plan 8 Section 4 closure; process-tree
termination and bounded capture are not presented as a complete OS resource sandbox.

## Machine-verifiable evidence

The host service must also ED25519-sign the canonical evidence body with a separately trusted host
attestation key. A self-consistent JSON envelope plus recomputed hashes is not accepted. The
independent verifier validates that host signature and independently derives resource observations,
blocker reasons, action verdicts and the final qualification verdict from the signed body.

The evidence envelope cross-binds:

- signed packet identity and detached-signature identity;
- signing key id;
- agent source SHA-256;
- exact target Git SHA;
- host-inventory identity;
- per-resource real/simulated status;
- per-action logical argv, return code, timings, pre/post Git state and bounded output hashes;
- bounded JSONL log hash/length;
- artifact hashes/sizes;
- final verdict and explicit blockers.

\`verify_qualification_evidence()\` independently rechecks the envelope identity, packet/signature
binding, source hash, real-vs-simulation semantics, required resources, exact action set, clean Git
states, JSONL output hashes and artifact bytes.

## Operator path and trust boundaries

The keyboard/reader-friendly foreground command line is `tools/plan8_physical_operator.py`.
It supports `run`, `status` and `verify` with text/JSON responses, deterministic exit codes,
no mouse/coordinates/color dependencies and Ctrl+C for process-tree stop. The operator is
one-shot and does not expose a background service or arbitrary host command execution.
The packet allows exact signed tracked pytest files only; evidence is Ed25519 signed by a
separate host key and independently verified against an external host trust store.

Run from an exact clean checkout with `PYTHONPATH=src` configured (Windows PowerShell:
`$env:PYTHONPATH='src'`). The operator requires a separately approved installation of
`cryptography` for Ed25519; absence fails closed. Example invocation, with all four key files
and packet provisioned **out of band**, never auto-generated by the operator:

```text
python tools/plan8_physical_operator.py run --packet packet.json --authority-keys authority.json --host-keys host.json --repo-root . --receipt receipt.json --log receipt.jsonl --host-private-key ../trusted-keys/host.key --host-key-id approved-host-1
python tools/plan8_physical_operator.py status --receipt receipt.json
python tools/plan8_physical_operator.py verify --packet packet.json --authority-keys authority.json --host-keys host.json --repo-root . --receipt receipt.json --log receipt.jsonl
```

A trust store has exactly `schema_version: 12-6.physical-trust-store.v1` and a `keys` map
from authorized key IDs to base64-encoded 32-byte Ed25519 *public* keys. The host private
key is a 32-byte raw Ed25519 secret stored **outside** the checkout, with owner-only POSIX
permissions. Do not commit keys/receipts, and do not share a host private signing key with
packet authorities. Host signer identity is verified before any signed action executes.
`status` explicitly labels raw receipts UNVERIFIED, not PASS. `verify` requires true
physical PASS by default; simulation can be read back only with explicit
`--allow-simulation`, never silently promoted.

## Remaining closure gates

This branch remains a candidate until the updated exact-head scoped tests, static/CI checks,
accepted-main integration/readback and terminal registry/Drive record are completed.
Real Windows/Linux/server HIL, production provisioning/trust custody, OS-level resource
quotas and final product/NVDA acceptance are not claimed from local fixtures. A production
server run with real physical PASS requires actual host observations and authorized keys;
NETWORK/MODEL/PROVIDER resource checks fail closed without concrete trusted adapters.

This code creates no corpus/tokenizer/training/final-test/paid-compute/scale-promotion authority and
does not manufacture a physical PASS from software simulation.
