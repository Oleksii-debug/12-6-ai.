# Section 5 — Physical Qualification Fabric / device-server agent

This draft implements the first fail-closed Section-5 protocol surface. It is stacked behind
Section 4 and is not closure evidence for Sections 0–5.

## Signed exact packet

A qualification authority supplies a canonical
\`12-6.signed-physical-qualification-packet.v1\` envelope. The signed bytes bind:

- exact target Git SHA;
- exact SHA-256 of the qualification-agent source;
- REAL_HOST versus SIMULATION execution mode;
- allowed Windows/Linux host families;
- a bounded validity window;
- canonical, no-shell pytest actions with per-action timeout/output limits;
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

CPU/RAM/disk and CUDA GPU can be proven by the v1 local probes. Network, model and provider
execution are deliberately NOT_PROBED until an authoritative adapter exists. A signed packet that
requires one of those resources therefore cannot receive physical PASS.

SIMULATION is a first-class evidence mode. Even if every test vector passes, simulation receives
SIMULATION_PASS and the independent verifier rejects it when real physical PASS is required.

## Bounded execution and exact-tree binding

Only checked-in \`tests/...\` pytest targets are accepted. Shell command strings, parent traversal
and arbitrary executables are outside the v1 packet language. Action count, timeout, captured
stdout/stderr and artifact count/bytes are capped.

The agent probes the exact Git SHA and tracked/index cleanliness before the run and before/after
every action. A dirty checkout, changed HEAD, output overflow, non-zero test exit, missing required
real resource or artifact substitution prevents physical PASS.

The current subprocess implementation enforces timeout and evidence-output caps. A hardened
device-service transport with OS-level process/memory/disk quotas is still required before final
Section-5 closure; this draft does not pretend that Python pipe capture is an OS resource sandbox.

## Machine-verifiable evidence

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

## Remaining closure gates

Section 5 stays IN_PROGRESS. No real Windows/Linux/server device has been qualified by this draft.
Closure still needs predecessor Sections 0–4, exact-head shared CI, production ED25519 trust-store
integration, hardened host-service process bounds, real Windows/Linux/server runs, and authoritative
NETWORK/MODEL/PROVIDER adapters where those resources are required.

This code creates no corpus/tokenizer/training/final-test/paid-compute/scale-promotion authority and
does not manufacture a physical PASS from software simulation.
