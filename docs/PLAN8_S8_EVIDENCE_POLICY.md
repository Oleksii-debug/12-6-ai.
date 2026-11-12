# Plan 8 Section 8 evidence policy

The policy file declares a required evidence level for all 26 capabilities in
the accepted registry. Load it with the independently trusted file SHA-256 and
the matching registry. Capability test scope in the existing registry remains
a separate classification.

| Level | Evidence admitted |
| --- | --- |
| Q0 | Component or contract observation for a trusted scenario and exact input checkout |
| Q1 | Independently verified SIL receipt for the exact package and capability registry |
| Q2 | Signed HIL scope and reverified physical packet, host receipt and log |
| Q3 | Signed supervised physical operator evidence for the declared capability |
| Q4 | Signed field or canary evidence for the declared capability |

Each level has its own scope. A signed level uses the trusted key set for that
level and subject. A Q0 receipt records source evidence. Package promotion
readiness also requires package-bound evidence and the existing AI QA gate
decision. An affirmative result remains ready for independent promotion.

Current evidence must match the candidate Git SHA. Package evidence must also
match the exact package SHA-256. When either changes, old package records drop
out of readiness. Q0 source evidence drops out when the source SHA changes.
Registry changes invalidate capability evidence admitted under the older
registry.

Scale readiness reads the seven exact scale configs in the order 500M, 750M,
1B, 2B, 3B, 4B and 5B. It reports five separate states for each config.
`CONFIG_DEFINED` comes from the exact checked-in config. Proxy, host, training
authorization and learned qualification each require separately scoped evidence.
Host verification also rechecks the signed exact-profile packet and its CPU,
RAM and disk requirements. A CUDA claim additionally requires GPU evidence.
The report identifies verified backends. Production trust stores, training
grants and learned results are supplied independently.
