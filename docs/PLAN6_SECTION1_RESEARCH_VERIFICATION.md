# Plan 6 / Section 1: immutable research trials and verification pyramid

The Plan-6-owned Python module twelve_six.research_engine implements canonical,
content-addressed trial identities over immutable protocol/model/data/artifact
hashes, seed, config and producer actor. It performs no model training.

The independently controlled verifier executes a stage twice against the same
sample and seed, checks a trusted expected output, and seals the result with a
verifier-owned HMAC key (minimum 32 bytes). This key and the expected holdout
outcomes MUST be kept outside the candidate/model process; the checked-in
fixture key grants no real-world trust.

All five tiers are mandatory, in order: deterministic, holdout, independent,
regression, replay. Missing, duplicate, unordered, changed-trial, forged,
self-issued, mismatch and unstable replay records fail closed. A retried
identical fixture produces the identical receipt. Changing any trial identity
component requires requalification.

Passing this component only produces a research qualification digest; it does
NOT authorize candidate promotion, Base checkpoint mutation, paid compute,
real corpus reads, or access to production/final-test holdouts. Those authorities
remain with their owning plans and their independently administered gates.

Automated unit and negative cases:
tests/test_research_engine_plan6_section1.py
