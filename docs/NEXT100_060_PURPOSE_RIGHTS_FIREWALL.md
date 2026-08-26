# NEXT100-060 purpose-specific data rights firewall

Worker: `NEXT100-060-EVAL-RIGHTS-FIREWALL`

## Scope

This authority adds one append-only firewall across six independent project purposes:
`training`, `tokenizer-fitting`, `selection-validation`, `final-test`,
`redistribution`, and `analysis`.

An upstream license is evidence, not a project-purpose decision. Every exact project object must
carry all six decisions explicitly. Omitted purposes never inherit `ALLOW` from public
availability, a permissive license, a training decision, or a redistribution decision.

## Exact-object authority

`src/twelve_six/data/rights_firewall.py` binds each decision to immutable object identity:
source ID, source family, upstream revision, object path/locator, content SHA-256, and optional
Git blob SHA-1. Evaluation use additionally requires an immutable reservation timestamp and
reservation commit.

The machine authority in
`configs/data/next100_060_purpose_rights_firewall_v1.json` binds twelve exact examples from the
live authority vector: a DATA-300 training object, all ten EVAL-303 selection membership objects,
and the EVAL-233 preserved final-test seed object. No final-test payload or outcome was read to
construct NEXT100-060; only committed provenance identities and firewall metadata were consumed.

## Irreversible reservation rules

A selection-reserved object can never later become training or tokenizer-fit material. A
final-test-reserved object can never later become training, tokenizer-fit, or selection material.
Final-test observations cannot influence training, tokenizer fitting, or selection.

A rights/evidence/reservation decision may change only through a successor authority for the same
exact source-object identity. The successor must name the immediately preceding authority, have a
later issuance time, and be bound to a distinct immutable commit. Successors cannot undo an
existing selection or final-test reservation.

## Concurrency intake

The authority consumes terminal DATA/EVAL rights state visible at seal time and separately records
concurrent source-rights PRs. Positive source claims whose current exact-head scientific workflow
is queued are recorded but cannot become enforcement inputs. RETEST and REJECT decisions remain
fail-closed. The exact-head-green Starlette training-only authority is consumed as a late terminal
rights input without mutating DATA-300 or the canonical registry.

## Regression contract

Focused tests prove:

- a training-only object cannot enter selection-validation or final-test;
- a selection-reserved object cannot later enter training/tokenizer fitting;
- a final-test object cannot influence selection;
- rights-state changes require a successor authority;
- broad upstream licensing does not fill omitted project-purpose decisions;
- evaluation permission requires an exact reservation timestamp and commit;
- final-test role separation is structural;
- analysis and redistribution remain independent dimensions.

Validation commands:

```bash
PYTHONPATH=src python tools/validate_next100_060_purpose_rights_firewall.py
PYTHONPATH=src python -m pytest -q tests/test_next100_060_purpose_rights_firewall.py
```

`LOCAL_FREE` only. No model/data training, tokenizer fitting, optimizer update, evaluation scoring,
or final-test outcome inspection is performed by NEXT100-060.
