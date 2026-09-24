# MODEL-341 finite semantic identity boundary

This D01 hardening package closes one fail-open identity edge on the canonical MODEL-341 carrier without changing architecture geometry, parameter algebra, initialization formula, or any finite canonical identity.

`ModelSpec.rope_theta`, `ModelSpec.norm_eps`, and `InitSpec.std` are identity-bearing floating-point semantics. Python's default JSON encoder accepts non-finite floats and emits `NaN` / `Infinity`, while the prior positivity guards allowed `NaN` and positive infinity to pass. A self-consistent stage packet could therefore obtain a stable-looking SHA-256 for a poisoned/non-standard semantic value before model construction.

The repaired contract is deliberately narrow: canonical JSON hashing uses `allow_nan=False`; `rope_theta`, `norm_eps`, and initialization `std` must be finite before the existing positive-value checks run. Existing finite serialization bytes do not change. The canonical MODEL-341 ModelSpec remains `fbff24d561a2818453554d58ca23fc6ace3303b078f1935a8576c4565bd92441`, InitSpec remains `86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5`, and the exact trainable parameter count remains 20,613,440.

Adversarial regression coverage includes NaN, positive infinity, and negative infinity for each affected semantic field; direct canonical-JSON rejection; poisoned stage-config loading; and the unchanged MODEL-341 hash/count proof. The explicit independent-Q-width semantics introduced by the earlier D01 ModelSpec contract are untouched.

This is mechanics/identity hardening only. It authorizes zero optimized targets, produces no learned weights, fits no tokenizer, reads no final-test outcomes, and uses no paid compute. Shared exact-head CI is required before integration; queued, running, absent, cancelled, or red CI is not PASS.
