# TRAINING / COMPUTE AUTHORIZATION

States: LOCAL_FREE, PREPARED_NOT_LAUNCHED, COMPUTE_AUTHORIZED, RUNNING, COMPLETED, AUDITED.

Before a paid run: exact Git SHA; branch/tag; ModelSpec hash; parameter count; tokenizer hash; dataset manifest hash; train/eval split identity; seed; optimizer/scheduler; precision; context; global batch; target tokens/steps; checkpoint interval; expected hardware; estimated cost/range; output bucket/path; cancellation criteria.

No worker may infer financial authorization from a general "continue" message. The owner must explicitly approve the paid run/budget, or a previously approved budget policy must cover it.
