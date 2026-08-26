# EVAL-289 Code Evaluation Rights Reservation

Worker: `EVAL-289-CODE-EVALUATION-RIGHTS-RESERVATION`

## Verdict

`BLOCKED_NO_PRISTINE_CODE_OBJECTS_WITH_EXPLICIT_EVALUATION_AUTHORITY`

The terminal Wave-1 code authority contains two independent external-real source families, but neither exact object is admissible as a pristine evaluation reservation.

## Exact Wave-1 evidence

DATA-227 exact head `8ebdb2e132ed7bae5245e9d4c140752640ab9885` completed dedicated workflow `32956209865` successfully. Artifact `9602093542` has archive SHA-256 `080f073327020cb3bbb05c7348f658223804684d23012d9b66ab9b798c4fed5d`; its DATA-227 report identity is `234ace1497b00495716d00a1502c323fc06b88458c49ec3d6aa58bc2cdd52294`.

The rights policy is bound by Git blob SHA-1 `0ce5223a1cade10031899bf27348a1a65121d4c6` and SHA-256 `905a567242c66dd24b3c5b1e73c4da2c0af4d077179db5739850f79eb6f8fbe3`. It explicitly records acquisition, storage, analysis, model-training and redistribution decisions. It does not separately admit evaluation use. EVAL-289 therefore does not infer evaluation permission from model-training permission.

The same terminal DATA-227 report records a successful four-optimizer-step Trainer proof with 252 optimized tokens and both exact code source IDs in the training stream. This independently destroys pristine held-out status even if a later evaluation-use review were added.

## Exact candidates

`code.encode.httpx._content`, family `github:encode/httpx`, is pinned at revision `b5addb64f0161ff6bfe94c124ef76f6a1fba5254`, path `httpx/_content.py`, Git blob SHA-1 `6f479a0885f723b7395843d41164a87041820776`, raw/normalized SHA-256 `2c61b3ac94d1dcebcde0c6f519554d2d7917247fbaa0a97002db4ef69e70ff28`, source-manifest SHA-256 `e267e157a12fbfa0d4586ad11e51daaa9caac0be10097054d2bc5c9cf7147dbb`, BSD-3-Clause license blob SHA-1 `ab79d16a3f4c6c894c028d1f7431811e8711b42b` and license SHA-256 `4ec59d544f12b5f539a3a716fd321ac58ccd8030b465221f2c880200cdf28d8d`.

`code.psf.requests._internal_utils`, family `github:psf/requests`, is pinned at revision `5460f467b02e49471c0fd6cfc9ca0adab6351f98`, path `src/requests/_internal_utils.py`, Git blob SHA-1 `0466a7d347db4ed34a37db51b75fc8e80bc06055`, raw/normalized SHA-256 `4c7d8d132c9898fc7d715e473f3ac74785ddc4ab96d2c9240f87835dc6d981ff`, source-manifest SHA-256 `88f99ac5cc7573e331033d8f698721c1c82504e0c9d19e354292b4554765f6ca`, Apache-2.0 license blob SHA-1 `67db8588217f266eb561f75fae738656325deac9` and license SHA-256 `09e8a9bcec8067104652c168685ab0931e7868f9c8284b66f5ae6edae5f1130b`.

Both are ineligible for two independent reasons: `NO_EXPLICIT_EVALUATION_USE_AUTHORITY` and `ALREADY_EXPOSED_TO_MODEL_TRAINING`.

## Reservation result

Observed Wave-1 source families: 2. Eligible pristine source families: 0. Reserved code objects: 0.

No contaminated object is written into a future-training exclusion reservation under the false claim that it remains evaluation-clean. EVAL-289 performs no model training and authorizes no training-corpus mutation.

## Unblock contract

A successor may create a code evaluation reservation only after at least two exact objects from independent source families have separately bound compatible evaluation-use evidence, zero historical tokenizer-fit/model-training exposure, and a pre-corpus-construction exclusion identity that prohibits those exact raw/normalized/content hashes from every future training corpus.

The machine authority is `evidence/eval289/code-evaluation-rights-reservation.json`, identity `de52b38690230bee8a912fc1265ad34b80a3abde46bd23401f15395749a4813c`.

LOCAL_FREE only. No model training.
