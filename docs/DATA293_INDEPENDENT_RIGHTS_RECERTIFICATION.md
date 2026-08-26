# DATA-293 Independent Rights Recertification

Worker: `DATA-293-INDEPENDENT-RIGHTS-RECERTIFICATION`

Scope: independent recertification of the exact external-source inventory currently eligible to feed the research corpus. This audit does not weaken any prior data-right gate and does not infer evaluation authority from a training or redistribution license.

## Authority boundary

The audit is anchored to:

- DATA-229 text registry head `90bc0b7f8b696ec35202532b13edf6ab29a662fe`, registry identity `1357a343eb4ea973950d8991913109cbea53fe4fa891f0be9745ab497eb59486`.
- DATA-227 code admission head `8ebdb2e132ed7bae5245e9d4c140752640ab9885`, successful dedicated run `32956209865`.
- DATA-228 candidate head `46a70c990dab6ff72bb84ddb54cff1156b491b40`, dedicated run `32957120454`, which failed during environment bootstrap before source materialization.
- No durable terminal DATA-278 authority was discovered and no DATA-278 source is invented into this inventory.

The canonical GitHub repository identity is `Oleksii-debug/12-6-ai.` including the trailing period.

## Recertified admitted inventory

| Exact source | Family | Binding | Training | Evaluation | Redistribution |
| --- | --- | --- | --- | --- | --- |
| `ua.rada.open-data.laws-texts.d23314` | `ua.rada.open-data.laws-texts` | bounded source label + raw SHA-256 `36eae31c...e1a4` | allowed with source attribution retained in provenance | **not separately admitted** | allowed with source attribution |
| `en.standardebooks.manual.8-typography` | `en.standardebooks.manual` | git `d1143a9...` + raw SHA-256 `21582c7f...f7860` | allowed | **not separately admitted** | allowed under CC0 |
| `en.standardebooks.manual.9-metadata` | `en.standardebooks.manual` | git `d1143a9...` + raw SHA-256 `7ac53dfb...f509` | allowed | **not separately admitted** | allowed under CC0 |
| `code.encode.httpx._content` | `github:encode/httpx` | git `b5addb6...` + source blob `6f479a0...20776` | allowed | **not separately admitted** | allowed subject to BSD copyright/conditions/disclaimer |
| `code.psf.requests._internal_utils` | `github:psf/requests` | git `5460f46...` + source blob `0466a7d...06055` | allowed | **not separately admitted** | allowed subject to Apache license and applicable NOTICE |

Totals: **5 training-admitted objects, 4 independent families, 0 evaluation-admitted objects**. Family counts are Ukrainian `1`, English `1`, code `2`.

The two Standard Ebooks records are two objects from one repository/source family. Counting them as two independent English families is an attribution error.

## Fail-closed candidate inventory

The following DATA-228 candidates have license-level evidence compatible with the described use conditions, but they are **not admitted to the current research corpus** because the dedicated terminal run failed before source materialization and no explicit corpus-purpose admission was completed:

- `uk.kubernetes.docs.concepts-index`, git `25f3dcbed7429ebe20174ccc7000428d0f0aedda`, source blob `ab9d757e99679b3db48a3230bf6eb07a997eec9c`, CC-BY-4.0 license blob `da6ab6cc8f333d7e89a99812866df8f24374d47c`.
- `en.python.docs.tutorial-introduction`, git `7f0ccd6c0e3f85fbaeceb2f67b06ab3631db0480`, source blob `465c32d0b72431cc446aae7edeb6b829c657b243`, PSF-2.0 license blob `20cf39097c68baa17cc566b64e76d34ebf034044`.

Their correct status is `NOT_ADMITTED_EVIDENCE_NOT_MATERIALIZED`, not “rights rejected”. Neither receives evaluation authority.

## Independent rights findings

### Rada

The retained immutable evidence requires source attribution/link for reuse. The live portal terms were re-read on 2026-08-26 and materially match that condition, so no stale-terms blocker was found.

The field `source_version = laws-texts/bounded-2026-08-25` is a project bounded label rather than an immutable upstream revision. This is not treated as an upstream revision. Exact admitted bytes are controlled by the retained raw SHA-256 `36eae31c3b0676ea7c02236fa05bd695c240c9a8eade5febc00457b8103ee1a4`.

Accordingly, redistribution is recorded as `ALLOWED_WITH_SOURCE_ATTRIBUTION`, not unconditional `ALLOWED`.

### Standard Ebooks

The exact upstream `LICENSE.md` at commit `d1143a9b459b5e6f9cdda93a7c1e04676bff4f6b` has Git blob `ecc3ab7a2a7d726cc225b51a0c85809a7b0274cb` and states repository contents are CC0 except `build-manual.py`. The two selected `.rst` objects are not that exception.

Source/version/license binding is exact; no stale or wrong-revision issue was found.

### HTTPX

At admitted commit `b5addb64f0161ff6bfe94c124ef76f6a1fba5254`, the selected source blob is `6f479a0885f723b7395843d41164a87041820776` and the BSD-3-Clause `LICENSE.md` blob is `ab79d16a3f4c6c894c028d1f7431811e8711b42b`.

Training remains admitted. Redistribution retains the BSD copyright notice, conditions, disclaimer and no-endorsement obligations.

### Requests

At admitted commit `5460f467b02e49471c0fd6cfc9ca0adab6351f98`, the selected source blob is `0466a7d347db4ed34a37db51b75fc8e80bc06055` and the Apache-2.0 `LICENSE` blob is `67db8588217f266eb561f75fae738656325deac9`.

DATA-293 additionally binds the exact `NOTICE` at that same commit: blob `1ff62db688277b77c83c1766dac7f165364d3528`. This closes a redistribution-evidence weakness without changing the training admission.

## Evaluation boundary

No source in this inventory has a separate project evaluation-purpose admission. Broad copyright permissions are not promoted into evaluation authority. Current evaluation-admitted count therefore remains **zero**.

Any evaluation dataset worker must obtain its own explicit evaluation-purpose authority and reserve exact records from future training before construction.

## Verdict

`RECERTIFIED_WITH_STRONGER_PURPOSE_AND_REDISTRIBUTION_CONDITIONS`

Admitted for research-corpus training: the five exact objects listed above only.

Not admitted: both DATA-228 candidates and every unmaterialized/unverified expansion candidate, including any purported DATA-278 inventory without durable terminal authority.

No data-right weakening was performed. LOCAL_FREE only.
