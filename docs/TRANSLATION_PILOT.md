# Scratch translation pilot

The pilot trains the native 12-6 decoder in both English–Ukrainian directions.
It uses byte BPE fitted on a deterministic sample of the training partition.
The training objective scores the target text and EOS. The source prompt and
right padding are excluded from the loss.

Prepare the pinned bitext using [the dataset instructions](TRANSLATION_PARALLEL_DATA.md).
From a checkout with the repository installed, run the following commands.
Keep the run directory and input bundle outside the repository.

```powershell
$env:PYTHONPATH = 'src;.'
$data = '../mt-tatoeba-en-uk-v1'
$pin = '127d1b5e62899049b9f4060242e6c249a58c08eb6ed6c433a9d97f6bc1a9136d'
$run = '../mt-run-en-uk-01'
python -m tools.train_translation_pilot train --data $data --data-sha256 $pin --run $run --steps 1800 --stop-after 300 --checkpoint-every 300 --batch-size 24 --eval-pairs 100
python -m tools.train_translation_pilot train --data $data --data-sha256 $pin --run $run --resume 300 --checkpoint-every 300
python -m tools.train_translation_pilot translate --run $run --step 1800 --direction en-uk --text 'Good morning.'
python -m tools.train_translation_pilot evaluate-tatoeba --run $run --step 1800 --data $data --split test --output "$run/tatoeba-test.json"
```

For the expanded CPU experiment, build the same snapshot with
`--train-count 100000`. The dataset keeps the original validation and test
partitions. Its manifest SHA-256 is
`ffffa19f1ff3b570bb3f554e1607b2936c4c52d969836727c3a917fda0872022`.

```powershell
python -m tools.train_translation_pilot train --data ../mt-tatoeba-en-uk-100k-v1 --data-sha256 ffffa19f1ff3b570bb3f554e1607b2936c4c52d969836727c3a917fda0872022 --run ../mt-run-en-uk-100k-01 --steps 8400 --stop-after 4200 --checkpoint-every 1400 --batch-size 24 --eval-pairs 100 --tokenizer-from "$run/run.json" --width 128 --layers 4 --length-window-batches 16
python -m tools.train_translation_pilot train --data ../mt-tatoeba-en-uk-100k-v1 --data-sha256 ffffa19f1ff3b570bb3f554e1607b2936c4c52d969836727c3a917fda0872022 --run ../mt-run-en-uk-100k-01 --resume 4200 --checkpoint-every 1400
```

The tokenizer reuse option restores only verified merge rules and records
their originating run digest. Model weights still start from the new seed.
Length grouping sorts bounded shuffled windows before batching, then shuffles
full batches and preserves the final partial batch. The recorded window size
and seed reproduce the order at resume.

A fresh optimizer phase accepts `--parent-run`, `--parent-step`, and
`--parent-checkpoint-id` to initialize from an exact verified checkpoint.
It inherits the parent's context by default. Pass `--context 192` to extend
a 96-token run. The child records its own model spec and keeps the parent
checkpoint identity for provenance. The native decoder uses rotary positions,
so this change leaves tensor shapes and parameter count intact. Quality at
longer lengths needs separate evaluation. Resuming an existing phase uses
its recorded context.

The training process records the model, tokenizer, schedule, data manifest and
Git revision in `run.json`. Each saved checkpoint includes model, optimizer,
scheduler, RNG and progress state. Resume reopens that state before selecting
the next deterministic batch. Training records validation loss and verifies
checkpoint readback. Score the selected checkpoint on the held-out test
partitions separately. Each direction reports corpus chrF2, exact matches,
empty predictions and source copies. [chrF paper](https://aclanthology.org/W15-3049/)

For independent evaluation, pass the pinned FLORES-200 JSONL and its SHA-256
to `evaluate`. That command evaluates only examples whose source fits the
model context and records any exclusions.

```powershell
python -m tools.train_translation_pilot evaluate --run $run --step 1800 --pairs '../mt-flores200-en-uk-devtest-v1/devtest.jsonl' --pairs-sha256 ed090446f653dbb6a1555c7deb52849b724faa374f7b0650f7a6b3a182f1088d --max-new-tokens 40 --output "$run/flores-test.json"
```
