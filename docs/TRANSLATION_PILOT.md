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
