from types import SimpleNamespace

import pytest
import torch

from tools import train_translation_pilot as pilot
from tools.plan2_s15_real_32k_bpe_v1 import ByteBPE32k
from twelve_six.model import BaseInitManifest, InitSpec, ModelSpec
from twelve_six.translation_evaluation import TranslationExample


def test_loss_labels_include_only_translation_and_eos():
    tokenizer = ByteBPE32k([], required_vocab_size=260)
    short = TranslationExample("pair", "en-uk", "Hi.", "Вітаю.")
    long = TranslationExample("long", "uk-en", "Доброго ранку.", "Good morning.")
    encoded, start = pilot._encode(tokenizer, short)
    batch = pilot.collate([(encoded, start), pilot._encode(tokenizer, long)], tokenizer.pad_id)
    expected = tokenizer.encode(short.reference, add_eos=True)

    assert encoded[:start] == pilot.prompt_ids(tokenizer, short.direction, short.source)
    assert batch["labels"][0, :start].eq(-100).all()
    assert batch["labels"][0, start:len(encoded)].tolist() == expected
    assert batch["labels"][0, len(encoded):].eq(-100).all()
    assert batch["labels"][0, 1:][start - 1].item() == expected[0]


def test_resume_schedule_handles_partial_batch_and_epoch_boundary():
    examples = list(range(5))
    original = pilot._batches(examples, 3, seed=8, start=0)
    expected = [next(original) for _ in range(8)]
    assert all(sorted(expected[i] + expected[i + 1]) == examples for i in range(0, 8, 2))
    for step in range(8):
        resumed = pilot._batches(examples, 3, seed=8, start=step)
        assert [next(resumed) for _ in range(8 - step)] == expected[step:]


def test_length_grouping_reduces_padding_and_preserves_resume_order():
    examples = [(list(range(n)), n) for n in range(4, 57)]
    batch_size = 8
    batches_per_epoch = 7
    plain = pilot._batches(examples, batch_size, seed=8, start=0)
    grouped = pilot._batches(examples, batch_size, seed=8, start=0, length_window_batches=8)
    original = [next(plain) for _ in range(batches_per_epoch)]
    expected = [next(grouped) for _ in range(batches_per_epoch * 2)]
    for start in (0, batches_per_epoch):
        epoch = expected[start:start + batches_per_epoch]
        assert sorted(item[1] for batch in epoch for item in batch) == list(range(4, 57))
        assert len(epoch[-1]) == 5
    plain_slots = sum(len(batch) * max(item[1] for item in batch) for batch in original)
    grouped_slots = sum(
        len(batch) * max(item[1] for item in batch) for batch in expected[:batches_per_epoch]
    )
    assert grouped_slots < plain_slots
    for step in (0, 3, 6, 7, 9):
        replayed = pilot._batches(examples, batch_size, seed=8, start=step, length_window_batches=8)
        assert [next(replayed) for _ in range(len(expected) - step)] == expected[step:]


def test_generation_uses_source_only_and_trims_eos():
    tokenizer = ByteBPE32k([], required_vocab_size=260)
    expected = pilot.prompt_ids(tokenizer, "uk-en", "Так.")

    class Model:
        class spec:
            max_seq_len = 64

        def eval(self):
            return self

        def generate(self, prompt, *, max_new_tokens, do_sample):
            assert prompt.tolist() == [expected]
            assert not do_sample
            assert max_new_tokens <= self.spec.max_seq_len - len(expected)
            return torch.tensor([expected + tokenizer.encode("Yes.", add_eos=True) + [65]])

    assert pilot.translate(Model(), tokenizer, "uk-en", "Так.") == "Yes."


def test_renamed_valid_checkpoint_is_refused_before_model_load(tmp_path, monkeypatch):
    verified = SimpleNamespace(manifest={"identity": {"step": 1}, "checkpoint_id": "a" * 64})
    monkeypatch.setattr(pilot, "prepare_checkpoint_load", lambda _: verified)
    with pytest.raises(ValueError, match="differs from the requested step"):
        pilot._restore_model(tmp_path, 2, {})


def test_batched_evaluation_groups_equal_prompts_and_preserves_output():
    tokenizer = ByteBPE32k([], required_vocab_size=260)
    examples = [
        TranslationExample("a", "en-uk", "Go.", "Іди."),
        TranslationExample("b", "en-uk", "Hi.", "Вітаю."),
        TranslationExample("c", "en-uk", "Go.", "Ідіть."),
        TranslationExample("d", "uk-en", "Йди.", "Go."),
    ]

    class Model:
        def __init__(self):
            self.spec = SimpleNamespace(max_seq_len=64)
            self.calls = []

        def eval(self):
            return self

        def generate(self, prompt, *, max_new_tokens, do_sample):
            assert not do_sample
            self.calls.append(tuple(prompt.shape))
            first_source_byte = prompt[:, len("en-uk\n") + 1:len("en-uk\n") + 2]
            eos = torch.full_like(first_source_byte, tokenizer.eos_id)
            return torch.cat((prompt, first_source_byte, eos), dim=1)

    model = Model()
    lookup = pilot._translation_lookup(model, tokenizer, examples, 8, 4)
    assert sorted(batch for batch, _ in model.calls) == [1, 2]
    for example in examples:
        assert lookup(example.direction, example.source) == pilot.translate(
            model, tokenizer, example.direction, example.source, 8,
        )


def test_continued_manifest_pins_weight_parent_and_new_schedule(tmp_path, monkeypatch):
    spec = ModelSpec(1, 260, 24, 16, 1, 2, 1, 8, 32, rope_rotary_dim=8)
    init = InitSpec()
    parent = {
        "schema": pilot.SCHEMA, "git_sha": "c" * 40, "data_sha256": "a" * 64,
        "tokenizer": {"version": "frozen"}, "model_spec": spec.to_dict(), "init_spec": init.to_dict(),
        "initialization": BaseInitManifest(1, spec.identity_sha256(), init.identity_sha256(), 1701).to_dict(),
        "training": {"seed": 1701},
    }
    (tmp_path / "run.json").write_text(pilot.json.dumps(parent), encoding="utf-8")
    monkeypatch.setattr(pilot, "detect_git_sha", lambda _: "d" * 40)
    args = SimpleNamespace(
        parent_run=tmp_path, parent_step=8400, parent_checkpoint_id="b" * 64,
        lr=0.0003, steps=2400, seed=1701, shuffle_seed=2001, data_sha256="a" * 64,
        batch_size=24, eval_pairs=100, threads=3, length_window_batches=16, context=None,
    )
    continued = pilot._continued_manifest(args)
    assert continued["weight_parent"] == {
        "run_manifest_sha256": pilot.hash_json(parent), "checkpoint_id": "b" * 64, "step": 8400,
    }
    assert continued["model_spec"] == parent["model_spec"]
    assert continued["initialization"] == parent["initialization"]
    assert continued["tokenizer"] == parent["tokenizer"]
    assert continued["training"]["learning_rate"] == 0.0003
    assert continued["shuffle_seed"] == 2001
    assert continued["training"]["seed"] == 1701
    args.context = 48
    extended = pilot._continued_manifest(args)
    assert extended["model_spec"] == {**parent["model_spec"], "max_seq_len": 48}
    original_model = pilot._model(parent)
    extended_model = pilot._model(extended)
    extended_model.load_state_dict(original_model.state_dict(), strict=True)
    prompt = torch.ones((1, 24), dtype=torch.long)
    torch.testing.assert_close(extended_model(prompt).logits, original_model(prompt).logits, rtol=0, atol=0)
    assert extended_model(torch.ones((1, 25), dtype=torch.long)).logits.shape == (1, 25, 260)
    args.context = 23
    with pytest.raises(ValueError, match="preserve or extend"):
        pilot._continued_manifest(args)
    args.parent_checkpoint_id = "unverified"
    with pytest.raises(ValueError, match="earlier translation run"):
        pilot._continued_manifest(args)
