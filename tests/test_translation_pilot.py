from types import SimpleNamespace

import pytest
import torch

from tools import train_translation_pilot as pilot
from tools.plan2_s15_real_32k_bpe_v1 import ByteBPE32k
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
