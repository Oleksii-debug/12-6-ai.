# Ready components selected for reuse — 2026-08-23

- PyTorch: tensor/autograd/GPU/runtime foundation.
- OLMo-core: transparent large-scale PyTorch model/trainer/data/checkpoint building blocks; official scratch-training scripts and random init support.
- TorchTitan: FSDP2, TP, PP, CP, distributed checkpointing, float8, SFT, checkpointable loading and distributed simulation/debug modes.
- Megatron Core: scale path for TP/PP/CP/EP/MoE across hundreds of billions/trillion-scale research.
- DataTrove: extraction/filtering/statistics/exact and near dedup, local/Slurm/Ray execution.
- HF Tokenizers or SentencePiece: train our own tokenizer from approved corpus; not foreign model weights.
- SafeTensors + HF-compatible config: portable model exchange.
- TRL: later SFT/DPO/reward/GRPO/RL tooling; keep separate from Base until enabled.
- verl: candidate for later large-scale RL if scale requires it.
- vLLM: ready OpenAI-compatible GPU serving; current Transformers backend can load compatible custom decoder/MoE models.
- llama.cpp: ready lightweight local server/web UI for GGUF models when conversion/support is implemented.
- lm-evaluation-harness/Lighteval: benchmark framework, with contamination-safe project policy.

Decision: early canonical code should expose a thin 12-6-owned ModelSpec and interfaces, while swapping infrastructure backends as scale increases. Avoid hard-coupling the model definition to only one training framework.
