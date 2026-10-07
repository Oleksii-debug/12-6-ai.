# INFERENCE / TESTING STRATEGY

1. S0-S2: canonical Python generate CLI + deterministic smoke prompts; no custom GUI.
2. Maintain Transformers-compatible config/model/export as early as practical so local directories can be loaded by standard tooling.
3. vLLM: target for GPU serving when our Transformers-compatible architecture path satisfies the modeling backend; it can serve a local model directory and expose OpenAI-compatible APIs.
4. llama.cpp: target for local Windows CPU/GPU testing after a correct GGUF converter and architecture support exist. llama-server already provides a web UI and OpenAI-compatible endpoints, so writing our own ChatGPT-like window is unnecessary at first.
5. A custom accessible desktop client is optional later; if built, it consumes our stable local API rather than embedding model logic.
6. Every converted/exported artifact must be compared against canonical checkpoint logits/generation within defined tolerances.
