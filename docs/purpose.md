# Purpose

**Kind:** chat messages → text (conversation)

**Group:** Conversation

**Tags:** `cpu`, `streaming`, `variants`

| | |
|---|---|
| Input | Chat messages, optional `model`, temperature and maximum tokens. |
| Output | JSON content, `model`, `quant`, finish reason and usage; NDJSON streaming available. |
| Runs on | CPU API. Inference runs on an external Ollama (LFM3, Spark X2.6, MiniCPM5) or a bitnet.cpp sidecar (BitNet b1.58 2B4T). The Models card lists each model's quantized builds with a memory estimate and whether this machine can run them. |

**Use for:** everyday conversation with one of the small chat models; pick the model per request with `model`.
**Not for:** project checkpoints, JSON-schema output or per-request thinking (422); `tools` only for models with the `tools` capability (`GET /v1/models`).
