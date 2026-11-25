# CNV_Model

Chat models over HTTP, port 29204. One service replaces `CNV_LFM3_Model`, `CNV_SparkX26_Model`, `CNV_MiniCPM52B_Model` and `CNV_BitNet2B_Model`: a request names the model, the service runs it on its engine. Admin settings, tokens, rate limits, metrics, streaming and answer caches are the shared model-service ones.

First start: open `/admin/` → **Models and builds** → **Download** on a build. Startup and requests load downloaded files only; a missing build returns 503 and keeps health and admin up.

## API

| Call | Purpose |
|---|---|
| `POST /v1/chat` | `messages`, `model`, optional `engine`, `temperature`, `max_tokens`, `stream`, `tools`; returns `content`, `tool_calls`, `finish_reason`, `model`, `quant`, `engine`, `usage`, `duration_ms`. |
| `GET /v1/models` | Each model's `name`, `title`, `capabilities`, the `quant` and `engine` it runs now, and its `builds` with the engines that can run each. Token required when auth is on. |
| `GET /health` | `status`, `ready` (some model is loaded), `models`, `num_thread`; no token required. |

`model` is required and is a catalog name (`lfm3`, `sparkx26`, `minicpm52b`, `bitnet2b`); there is no default model. A missing `model` is 422, an unknown name 404. `tools` on a model without the `tools` capability, and `schema`, `project` or `thinking` in any request, are 422. Thinking is an admin setting. Messages, limits, streaming (NDJSON `content` lines then `done`), cache (`X-Cache`, temperature 0 only; the key includes model and quant) and tokens work as in the old services.

## Catalog

One JSON file per model in `configs/models/`; see [configs/README.md](configs/README.md). Add a row to `builds` to list another quantization.

## Engines

A model lists its builds (original or quantized), and each build lists the engines that can run it. Nothing is fixed to an engine: the user picks which engine runs a build, on the admin Models card (**Run on <engine>**, saved for every replica) or per request with `engine` (it must be one the build's current catalog entry lists, else 422).

| Engine | Where it runs |
|---|---|
| `ollama` | An Ollama server; its address is the **Ollama server address** setting. Download is `ollama pull`. |
| `llamacpp` | A `llama-server` sidecar that holds one GGUF (`sidecars/bitnet/` for BitNet), at the `url` in the build's catalog entry. Weights are GGUF files in the **GGUF folder** setting (`/models` in Docker; the host folder is `SERVICE_MODELS_PATH` in compose), shared with the sidecar. |

Today `lfm3`, `sparkx26` and `minicpm52b` list only `ollama`, and `bitnet2b` only `llamacpp` (bitnet.cpp is the only engine that reads its `i2_s` weights). A second engine for a build is one more entry in its `engines`, plus a running sidecar for it.

## Run

```bash
docker compose --env-file .env.compose up -d --build
```

Local: `run.bat run` (Ollama models only unless you start a `llama-server` yourself).
