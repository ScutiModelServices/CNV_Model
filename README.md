# CNV_Model

Chat models over HTTP, port 29204. One service replaces `CNV_LFM3_Model`, `CNV_SparkX26_Model`, `CNV_MiniCPM52B_Model` and `CNV_BitNet2B_Model`: a request names the model, the service runs it on its engine. Admin settings, tokens, rate limits, metrics, streaming and answer caches are the shared model-service ones.

First start: open `/admin/` → **Models and builds** → **Download** on a build. Startup and requests load downloaded files only; a missing build returns 503 and keeps health and admin up.

## API

| Call | Purpose |
|---|---|
| `POST /v1/chat` | `messages`, `model`, optional `temperature`, `max_tokens`, `stream`, `tools`; returns `content`, `tool_calls`, `finish_reason`, `model`, `quant`, `usage`, `duration_ms`. |
| `GET /v1/models` | Each model's `name`, `title`, `engine`, `capabilities`, current `quant`, `default`. Token required when auth is on. |
| `GET /health` | `status`, `ready` (some model is loaded), `models`, `num_thread`; no token required. |

`model` is required and is a catalog name (`lfm3`, `sparkx26`, `minicpm52b`, `bitnet2b`); there is no default model. A missing `model` is 422, an unknown name 404. `tools` on a model without the `tools` capability, and `schema`, `project` or `thinking` in any request, are 422. Thinking is an admin setting. Messages, limits, streaming (NDJSON `content` lines then `done`), cache (`X-Cache`, temperature 0 only; the key includes model and quant) and tokens work as in the old services.

## Catalog

One JSON file per model in `configs/models/`; see [configs/README.md](configs/README.md). Add a row to `builds` to list another quantization.

## Engines

| Engine | Models | Where it runs |
|---|---|---|
| `ollama` | `lfm3`, `sparkx26`, `minicpm52b` | An Ollama server, `SERVICE_OLLAMA_URL`. Download is `ollama pull`. |
| `llamacpp` | `bitnet2b` | The bitnet.cpp `llama-server` sidecar (`sidecars/bitnet/`), at the `url` in `configs/models/bitnet2b.json`. Weights are GGUF files in `SERVICE_MODELS_DIR`, shared with the sidecar. |

## Run

```bash
docker compose --env-file .env.compose up -d --build
```

Local: `run.bat run` (Ollama models only unless you start a `llama-server` yourself).
