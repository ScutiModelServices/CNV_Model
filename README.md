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

## Download progress and delete

While a download runs, the Models page shows a progress bar with the bytes done and, once known, the total and percent (`GET /admin/progress` returns `{active, label, done, total}`; every replica shows the same numbers through the shared data volume, and the bar comes back after a page reload). Only one download runs at a time: a second **Download** gets 409.

**Delete** (`POST /admin/models/{name}/delete`) removes a downloaded build from its engine: `ollama` deletes it from Ollama's storage, `llamacpp` deletes the GGUF file. It asks for confirmation. The build a running model uses cannot be deleted (409): stop the model or run another build first. The bitnet.cpp sidecar keeps its open copy of a deleted file until `docker compose restart cnv_bitnet_server`, and then waits for the file again.

## Start, stop, restart

Each model has Start, Stop and Restart on the admin Models page (`POST /admin/models/{name}/start|stop|restart`). **Stop** frees the model (Ollama is told to unload it; the bitnet.cpp sidecar keeps running, so only the service's handle is dropped) and refuses its requests with 503 until it is started. **Start** loads it now and allows requests again; **Restart** is both. A model's state is `stopped`, `running` (loaded by this replica) or `ready` (loads on its first request). Stopped models are saved, so every replica and a restart keep them stopped.

## Marketplace

The admin **Marketplace** page lists the quantizations that exist in each supported model's Hugging Face repository and that its catalog file does not list, with their size and whether this machine can run them. **Add** puts one under the model's builds (saved in `marketplace.json` in the data volume, shared by replicas); download and run it on the Models page. **Remove** takes an added variant out again (its files stay on disk; the build in use cannot be removed). Added variants run on Ollama only; a model that runs on its own llama.cpp sidecar (`bitnet2b`) has none, since a sidecar holds one file. The listing needs the Hub; offline it says so and the catalog builds still work.

## What fits

A build is downloaded, run or loaded only if this machine can run it: its estimated memory (file size × 1.1 + 400 MB) must fit the free GPU memory or free RAM. A build that does not fit gets 409 on download, run, start and marketplace add, and 503 if a request would load it. When the fit cannot be checked (the engine runs on another machine, or memory is unreadable) the build is allowed and labelled "could not check". The build already running is not re-checked on download.

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
