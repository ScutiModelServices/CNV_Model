# configs

`models/<name>.json` is one chat model. The file name must equal `name`. The service reads the folder at start; restart after a change.

| Field | Meaning |
|---|---|
| `name` | The value of the `model` field in `POST /v1/chat`. |
| `title` | The label on the admin page. |
| `engine` | `ollama` (the Ollama server, `SERVICE_OLLAMA_URL`) or `llamacpp` (a `llama-server` sidecar that holds this model's GGUF; needs `url`). |
| `url` | `llamacpp` only: the sidecar's address, for example `http://cnv_bitnet_server:8080`. One sidecar serves one model, so every `llamacpp` model has its own. Not allowed for `ollama`. |
| `repo` | HuggingFace repository of the GGUF files. |
| `default_quant` | The build used until an admin picks another. Must be one of `builds`. |
| `capabilities` | Any of `tools`, `thinking`, `schema`, `project`. A request that uses one the model lacks gets 422. |
| `builds` | The quantized builds listed under the model on the admin page: `quant`, `file_mb` (stored size, replaced by the live Hub size on refresh), `note`, and for `llamacpp` the GGUF file name in `ref`. |

To list a new quantization, add a row to `builds`. Weights are downloaded only from the admin page.
