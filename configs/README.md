# configs

`models/<name>.json` is one chat model. The file name must equal `name`. The service reads the folder at start; restart after a change.

| Field | Meaning |
|---|---|
| `name` | The value of the `model` field in `POST /v1/chat`. |
| `title` | The label on the admin page. |
| `repo` | HuggingFace repository of the GGUF files. |
| `capabilities` | Any of `tools`, `thinking`, `schema`, `project`. A request that uses one the model lacks gets 422. |
| `builds` | The builds listed under the model on the admin page, original or quantized: `quant`, `file_mb` (stored size, replaced by the live Hub size on refresh), `note`, and `engines`. |

`engines` is an object per build: each key is an engine that can run that build, with the options that engine needs. The catalog only lists them; the user picks which one runs (admin page, or `engine` in a request). The first one listed is used until the user picks.

| Engine | Options | Runs |
|---|---|---|
| `ollama` | none (`{}`); optional `renderer` and `parser`: Ollama's built-in names for the model (`"inkling2"`). After the admin Download pulls the build, the service recreates it from the same weights with them (`/api/create`), because a plain Hub import gets generic stop tokens that end the answer after one token. Needs an Ollama that has that renderer. | The build through your Ollama server, as `hf.co/<repo>:<quant>`. |
| `llamacpp` | `ref`: the GGUF file name in `repo`; `url`: the `llama-server` sidecar that holds it, for example `http://cnv_bitnet_server:8080` (one sidecar holds one GGUF); optional `prompt`: `"bitnet"` for a server that cannot read a chat template (the bitnet.cpp build): the service then writes the prompt itself and calls `/completion`. optional `cpu`: a CPU flag the build needs (`"avx2"` for bitnet.cpp's i2_s kernel); the build then counts as not fitting a CPU without it, so it is not downloaded, run, started or loaded. Without `prompt` the service calls `/v1/chat/completions` and the server applies the model's own chat template. | The file through that sidecar. |

```json
"builds": [{"quant": "Q4_K_M", "file_mb": 1048, "engines": {"ollama": {}, "llamacpp": {"ref": "m-Q4_K_M.gguf", "url": "http://cnv_m_server:8080"}}}]
```

To list a new quantization, add a row to `builds`. Weights are downloaded only from the admin page.
