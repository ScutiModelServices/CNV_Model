# Deployment

- Port 29200 (`SERVICE_PORT`), bound on `0.0.0.0` by default so ServiceHub can reach it. Do not set `SERVICE_BIND` to a loopback address.
- Compose runs the API replicas (`SERVICE_REPLICAS`), an nginx load balancer, and the bitnet.cpp sidecar `cnv_bitnet_server` (build `sidecars/bitnet`). Weights live in the `cnv_model_files` volume (GGUF files for the sidecar) and in your Ollama (Ollama models).
- Needs: an Ollama server (`SERVICE_OLLAMA_URL`) for `lfm3`, `sparkx26`, `minicpm52b`.
- Port 29200 was `CNV_MiniCPM52B_Model`'s. Stop it before starting this service.
- The `bitnet2b` sidecar starts only with weights present: press Download on its build in `/admin/`, then restart `cnv_bitnet_server`.
- Details: [README.md](README.md), [docs/guide-docker-same-device.md](docs/guide-docker-same-device.md), [docs/migration.md](docs/migration.md).
