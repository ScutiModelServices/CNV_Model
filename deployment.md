# Deployment

- Port 29204 (`SERVICE_PORT`), bound on `0.0.0.0` by default so ServiceHub can reach it. Do not set `SERVICE_BIND` to a loopback address.
- Compose runs the API replicas (`SERVICE_REPLICAS`), an nginx load balancer, and the bitnet.cpp sidecar `cnv_bitnet_server` (build `sidecars/bitnet`). Weights live in the `cnv_model_files` volume, or in the host folder `SERVICE_MODELS_PATH` when set (GGUF files for the sidecar) and in your Ollama (Ollama models).
- Needs: an Ollama server (address in the admin Settings card; `SERVICE_OLLAMA_URL` only seeds the first start) for `lfm3`, `sparkx26`, `minicpm52b`.
- Port 29204 is separate from the old CNV services (29200–29203), so both can run side by side while callers move.
- The `bitnet2b` sidecar does not start on a CPU without AVX2 (it exits 0 and is not restarted; `docker compose ps` shows it exited). It starts only with weights present: press Download on its build in `/admin/`, then restart `cnv_bitnet_server`.
- Details: [README.md](README.md), [docs/guide-docker-same-device.md](docs/guide-docker-same-device.md), [docs/migration.md](docs/migration.md).
