# CNV_Model: without Docker, on Windows

First start: open `/admin/` → **Updates** → **Download**; see [the web UI guide](guide-web-ui.md#updating-the-model-files). Startup and requests load downloaded files only; an empty installation keeps health/admin available and returns 503 for inference.

Run the API directly on `127.0.0.1:29204`, with inference in a running Ollama.

## Requirements

- Python 3.11 or newer and uv (`winget install astral-sh.uv`).
- Running Ollama with the configured model already pulled; model memory belongs to Ollama.

## Start

```bash
curl -X POST http://127.0.0.1:29204/admin/models/lfm3/download -H "X-Internal-Token: <ADMIN_PASSWORD>" -H "Content-Type: application/json" -d '{"quant": "Q4_K_M"}'
```

Use `run.bat run`, or choose Run from its menu. It syncs `.venv` and starts `python src/app.py --dotenv`. Copy `.env.example` to `.env.local` only if needed, change `SERVICE_OLLAMA_URL` to `http://127.0.0.1:11434`, and set an admin password. When the URL is unset, the local default is `http://127.0.0.1:11434`; explicitly empty is an error.

`--dotenv` loads this folder's `.env.local`, then `.env` (the first file to set a name wins) and prints names only; real environment variables win. A plain invocation reads neither:

```bash
uv run python src/app.py
```

`SERVICE_HOST` defaults to `127.0.0.1`. Other startup values are listed in [Docker settings](guide-docker-same-device.md#settings); limits and reasoning controls are on the admin page. Model storage is managed by the separate Ollama process.

## Check

```bash
curl http://127.0.0.1:29204/health
```

The API starts even when Ollama is down. `ready` stays false and chat returns 503 with a admin Updates Download hint. Open `http://127.0.0.1:29204/admin/` and create a token.

## Stop

Press Ctrl+C. Do not run the Docker stack on the same port simultaneously.

## Tests

```bash
uv run pytest
```

## Problems

See [common problems](guide-docker-same-device.md#problems). Real Ollama inference has not been verified for this migration.
