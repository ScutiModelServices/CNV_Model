# CNV_Model

Chat models served over HTTP through a JSON catalog and engine adapters, behind the shared model-service admin page, tokens and settings. Port 29204, env prefix `SERVICE_`, token prefix `cnv_`. Replaces `CNV_LFM3_Model`, `CNV_SparkX26_Model`, `CNV_MiniCPM52B_Model`, `CNV_BitNet2B_Model` (design: `AI/ModelServices/_docs/specs/2026-11-24-group-services-design.md`).

## Agent instructions
This file is the one instruction file for every coding agent; `CLAUDE.md` only imports it (`@AGENTS.md`). Edit this file, never `CLAUDE.md`.

## Commands (from this folder, venv `.venv`, managed by uv)
- Install: `uv sync` (`winget install astral-sh.uv` if uv is missing)
- Test: `uv run pytest -q`. `tests/test_admin_ui.py` needs Playwright and its browser and is skipped without it. On Windows, if uv fails with "cannot move the file to a different disk drive", set `UV_CACHE_DIR` to a folder on the same drive and `UV_LINK_MODE=copy`.
- Run: `run.bat run`, or `uv run python src/app.py` (add `--dotenv` to load `.env.local` and `.env`), or `docker compose --env-file .env.compose up -d --build`

## Rules
- Load downloaded weights only; no startup, warm-up or request downloads. Only the admin per-build Download downloads, then replica markers invalidate old weights. Missing files keep health/admin available and inference returns 503 with an admin Download hint.
- Models are catalog files, not code: `configs/models/<name>.json` (`src/catalog.py`). Engines are adapters in `src/adapters/` (`ollama`, `llamacpp`) behind `src/router.py`; a new engine is a new adapter plus an entry in `ENGINES`. Keep injected fake adapters for tests (`tests/conftest.py`).
- `num_thread` is saved and reported only; Ollama chooses its threads. A 503 from an engine drops that loaded model; the next request loads it again.
- Follow the `model-service-parity` skill and `AI/ModelServices/_docs/SHARED-FEATURES.md` (what must match in every model service) and `AI/ModelServices/_docs/SERVICE-SPECIFIC.md` (what differs on purpose). A change to a shared feature goes to every service; a new difference goes into `SERVICE-SPECIFIC.md` in the same change.
- `web/index.html`, `src/tokens.py`, `src/settings.py`, `src/updates.py` and `.dockerignore` are copies shared with the other model services. Change one, copy it to the others, run `python tools/check_sync.py` in each repo.
- Builds: each catalog `builds` row is a quantization listed under its model; `src/builds.py` adds the memory estimate, the fit check (`src/hardware.py`) and live Hub sizes (cached in `hub-sizes.json` in the data volume; offline falls back to the stored size). The chosen build per model is the saved `quants` setting. There is no default model and no warm-up: every request names its `model`. Only `POST /admin/models/{name}/download` fetches weights.
- `POST /v1/chat` requires `model` (422 when missing, 404 when unknown); `tools` need the model's `tools` capability (else 422); `schema`, request-level `thinking` and `project` are 422 (not carried over from MiniCPM and BitNet yet). Adapters get `tools=` only when a request has tools.
- Port 29204 is the next free port in the CNV band; the four old CNV services keep 29200–29203 until their callers move (`docs/migration.md`).
- Behaviour changes update `README.md`, `deployment.md`, `docs/guide-web-ui.md`, `.env.example` and `CHANGELOG.md` in the same commit.
- Never read or print `.env`, `.env.local`, `.env.compose`, `tokens.json` or `settings.json`.
- Commit with the noreply author email `214214011+scutiii2@users.noreply.github.com` (another address is rejected on push). Push `main`, then run `python tools/publish_live.py`. Build Docker only when the Dockerfile or the dependencies changed.
