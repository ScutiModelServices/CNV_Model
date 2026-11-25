# CNV_Model: the admin web UI

The admin UI at `/admin/` has separate sidebar pages for health, access and model configuration. It works with every deployment mode. Only the address differs:

| Run | Address |
|---|---|
| Docker or `run.bat` on this PC | `http://127.0.0.1:29204/admin/` |
| Docker on the deployment box | `http://<box-ip>:29204/admin/` |

If `SERVICE_PORT` is changed, use that port.

## Before you open it

The page needs the admin password: the value of `SERVICE_ADMIN_PASSWORD`.
- Docker Desktop on this device: set it in `.env.compose` next to `docker-compose.yml`, then `docker compose --env-file .env.compose up -d`.
- Deployment device: set it in `.env` next to `docker-compose.yml` (Deploy option 2 copies it), then `docker compose up -d`.
- No Docker: set it in `.env.local` (or `.env`) in this folder and start with `run.bat` (it passes `--dotenv`), or set it in the environment first; a real environment variable wins over both files and `.env.local` wins over `.env`. A plain `python src/app.py` reads neither.
- Empty password = no auth at all and the page answers `Admin password required (set SERVICE_ADMIN_PASSWORD)`. Set
  one to use the page.

## Sign in and service status

Type the admin password and press **Sign in**. A wrong password shows an inline `Wrong admin password` error.
The password stays in `sessionStorage` for this browser tab, so reloading restores the session. **Sign out** clears
the password and any freshly created token. If browser storage is blocked, sign in again after reloading.

The header shows the service address and a health badge with an icon and text: **Healthy**, **Starting**, or
**Unreachable**. Health is checked without authentication every 10 seconds while signed in. **Starting** means the model
has not loaded through Ollama yet, or Ollama is unavailable. Tiles show model readiness, replicas, Ollama parallel requests, CPU threads, and active tokens.
Use **Theme** to cycle between system, light, and dark appearance; this preference is saved in the browser.

## Tokens

Each calling project gets its own token, so one can be revoked without touching the others.
- **Create:** type a project name (up to 64 characters) and press **Create token**. The token appears once in
  an accent-tinted box with a **Copy** button and the reminder “Shown once. Copy it before you leave.”
  The box is cleared on the next create attempt or sign-out. Clipboard access has a fallback for plain HTTP;
  if both methods are blocked, select the token and copy it manually.
- **Use:** the caller sends it in the `X-Internal-Token` header on every call except `/health`.
- **List:** the table shows **Project**, **Created**, **Last used**, and **Requests** (authenticated calls so far). Last used is **never** until a token
  authenticates a request, then a relative time; hover a timestamp for its exact local time. The list refreshes
  every 10 seconds. Token values are not shown again.
- **Revoke:** press the red-outlined **Revoke** pill and confirm in the in-app dialog, which names the project.
  **Cancel**, Escape, or a click on the backdrop closes the dialog. The caller gets `401 Invalid token` from then on. Lost
  token = revoke it and create a new one.

Last-used timestamps are Unix seconds or null. They are persisted at most once per minute per token, so they
are approximate. Request counts are added to the file with that same write; uses between writes are
counted in memory, so a crash can lose up to a minute of them. Old token files without the field still load and show **never** until next use. A companion
`tokens.json.lock` file serializes updates across replicas without adding dependencies.

Tokens live in `tokens.json` (`SERVICE_TOKENS_FILE`). In Docker that is the `cnv_model_data` volume at `/data`, so they
survive `docker compose down` and rebuilds. Without Docker it is `tokens.json` in the folder the service starts from.
Tokens are enforced only while `SERVICE_ADMIN_PASSWORD` is set.

## Models and builds

The **Models** card lists every model in `configs/models/` (LFM3, Spark X2.6, MiniCPM5, BitNet and Inkling 2 Mini), and under each model its quantized builds. Each row shows the estimated memory it takes (the file size plus a margin; not measured), the download size (the live size from the Hub when it can be read, else the stored size), whether the engine already has it, and whether it would run on the GPU or the CPU (slower). The top line shows what this machine has. A build that fits neither the free GPU memory nor the free RAM is greyed with the reason and cannot be chosen. When the engine runs on another machine its memory cannot be seen, so nothing is blocked and the card says so. Each model is one card: a header (name, the selected build and engine, capabilities, a state pill with a `?` that explains it, a Turn on or Turn off button and Preload) over a table with one row per build and engine (build, note, engine, memory estimate, download size, and three icon buttons: Select, Download and Uninstall). A button that is already true (the selected build, downloaded weights) turns green and cannot be pressed. A disabled button says why in its tooltip. Click a row for the build's details in a side drawer. Uninstall (the trash icon) asks for confirmation. A running download shows a progress bar with its percentage at the bottom of its row. While a model is in memory, its Select, Download and Uninstall are disabled (turn it off first); other models stay usable. Each card shows its state: **Off** (refuses requests), **On, not in memory** (takes requests; the first one loads the model, which can take minutes for a large one, and Ollama frees it again after its keep-alive) or **On, in memory** (answers at once). The page asks Ollama which models are in memory, so the pill does not stay on "in memory" after Ollama freed the model. **Turn on** allows requests and loads nothing; **Preload** loads the selected build now (disabled when Off or already in memory); **Turn off** unloads it and refuses requests. Under each build are the engines that can run it. **Select** (only for a downloaded build; nothing is selected by default) saves that build and engine for the model and every replica applies it within a second; **Download** (per engine) fetches the weights (Ollama pull or a GGUF file) and is the only place weights are fetched. The build that is already loaded counts as fitting.

## CPU threads

The shared card saves values from 1 to 256 and sends them to Ollama as `num_thread` with every request, so a change applies to the next question on every replica within a few seconds. Ollama reloads the model once when the value changes, so that first answer is slower. Set it to the number of physical CPU cores: Ollama can default to logical CPUs and run slower on a small CPU. Leave it unset to let Ollama decide.

## Replicas and parallel requests (read-only)

The tiles show replicas and parallel generations from `SERVICE_REPLICAS` and `SERVICE_OLLAMA_PARALLEL`: each replica runs that many generations at once (default 2; set it to the `OLLAMA_NUM_PARALLEL` of your Ollama).
The muted line says **Replicas: N. Change SERVICE_REPLICAS in .env.compose (or .env on the deployment device), then rebuild.** The
page cannot change them. To change, set them in `.env.compose` (or `.env` on the deployment device), then:
```bash
docker compose --env-file .env.compose up -d --build
docker compose restart cnv_model_lb
```
`unknown` means the variable is not set, which is normal without Docker. Without Docker there is one process.

## Problems

| Symptom | Fix |
|---|---|
| Page does not load | Service not running, wrong port, or blocked by a firewall. Test `/health`. |
| `Wrong admin password` | Password differs from `SERVICE_ADMIN_PASSWORD` in the running container. Recreate it after editing `.env.compose` or `.env`. |
| `Admin password required (set SERVICE_ADMIN_PASSWORD)` | The password is empty. Set it and restart. |
| Token lost | Revoke it and create a new one. |
| `/health` shows `ready: false` | The model is still loading in Ollama, Ollama is unavailable, or the model has not been pulled (see the container logs). |
| Never expose port 29204 to the internet | It is a LAN tool; use a firewall or VPN for remote access. |

## Legacy token scopes

The shared token UI still has a scope input, but this service refuses any project list with 422 because all calls use one model. Leave it empty: omitted or null `projects` is accepted for editor compatibility. Legacy stored scope labels are hidden. The removed `project` request field also returns 422.

## Rate limit, expiry, edits, usage and metrics

- **Requests per minute** (optional, when creating or editing a token): over the limit a call gets `429` with `Retry-After`. The count is per replica, so with several replicas behind the balancer the real limit can be up to that many times higher.
- **Expires in (days)** (optional, only when creating): after that the token is refused like a revoked one, and the row shows "Expired". Create a new token to continue.
- **Edit** changes the rate limit without revoking the token; leave the inherited project scope field empty.
- **Usage:** each row shows errors (answers with status 500 or above) and the average time per call. The counters are written at most once a minute.
- **Metrics:** `GET /metrics` (send the admin password in `X-Internal-Token`) returns the same numbers as Prometheus text, one series per token. It reads the shared token file, so any replica answers with the same data. `cnv_request_seconds` is a histogram of how long each `/v1` call took on the replica that answers (to the first byte for a streamed reply); it starts from 0 when the replica restarts.

## Settings

The **Settings** card changes options without editing `.env.compose`, `.env` or restarting: Ollama address, GGUF folder, parallel generations, caches and limits, thinking (off/on) and hide reasoning. Each row shows where its value comes from (saved, environment or default). A value saved here beats the environment variable; **Reset** removes it. Every replica reads the shared settings file, so a change applies within a few seconds. Rows marked "applies after a restart" are read once at start. The admin password, ports, replica count and other Docker settings stay in `.env` or the compose file, because the service cannot change its own container.

## Download progress and delete

Pressing **Download** shows a progress bar at the bottom of that build's row: percent, bytes done and the total once the engine reports them. It survives a page reload, and only one download runs at a time. **Uninstall** (the trash icon, enabled once a build is downloaded) removes it from Ollama or deletes the GGUF file, after a confirmation. The build a running model uses cannot be uninstalled, and a model that is in memory cannot change build, download or uninstall at all: turn it off first. For BitNet, restart `cnv_bitnet_server` afterwards to free its memory.

## Start, stop and restart

Each model on the Models page has **Start**, **Stop** and **Restart**. Stop unloads it from the engine and refuses its requests (503) until you start it; Start loads it now. The state (`stopped`, `running`, `ready`) is shown next to the title and saved for every replica.

## Marketplace

The **Marketplace** page lists the other quantizations in each supported model's Hugging Face repository, with size and whether this machine can run them. **Add** puts one under the model on the Models page, where you download and run it; **Remove** takes it out again. Variants that do not fit are greyed with the reason and cannot be added, downloaded or run. If the fit cannot be checked the row says so and is allowed.

## Updating the model files

The service loads only downloaded weights and tokenizer files. Startup and requests never download; there is no warm-up and no default model. With no downloaded model, `/health` and `/admin/` answer, `ready` is false and inference returns 503 with an admin Download hint. First start: open `/admin/` → **Models and builds** → **Download** on a build, then **Select** it. Only Download fetches files (`ollama pull`, or a GGUF file for llama.cpp); a successful download makes every replica drop its loaded copy through shared storage and reload on its next request.

Admin curl alternative:

```bash
curl -X POST http://127.0.0.1:29204/admin/models/lfm3/download -H "X-Internal-Token: <ADMIN_PASSWORD>" -H "Content-Type: application/json" -d '{"quant": "Q4_K_M"}'
```

## Connect to this service

The **Connect to this service** card lists the addresses other services can use: the address you opened the page with, the same PC (`127.0.0.1`), another Docker container on the same network (the `_lb` name from the compose file), and a Docker container calling this PC (`host.docker.internal`), then every endpoint with a **Copy** button. Callers send a token from the tokens card in `X-Internal-Token`.

## Try it: send a real request

1. Sign in to `/admin/` and find **Try it**.
2. The token field is hidden. Signing in to the admin page makes a temporary public token named `playground (temporary)` (valid 1 day) and fills it in, and choices such as voices load by themselves. It is the public client token, not the admin password.
3. Choose an operation, complete its labeled fields and send it. Requests go to this service's public route, with the same token scope, rate/admission limits and cache as a separate client.
4. Read the HTTP status, elapsed time and response. NDJSON/SSE text appears while it arrives; JSON is formatted. HTTP errors, including 401, 422, 429 and 503, keep their response body visible. Use **Cancel** to abort an active request.
5. Copy the curl equivalent to repeat the request elsewhere. The token is always a placeholder; fill it locally. For an upload, provide the indicated input file. Multi-step operations show their follow-up request too.

Authentication is **a temporary public token made from the admin session and held only in memory**. The playground does not persist it in browser storage, put it in a URL, log it or include it in curl. Sign out or close the tab revokes it; opening the admin page in a second tab replaces it. It appears in the tokens list and counts as an active token while it lives. The existing admin sign-in is separate. A playground run consumes normal service quotas and can generate retained files where the public API normally does so; cancelling the browser fetch does not promise to undo work already accepted by a backend.

The browser caps uploads at **10 MiB**, displayed text at **256 KiB**, and binary audio/image responses at **32 MiB**. Server limits still apply and may be lower. Media uses bounded Blob URLs, revoked when replaced or cleared. Text is rendered as text, never executed as HTML. Audio players become available after the bounded audio download finishes. No new environment setting is needed.

### This service's request

Chat sends messages to `POST /v1/chat`, with optional temperature, max_tokens and streaming. LFM3 rejects schema and thinking with 422; its form does not send those MiniCPM-only fields or tools (call the API for tools).

## Admin pages and navigation

Open `/admin/` and sign in. Overview opens first; each sidebar destination has its own URL fragment and supports browser Back/Forward and direct links:

- **Overview** (`#overview`) shows readiness, replicas, token usage, caller addresses and API endpoints.
- **Models** (`#variants`) shows lifecycle controls and expandable builds with fit, download, select and uninstall actions.
- **Marketplace** (`#marketplace`) lists extra repository builds. Add a build through its detail drawer, then download it in Models.
- **Tokens** (`#tokens`) lets you search, create, edit and revoke project tokens. Create and edit open in a right-side drawer.
- **Settings** (`#runtime`) shows CPU threads, replica information and runtime settings.
- **Updates** (`#updates`) shows installed builds and download activity. Individual downloads start in Models.
- **Try it** (`#playground`) sends a request for a model ID you select explicitly.

Use the header Sidebar button or Ctrl+B to switch between a labeled navigation list and an icon rail on desktop. On phones, the button opens navigation above the content. Profile holds Light, Dark, System and Sign out. Closing a detail drawer returns focus to its opener. Sign out clears credentials and cancels browser requests.
