# CNV_Model: the admin web UI

The admin UI at `/admin/` has a page for each job: service status, models, tokens and settings. It works with every deployment mode. Only the address differs:

| Run | Address |
|---|---|
| Docker or `run.bat` on this PC | `http://127.0.0.1:29204/admin/` |
| Docker on the deployment box | `http://<box-ip>:29204/admin/` |

If `SERVICE_PORT` is changed, use that port.

## Before you open it

The page needs the admin password: the value of `SERVICE_ADMIN_PASSWORD`.

- Docker Desktop on this device: set it in `.env.compose` next to `docker-compose.yml`, then `docker compose --env-file .env.compose up -d`.
- Deployment device: set it in `.env` next to `docker-compose.yml` (Deploy option 2 copies it), then `docker compose up -d`.
- No Docker: set it in `.env.local` (or `.env`) in this folder and start with `run.bat` (it passes `--dotenv`), or set it in the environment first. A real environment variable wins over both files, and `.env.local` wins over `.env`. A plain `python src/app.py` reads neither file.
- An empty password turns auth off, and the page answers `Admin password required (set SERVICE_ADMIN_PASSWORD)`. Set one to use the page.

## First start

1. Open `/admin/` and sign in.
2. Open **Models**.
3. Press **Download** on a build that fits.
4. Press **Select** on that build.
5. Send a request, for example on **Try it**. The first request loads the model.

Nothing is downloaded or selected by default.

## Sign in and the header

Type the admin password and press **Sign in**. A wrong password shows `Wrong admin password`. The password stays in `sessionStorage` for this browser tab, so a reload restores the session. If browser storage is blocked, sign in again after each reload.

The header holds:

- the sidebar button (also Ctrl+B). The sidebar is a labelled list or an icon rail. Narrow screens start with the rail. On phones the button opens the navigation, and Escape closes it.
- the service address and the page title.
- a health badge: **Healthy** (the service answers) or **Unreachable**. Health is checked without a password every 10 seconds.
- **?**, which describes the current page.
- the account menu: **Theme** (Dark, Light or System, saved in the browser) and **Sign out**. Sign out clears the password and any freshly created token.

The **?** buttons on panels open their help in a drawer on the right.

## Pages

Each page has its own URL fragment and supports browser Back, Forward and direct links.

| Page | Fragment | Use it to |
|---|---|---|
| Overview | `#overview` | Check readiness, tiles, service details, token usage and caller addresses. |
| Models | `#variants` | Download, select and uninstall builds. Turn models on and off. |
| Marketplace | `#marketplace` | Add other quantizations from the Hub. |
| Tokens | `#tokens` | Create, edit and revoke project tokens. |
| Settings | `#runtime` | Change CPU threads and runtime settings. |
| Updates | `#updates` | See how many builds are installed and whether a download runs. |
| Try it | `#playground` | Send a real request. |

## Overview

The **Models in memory** tile counts the loaded models out of all models; it refreshes every 10 seconds.

The readiness panel says **Ready to serve** when at least one model is in memory. Otherwise it says **Model not ready**. That is normal right after a start, even with weights downloaded, because nothing loads until the first request or **Preload**.

**Connect to this service** lists the addresses other services can use: the address you opened the page with, the same PC (`127.0.0.1`), another Docker container on the same network (the `_lb` name from the compose file) and a Docker container calling this PC (`host.docker.internal`). It also lists each endpoint with a **Copy** button. Callers send a token in `X-Internal-Token`.

## Models

The top line shows what this machine has. Below it, each model in `configs/models/` has one card: LFM3, Spark X2.6, MiniCPM5, BitNet and Inkling 2 Mini.

### Card header

The header shows the model name, the selected build and engine (or **none**) and a state:

| State | Meaning |
|---|---|
| **Off** | The model refuses every request. |
| **On, not in memory** | It takes requests. The first one loads the model, which can take minutes for a large one. Ollama frees it again after its keep-alive. |
| **On, in memory** | It answers without a load step. |

The page asks Ollama which models are in memory, so the state stays correct after Ollama frees a model.

The header buttons are:

- **?**: explains the current state.
- the arrow: shows or hides the builds. The first model starts open. A card stays open while one of its builds downloads.
- **Turn on** or **Turn off**: **Turn on** allows requests and loads nothing. **Turn off** unloads the model and refuses its requests (503). Off models stay off after a restart, on every replica.
- **Preload**: loads the selected build now. It is disabled when the model is off or already in memory. It fails with a message when no build is selected.

### Build rows

Each row is one build on one engine. It shows the build name, a **Selected** tag where it applies, a note, and chips for the engine, the memory estimate, the download size and whether it fits (**Fits on GPU**, **Fits on CPU**, **Does not fit** or **Fit unknown**). The line above the buttons says why an action is not available.

The memory estimate is the file size plus a margin. It is not measured. The download size is the live size from the Hub when it can be read, else the stored size. A build that does not fit cannot be downloaded, selected, loaded or run. When the engine runs on another machine, its memory cannot be seen, so nothing is blocked.

| Button | What it does |
|---|---|
| **Select** | Saves this build and engine for the model. Every replica applies it within a second. It needs a downloaded build. |
| **Download** | Fetches the weights: an Ollama pull, or a GGUF file for llama.cpp. This is the only place that fetches weights. |
| **Uninstall** | Removes the downloaded build after a confirmation. Ollama deletes it from its storage; llama.cpp deletes the GGUF file. |

**Select** and **Download** show as pressed when already true.

A build added from the Marketplace has an **Added** tag and a **Remove** button, which takes it out of the model's list. Uninstall its weights first. The selected build cannot be removed.

While a model is in memory, all three buttons are disabled for that model. Press **Turn off** first. Other models stay usable. The selected build of a model that is on cannot be uninstalled: turn the model off or select another build first.

### Downloads

A running download shows a progress bar under that row's buttons, with the percent and the bytes once the engine reports them. The bar comes back after a page reload. Only one download runs at a time. After a download or an uninstall, every replica drops its loaded copies and loads again on the next request.

For BitNet, the `cnv_bitnet_server` sidecar starts serving by itself once its file is downloaded. After an uninstall, run `docker compose restart cnv_bitnet_server` to free its memory.

The admin API can do the same two steps:

```bash
curl -X POST http://127.0.0.1:29204/admin/models/lfm3/download -H "X-Internal-Token: <ADMIN_PASSWORD>" -H "Content-Type: application/json" -d '{"quant": "Q4_K_M", "engine": "ollama"}'
curl -X PUT http://127.0.0.1:29204/admin/models/lfm3/run -H "X-Internal-Token: <ADMIN_PASSWORD>" -H "Content-Type: application/json" -d '{"quant": "Q4_K_M", "engine": "ollama"}'
```

## Marketplace

The page lists the quantizations in each supported model's Hugging Face repository that the model does not have yet, with their size and whether this machine can run them. **Search builds** filters the list. **Add** puts a variant under its model on **Models**, where you download and select it. A variant that does not fit cannot be added. If the fit cannot be checked, the row says so and adding is allowed.

An added variant leaves the Marketplace list. To remove it, use **Remove** on its row on **Models**; it then shows in the Marketplace again.

Added variants run on Ollama only. BitNet has none, because its sidecar holds one file. The listing needs the Hub; offline, the page says so and offers a retry.

## Tokens

Each calling project gets its own token, so one can be revoked without touching the others. **Search tokens** filters the table.

- **Create token:** opens a drawer. Type a project name (up to 64 characters) and press **Create token**. The token appears once, with a **Copy** button. If both copy methods are blocked, select the token and copy it by hand.
- **Use:** the caller sends it in the `X-Internal-Token` header on every call except `/health`.
- **Table:** **Project**, **Created**, **Last used** and **Requests**. Last used is **never** until the token authenticates a request. Hover a time for the exact value. Token values are not shown again.
- **Edit:** changes the rate limit without revoking the token.
- **Revoke:** asks for confirmation. The caller gets `401 Invalid token` from then on. For a lost token, revoke it and create a new one.
- **Requests per minute** (optional): over the limit, a call gets `429` with `Retry-After`. The count is per replica, so with several replicas the real limit can be that many times higher.
- **Expires in (days)** (optional, only when creating): afterwards the token is refused and the row shows **Expired**.
- **Restrict to specific projects:** leave it off. This service refuses a project list with 422.

Each row also shows errors (status 500 or above) and the average time per call. Counters are written at most once a minute, so a crash can lose up to a minute of them.

Tokens live in `tokens.json` (`SERVICE_TOKENS_FILE`). In Docker that is the `cnv_model_data` volume at `/data`, so they survive `docker compose down` and rebuilds. Without Docker it is `tokens.json` in the folder the service starts from. Tokens are enforced only while `SERVICE_ADMIN_PASSWORD` is set.

`GET /metrics` (admin password in `X-Internal-Token`) returns the same numbers as Prometheus text, one series per token. `cnv_request_seconds` is a histogram of `/v1` call times on the replica that answers (to the first byte for a streamed reply); it restarts from 0 with the replica.

## Settings

The page has runtime settings with a search box: Ollama address, GGUF folder, parallel generations, caches and limits, thinking and hide reasoning. Each row shows where its value comes from (saved, environment or default). Change values, then press **Save changes** or **Discard**. A saved value beats the environment variable; **Reset** removes it. Every replica applies a change within a few seconds. Rows marked "applies after a restart" are read once at start. The admin password, ports and replica count stay in `.env` or the compose file.

**CPU threads** saves a value from 1 to 256 and sends it to Ollama as `num_thread` with every request, so it applies from the next question on every replica. Ollama reloads a model once when the value changes, so that answer is slower. Set it to the number of physical CPU cores. Unset lets Ollama decide. BitNet's sidecar ignores it.

The page also notes the replica count, which it cannot change. To change it, set `SERVICE_REPLICAS` in `.env.compose` (or `.env` on the deployment device), then:

```bash
docker compose --env-file .env.compose up -d --build
docker compose restart cnv_model_lb
```

`unknown` means the variable is not set, which is normal without Docker.

## Updates

The page shows the number of installed builds and any running download, with **Refresh activity** and a link to **Models**. Downloads start on **Models**. There is no warm-up and no automatic download.

## Try it: send a real request

1. Open **Try it**.
2. The token is filled in for you. Signing in makes a temporary public token named `playground (temporary)`, valid for 1 day.
3. Enter **Model** (a catalog ID such as `lfm3`). **Engine** is optional; empty means the one you selected.
4. Set the message and, if you want, role, temperature, maximum tokens, streaming and tools (JSON, only for models with the `tools` capability).
5. Send. The page shows the HTTP status, the time and the response. Streamed text appears while it arrives. Errors such as 401, 422, 429 and 503 keep their response body visible. **Cancel** aborts a running request.
6. Copy the curl equivalent to repeat the request elsewhere. It holds a token placeholder; fill it in locally.

The playground token is held only in memory. It is never put in browser storage, a URL, a log or the curl text. Signing out or closing the tab revokes it, and opening the admin page in a second tab replaces it. It appears in the token list while it lives. A playground run uses normal service quotas, and cancelling does not undo work the service already accepted.

The browser caps uploads at 10 MiB and displayed text at 256 KiB. Server limits still apply. Responses are shown as text and never run as HTML.

## Problems

| Symptom | Fix |
|---|---|
| Page does not load | The service is not running, the port is wrong or a firewall blocks it. Test `/health`. |
| `Wrong admin password` | The password differs from `SERVICE_ADMIN_PASSWORD` in the running container. Recreate it after editing `.env.compose` or `.env`. |
| `Admin password required (set SERVICE_ADMIN_PASSWORD)` | The password is empty. Set it and restart. |
| Token lost | Revoke it and create a new one. |
| `/health` shows `ready: false` | No model is in memory. Send a request or press **Preload**. If that fails, check Ollama and the downloaded builds. |
| A request answers 503 | Read the message: the model is off, has no selected build or cannot run on this machine. |

Never expose port 29204 to the internet. It is a LAN tool; use a firewall or VPN for remote access.
