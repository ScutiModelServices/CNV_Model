# Changelog

## 0.1.0

- Fix: a stopped model refuses requests even when the answer is cached (the cache was checked before the stop).
- Fix: the sidebar collapse button did nothing on screens 1000 px wide or less (the rail was forced to 60 px). It now toggles between the icon rail and a 168 px sidebar there; narrow screens start as the rail.
- Whole admin page restyled like the Citadel admin apps: dark top bar (service, health, profile menu with the theme choice and Sign out), flat sidebar with the collapse button at its bottom (Ctrl+B), a page title with a one-line lede, panels and stat tiles on every page. Models and Marketplace get one panel per model with a builds table, state and build pills, helper popovers, a build-details drawer, a delete confirmation dialog and a download bar. `web/index.html` now differs from the copies in the other model services (`tools/check_sync.py` will report it).
- Models card redesigned: one card per model with a state chip (Stopped, Unloaded = takes requests but not in memory, Loaded = in memory) and per-build chips (Selected, Downloaded, Not downloaded, Runs on). The old "Running" button on a build meant only "selected".
- First release: one chat service for the models of `CNV_LFM3_Model`, `CNV_SparkX26_Model`, `CNV_MiniCPM52B_Model` and `CNV_BitNet2B_Model`, chosen by the `model` field.
- Catalog in `configs/models/*.json`; engines `ollama` and `llamacpp` (bitnet.cpp sidecar).
- Admin **Models** card: quantized builds under each model, live sizes from the Hub, per-build Use and Download.
- Ollama address, GGUF folder and parallel generations are live admin settings (env variables only seed the first start).
- `llamacpp` models carry their own sidecar `url` in the catalog; there is no `SERVICE_LLAMACPP_URL`.
- The bitnet.cpp sidecar exits (not restarted) on a CPU without AVX2.
- A build can require a CPU feature (`cpu` in the catalog): `bitnet2b` needs AVX2, so it is refused on CPUs without it instead of answering `!!!!`.
- Fix: the bitnet.cpp sidecar rejected `--jinja`; it now starts without chat-template flags and the service sends the BitNet prompt to `/completion` (catalog option `prompt: bitnet`).
- Download progress bar on the Models page (one download at a time) and a Delete button for downloaded builds.
- Start, Stop and Restart per model (stopped models refuse requests with 503); a Marketplace page that lists and adds other Hub quantizations of the supported models; builds that do not fit this machine cannot be downloaded, run, started or added.
- A build lists the engines that can run it and the user chooses: **Select** on the Models card, or `engine` in a request. Several models can stay loaded at once.
- No default model: every request names its `model`, so there is no default setting, no warm-up and no Updates card (per-build Download replaces it).
- Not carried over from the old services: JSON-schema output and project checkpoints (MiniCPM, BitNet), the transformers engine (BitNet).
