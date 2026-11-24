# Changelog

## 0.1.0

- First release: one chat service for the models of `CNV_LFM3_Model`, `CNV_SparkX26_Model`, `CNV_MiniCPM52B_Model` and `CNV_BitNet2B_Model`, chosen by the `model` field.
- Catalog in `configs/models/*.json`; engines `ollama` and `llamacpp` (bitnet.cpp sidecar).
- Admin **Models** card: quantized builds under each model, live sizes from the Hub, per-build Use and Download.
- Ollama address, GGUF folder and parallel generations are live admin settings (env variables only seed the first start).
- `llamacpp` models carry their own sidecar `url` in the catalog; there is no `SERVICE_LLAMACPP_URL`.
- No default model: every request names its `model`, so there is no default setting, no warm-up and no Updates card (per-build Download replaces it).
- Not carried over from the old services: JSON-schema output and project checkpoints (MiniCPM, BitNet), the transformers engine (BitNet).
