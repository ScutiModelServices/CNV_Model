# Changelog

## 0.1.0

- First release: one chat service for the models of `CNV_LFM3_Model`, `CNV_SparkX26_Model`, `CNV_MiniCPM52B_Model` and `CNV_BitNet2B_Model`, chosen by the `model` field.
- Catalog in `configs/models/*.json`; engines `ollama` and `llamacpp` (bitnet.cpp sidecar).
- Admin **Models** card: quantized builds under each model, live sizes from the Hub, per-build Use and Download.
- Not carried over from the old services: JSON-schema output and project checkpoints (MiniCPM, BitNet), the transformers engine (BitNet).
