#!/bin/sh
# 1. Stop at once on a CPU the bitnet.cpp kernel cannot run on (exit 0: compose does not restart it, `docker compose ps`
#    shows it as exited). On such a CPU the server answers "!!!!" forever.
# 2. Wait for the GGUF (the admin Download button writes it, renamed into place only when complete), then run the server.
#    Without this the container would exit and restart in a loop until the weights exist.
CPUINFO="${CPUINFO:-/proc/cpuinfo}"
MODEL_FILE="${MODEL_FILE:-/models/microsoft/BitNet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf}"
if ! grep -qw avx2 "$CPUINFO" 2>/dev/null; then
  echo "Not started: the bitnet.cpp i2_s kernel needs a CPU with AVX2 (the catalog requires it for bitnet2b), and this CPU has none."
  exit 0
fi
until [ -f "$MODEL_FILE" ]; do
  echo "Waiting for $MODEL_FILE (press Download on the bitnet2b build in /admin/)"
  sleep 10
done
exec llama-server --host 0.0.0.0 --port 8080 -c 4096 -np 1 --special -m "$MODEL_FILE"
