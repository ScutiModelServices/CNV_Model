#!/bin/sh
# Wait for the GGUF (the admin Download button writes it, renamed into place only when complete), then run the server.
# Without this the container would exit and restart in a loop until the weights exist.
MODEL_FILE="${MODEL_FILE:-/models/microsoft/BitNet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf}"
until [ -f "$MODEL_FILE" ]; do
  echo "Waiting for $MODEL_FILE (press Download on the bitnet2b build in /admin/)"
  sleep 10
done
exec llama-server --host 0.0.0.0 --port 8080 -c 4096 -np 1 --special --jinja --chat-template-file /etc/bitnet.jinja -m "$MODEL_FILE"
