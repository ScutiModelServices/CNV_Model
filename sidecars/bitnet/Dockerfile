# syntax=docker/dockerfile:1
# bitnet.cpp's llama-server for the `bitnet2b` model. Static, no OpenMP, so the slim image needs no extra libraries.
# Same build as CNV_BitNet2B_Model/Dockerfile (pinned commit: later BitNet versions changed the i2_s kernels).
FROM python:3.12-slim AS cpp
RUN apt-get update && apt-get install -y --no-install-recommends git cmake clang make ca-certificates && rm -rf /var/lib/apt/lists/*
ARG BITNET_CPP_REF=404980eecae38affa4871c3e419eae3f44536a95
RUN git clone https://github.com/microsoft/BitNet /bitnet && cd /bitnet && git checkout "$BITNET_CPP_REF" && git submodule update --init --recursive --depth 1
WORKDIR /bitnet
RUN if [ "$(uname -m)" = x86_64 ]; then python utils/codegen_tl2.py --model bitnet_b1_58-3B --BM 160,320,320 --BK 96,96,96 --bm 32,32,32; else python utils/codegen_tl1.py --model bitnet_b1_58-3B --BM 160,320,320 --BK 64,128,64 --bm 32,64,32; fi
RUN CC=clang CXX=clang++ cmake -B build -DBUILD_SHARED_LIBS=OFF -DGGML_OPENMP=OFF -DBITNET_X86_TL2=OFF -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF && cmake --build build --target llama-server -j"$(nproc)"

FROM python:3.12-slim
COPY --from=cpp /bitnet/build/bin/llama-server /usr/local/bin/llama-server
# No chat template here: this bitnet.cpp server cannot read one, so the service writes the prompt and calls /completion (catalog: prompt "bitnet").
# One server, one model: the file the catalog's `bitnet2b` points at (shared /models volume, filled by the admin Download button).
# The entrypoint waits until that file exists, so the container does not restart in a loop before the first download.
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
