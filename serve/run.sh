#!/bin/bash
# Entry point for the STT service. Runs in the reference venv, where torch and
# openai-whisper live; rawxrt itself only needs numpy, ml_dtypes and pyxrt.
# Paths derive from this script's location, so the checkout can live anywhere.
set -eu

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

: "${MLIR_AIE_DIR:=/opt/mlir-aie}"
# pyxrt is a distro package outside both virtualenvs.
export PYTHONPATH="${PYTHONPATH:-}${PYTHONPATH:+:}/usr/lib/python3/dist-packages"
export PEANO_INSTALL_DIR="${PEANO_INSTALL_DIR:-$MLIR_AIE_DIR/ironenv/lib/python3.13/site-packages/llvm-aie}"

if [ ! -x "$HERE/refenv/bin/python" ]; then
  echo "serve/run.sh: $HERE/refenv is missing." >&2
  echo "Build the image with --build-arg WITH_REFERENCE=1 (torch + openai-whisper)." >&2
  exit 1
fi

# Encoder weights are lifted out of the whisper checkpoint on the first start,
# which downloads ~1.6 GB into the openai-whisper cache.
exec "$HERE/refenv/bin/python" -u serve/npu_stt_server.py
