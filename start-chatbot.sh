#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.9}"
export CUDAToolkit_ROOT="${CUDAToolkit_ROOT:-$CUDA_HOME}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

VENV_DIR="$SCRIPT_DIR/.venv-wsl"
PYTHON_BIN="$VENV_DIR/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: WSL virtual environment not found at $VENV_DIR"
  echo "Run ./setup.sh --with-cuda from WSL, then start again."
  exit 1
fi

export VIRTUAL_ENV="$VENV_DIR"
export PATH="$VENV_DIR/bin:$PATH"

"$PYTHON_BIN" -m app.main

exec bash
