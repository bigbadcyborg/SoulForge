#!/usr/bin/env bash
# SoulForge WSL installer — run from install-windows.ps1 or directly inside WSL.
#
# Usage:
#   ./install-wsl.sh [/path/to/project] [--with-cuda]
#
# Options:
#   --with-cuda   Rebuild llama-cpp-python with CUDA (requires CUDA Toolkit in WSL)

set -euo pipefail

PROJECT_DIR=""
WITH_CUDA=0
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.9}"

for arg in "$@"; do
  case "$arg" in
    --with-cuda)
      WITH_CUDA=1
      ;;
    *)
      if [[ -z "$PROJECT_DIR" ]]; then
        PROJECT_DIR="$arg"
      fi
      ;;
  esac
done

if [[ -z "$PROJECT_DIR" ]]; then
  PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

cd "$PROJECT_DIR"

echo "=== SoulForge WSL install ==="
echo "Project: $PROJECT_DIR"
echo ""

echo ">>> Installing system packages (may prompt for sudo)..."
sudo apt update
sudo apt install -y \
  python3 \
  python3-pip \
  python3-venv \
  build-essential \
  cmake \
  ninja-build \
  git \
  tesseract-ocr \
  poppler-utils

VENV_DIR="$PROJECT_DIR/.venv-wsl"
PYTHON_BIN="$VENV_DIR/bin/python"

if [[ ! -d "$VENV_DIR" ]]; then
  echo ">>> Creating Python virtual environment (.venv-wsl)..."
  python3 -m venv "$VENV_DIR"
else
  echo ">>> Using existing virtual environment (.venv-wsl)"
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python executable not found at $PYTHON_BIN"
  echo "Remove .venv-wsl and rerun setup if the virtual environment is corrupted."
  exit 1
fi

export VIRTUAL_ENV="$VENV_DIR"
export PATH="$VENV_DIR/bin:$PATH"
hash -r 2>/dev/null || true

echo ">>> Upgrading pip..."
"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel

echo ">>> Installing Python dependencies from requirements.txt..."
"$PYTHON_BIN" -m pip install -r requirements.txt

if [[ "$WITH_CUDA" -eq 1 ]]; then
  echo ">>> Building llama-cpp-python with CUDA support..."
  if [[ ! -d "$CUDA_HOME" ]]; then
    echo "ERROR: CUDA not found at $CUDA_HOME"
    echo "Install CUDA Toolkit in WSL or set CUDA_HOME before running with --with-cuda"
    exit 1
  fi

  export CUDAToolkit_ROOT="$CUDA_HOME"
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

  "$PYTHON_BIN" -m pip uninstall -y llama-cpp-python || true

  CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=120 -DGGML_NATIVE=OFF"
  export CMAKE_ARGS
  export FORCE_CMAKE=1

  "$PYTHON_BIN" -m pip install --no-cache-dir --force-reinstall --no-binary llama-cpp-python llama-cpp-python

  echo ">>> Verifying CUDA linkage..."
  LLAMA_SO="$("$PYTHON_BIN" -c "import llama_cpp, pathlib; p=pathlib.Path(llama_cpp.__file__).parent/'lib'/'libllama.so'; print(p)" 2>/dev/null || true)"
  if [[ -n "$LLAMA_SO" && -f "$LLAMA_SO" ]]; then
    ldd "$LLAMA_SO" | grep -i cuda || echo "WARNING: CUDA libraries not detected in libllama.so"
  fi
else
  echo ""
  echo "NOTE: llama-cpp-python was installed from PyPI (may be CPU-only)."
  echo "      For GPU offload, re-run with: ./install-wsl.sh --with-cuda"
  echo "      (Requires CUDA Toolkit 12.9+ in WSL — see README)"
fi

mkdir -p docs models chromaDb

echo ""
echo "=== Install complete ==="
echo ""
echo "Next steps:"
echo "  1. Place GGUF models in: $PROJECT_DIR/models/"
echo "  2. Place RAG documents in: $PROJECT_DIR/docs/"
echo "  3. From Windows PowerShell: .\\start-chatbot-windows.ps1"
echo "     Or inside WSL: ./start-chatbot.sh"
echo ""
