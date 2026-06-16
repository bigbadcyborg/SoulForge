#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

FAILS=0
WARNS=0

ok() {
  echo "[OK]   $1"
}

warn() {
  echo "[WARN] $1"
  WARNS=$((WARNS + 1))
}

fail() {
  echo "[FAIL] $1"
  FAILS=$((FAILS + 1))
}

echo "=== SoulForge Doctor ==="
echo "Project: $ROOT_DIR"
echo

# Environment checks
if [[ "$(uname -s)" == "Linux" ]]; then
  ok "Running in Linux environment"
else
  fail "Not running in Linux/WSL shell. Run this from Ubuntu on WSL."
fi

if grep -qi microsoft /proc/version 2>/dev/null; then
  ok "WSL environment detected"
else
  warn "WSL marker not detected in /proc/version (may still be fine on native Linux)"
fi

# Venv + Python checks
if [[ -d ".venv-wsl" ]]; then
  ok ".venv-wsl exists"
else
  fail ".venv-wsl missing. Run ./setup.sh"
fi

if [[ -f ".venv-wsl/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv-wsl/bin/activate
  ok "Activated .venv-wsl"
else
  fail "Cannot activate .venv-wsl/bin/activate"
fi

PYTHON_BIN="$(command -v python || true)"
if [[ -n "$PYTHON_BIN" && "$PYTHON_BIN" == *".venv-wsl/bin/python"* ]]; then
  ok "python points to virtual environment"
else
  warn "python is not from .venv-wsl ($PYTHON_BIN)"
fi

python - <<'PY'
import importlib
modules = ["yaml", "textual", "chromadb", "llama_cpp"]
missing = [m for m in modules if importlib.util.find_spec(m) is None]
if missing:
    print("MISSING:" + ",".join(missing))
else:
    print("MISSING:")
PY

MISSING_MODULES="$(python - <<'PY'
import importlib
modules = ["yaml", "textual", "chromadb", "llama_cpp"]
missing = [m for m in modules if importlib.util.find_spec(m) is None]
print(",".join(missing))
PY
)"
if [[ -z "$MISSING_MODULES" ]]; then
  ok "Required Python modules import successfully"
else
  fail "Missing Python modules: $MISSING_MODULES (run ./setup.sh)"
fi

# CUDA checks (non-fatal)
if command -v nvidia-smi >/dev/null 2>&1; then
  ok "nvidia-smi found"
else
  warn "nvidia-smi not found (GPU checks skipped)"
fi

LLAMA_SO="$(python - <<'PY'
try:
    import llama_cpp
    import pathlib
    p = pathlib.Path(llama_cpp.__file__).parent / "lib" / "libllama.so"
    print(p)
except Exception:
    print("")
PY
)"

if [[ -n "$LLAMA_SO" && -f "$LLAMA_SO" ]]; then
  if ldd "$LLAMA_SO" | grep -qi cuda; then
    ok "libllama.so links CUDA libraries"
  else
    warn "libllama.so does not appear CUDA-linked (CPU mode likely)"
  fi
else
  warn "libllama.so not found for CUDA linkage check"
fi

# Config/project checks
if [[ -f "config.yaml" ]]; then
  ok "config.yaml exists"
else
  fail "config.yaml missing (copy examples/config.example.yaml)"
fi

MODEL_CHECK="$(python - <<'PY'
import yaml
from pathlib import Path
cfg_path = Path("config.yaml")
if not cfg_path.exists():
    print("CONFIG_MISSING")
    raise SystemExit(0)
cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
model = cfg.get("model", {}) if isinstance(cfg.get("model"), dict) else {}
features = cfg.get("features", {}) if isinstance(cfg.get("features"), dict) else {}
chat = model.get("chatModelPath", "./models/NemoMix-Unleashed-12B-Q4_K_M.gguf")
emb = model.get("embeddingModelPath", "./models/embedding-model.gguf")
rag_on = bool(features.get("rag", False))
print(f"CHAT={Path(chat).exists()}")
print(f"EMB={Path(emb).exists()}")
print(f"RAG={rag_on}")
print(f"CHAT_PATH={chat}")
print(f"EMB_PATH={emb}")
PY
)"

if [[ "$MODEL_CHECK" == *"CONFIG_MISSING"* ]]; then
  :
else
  CHAT_EXISTS="$(echo "$MODEL_CHECK" | awk -F= '/^CHAT=/{print $2}')"
  EMB_EXISTS="$(echo "$MODEL_CHECK" | awk -F= '/^EMB=/{print $2}')"
  RAG_ON="$(echo "$MODEL_CHECK" | awk -F= '/^RAG=/{print $2}')"
  CHAT_PATH="$(echo "$MODEL_CHECK" | awk -F= '/^CHAT_PATH=/{print $2}')"
  EMB_PATH="$(echo "$MODEL_CHECK" | awk -F= '/^EMB_PATH=/{print $2}')"

  if [[ "$CHAT_EXISTS" == "True" ]]; then
    ok "Chat model path exists ($CHAT_PATH)"
  else
    fail "Chat model missing at $CHAT_PATH"
  fi
  if [[ "$RAG_ON" == "True" ]]; then
    if [[ "$EMB_EXISTS" == "True" ]]; then
      ok "Embedding model path exists ($EMB_PATH)"
    else
      warn "RAG enabled but embedding model missing at $EMB_PATH"
    fi
  fi
fi

if [[ -d "docs" ]]; then
  ok "docs/ exists"
else
  warn "docs/ missing (create it for RAG sources)"
fi

# Git hygiene
if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  BAD_PATHS="$(git status --porcelain | awk '{print $2}' | grep -E '^(models/|chromaDb/|\.venv-wsl/|logs/)' || true)"
  if [[ -z "$BAD_PATHS" ]]; then
    ok "Git hygiene looks clean for models/chromaDb/.venv-wsl/logs"
  else
    fail "Large/local paths staged or modified in git:\n$BAD_PATHS"
  fi
else
  warn "Not a git repository (skipping git hygiene checks)"
fi

echo
echo "=== Summary ==="
echo "Warnings: $WARNS"
echo "Failures: $FAILS"
echo
echo "Next steps:"
echo "  1. Copy starter config: cp examples/config.example.yaml config.yaml"
echo "  2. Place GGUF models under models/"
echo "  3. Re-run: ./doctor.sh"
echo "  4. Start app: ./start-chatbot.sh or .\\start-chatbot-windows.ps1"

if [[ "$FAILS" -gt 0 ]]; then
  exit 1
fi
