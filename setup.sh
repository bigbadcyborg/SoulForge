#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== SoulForge setup ==="
echo "This command wraps install-wsl.sh."
echo "Tip: use --with-cuda to rebuild llama-cpp-python with CUDA support."
echo

"$SCRIPT_DIR/install-wsl.sh" "$@"
