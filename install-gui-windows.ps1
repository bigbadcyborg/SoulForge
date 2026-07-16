# Install the SoulForge desktop GUI into a native Windows Python venv.
#
# The GUI runs on Windows (screen/audio/hotkeys); the model + API server run in
# WSL. This script only sets up the Windows side.
#
# Run:  powershell -ExecutionPolicy Bypass -File .\install-gui-windows.ps1

$ProjectRoot = $PSScriptRoot
$VenvDir = Join-Path $ProjectRoot ".venv-gui"
$Requirements = Join-Path $ProjectRoot "gui\requirements-windows.txt"

$python = (Get-Command python -ErrorAction SilentlyContinue)
if (-not $python) {
    Write-Error "Python not found on PATH. Install Python 3.11+ for Windows first."
    exit 1
}

if (-not (Test-Path $VenvDir)) {
    Write-Host ">>> Creating GUI virtual environment (.venv-gui)..."
    & python -m venv $VenvDir
} else {
    Write-Host ">>> Using existing GUI virtual environment (.venv-gui)"
}

$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
Write-Host ">>> Upgrading pip..."
& $VenvPython -m pip install --upgrade pip

Write-Host ">>> Installing GUI dependencies..."
& $VenvPython -m pip install -r $Requirements

Write-Host ">>> Done. Start the GUI with .\start-gui-windows.ps1"
