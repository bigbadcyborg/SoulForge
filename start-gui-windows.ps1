# Launch the SoulForge desktop GUI (Windows) against the WSL API server.
#
# 1. Starts the API server inside WSL Ubuntu (background) unless it is already up.
# 2. Waits for the server to answer on localhost.
# 3. Launches the native Windows GUI from its own venv (.venv-gui).
#
# Run:  powershell -ExecutionPolicy Bypass -File .\start-gui-windows.ps1

$ProjectRoot = $PSScriptRoot
$Port = if ($env:SOULFORGE_PORT) { $env:SOULFORGE_PORT } else { 8765 }
$PingUrl = "http://127.0.0.1:$Port/api/ping"

function Convert-ToWslPath {
    param([string]$WindowsPath)
    $resolved = (Resolve-Path -LiteralPath $WindowsPath).Path
    if ($resolved -match '^([A-Za-z]):\\(.*)$') {
        $drive = $matches[1].ToLower()
        $rest = $matches[2] -replace '\\', '/'
        return "/mnt/$drive/$rest"
    }
    throw "Could not convert path: $WindowsPath"
}

function Test-Server {
    try {
        $null = Invoke-WebRequest -Uri $PingUrl -TimeoutSec 3 -UseBasicParsing
        return $true
    } catch {
        return $false
    }
}

if (Test-Server) {
    Write-Host ">>> API server already running on port $Port."
} else {
    Write-Host ">>> Starting SoulForge API server in WSL..."
    $wslPath = Convert-ToWslPath -WindowsPath $ProjectRoot
    Start-Process -WindowStyle Minimized wsl -ArgumentList @(
        "-d", "Ubuntu", "--", "bash", "-lc", "cd '$wslPath' && ./start-server.sh"
    )

    Write-Host ">>> Waiting for the model to load (this can take a minute)..."
    $ready = $false
    for ($i = 0; $i -lt 120; $i++) {
        if (Test-Server) { $ready = $true; break }
        Start-Sleep -Seconds 2
    }
    if (-not $ready) {
        Write-Error "Server did not become ready. Check the WSL window for errors."
        exit 1
    }
}

$GuiPython = Join-Path $ProjectRoot ".venv-gui\Scripts\python.exe"
if (-not (Test-Path $GuiPython)) {
    Write-Error "GUI venv not found. Run .\install-gui-windows.ps1 first."
    exit 1
}

Write-Host ">>> Launching SoulForge GUI..."
& $GuiPython -m gui.app
