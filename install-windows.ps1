# SoulForge Windows installer
#
# Checks for WSL (installs Ubuntu if missing), then runs install-wsl.sh inside WSL.
#
# Usage (PowerShell):
#   .\install-windows.ps1
#   .\install-windows.ps1 -WithCuda
#
# WSL installation requires Administrator and may require a reboot.

[CmdletBinding()]
param(
    [switch]$WithCuda,
    [string]$Distro = "Ubuntu"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host ">>> $Message" -ForegroundColor Cyan
}

function Test-IsAdministrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]$identity
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Convert-ToWslPath {
    param([string]$WindowsPath)
    $resolved = (Resolve-Path -LiteralPath $WindowsPath).Path
    if ($resolved -match '^([A-Za-z]):\\(.*)$') {
        $drive = $matches[1].ToLower()
        $rest = $matches[2] -replace '\\', '/'
        return "/mnt/$drive/$rest"
    }
    throw "Could not convert path to WSL format: $WindowsPath"
}

function Test-WslAvailable {
    if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
        return $false
    }
    try {
        $null = wsl.exe --status 2>&1
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    }
}

function Get-InstalledWslDistros {
    $raw = wsl.exe -l -q 2>&1
    if ($LASTEXITCODE -ne 0) {
        return @()
    }
    $distros = @()
    foreach ($line in ($raw -split "`n")) {
        $name = $line.Trim()
        if ($name) {
            $distros += $name
        }
    }
    return $distros
}

function Install-WslUbuntu {
    Write-Step "WSL is not installed or not ready"

    if (-not (Test-IsAdministrator)) {
        Write-Host ""
        Write-Host "Administrator privileges are required to install WSL." -ForegroundColor Yellow
        Write-Host "Re-run PowerShell as Administrator, then execute:" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "  Set-Location '$ProjectRoot'" -ForegroundColor White
        Write-Host "  .\install-windows.ps1$(if ($WithCuda) { ' -WithCuda' })" -ForegroundColor White
        Write-Host ""
        exit 1
    }

    Write-Host "Installing WSL with $Distro (this may take several minutes)..." -ForegroundColor Yellow
    Write-Host "You may be prompted to reboot when installation finishes." -ForegroundColor Yellow
    Write-Host ""

    wsl.exe --install -d $Distro
    $exitCode = $LASTEXITCODE

    Write-Host ""
    if ($exitCode -ne 0) {
        Write-Host "WSL installation returned exit code $exitCode." -ForegroundColor Red
        Write-Host "Try manually: wsl --install -d Ubuntu" -ForegroundColor Yellow
        exit $exitCode
    }

    Write-Host "WSL installation initiated successfully." -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "  1. Reboot if Windows prompts you"
    Write-Host "  2. Open Ubuntu from the Start menu and complete first-time setup (username/password)"
    Write-Host "  3. Run this installer again from PowerShell:"
    Write-Host "       .\install-windows.ps1$(if ($WithCuda) { ' -WithCuda' })"
    Write-Host ""
    exit 0
}

function Resolve-WslDistro {
    param([string[]]$Installed)

    foreach ($name in $Installed) {
        if ($name -eq $Distro) {
            return $Distro
        }
    }

    foreach ($name in $Installed) {
        if ($name -like "*Ubuntu*") {
            return $name
        }
    }

    if ($Installed.Count -gt 0) {
        return $Installed[0]
    }

    return $null
}

Write-Host "SoulForge Windows Installer" -ForegroundColor Green
Write-Host "Project: $ProjectRoot"

if (-not (Test-WslAvailable)) {
    Install-WslUbuntu
}

$installed = Get-InstalledWslDistros
$targetDistro = Resolve-WslDistro -Installed $installed

if (-not $targetDistro) {
    Write-Step "No WSL Linux distribution found - installing $Distro"
    if (-not (Test-IsAdministrator)) {
        Write-Host "Administrator privileges required. Re-run PowerShell as Administrator." -ForegroundColor Yellow
        exit 1
    }
    wsl.exe --install -d $Distro
    Write-Host ""
    Write-Host "Complete Ubuntu first-time setup, then run this installer again." -ForegroundColor Yellow
    exit 0
}

Write-Step "Using WSL distribution: $targetDistro"

$wslProjectPath = Convert-ToWslPath -WindowsPath $ProjectRoot
$installScript = "$wslProjectPath/install-wsl.sh"

Write-Step "Checking install script in WSL"
$checkCmd = "test -f " + [char]39 + $installScript + [char]39 + "; if [ -f " + [char]39 + $installScript + [char]39 + " ]; then echo ok; fi"
$scriptCheck = wsl.exe -d $targetDistro -- bash -lc $checkCmd
if ($scriptCheck -ne "ok") {
    Write-Host "ERROR: install-wsl.sh not found at $installScript" -ForegroundColor Red
    exit 1
}

Write-Step "Normalizing line endings for install-wsl.sh"
$normalizeCmd = 'sed -i ''s/' + [char]13 + '$//' + ''' ' + [char]39 + $installScript + [char]39 + '; chmod +x ' + [char]39 + $installScript + [char]39
wsl.exe -d $targetDistro -- bash -lc $normalizeCmd

$cudaFlag = ""
if ($WithCuda) {
    $cudaFlag = "--with-cuda"
    Write-Host "CUDA build enabled (llama-cpp-python will compile from source)" -ForegroundColor Yellow
}

Write-Step "Running WSL install ($targetDistro)"
Write-Host "This installs apt packages, creates .venv-wsl, and pip installs dependencies."
Write-Host ""

$runCmd = "cd " + [char]39 + $wslProjectPath + [char]39 + "; ./install-wsl.sh " + [char]39 + $wslProjectPath + [char]39 + " $cudaFlag"
wsl.exe -d $targetDistro -- bash -lc $runCmd
$exitCode = $LASTEXITCODE

Write-Host ""
if ($exitCode -eq 0) {
    Write-Host "=== Windows-side install finished ===" -ForegroundColor Green
    Write-Host ""
    Write-Host "Start the chatbot:" -ForegroundColor Cyan
    Write-Host "  .\start-chatbot-windows.ps1" -ForegroundColor White
} else {
    Write-Host "Install failed with exit code $exitCode" -ForegroundColor Red
    exit $exitCode
}
