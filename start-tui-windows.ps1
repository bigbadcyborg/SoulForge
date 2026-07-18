$ProjectRoot = $PSScriptRoot

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

$wslPath = Convert-ToWslPath -WindowsPath $ProjectRoot
wsl -d Ubuntu -- bash -lc "cd '$wslPath' && ./start-chatbot.sh"
