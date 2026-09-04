<#
.SYNOPSIS
    Loads .env into the environment, then starts Claude Code.

.DESCRIPTION
    Claude Code expands ${VAR} in .mcp.json from the system environment, but it
    does not read .env files. This launcher bridges that gap so the only thing
    you need on a new machine is a .env file.

    Any arguments are forwarded to claude, e.g.  .\start.ps1 --resume
#>

$ErrorActionPreference = "Stop"
$envFile = Join-Path $PSScriptRoot ".env"

if (-not (Test-Path $envFile)) {
    Write-Host "No .env found at $envFile" -ForegroundColor Yellow
    Write-Host "Create one with:  Copy-Item .env.example .env   (then fill in values)"
    exit 1
}

$loaded = @()
foreach ($line in Get-Content $envFile) {
    $trimmed = $line.Trim()
    # Skip blanks and comments
    if ($trimmed -eq "" -or $trimmed.StartsWith("#")) { continue }
    if ($trimmed -notmatch "^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$") { continue }

    $name  = $Matches[1]
    $value = $Matches[2].Trim()
    # Strip surrounding quotes if present
    if ($value.Length -ge 2 -and
        (($value.StartsWith('"') -and $value.EndsWith('"')) -or
         ($value.StartsWith("'") -and $value.EndsWith("'")))) {
        $value = $value.Substring(1, $value.Length - 2)
    }
    if ($value -eq "") { continue }

    Set-Item -Path "Env:$name" -Value $value
    $loaded += $name
}

if ($loaded.Count -eq 0) {
    Write-Host "Warning: .env contained no usable variables." -ForegroundColor Yellow
} else {
    Write-Host "Loaded from .env: $($loaded -join ', ')" -ForegroundColor DarkGray
}

& claude @args
