[CmdletBinding()]
param(
    [ValidateSet("3.11", "3.12")]
    [string]$PythonVersion = "3.11",
    [switch]$SkipBrowser
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$virtualEnvironment = Join-Path $projectRoot ".venv"
$virtualEnvironmentPython = Join-Path $virtualEnvironment "Scripts\python.exe"
$lockFile = Join-Path $projectRoot "requirements-py$($PythonVersion.Replace('.', '')).lock"

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command,
        [Parameter(Mandatory = $true, ValueFromRemainingArguments = $true)]
        [string[]]$Arguments
    )

    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: ${Command} $($Arguments -join ' ')"
    }
}

Push-Location $projectRoot
try {
    if (-not (Test-Path -LiteralPath $virtualEnvironmentPython -PathType Leaf)) {
        if (Test-Path -LiteralPath $virtualEnvironment) {
            throw "Existing .venv is incomplete. Remove it manually, then rerun this script."
        }
        $pythonLauncher = (Get-Command py -ErrorAction Stop).Source
        Invoke-Checked $pythonLauncher "-$PythonVersion", "-m", "venv", ".venv"
    }

    $installedVersion = (& $virtualEnvironmentPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
    if ($LASTEXITCODE -ne 0 -or $installedVersion -ne $PythonVersion) {
        throw ".venv must use Python ${PythonVersion}; it reports ${installedVersion}. Remove .venv manually before switching versions."
    }
    if (-not (Test-Path -LiteralPath $lockFile -PathType Leaf)) {
        throw "Missing runtime lock file: ${lockFile}"
    }

    Invoke-Checked $virtualEnvironmentPython "-m", "pip", "install", "--upgrade", "pip==25.1.1"
    Invoke-Checked $virtualEnvironmentPython "-m", "pip", "install", `
        "--find-links", "https://download.pytorch.org/whl/cpu/torch/", `
        "--require-hashes", "-r", $lockFile
    Invoke-Checked $virtualEnvironmentPython "-m", "pip", "install", "-r", "requirements-dev.txt"
    Invoke-Checked $virtualEnvironmentPython "-m", "pip", "install", "-e", ".", "--no-deps"
    Invoke-Checked $virtualEnvironmentPython "-m", "pip", "check"
    Invoke-Checked $virtualEnvironmentPython "scripts/verify_dependency_lock.py"

    if (-not $SkipBrowser) {
        $nodeVersion = (& node --version).Trim()
        if ($nodeVersion -notmatch '^v24\.') {
            throw "Node 24 is required for the browser regression suite; found ${nodeVersion}."
        }
        Invoke-Checked "npm" "ci", "--cache", ".npm-cache"
        Invoke-Checked "npx" "playwright", "install", "chromium"
    }

    Write-Output "Development environment is ready. Run: powershell -ExecutionPolicy Bypass -File scripts/check.ps1"
}
finally {
    Pop-Location
}
