$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$virtualEnvironmentPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$configuredPython = $env:RAG_PYTHON
$pythonExecutable = if ($configuredPython) {
    if (-not (Test-Path -LiteralPath $configuredPython -PathType Leaf)) {
        throw "RAG_PYTHON must name an existing Python executable: ${configuredPython}"
    }
    $configuredPython
} elseif (Test-Path -LiteralPath $virtualEnvironmentPython -PathType Leaf) {
    $virtualEnvironmentPython
} else {
    throw @"
Missing project virtual environment. Create the supported development environment first:
  py -3.11 -m venv .venv
  .\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-dev.txt

To use another already-prepared Python 3.11 or 3.12 environment explicitly, set RAG_PYTHON to its executable path.
"@
}

$pythonVersion = (& $pythonExecutable -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Unable to determine Python version for ${pythonExecutable}."
}
if ($pythonVersion -notin @("3.11", "3.12")) {
    throw "scripts/check.ps1 requires Python 3.11 or 3.12; selected ${pythonExecutable} reports ${pythonVersion}."
}

function Invoke-CheckedPython {
    param(
        [Parameter(Mandatory = $true, ValueFromRemainingArguments = $true)]
        [string[]] $Arguments
    )

    & $pythonExecutable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

function Invoke-CheckedGit {
    param(
        [Parameter(Mandatory = $true, ValueFromRemainingArguments = $true)]
        [string[]] $Arguments
    )

    & git @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Git command failed with exit code ${LASTEXITCODE}: git $($Arguments -join ' ')"
    }
}

Push-Location $projectRoot
try {
    $nodeVersion = (& node --version).Trim()
    if ($nodeVersion -notmatch '^v24\.') {
        throw "Node 24 is required for the browser regression suite; found ${nodeVersion}."
    }
    $playwrightCli = Join-Path $projectRoot "node_modules\@playwright\test\cli.js"
    if (-not (Test-Path -LiteralPath $playwrightCli -PathType Leaf)) {
        throw "Browser test dependencies are missing. Run npm ci before scripts/check.ps1."
    }
    Invoke-CheckedPython -m compileall -q rag_system tests scripts
    Invoke-CheckedPython scripts/scan_secrets.py
    Invoke-CheckedPython scripts/verify_dependency_lock.py
    Invoke-CheckedPython scripts/audit_dependencies.py
    Invoke-CheckedPython -m ruff check .
    Invoke-CheckedPython -m mypy
    Invoke-CheckedPython scripts/verify_wheel.py
    Invoke-CheckedPython scripts/verify_openapi_contract.py
    Invoke-CheckedPython scripts/benchmark_sparse.py `
        evals/retrieval_cases.jsonl `
        evals/corpus/rag.md `
        evals/corpus/retrieval.md `
        evals/corpus/safety.md `
        evals/corpus/storage.md `
        --top-k 5 `
        --quality-gate evals/gates/bm25-smoke.json `
        --json-output reports/bm25-smoke.json `
        --markdown-output reports/bm25-smoke.md
    Invoke-CheckedPython scripts/validate_retrieval_suite.py `
        evals/retrieval_suite.json `
        --contract evals/gates/retrieval-suite.json `
        --json-output reports/retrieval-suite.json `
        --markdown-output reports/retrieval-suite.md
    Invoke-CheckedPython scripts/validate_answer_suite.py `
        evals/answer_suite.json `
        --contract evals/gates/answer-suite.json `
        --json-output reports/answer-suite.json `
        --markdown-output reports/answer-suite.md
    Invoke-CheckedPython scripts/benchmark_sparse.py `
        evals/retrieval_suite.json `
        --top-k 5 `
        --quality-gate evals/gates/bm25-foundation.json `
        --json-output reports/bm25-foundation.json `
        --markdown-output reports/bm25-foundation.md
    Invoke-CheckedPython -m coverage run -m unittest discover -s tests -v
    Invoke-CheckedPython -m coverage report
    & npm run test:browser
    if ($LASTEXITCODE -ne 0) {
        throw "Browser end-to-end test command failed with exit code ${LASTEXITCODE}."
    }
    Invoke-CheckedGit diff --check
}
finally {
    Pop-Location
}
