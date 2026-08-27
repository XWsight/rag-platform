$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$virtualEnvironmentPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$pythonExecutable = if (Test-Path -LiteralPath $virtualEnvironmentPython -PathType Leaf) {
    $virtualEnvironmentPython
} else {
    (Get-Command python -ErrorAction Stop).Source
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
    Invoke-CheckedPython -m compileall -q rag_system tests scripts
    Invoke-CheckedPython scripts/scan_secrets.py
    Invoke-CheckedPython scripts/verify_dependency_lock.py
    Invoke-CheckedPython scripts/audit_dependencies.py
    Invoke-CheckedPython -m ruff check .
    Invoke-CheckedPython -m mypy
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
