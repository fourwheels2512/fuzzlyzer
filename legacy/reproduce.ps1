#!/usr/bin/env pwsh
$ErrorActionPreference = "Stop"

if (-not $env:ANTHROPIC_API_KEY -and -not $env:GEMINI_API_KEY -and -not $env:OPENAI_API_KEY) {
    Write-Error "Set one of: ANTHROPIC_API_KEY, GEMINI_API_KEY, OPENAI_API_KEY"
    exit 1
}

python -m pip install -r requirements.txt --quiet
python pramana_test.py @args
Write-Output "Done. Results in pramana_test_results.json"
