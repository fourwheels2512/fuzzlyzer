#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${ANTHROPIC_API_KEY:-}" && -z "${GEMINI_API_KEY:-}" && -z "${OPENAI_API_KEY:-}" ]]; then
    echo "Set one of: ANTHROPIC_API_KEY, GEMINI_API_KEY, OPENAI_API_KEY" >&2
    exit 1
fi

python -m pip install -r requirements.txt --quiet
python pramana_test.py "$@"
echo "Done. Results in pramana_test_results.json"
