# legacy/

Pre-rename artifacts from before `c2632de` (modelbrew-core → fuzzeval) and the v5+
restructure. Kept in-tree for historical reference; **not exercised by CI** and not
imported by `topofuzz` / `fuzzeval`.

Contents
--------

- `pramana_test.py` — early pramana-eval validation script.
- `test_scorers.py` — pramana scorer tests; imports `pramana_test` via the
  bare module name. Run from inside this directory (`cd legacy && pytest test_scorers.py`)
  so Python's sys.path includes the local `pramana_test.py`.
- `validate_results.py` — pramana results validator.
- `reproduce.sh`, `reproduce.ps1` — original reproduce harness pointing at
  `pramana_test.py`.

Why kept
--------

These artifacts back the published pramana-eval results (Gemini 2.5 Pro 9/10 format /
10/10 semantic) referenced in the v5+ launch playbook. Deleting them would lose the
reproduction story; keeping them in `legacy/` keeps `git log --follow` history while
excluding them from the active build surface.

If you're looking for the current build, see `packages/topofuzz/` and
`packages/fuzzeval/`.
