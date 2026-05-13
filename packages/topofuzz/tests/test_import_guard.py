"""§11a verify-path safety: ``import topofuzz.cli`` MUST NOT transitively
load the run-time stack (LangGraph, LLM SDKs, plotting, numpy).

This guard exists so a ``pip install topofuzz`` user can run
``topofuzz verify`` / ``topofuzz keygen`` on a verify-only host
without pulling in ~400 MB of optional deps. The lazy-import policy
in :mod:`topofuzz.cli` makes this true today; the test pins it down
so a future refactor that moves a heavy import to module scope fails
loudly in CI.

The check runs in a **subprocess** with a fresh interpreter — pytest
itself imports numpy and others elsewhere in the test session, so
introspecting the parent's ``sys.modules`` would yield false
positives. The subprocess imports only ``topofuzz.cli`` and then
introspects its own (clean) module table.

Test parametrisation covers ``topofuzz.cli`` (the CLI surface),
:mod:`topofuzz.attest` (the verify glue), and ``fuzzeval.attestation``
(the cryptographic primitives) — each of these is on a verify-path
load order and must stay clean. ``topofuzz`` and ``topofuzz.cli``
are not equivalent: a future ``topofuzz/__init__.py`` might
intentionally pull a heavy submodule, in which case this test would
need updating; ``topofuzz.cli`` is the codepath the OS shell hits.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

# Modules that MUST NOT be loaded after a verify-path import. These are
# the run-time stack pulled by ``topofuzz[full]``. ``google.genai`` uses
# a dot in its name; we check both top-level and dotted forms.
FORBIDDEN = (
    "langgraph",
    "litellm",
    "anthropic",
    "openai",
    "google.genai",
    "matplotlib",
    "jinja2",
    "numpy",
    "scipy",
)


VERIFY_PATH_MODULES = (
    "topofuzz.cli",
    "topofuzz.attest",
    "fuzzeval.attestation",
    "fuzzeval.keys",
)


@pytest.mark.parametrize("target", VERIFY_PATH_MODULES)
def test_verify_path_import_is_lean(target: str) -> None:
    """Importing ``target`` in a fresh Python must not transitively load
    any module in :data:`FORBIDDEN`.

    The subprocess approach is necessary: pytest itself imports numpy
    (via fuzzeval.stats) and matplotlib (via some report skills),
    contaminating the parent process's ``sys.modules``. A fresh
    interpreter avoids that.
    """
    script = textwrap.dedent(
        f"""
        import sys, importlib
        importlib.import_module({target!r})
        forbidden = {FORBIDDEN!r}
        offenders = sorted(
            m for m in forbidden
            if m in sys.modules
            or any(loaded.startswith(m + ".") for loaded in sys.modules)
        )
        if offenders:
            print("LEAKED:", ",".join(offenders))
            sys.exit(1)
        sys.exit(0)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        # 60s rather than 30: cold Windows CI under Defender real-time
        # scanning has been observed at 15–25s for a fresh Python startup
        # with cryptography wheel cold-load. Local runs finish under 2s
        # so the slack is invisible most of the time.
        timeout=60,
    )
    assert result.returncode == 0, (
        f"importing {target} transitively loaded forbidden modules.\n"
        f"stdout: {result.stdout!r}\n"
        f"stderr: {result.stderr!r}"
    )
