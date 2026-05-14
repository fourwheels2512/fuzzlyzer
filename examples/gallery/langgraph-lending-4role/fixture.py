"""langgraph-lending-4role: gallery wrapper around the home demo fixture.

Reuses the build_graph from ``topofuzz.fixtures.lending_4role`` (the
canonical 4-agent lending-decision pipeline shipped with topofuzz)
but overrides LLM_CONFIGS to all-Flash to keep the gallery run cost
modest. The packaged ``topofuzz.fixtures.lending_4role`` uses a mixed
flash + pro setup tuned for §6 role_swap pair coverage; for the
gallery, all-flash gives more swappable pairs AND ~25x lower per-call
cost. Observed cost on the v0.5 reference run: ~$1.20 at
``--n 8 --variants 4`` (BLUEPRINT 19a's ``<$1`` target was missed by
~20%; the cost estimator under-estimates real Gemini Flash output cost
by ~3-5x).

The eval set + rubric live in this directory (``eval.csv`` and
``rubric.txt``, both byte-copies of the packaged
``topofuzz.fixtures.lending_eval.csv`` and
``topofuzz.prompts.lending_judge.txt`` respectively) so the gallery
entry is self-contained.
"""

from __future__ import annotations

from topofuzz.fixtures.lending_4role import build_graph as _build_graph
from topofuzz.ir import LLMConfig


AGENT_IDS: tuple[str, ...] = ("intake", "risk", "policy", "decision")

LLM_CONFIGS: dict[str, LLMConfig] = {
    "intake": LLMConfig(model="google/gemini-2.5-flash", temperature=0.0, max_tokens=4000),
    "risk": LLMConfig(model="google/gemini-2.5-flash", temperature=0.0, max_tokens=4000),
    "policy": LLMConfig(model="google/gemini-2.5-flash", temperature=0.0, max_tokens=4000),
    "decision": LLMConfig(model="google/gemini-2.5-flash", temperature=0.0, max_tokens=4000),
}


def build_graph():
    """Return the canonical lending_4role graph (delegated to the package)."""
    return _build_graph()
