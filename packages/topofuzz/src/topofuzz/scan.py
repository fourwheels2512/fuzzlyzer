# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""Static risk scan (BLUEPRINT_v5plus.md §5a) — the gateway-drug feature.

Walks a :class:`topofuzz.ir.Topology` and emits a ``HIGH`` / ``MEDIUM`` /
``LOW`` risk badge from a fixed table of rule-based structural patterns
derived from the Bajaj 2026 finding. Zero LLM cost. Pure-Python. Runs in
milliseconds.

Public surface:

* :func:`scan` — main entry, ``Topology -> ScanResult``.
* :class:`ScanResult` — frozen dataclass with ``badge``,
  ``matched_patterns`` (the ordered list of rule ids that fired), and
  ``rationale`` (single-line summary suitable for embedding in JSON).

Verify-path safety (BLUEPRINT §11a): this module imports only
:mod:`topofuzz.ir`. It does NOT pull langgraph, LLM SDKs, or any of the
``topofuzz[full]`` stack. ``test_import_guard.py`` enforces that
``topofuzz.cli`` (which now imports this module lazily inside ``scan()``)
stays verify-path lean.

The rule table is BLUEPRINT-literal — see ``§5a``. Two notes on
interpretation that the BLUEPRINT under-specifies:

* "No judge node" is treated as "no aggregator node" because the IR has
  no ``kind="judge"`` and §5a line 152 itself uses ``aggregator
  (vote/judge)`` parenthetically — the two are synonyms at the IR
  level. The LLM judge is external to the topology (see :mod:`fuzzeval.judge`).
* "Linear chain of N agents" counts every node with ``kind="agent"``.
  The default :func:`topofuzz.adapter.from_langgraph` labels every
  LangGraph node as ``agent``; router/aggregator/human kinds appear
  only when a fixture constructs the IR directly. The scanner is
  designed to give the BLUEPRINT-literal answer in both cases.

The scan is a heuristic, not a measurement. False positives and false
negatives both exist. ``topofuzz run`` is the ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from topofuzz.ir import Node, Topology

Badge = Literal["LOW", "MEDIUM", "HIGH"]

# Risk ladder. The +1 RISK STEP promotions in §5a operate on this index.
# Order is significant — _STEP_TO_BADGE[i+1] is the promotion target of
# _STEP_TO_BADGE[i]. Capped at HIGH (no overflow above index 2).
_STEP_TO_BADGE: tuple[Badge, ...] = ("LOW", "MEDIUM", "HIGH")
_BADGE_TO_STEP: dict[Badge, int] = {b: i for i, b in enumerate(_STEP_TO_BADGE)}
_HIGH_STEP = _BADGE_TO_STEP["HIGH"]

# Pattern ids — stable strings, safe to depend on from JSON consumers.
# Renaming any of these is a breaking change to the scan.json shape.
_P_LINEAR_3PLUS_NO_AGG = "linear_chain_3plus_no_aggregator"
_P_LINEAR_2_NO_AGG = "linear_chain_2_no_aggregator"
_P_TERMINAL_AGGREGATOR = "chain_with_terminal_aggregator"
_P_MIDFLOW_ROUTER = "chain_with_router_midflow"
_P_ALL_SHARE_MODEL = "all_agents_share_model"
_P_TEMPERATURE_GT_0 = "temperature_gt_0"

# Headline rationale per base pattern. The promotion patterns get their
# own one-liner appended downstream. The Bajaj citations are BLUEPRINT-
# literal (§3.2 for the 31% number, §4.1 for the heterogeneity claim).
_BASE_HEADLINES: dict[str, str] = {
    _P_LINEAR_3PLUS_NO_AGG: (
        "{n}-agent linear chain with no aggregator or judge node. Per "
        "Bajaj 2026 §3.2, this topology family showed up to 31% decision "
        "flips when agents were reordered."
    ),
    _P_LINEAR_2_NO_AGG: (
        "2-agent linear chain with no aggregator. Smaller surface than "
        "3+ chains but still ordering-dependent."
    ),
    _P_TERMINAL_AGGREGATOR: (
        "Chain terminates in an aggregator (vote/judge) node — the "
        "aggregator absorbs upstream variance."
    ),
    _P_MIDFLOW_ROUTER: (
        "Router node mid-flow — router decisions may flip on permutation."
    ),
}

_PROMOTION_HEADLINES: dict[str, str] = {
    _P_ALL_SHARE_MODEL: (
        "all agents share llm_config.model (heterogeneity is the "
        "variance amplifier per Bajaj §4.1)"
    ),
    _P_TEMPERATURE_GT_0: (
        "≥1 agent has temperature > 0 (decoding noise stacks with "
        "ordering noise)"
    ),
}


@dataclass(frozen=True)
class ScanResult:
    """Result of a §5a static risk scan.

    Attributes:
        badge: ``"LOW"`` | ``"MEDIUM"`` | ``"HIGH"`` — the final risk
            badge after base-rule selection and promotion.
        matched_patterns: Tuple of pattern ids that fired, in the
            order they were checked. Base rules first, then promotions.
            Stable strings — safe to test against in CI.
        rationale: Single-line human summary. Embedded verbatim in the
            scan.json. The CLI banner re-formats this with newline
            wrapping + a Heuristic-only disclaimer.
    """

    badge: Badge
    matched_patterns: tuple[str, ...]
    rationale: str

    def to_json_dict(self) -> dict[str, object]:
        return {
            "badge": self.badge,
            "matched_patterns": list(self.matched_patterns),
            "rationale": self.rationale,
        }


def scan(topology: Topology) -> ScanResult:
    """Run the §5a rule scan against a Topology.

    Pure function of the IR; zero side effects. Runs in O(nodes + edges).
    """
    agents = [n for n in topology.nodes if n.kind == "agent"]
    has_aggregator = any(n.kind == "aggregator" for n in topology.nodes)
    is_linear = _is_linear_chain(topology)
    n_agents = len(agents)

    matched: list[str] = []
    base_step = 0

    # Base patterns. We collect every applicable rule rather than first-
    # match-wins because §5a's 4-row table is structured around features
    # (chain length, terminal kind, mid-flow kind) that can co-occur on
    # the same topology. Final base risk is max() across matches —
    # natural for "if any high-risk pattern applies, badge is HIGH".
    if is_linear and not has_aggregator and n_agents >= 3:
        matched.append(_P_LINEAR_3PLUS_NO_AGG)
        base_step = max(base_step, _BADGE_TO_STEP["HIGH"])
    elif is_linear and not has_aggregator and n_agents == 2:
        # Predicates ``n_agents >= 3`` and ``n_agents == 2`` are mutually
        # exclusive on ``int``; ``elif`` makes the §5a table ↔ code
        # mapping read top-down.
        matched.append(_P_LINEAR_2_NO_AGG)
        base_step = max(base_step, _BADGE_TO_STEP["MEDIUM"])

    # Terminal-aggregator rule requires a chain shape and at least one
    # non-aggregator upstream — "the aggregator absorbs upstream variance"
    # has no semantic force on a fan-in topology, a lone aggregator with
    # no chain, or a topology with no agents at all. Audit fix:
    # CR-MAJOR-2 / RT-M1 (pass-1) — without these guards the rule fires
    # on degenerate single-node aggregator topologies + non-linear
    # fan-in shapes, producing a LOW badge with a wrong-by-rationale
    # headline.
    if (
        is_linear
        and n_agents >= 1
        and _all_terminals_are_aggregator(topology)
    ):
        matched.append(_P_TERMINAL_AGGREGATOR)
        base_step = max(base_step, _BADGE_TO_STEP["LOW"])

    if _has_midflow_router(topology):
        matched.append(_P_MIDFLOW_ROUTER)
        base_step = max(base_step, _BADGE_TO_STEP["MEDIUM"])

    # +1 RISK STEP modifiers. Each fires independently and bumps the
    # final step by 1 (capped at HIGH). A topology with no base pattern
    # but a temperature>0 agent still promotes LOW → MEDIUM, matching
    # the BLUEPRINT-literal reading "+1 RISK STEP" as an unconditional
    # bump of whatever base step was selected.
    #
    # Asymmetry between the two promotion rules is deliberate (CR-MINOR-4
    # audit fix): ``temperature_gt_0`` is an *existential* — one agent
    # with temp>0 is enough, and agents with ``llm_config=None`` simply
    # cannot satisfy the predicate (they contribute nothing). The
    # all-share rule is a *universal* — every agent's config must be
    # readable, so any ``llm_config=None`` forces a conservative skip
    # rather than a guess.
    promotions = 0
    # The all-share rule is plural-by-rationale (§5a-extension): the
    # Bajaj §4.1 "heterogeneity is the variance amplifier" claim is
    # vacuous on a 1-agent topology where the set has cardinality 1 by
    # construction. Requiring ``len(agents) >= 2`` keeps single-agent
    # topologies out of the MEDIUM tier on a rule that has no semantic
    # force there.
    if len(agents) >= 2 and _all_agents_share_model(agents):
        matched.append(_P_ALL_SHARE_MODEL)
        promotions += 1
    if any(
        n.llm_config is not None and n.llm_config.temperature > 0
        for n in agents
    ):
        matched.append(_P_TEMPERATURE_GT_0)
        promotions += 1

    final_step = min(base_step + promotions, _HIGH_STEP)
    badge = _STEP_TO_BADGE[final_step]
    rationale = _format_rationale(badge, matched, n_agents=n_agents)
    return ScanResult(
        badge=badge,
        matched_patterns=tuple(matched),
        rationale=rationale,
    )


# --- helpers -----------------------------------------------------------


def _is_linear_chain(topology: Topology) -> bool:
    """True iff the topology is a single path from entrypoint to one
    terminal: every non-entry node has exactly 1 predecessor, every
    non-terminal node has exactly 1 successor, and the edge count is
    ``n_nodes - 1`` (which excludes disconnected components and cycles).

    The single-node degenerate case (``entrypoint == terminal[0]``,
    ``n_nodes == 1``, ``edges == ()``) is treated as a length-1 linear
    chain; the §5a base rules require at least 2 agents to fire, so the
    semantic effect is identical to "not a chain".
    """
    if len(topology.terminal) != 1:
        return False
    n_nodes = len(topology.nodes)
    if len(topology.edges) != n_nodes - 1:
        return False
    # Audit fix RT-M2 (pass-1): reject the entrypoint==terminal cycle
    # case explicitly rather than relying on the degree-walk to catch it
    # by accident. The singleton n_nodes==1 case is the only legitimate
    # entrypoint==terminal shape (a 1-node "chain" with no edges).
    if topology.entrypoint == topology.terminal[0] and n_nodes > 1:
        return False
    in_deg: dict[str, int] = {n.id: 0 for n in topology.nodes}
    out_deg: dict[str, int] = {n.id: 0 for n in topology.nodes}
    for e in topology.edges:
        out_deg[e.from_id] = out_deg.get(e.from_id, 0) + 1
        in_deg[e.to_id] = in_deg.get(e.to_id, 0) + 1
    terminal_id = topology.terminal[0]
    for n in topology.nodes:
        expected_in = 0 if n.id == topology.entrypoint else 1
        expected_out = 0 if n.id == terminal_id else 1
        if in_deg[n.id] != expected_in or out_deg[n.id] != expected_out:
            return False
    return True


def _all_terminals_are_aggregator(topology: Topology) -> bool:
    """True iff there is at least one terminal AND every terminal node
    has ``kind="aggregator"``.

    Multi-terminal topologies only qualify when *every* terminal absorbs
    variance — a single non-aggregator branch breaks the §5a "aggregator
    absorbs upstream variance" guarantee.
    """
    terminal_ids = set(topology.terminal)
    if not terminal_ids:
        return False
    terminal_nodes = [n for n in topology.nodes if n.id in terminal_ids]
    if len(terminal_nodes) != len(terminal_ids):
        # IR construction would already have rejected this — defensive
        # only. If a terminal id is missing from nodes the structural
        # validator at Topology() construction time raises.
        return False
    return all(n.kind == "aggregator" for n in terminal_nodes)


def _has_midflow_router(topology: Topology) -> bool:
    """True iff any node with ``kind="router"`` is neither the
    entrypoint nor a terminal. A router *at* the boundary is not
    "mid-flow" — it's the entry/exit shape — so §5a's rationale
    ("router decisions may flip on permutation") doesn't apply.
    """
    terminal_ids = set(topology.terminal)
    for n in topology.nodes:
        if n.kind != "router":
            continue
        if n.id == topology.entrypoint:
            continue
        if n.id in terminal_ids:
            continue
        return True
    return False


def _all_agents_share_model(agents: list[Node]) -> bool:
    """True iff every agent has a non-None ``llm_config`` AND they all
    share the same ``llm_config.model`` string.

    Returns False if any agent has ``llm_config=None`` — the rule
    cannot be evaluated without complete configs, so we refuse to
    promote rather than guess. The CLI surfaces this case via the
    ``matched_patterns`` (the promotion id simply does not appear).
    """
    if not agents:
        return False
    models: set[str] = set()
    for a in agents:
        if a.llm_config is None:
            return False
        models.add(a.llm_config.model)
    return len(models) == 1


def _format_rationale(
    badge: Badge, matched: list[str], *, n_agents: int
) -> str:
    """Compose a single-line rationale. Embedded verbatim into
    ``scan.json``; the CLI banner wraps it across multiple lines.
    """
    if not matched:
        return (
            f"{badge} risk — no §5a structural patterns matched. "
            "Default-LOW baseline."
        )
    base_matches = [m for m in matched if m in _BASE_HEADLINES]
    promo_matches = [m for m in matched if m in _PROMOTION_HEADLINES]
    if base_matches:
        # Headline = first base pattern (insertion order = check order =
        # §5a table top-down). For ties on risk step the deterministic
        # check order gives a stable headline.
        headline_tpl = _BASE_HEADLINES[base_matches[0]]
        headline = headline_tpl.format(n=n_agents)
    else:
        # Only promotions fired. Synthesize a neutral headline so the
        # rationale still reads as a sentence.
        headline = (
            "no base structural pattern matched; risk promoted by "
            "decoding/heterogeneity signals."
        )
    parts = [f"{badge} risk — {headline}"]
    if promo_matches:
        promos_text = "; ".join(
            _PROMOTION_HEADLINES[m] for m in promo_matches
        )
        parts.append(f"Promoted by: {promos_text}.")
    return " ".join(parts)
