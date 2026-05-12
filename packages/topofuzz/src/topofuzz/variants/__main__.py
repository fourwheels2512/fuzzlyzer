"""Entrypoint for ``python -m topofuzz.variants`` — the §6 demo print.

Builds the lending_4role IR with realistic LLM configs and emits every
variant from :func:`topofuzz.variants.generate_variants` so a developer
can eyeball that the §6 envelope (8–16 variants) holds end-to-end.

This module requires ``langgraph`` (via the fixture builder). The
``topofuzz[verify]`` minimal install profile does not include it; run
the full ``topofuzz`` install to use the demo.
"""

from __future__ import annotations

from topofuzz.adapter import from_langgraph
from topofuzz.fixtures.lending_4role import LLM_CONFIGS, build_graph
from topofuzz.variants import generate_variants


def main() -> None:
    topo = from_langgraph(
        build_graph(), name="lending_4role", llm_configs=LLM_CONFIGS
    )
    variants = generate_variants(topo)
    print(f"lending_4role variants: {len(variants)}")
    for i, v in enumerate(variants):
        ids = ",".join(n.id for n in v.nodes)
        roles = ",".join(n.role for n in v.nodes)
        temps = ",".join(
            f"{n.llm_config.temperature:.1f}" if n.llm_config else "-"
            for n in v.nodes
        )
        print(
            f"  v{i:>3d}  hash={v.canonical_hash()[:12]}  "
            f"ids=[{ids}]  roles=[{roles}]  temps=[{temps}]"
        )


if __name__ == "__main__":
    main()
