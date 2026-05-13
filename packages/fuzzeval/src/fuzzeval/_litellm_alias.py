"""LiteLLM model-id alias at the SDK call boundary.

Lives in :mod:`fuzzeval` (the upstream package) so both
:mod:`topofuzz.runner` (agent rollouts) and :mod:`fuzzeval.judge`
(SingleJudge) can apply the same mapping. Used **only** at the
``litellm.completion`` / ``litellm.acompletion`` call site — every
internal cache key, manifest field, attestation payload, and §8
``judge_independence.vendor_overlap`` metric keeps the canonical name
(e.g. ``"google/gemini-2.5-flash"``).

Why this exists: the project's canonical model ids are ``<vendor>/<family>-<version>``
(see :func:`topofuzz.orchestrator._vendor_from_model_id`), but
LiteLLM's actual provider prefix for Google AI Studio is ``gemini/``.
Without this alias the live call raises
``litellm.BadRequestError: LLM Provider NOT provided``.

A future fixture that opts into Vertex AI routing should spell out
the LiteLLM-native prefix (``vertex_ai/gemini-2.5-pro``) — the alias
passes such ids through unchanged.
"""

from __future__ import annotations

# Mapping from project-canonical provider prefix (the vendor string
# used in cache keys, manifest, attestation, and the §8
# ``judge_independence.vendor_overlap`` metric) to the actual LiteLLM
# provider prefix at the SDK call boundary. Only entries where the two
# differ are listed; an unmapped model id is passed through unchanged.
_LITELLM_MODEL_PREFIX_ALIASES: dict[str, str] = {
    # Google AI Studio direct — our canonical is ``google/`` (the
    # vendor), LiteLLM's prefix is ``gemini/`` (the family). Vertex AI
    # is ``vertex_ai/`` in LiteLLM; fixtures that want Vertex routing
    # should spell that out explicitly rather than relying on this map.
    "google/": "gemini/",
}


def litellm_model_id(canonical: object) -> object:
    """Translate a project-canonical model id to LiteLLM's call form.

    The mapping is **prefix-based**: the first ``"<vendor>/"`` segment
    is rewritten via :data:`_LITELLM_MODEL_PREFIX_ALIASES` and the
    suffix is preserved. Non-string inputs pass through unchanged so a
    fake ``llm_callable`` injected by a test can use any sentinel value
    it wants (the runner / judge call helpers only invoke this on the
    real-LiteLLM branch, but defensiveness is cheap).

    Examples::

        litellm_model_id("google/gemini-2.5-flash")
        # → "gemini/gemini-2.5-flash"

        litellm_model_id("anthropic/claude-opus-4-7")
        # → "anthropic/claude-opus-4-7"   (passthrough)

        litellm_model_id("gpt-4o")
        # → "gpt-4o"                       (no slash, passthrough)

        litellm_model_id("vertex_ai/gemini-2.5-pro")
        # → "vertex_ai/gemini-2.5-pro"     (LiteLLM-native, passthrough)
    """
    if not isinstance(canonical, str):
        return canonical
    for src, dst in _LITELLM_MODEL_PREFIX_ALIASES.items():
        if canonical.startswith(src):
            return dst + canonical[len(src):]
    return canonical
