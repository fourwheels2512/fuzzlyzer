# Copyright (c) 2026 Kiran Nayudu. All rights reserved.
# Licensed under the Business Source License 1.1.
# See LICENSE file in the project root for full terms.
# <!-- PATENT_PENDING_NOTICE_GOES_HERE -->

"""One true canonical-hash helper.

Both ``topofuzz.ir.Topology.canonical_hash`` and the §7 cache key
producer in ``fuzzeval.cache`` route through this function so the
hashing recipe (RFC 8785 JCS canonicalization + SHA-256) cannot drift
between modules. A drift between the two would silently re-bucket cache
keys (e.g. ``Topology`` writes under hash X, ``LLMCache`` reads under
hash Y) and corrupt the §7 cache identity invariant.

The function accepts any JSON-serializable structure (``dict``,
``list``, ``str``, ``int``, ``float``, ``bool``, ``None``). Non-JSON
values (``bytes``, ``set``, datetimes, numpy scalars) raise ``TypeError``
with a helpful message — silently coercing them to strings is exactly
how cache poisoning is born.

Why JCS, not ``json.dumps(sort_keys=True)``: JCS (RFC 8785) is a
deterministic byte-level canonical form that also normalises numbers
(``-0.0`` → ``0``, scientific notation, etc.) and Unicode (NFC). The
plain stdlib serializer agrees on key order but diverges on those edge
cases, so two implementations using ``sort_keys=True`` can disagree on
byte output for the same logical input.
"""

from __future__ import annotations

import hashlib
from typing import Any

import rfc8785


def canonical_hash(obj: Any) -> str:
    """Return the SHA-256 hex digest of the JCS canonicalization of ``obj``.

    Args:
        obj: Any JSON-serializable Python value. Pydantic callers should
            pass ``model.model_dump(mode="json")`` so enums/datetimes are
            already in their JSON form before reaching this helper.

    Returns:
        A lowercase 64-character hex string.

    Raises:
        TypeError: ``obj`` contains values that JCS / JSON cannot
            represent (e.g. ``bytes``, ``set``, NaN/Inf floats). The
            error message names the offending value so the caller can
            locate the bug quickly. Note: ``bool`` is a subclass of
            ``int`` in Python and is accepted (encoded as ``true`` /
            ``false`` per JSON).
    """
    try:
        blob = rfc8785.dumps(obj)
    except (TypeError, ValueError) as e:
        raise TypeError(
            f"canonical_hash input must be JSON / JCS serializable; got: {e}"
        ) from e
    return hashlib.sha256(blob).hexdigest()
