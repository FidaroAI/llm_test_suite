"""The cache key — the organizing principle of the whole suite.

A cache key encapsulates *everything about the system under test* that the user
decides matters. It is computed from a namespace::

    namespace = {"model": model, "stream": stream, **params, **extra}

where ``params`` are call parameters (temperature, max_tokens, ...) and ``extra``
is arbitrary identity that isn't an API param (backend_version, system_prompt_id).
The user picks exactly which fields form the key via ``fields``; everything else is
ignored, so two configs that differ only in an ignored field collide on purpose.

``model`` and ``stream`` are reserved: they are structural fields of a provider config
rather than free-form parameters, so they are placed in the namespace directly and may
not be shadowed from ``params`` or ``extra``. ``stream`` is *selectable* but usually not
selected — an aggregated stream and a non-streamed response are the same answer, so a
config that streams generally wants to share a key with one that doesn't.

Both the hash (a stable join key) and the selected ``fields`` dict (for grouping and
human-readable reports) are kept.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

_RESERVED = ("model", "stream")


@dataclass(frozen=True)
class CacheKey:
    fields: Mapping[str, Any]  # the selected subset that defines identity
    canonical: str  # sorted-key JSON of ``fields``
    hash: str  # sha256 hex of ``canonical``

    @property
    def short(self) -> str:
        return self.hash[:12]


def build_namespace(
    model: str,
    params: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
    stream: bool = False,
) -> dict[str, Any]:
    """Merge the structural fields, params and extra into one flat namespace.

    Rejects any attempt to shadow a reserved name from ``params``/``extra``, and any
    key defined in both — either would make the namespace depend on merge order.
    """
    params = dict(params or {})
    extra = dict(extra or {})
    shadowed = [r for r in _RESERVED if r in params or r in extra]
    if shadowed:
        raise ValueError(f"reserved and may not appear in params/extra: {shadowed}")
    overlap = set(params) & set(extra)
    if overlap:
        raise ValueError(f"params and extra must be disjoint; overlap: {sorted(overlap)}")
    return {"model": model, "stream": stream, **params, **extra}


def compute_cache_key(
    model: str,
    params: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
    fields: Sequence[str] | None = None,
    stream: bool = False,
) -> CacheKey:
    """Compute the cache key for a system under test.

    ``fields=None`` keys on the whole namespace. Otherwise only the named fields
    contribute; a named field absent from the namespace is an error (the user asked
    to key on something that doesn't exist).
    """
    namespace = build_namespace(model, params, extra, stream)

    if fields is None:
        selected = dict(namespace)
    else:
        missing = [f for f in fields if f not in namespace]
        if missing:
            raise ValueError(f"cache_key_fields not present in namespace: {missing}")
        selected = {f: namespace[f] for f in fields}

    canonical = json.dumps(selected, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return CacheKey(fields=selected, canonical=canonical, hash=digest)
