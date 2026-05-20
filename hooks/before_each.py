"""Promptfoo extension hook: reconfigure a model before each test.

Reads `metadata.reconfigure` from the test definition and applies provider-
specific reconfiguration. Tests that don't set `metadata.reconfigure` are
unaffected.

Example test usage:

    - description: "A/B temperature"
      metadata:
        reconfigure: ollama_local
        options:
          temperature: 0.0
      vars: {...}
"""

import os


def before_each(hook_name, context):
    test = (context or {}).get("test") or {}
    meta = test.get("metadata") or {}
    target = meta.get("reconfigure")
    if not target:
        return context

    options = meta.get("options") or {}

    if target == "ollama_local":
        # Ollama accepts per-request `options` via the OpenAI-compatible API.
        # Merge them into the test's provider config so the next call picks
        # them up.
        provider_cfg = (
            test.setdefault("options", {})
                .setdefault("provider", {})
                .setdefault("config", {})
        )
        merged = dict(provider_cfg.get("options") or {})
        merged.update(options)
        provider_cfg["options"] = merged

    elif target == "vllm_lan":
        admin = os.environ.get("VLLM_ADMIN")
        if not admin:
            return context
        import requests
        resp = requests.post(f"{admin}/reconfigure", json=options, timeout=5)
        resp.raise_for_status()

    return context
