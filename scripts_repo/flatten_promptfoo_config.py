#!/usr/bin/env python3
"""Flatten a promptfoo config by inlining all `file://` data references.

Walks the YAML tree starting at INPUT (default: promptfooconfig.yaml) and
recursively resolves `file://` references:

  * `.yaml` / `.yml` / `.json` are parsed and inlined as structured values.
  * `.txt` and any other unknown extension are inlined as plain-text strings.
  * `.py`, `.js`, `.ts`, `.mjs`, `.cjs` are kept as `file://` URIs (they
    reference runtime callables, not data) but rewritten to absolute paths so
    the flattened output is relocatable.

When a list element is itself a `file://` ref whose target resolves to a list,
the resolved list is spliced into the parent rather than nested — matching how
promptfoo treats `tests:` and `providers:`.

Usage:
    scripts/flatten_promptfoo_config.py [INPUT] [-o OUTPUT]
    scripts/flatten_promptfoo_config.py -o flat.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml


CODE_EXTS = {".py", ".js", ".mjs", ".cjs", ".ts", ".tsx"}
YAML_EXTS = {".yaml", ".yml"}
JSON_EXTS = {".json"}


def _split_file_uri(uri: str) -> tuple[str, str | None]:
    """Split `file://path[:fragment]` into (path, fragment-or-None).

    The fragment is the optional `:funcname` suffix promptfoo uses for
    callables (e.g. `file://hooks/before_each.py:before_each`). We only treat a
    trailing `:token` as a fragment when `token` looks like an identifier, so
    that incidental colons inside a path (rare, but possible) don't mis-split.
    """
    assert uri.startswith("file://")
    body = uri[len("file://"):]
    if ":" in body:
        path, _, frag = body.rpartition(":")
        if path and frag and frag.replace("_", "").isalnum():
            return path, frag
    return body, None


def _is_file_uri(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("file://")


def _resolve(uri: str, root: Path, visited: frozenset[Path]) -> Any:
    raw_path, fragment = _split_file_uri(uri)
    target = (root / raw_path).resolve()

    if not target.exists():
        raise FileNotFoundError(
            f"file:// reference not found: {target} (resolved from {root})"
        )

    suffix = target.suffix.lower()

    if suffix in CODE_EXTS:
        # Runtime handle — keep as a URI but make the path absolute so the
        # flattened YAML still works if it's moved elsewhere.
        normalized = f"file://{target}"
        return f"{normalized}:{fragment}" if fragment else normalized

    if target in visited:
        raise ValueError(f"Cycle detected while resolving {target}")
    next_visited = visited | {target}

    if suffix in YAML_EXTS:
        with target.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    elif suffix in JSON_EXTS:
        with target.open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        return target.read_text(encoding="utf-8")

    return _walk(data, root, next_visited)


def _walk(node: Any, root: Path, visited: frozenset[Path]) -> Any:
    if isinstance(node, dict):
        return {k: _walk(v, root, visited) for k, v in node.items()}
    if isinstance(node, list):
        out: list[Any] = []
        for item in node:
            if _is_file_uri(item):
                resolved = _resolve(item, root, visited)
                # Splice when the ref's target is itself a list — otherwise a
                # `tests:` block referencing per-suite list files would become
                # a list of lists.
                if isinstance(resolved, list):
                    out.extend(resolved)
                else:
                    out.append(resolved)
            else:
                out.append(_walk(item, root, visited))
        return out
    if _is_file_uri(node):
        return _resolve(node, root, visited)
    return node


def flatten(input_path: Path) -> Any:
    """Promptfoo resolves `file://` paths relative to the config's directory,
    no matter how deeply nested the reference is. We pass that single root
    through every recursive call instead of using the parent of whichever
    file the URI happens to appear in.
    """
    input_path = input_path.resolve()
    root = input_path.parent
    with input_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return _walk(data, root, frozenset({input_path}))


def _str_representer(dumper: yaml.Dumper, data: str) -> yaml.ScalarNode:
    # Render multi-line strings as `|` block scalars so inlined .txt prompts
    # and JSON-as-string blobs stay readable.
    style = "|" if "\n" in data else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


yaml.add_representer(str, _str_representer, Dumper=yaml.SafeDumper)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="promptfooconfig.yaml",
        type=Path,
        help="Promptfoo config to flatten (default: promptfooconfig.yaml).",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Write the flattened YAML here instead of stdout.",
    )
    args = parser.parse_args(argv)

    flat = flatten(args.input)
    text = yaml.safe_dump(
        flat,
        sort_keys=False,
        allow_unicode=True,
        width=100,
        default_flow_style=False,
    )
    header = f"# Flattened from {args.input} — file:// data refs inlined.\n"
    output = header + text

    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        sys.stdout.write(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
