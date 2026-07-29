"""Porcelain: reporting tools built on the ``llmeval`` CLI and SQLite store.

This package sits *outside* ``llmeval`` on purpose. Per ``rewrite/CLAUDE.md``, the
package is plumbing (capabilities, exposed explicitly) and anything that encodes a
workflow — including dashboards like these — is porcelain that lives on top of the
plumbing's public contracts.

So the dependency runs one way only: ``reporting`` may import ``llmeval``; ``llmeval``
must never import ``reporting``.

* :mod:`reporting.csv_table` is the generic layer — arbitrary rows or a CSV file become a
  standalone HTML page with per-column filtering and column show/hide.
* :mod:`reporting.run_report` is the first tool built on it: everything one run produced.

Tools are run as modules from the ``rewrite/`` directory, e.g.
``python -m reporting.run_report run_2026 -o run.html``. There is deliberately no
console-script entry point — that would install porcelain alongside the plumbing.
"""
