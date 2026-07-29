"""Porcelain: reporting tools built on the ``llmeval`` CLI and SQLite store.

This package sits *outside* ``llmeval`` on purpose. Per ``rewrite/CLAUDE.md``, the
package is plumbing (capabilities, exposed explicitly) and anything that encodes a
workflow — including dashboards like these — is porcelain that lives on top of the
plumbing's public contracts.

So the dependency runs one way only: ``reporting`` may import ``llmeval``; ``llmeval``
must never import ``reporting``.

* :mod:`reporting.csv_table` is the generic layer — arbitrary rows or a CSV file become a
  standalone HTML page with per-column filtering, column show/hide and sorting, opened in
  a browser when run from the command line.

Tools are run as modules from the ``rewrite/`` directory, e.g.
``python -m reporting.csv_table results.csv -o results.html``. There is deliberately no
console-script entry point — that would install porcelain alongside the plumbing.

Row *building* lives in the plumbing (:mod:`llmeval.resultrows`), not here: which rows a
report contains is a capability, while turning them into a page is a workflow. So the
pipeline is ``llmeval report`` for the data and this package for the view.
"""
