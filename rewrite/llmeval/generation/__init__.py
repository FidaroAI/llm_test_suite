"""Shared machinery for test-case plugins — strictly separate from running.

Nothing here generates a suite of its own any more. The suites live in ``testcases/`` as
plugins (see :mod:`llmeval.plugins`); this package is the library they build on, so that
"parse a CSV" or "page a Hugging Face dataset" is written once rather than per plugin:

* :mod:`~llmeval.generation.common` — stable local test ids
* :mod:`~llmeval.generation.csv_source` — CSV rows to test-case dicts
* :mod:`~llmeval.generation.csv_plugin` — a whole CSV-backed plugin
* :mod:`~llmeval.generation.hf_rows` — the Hugging Face datasets-server paging loop
* :mod:`~llmeval.generation.dataset_plugin` — a whole dataset-backed plugin

The dependency runs one way: plugins import this, and nothing here knows which plugins
exist.
"""
