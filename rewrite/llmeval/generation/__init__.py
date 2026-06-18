"""Test generation — strictly separate from running.

Generators transform a source (CSV, dataset, hand-written) into the standardized
test-case JSON consumed by the runner. The generated data lives in its own directory
(e.g. ``testcases/``) so it can be inspected before any run — unlike opaque
``*_gen.py`` test factories that only materialise at run time.
"""
