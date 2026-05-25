"""Shared helper: turn a promptfoo CSV test file into generated test cases.

Promptfoo can read tests straight from a CSV (``tests: file://foo.csv``). This
helper reproduces that mapping in Python so a CSV can instead be driven by a
``tests/<name>_gen.py`` generator, gaining the repo's generator envelope (suite
naming, classification labels, config-driven selection) without changing the
underlying data. Point a generator at its CSV with :func:`generate_from_csv`.

Promptfoo's CSV column conventions, reproduced here (see promptfoo's
``testCaseFromCsvRow`` / ``assertionFromString``):

* a plain column header is a prompt variable (e.g. ``user``);
* ``__expected`` / ``__expected<N>`` hold a ``type:value`` assertion, e.g.
  ``icontains:Microsoft`` -> ``{"type": "icontains", "value": "Microsoft"}``.
  ``javascript:``/``fn:``/``eval:``, ``grade:``/``llm-rubric:`` and ``python:``
  are special-cased, ``contains-all``/``-any`` split their value on commas, and a
  cell whose prefix isn't a known assertion type becomes an ``equals``;
* ``__metadata:key`` populates ``metadata.key`` (``key[]`` splits on commas);
* ``__description`` sets the test description.

On top of that we apply the repo convention that the suite name is the
generator's filename minus ``_gen`` (``tests/suite_config.py``). That suite is
stamped onto every test's ``metadata.suite``, deliberately overriding any
``__metadata:suite`` column in the CSV, and each prompt is run through
``classification.augment``. Finally ``SuiteConfig.select`` applies the
config-driven (optionally randomized/stratified) selection, exactly like the
dataset-backed generators.
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import classification  # noqa: E402
import suite_config  # noqa: E402

# The base assertion types promptfoo recognises as a ``type:value`` prefix
# (BaseAssertionTypesSchema). A cell whose prefix isn't in here falls back to an
# ``equals`` assertion, matching promptfoo's behaviour.
_BASE_ASSERTION_TYPES = {
    "answer-relevance", "bleu", "classifier", "contains", "contains-all",
    "contains-any", "contains-html", "contains-json", "contains-sql",
    "contains-xml", "context-faithfulness", "context-recall",
    "context-relevance", "conversation-relevance", "cost", "equals",
    "factuality", "finish-reason", "g-eval", "gleu", "guardrails", "icontains",
    "icontains-all", "icontains-any", "is-html", "is-json", "is-refusal",
    "is-sql", "is-valid-function-call", "is-valid-openai-function-call",
    "is-valid-openai-tools-call", "is-xml", "javascript", "latency",
    "levenshtein", "llm-rubric", "pi", "meteor", "model-graded-closedqa",
    "model-graded-factuality", "moderation", "perplexity", "perplexity-score",
    "python", "regex", "rouge-n", "ruby", "similar", "similar:cosine",
    "similar:dot", "similar:euclidean", "starts-with", "tool-call-f1",
    "skill-used", "trajectory:goal-success", "trajectory:tool-args-match",
    "trajectory:step-count", "trajectory:tool-sequence", "trajectory:tool-used",
    "trace-error-spans", "trace-span-count", "trace-span-duration",
    "search-rubric", "webhook", "word-count",
}

# Types whose value is a comma-separated list of strings.
_LIST_VALUE_TYPES = {"contains-all", "contains-any", "icontains-all", "icontains-any"}


def assertion_from_string(expected):
    """Parse a promptfoo ``__expected`` cell into an assertion dict.

    Mirrors promptfoo's ``assertionFromString``: explicit ``javascript:``/
    ``llm-rubric:``/``python:`` prefixes, then a ``[not-]<type>[(threshold)]:value``
    form over the known assertion types, else a plain ``equals``.
    """
    if expected.startswith(("javascript:", "fn:", "eval:")):
        prefix_len = {"javascript:": 11, "fn:": 3, "eval:": 5}
        for prefix, length in prefix_len.items():
            if expected.startswith(prefix):
                return {"type": "javascript", "value": expected[length:].strip()}
    if expected.startswith("grade:"):
        return {"type": "llm-rubric", "value": expected[6:]}
    if expected.startswith("llm-rubric:"):
        return {"type": "llm-rubric", "value": expected[11:]}
    if expected.startswith("python:"):
        return {"type": "python", "value": expected[7:].strip()}

    prefix, sep, value = expected.partition(":")
    base, _, _threshold = prefix.partition("(")  # strip an optional (threshold)
    type_name = base[4:] if base.startswith("not-") else base
    if sep and type_name in _BASE_ASSERTION_TYPES:
        full_type = prefix.split("(", 1)[0]
        if type_name in _LIST_VALUE_TYPES:
            return {"type": full_type, "value": [v.strip() for v in value.split(",")] if value else value}
        return {"type": full_type, "value": value.strip()}

    return {"type": "equals", "value": expected}


def _row_to_test(row, suite, prompt_var):
    """Build one promptfoo test case from a CSV row, the way promptfoo would."""
    vars_ = {}
    asserts = []
    metadata = {}
    description = None

    for key, value in row.items():
        key = (key or "").strip()
        value = value if value is not None else ""
        if key.startswith("__expected"):
            if value.strip():
                asserts.append(assertion_from_string(value.strip()))
        elif key == "__description":
            description = value
        elif key.startswith("__metadata:"):
            meta_key = key[len("__metadata:"):]
            if meta_key.endswith("[]"):
                if value.strip():
                    metadata[meta_key[:-2]] = [v.strip() for v in value.split(",")]
            elif value.strip():
                metadata[meta_key] = value
        elif key.startswith(("__metadata", "__config:", "__prefix", "__suffix",
                             "__providerOutput", "__metric", "__threshold")):
            # Other promptfoo special columns: not used by our CSVs. Ignore
            # rather than treat them as vars. Add handling here if a CSV needs it.
            continue
        else:
            vars_[key] = value

    # Suite name follows the generator filename, overriding any CSV value.
    metadata["suite"] = suite

    test = {"vars": vars_, "assert": asserts, "metadata": metadata}
    if description is not None:
        test["description"] = description

    prompt = vars_.get(prompt_var, "")
    return classification.augment(test, suite, prompt)


def generate_from_csv(generator_file, csv_path, prompt_var="user"):
    """Generate the test cases for the CSV-backed suite owning ``generator_file``.

    ``generator_file`` is the calling ``*_gen.py``'s ``__file__``; its name
    (minus ``_gen``) is the suite. ``prompt_var`` names the column used as the
    prompt text for classification (the repo's templates use ``user``).
    """
    suite = suite_config.suite_name(generator_file)
    cfg = suite_config.load(generator_file)
    with Path(csv_path).open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    tests = [_row_to_test(row, suite, prompt_var) for row in rows]
    return cfg.select(tests)
