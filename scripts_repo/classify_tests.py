#!/usr/bin/env python3
"""Classify each suite's prompts onto the shared request_type / domain axes.

Reads the downloaded raw datasets, asks an LLM (Bedrock Claude, the same family
used to grade rubrics) to label every prompt with one ``request_type`` and one
``domain`` from the controlled vocabularies in ``tests/classification.py``, and
writes the result to ``data/classifications/<suite>.json``. The raw datasets are
never modified; generators merge these labels into test metadata at generation
time via ``classification.augment``.

Output shape::

    {
      "_meta": {"suite": ..., "model": ..., "generated_at": ..., "count": ...},
      "classifications": {
        "<sha1(prompt)>": {"request_type": ..., "domain": ..., "native_hint": ...}
      }
    }

The run is idempotent: prompts already present in the output file are skipped
unless ``--force`` is given, so re-running only classifies new prompts.

Usage::

    python scripts_repo/classify_tests.py                 # all suites
    python scripts_repo/classify_tests.py --suite multifaceted --limit 20
    python scripts_repo/classify_tests.py --force          # re-classify everything

Auth: uses the standard AWS chain plus AWS_BEARER_TOKEN_BEDROCK, exactly like
the promptfoo grader. Region comes from AWS_REGION (default us-east-1).
"""

import argparse
import datetime as dt
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tests"))
import classification  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
DEFAULT_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


def _load_dotenv():
    """Populate os.environ from a repo-root .env (without overriding real env)."""
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


# Per-suite adapters: where the raw data is and how to pull a prompt + a short
# native hint (the dataset's own label) to give the classifier extra context.
def _rr_records(rows):
    for r in rows:
        yield r["prompt"], f"dataset domain: {r.get('domain', '?')}"


def _ah_records(rows):
    for r in rows:
        yield r["prompt"], f"harm category: {r.get('category', '?')}"


def _mf_records(rows):
    for r in rows:
        yield r["prompt"], f"benchmark source: {r.get('source', '?')}"


SUITES = {
    "research_rubrics": ("researchrubrics.json", _rr_records),
    "agentharm_refusal": ("agentharm.json", _ah_records),
    "multifaceted": ("multifaceted.json", _mf_records),
}


def _enum_lines(vocab):
    return "\n".join(f"- {name}: {desc}" for name, desc in vocab.items())


SYSTEM_PROMPT = (
    "You label user prompts for an LLM test suite on two independent axes:\n\n"
    "request_type — what the user is trying to DO:\n"
    f"{_enum_lines(classification.REQUEST_TYPES)}\n\n"
    "domain — the subject AREA the request is about:\n"
    f"{_enum_lines(classification.DOMAINS)}\n\n"
    "Pick the single best value for each axis based on the prompt's dominant "
    "intent and subject. A provided dataset hint is advisory only; judge from "
    "the prompt itself. Use general_other / text_transformation only when "
    "nothing more specific fits. Harmful or adversarial prompts are still "
    "classified by their underlying task and subject."
)

TOOL = {
    "toolSpec": {
        "name": "classify",
        "description": "Record the request_type and domain for the prompt.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "request_type": {
                        "type": "string",
                        "enum": list(classification.REQUEST_TYPES),
                    },
                    "domain": {
                        "type": "string",
                        "enum": list(classification.DOMAINS),
                    },
                },
                "required": ["request_type", "domain"],
            }
        },
    }
}

_thread_local = threading.local()


def _client():
    """One bedrock-runtime client per worker thread (clients aren't shareable)."""
    if not hasattr(_thread_local, "client"):
        kwargs = {"region_name": os.environ.get("AWS_REGION", "us-east-1")}
        # When a Bedrock API key is present it authenticates the request via a
        # bearer token, so SigV4 credentials are unused. Hand boto3 placeholders
        # to stop it walking the credential-provider chain (which can fail on
        # providers needing optional deps like botocore[crt]).
        if os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
            kwargs["aws_access_key_id"] = "bedrock"
            kwargs["aws_secret_access_key"] = "bedrock"
        _thread_local.client = boto3.client("bedrock-runtime", **kwargs)
    return _thread_local.client


def _invoke(model, user):
    resp = _client().converse(
        modelId=model,
        system=[{"text": SYSTEM_PROMPT}],
        messages=[{"role": "user", "content": [{"text": user}]}],
        toolConfig={"tools": [TOOL], "toolChoice": {"tool": {"name": "classify"}}},
        inferenceConfig={"temperature": 0, "maxTokens": 200},
    )
    for block in resp["output"]["message"]["content"]:
        if "toolUse" in block:
            return block["toolUse"]["input"]
    raise RuntimeError("model did not call the classify tool")


def _valid(labels):
    return (labels.get("request_type") in classification.REQUEST_TYPES
            and labels.get("domain") in classification.DOMAINS)


def _classify_one(model, prompt, hint):
    # Bedrock treats the tool-schema enums as advisory, so validate and, if the
    # model invents a label, retry once telling it to choose from the lists.
    user = f"Dataset hint: {hint}\n\nPrompt:\n{prompt[:6000]}"
    labels = _invoke(model, user)
    if not _valid(labels):
        retry = (
            f"{user}\n\nYour previous answer ({labels}) used a value outside the "
            "allowed lists. request_type must be one of "
            f"{sorted(classification.REQUEST_TYPES)} and domain one of "
            f"{sorted(classification.DOMAINS)}."
        )
        labels = _invoke(model, retry)
    if not _valid(labels):
        raise ValueError(f"invalid labels after retry: {labels}")
    return {"request_type": labels["request_type"], "domain": labels["domain"]}


def _load_existing(suite):
    path = classification.classifications_path(suite)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8")).get("classifications", {})
    return {}


def _write(suite, classifications, model):
    path = classification.classifications_path(suite)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "_meta": {
            "suite": suite,
            "model": model,
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "count": len(classifications),
        },
        # Sort keys for stable, review-friendly diffs.
        "classifications": dict(sorted(classifications.items())),
    }
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return path


def classify_suite(suite, model, workers, force, limit):
    data_file, adapter = SUITES[suite]
    rows = json.loads((DATA_DIR / data_file).read_text(encoding="utf-8"))

    existing = {} if force else _load_existing(suite)
    # De-duplicate by prompt_key; keep the first hint seen for each.
    todo = {}
    for prompt, hint in adapter(rows):
        key = classification.prompt_key(prompt)
        if key not in existing and key not in todo:
            todo[key] = (prompt, hint)
    if limit:
        todo = dict(list(todo.items())[:limit])

    print(f"[{suite}] {len(existing)} already labelled, {len(todo)} to classify")
    results = dict(existing)
    done = 0
    errors = 0

    def work(item):
        key, (prompt, hint) = item
        try:
            labels = _classify_one(model, prompt, hint)
        except Exception as exc:  # noqa: BLE001 - keep going; re-run fills gaps
            return key, None, str(exc)
        labels["native_hint"] = hint
        return key, labels, None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for key, labels, err in pool.map(work, todo.items()):
            done += 1
            if err is not None:
                errors += 1
                print(f"[{suite}] ERROR on {key[:10]}: {err}")
                continue
            results[key] = labels
            if done % 25 == 0:
                print(f"[{suite}] {done}/{len(todo)}")

    path = _write(suite, results, model)
    print(f"[{suite}] wrote {len(results)} labels to {path} ({errors} errors)")


def main():
    _load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", choices=sorted(SUITES), help="default: all suites")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="re-classify everything")
    ap.add_argument("--limit", type=int, help="cap prompts per suite (smoke test)")
    args = ap.parse_args()

    suites = [args.suite] if args.suite else sorted(SUITES)
    for suite in suites:
        classify_suite(suite, args.model, args.workers, args.force, args.limit)


if __name__ == "__main__":
    main()
