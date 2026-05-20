# Reasoning-Aware Assertions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make built-in promptfoo assertions operate on cleaned final-answer text and add a reasoning-content assertion, so the suite can test reasoning-model outputs both at the surface and inside the chain-of-thought. Add a versioned vLLM launcher so reasoning-parser flags are not lost.

**Architecture:** Two-layer split. Layer 1 = provider config (vLLM `--reasoning-parser`, Ollama `think: true`) so reasoning is surfaced as structured response fields. Layer 2 = a single JS response transform registered globally that strips inline `<think>` artifacts from `output`. Reasoning content stays in `context.response.{reasoning_content,thinking,content[]}` for custom Python assertions.

**Tech Stack:** Node 24.15 + promptfoo 0.121.x for the runtime and JS transform; Python 3.11+ + pytest for custom-assertion unit tests; bash for the launch script.

**Spec:** [docs/superpowers/specs/2026-05-07-reasoning-aware-assertions-design.md](../specs/2026-05-07-reasoning-aware-assertions-design.md)

---

## Task 1: Python test infrastructure

**Files:**
- Modify: `requirements.txt`
- Create: `tests/python/conftest.py`
- Create: `tests/python/__init__.py` (empty)

- [ ] **Step 1: Add pytest to `requirements.txt`**

Append to existing `requirements.txt`:

```
pytest>=8.0
```

- [ ] **Step 2: Install pytest**

```bash
cd /Users/badger/dev/llm_test_suite
source .venv/bin/activate
pip install -r requirements.txt
```

Expected: pytest installs. Confirm with `pytest --version`.

- [ ] **Step 3: Create `tests/python/__init__.py`**

```bash
touch tests/python/__init__.py
```

- [ ] **Step 4: Create `tests/python/conftest.py`**

```python
"""Make project root importable so tests can `from assertions.foo import ...`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
```

- [ ] **Step 5: Verify pytest can discover the empty test directory**

```bash
pytest tests/python -q
```

Expected: `no tests ran` (exit code 5 is OK; we just want no import errors).

- [ ] **Step 6: Commit**

```bash
git add requirements.txt tests/python/__init__.py tests/python/conftest.py
git commit -m "chore: add pytest and tests/python scaffolding"
```

---

## Task 2: `assert_reasoning_contains.py` — whole-text matching

**Files:**
- Create: `tests/python/test_assert_reasoning_contains.py`
- Create: `assertions/assert_reasoning_contains.py`

This task covers the no-`step` config path: substring/regex matching against the full reasoning text, with the source-priority lookup (`reasoning_content` → `thinking` → Claude `content[]`) and the absent-reasoning fallback.

- [ ] **Step 1: Write failing tests**

Create `tests/python/test_assert_reasoning_contains.py`:

```python
"""Tests for assertions.assert_reasoning_contains.get_assert.

Covers whole-text matching (no `step` config). Step-mode tests live in the
same file but are added in Task 3.
"""

from assertions.assert_reasoning_contains import get_assert


def _ctx(response, **cfg):
    return {"response": response, "config": cfg}


def test_substring_in_reasoning_content_passes():
    ctx = _ctx({"reasoning_content": "I should reply with pong."}, value="pong")
    result = get_assert("pong", ctx)
    assert result["pass"] is True
    assert result["score"] == 1.0


def test_substring_missing_fails():
    ctx = _ctx({"reasoning_content": "I should reply with pong."}, value="ping")
    result = get_assert("pong", ctx)
    assert result["pass"] is False
    assert result["score"] == 0.0
    assert "not found" in result["reason"].lower()


def test_regex_mode_passes():
    ctx = _ctx(
        {"reasoning_content": "Step 1: parse. Step 2: compute."},
        value=r"Step\s+\d+",
        regex=True,
    )
    result = get_assert("ok", ctx)
    assert result["pass"] is True


def test_thinking_field_used_when_reasoning_content_absent():
    ctx = _ctx({"thinking": "Let me reason about pong."}, value="pong")
    result = get_assert("pong", ctx)
    assert result["pass"] is True


def test_claude_content_blocks_used_when_others_absent():
    response = {
        "content": [
            {"type": "thinking", "thinking": "First I plan."},
            {"type": "thinking", "thinking": "Then I conclude pong."},
            {"type": "text", "text": "pong"},
        ]
    }
    ctx = _ctx(response, value="conclude pong")
    result = get_assert("pong", ctx)
    assert result["pass"] is True


def test_claude_blocks_joined_for_whole_text_match():
    """A substring spanning the natural join between two blocks should NOT match;
    each block stays separated. But matching within one block (via whole-text
    join) works because we join on \\n\\n."""
    response = {
        "content": [
            {"type": "thinking", "thinking": "alpha"},
            {"type": "thinking", "thinking": "beta"},
        ]
    }
    # alpha\n\nbeta — substring "alpha\n\nbeta" should match
    ctx = _ctx(response, value="alpha\n\nbeta")
    assert get_assert("ok", ctx)["pass"] is True


def test_no_reasoning_returns_fail_with_reason():
    ctx = _ctx({}, value="anything")
    result = get_assert("ok", ctx)
    assert result["pass"] is False
    assert result["reason"] == "no reasoning available"


def test_missing_value_config_fails_clearly():
    ctx = _ctx({"reasoning_content": "x"})  # no value
    result = get_assert("ok", ctx)
    assert result["pass"] is False
    assert "value" in result["reason"].lower()


def test_empty_reasoning_string_treated_as_absent():
    ctx = _ctx({"reasoning_content": "   "}, value="x")
    result = get_assert("ok", ctx)
    assert result["pass"] is False
    assert result["reason"] == "no reasoning available"
```

- [ ] **Step 2: Run tests, verify they fail with import error**

```bash
pytest tests/python/test_assert_reasoning_contains.py -v
```

Expected: `ModuleNotFoundError: No module named 'assertions.assert_reasoning_contains'` (or collection error). All tests fail.

- [ ] **Step 3: Write minimal implementation**

Create `assertions/assert_reasoning_contains.py`:

```python
"""Assert that a value appears in the model's reasoning content.

Reads reasoning from the provider response in priority order:
  1. context["response"]["reasoning_content"] (vLLM with reasoning parser)
  2. context["response"]["thinking"]          (Ollama with think:true)
  3. context["response"]["content"][i] where item type == "thinking" (Claude)

Config keys (read from `context["config"]`):
    value  — substring or regex pattern to match (required)
    regex  — bool, treat value as regex (default: false)
    step   — "any" | int | omitted
             omitted: match against full reasoning text (blocks joined with \\n\\n)
             "any":   match if any block contains the value
             int:     match against block at that index (0-based)

If no reasoning is surfaced, returns pass=False, reason="no reasoning available".
"""

import re


def _split_paragraphs(text):
    parts = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    return parts or [text.strip()]


def _reasoning_blocks(context):
    resp = (context or {}).get("response") or {}

    rc = resp.get("reasoning_content")
    if isinstance(rc, str) and rc.strip():
        return _split_paragraphs(rc)

    t = resp.get("thinking")
    if isinstance(t, str) and t.strip():
        return _split_paragraphs(t)

    blocks = []
    for item in resp.get("content", []) or []:
        if isinstance(item, dict) and item.get("type") == "thinking":
            text = item.get("thinking") or item.get("text") or ""
            if text.strip():
                blocks.append(text)
    return blocks


def _match(value, text, use_regex):
    if use_regex:
        return re.search(value, text) is not None
    return value in text


def get_assert(output, context):
    cfg = (context or {}).get("config") or {}
    value = cfg.get("value")
    use_regex = bool(cfg.get("regex", False))

    if value is None:
        return {"pass": False, "score": 0.0, "reason": "missing required config 'value'"}

    blocks = _reasoning_blocks(context)
    if not blocks:
        return {"pass": False, "score": 0.0, "reason": "no reasoning available"}

    whole = "\n\n".join(blocks)
    ok = _match(value, whole, use_regex)
    return {
        "pass": ok,
        "score": 1.0 if ok else 0.0,
        "reason": (
            f"value {'matched' if ok else 'not found'} in reasoning "
            f"(regex={use_regex})"
        ),
    }
```

- [ ] **Step 4: Run tests, verify they pass**

```bash
pytest tests/python/test_assert_reasoning_contains.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/python/test_assert_reasoning_contains.py assertions/assert_reasoning_contains.py
git commit -m "feat: add assert_reasoning_contains (whole-text matching)"
```

---

## Task 3: `assert_reasoning_contains.py` — step-mode matching

**Files:**
- Modify: `tests/python/test_assert_reasoning_contains.py`
- Modify: `assertions/assert_reasoning_contains.py`

Adds `step: "any"` and `step: <int>` config behavior.

- [ ] **Step 1: Write failing tests**

Append to `tests/python/test_assert_reasoning_contains.py`:

```python
def test_step_any_matches_when_one_block_contains():
    response = {
        "content": [
            {"type": "thinking", "thinking": "first thought"},
            {"type": "thinking", "thinking": "pong is the answer"},
        ]
    }
    ctx = _ctx(response, value="pong", step="any")
    result = get_assert("pong", ctx)
    assert result["pass"] is True
    assert "block 1" in result["reason"]


def test_step_any_fails_when_no_block_contains():
    response = {
        "content": [
            {"type": "thinking", "thinking": "first"},
            {"type": "thinking", "thinking": "second"},
        ]
    }
    ctx = _ctx(response, value="missing", step="any")
    result = get_assert("ok", ctx)
    assert result["pass"] is False


def test_step_int_indexes_specific_block():
    response = {
        "content": [
            {"type": "thinking", "thinking": "alpha"},
            {"type": "thinking", "thinking": "bravo"},
            {"type": "thinking", "thinking": "charlie"},
        ]
    }
    # value lives in block 1 only
    ctx = _ctx(response, value="bravo", step=1)
    assert get_assert("ok", ctx)["pass"] is True

    # same value, wrong index
    ctx = _ctx(response, value="bravo", step=0)
    assert get_assert("ok", ctx)["pass"] is False


def test_step_int_out_of_range_fails_with_reason():
    ctx = _ctx({"reasoning_content": "only one block"}, value="x", step=5)
    result = get_assert("ok", ctx)
    assert result["pass"] is False
    assert "out of range" in result["reason"]


def test_step_with_paragraph_split_reasoning_content():
    """A single reasoning_content string with double-newline boundaries
    should be split into multiple blocks for step-mode matching."""
    text = "First step.\n\nSecond step about pong.\n\nThird step."
    ctx = _ctx({"reasoning_content": text}, value="pong", step=1)
    assert get_assert("ok", ctx)["pass"] is True

    # not in step 0
    ctx = _ctx({"reasoning_content": text}, value="pong", step=0)
    assert get_assert("ok", ctx)["pass"] is False


def test_step_any_with_regex():
    response = {
        "content": [
            {"type": "thinking", "thinking": "no numbers here"},
            {"type": "thinking", "thinking": "Step 42 of N"},
        ]
    }
    ctx = _ctx(response, value=r"Step\s+\d+", regex=True, step="any")
    assert get_assert("ok", ctx)["pass"] is True
```

- [ ] **Step 2: Run tests, verify the new ones fail (existing pass)**

```bash
pytest tests/python/test_assert_reasoning_contains.py -v
```

Expected: 9 pass (from Task 2), 6 fail (step-mode behavior not implemented yet — `step` config is ignored, so they currently match against whole text or get wrong reasons).

- [ ] **Step 3: Update `assertions/assert_reasoning_contains.py`**

Replace the body of `get_assert` with this fuller version (keeping helpers above unchanged):

```python
def get_assert(output, context):
    cfg = (context or {}).get("config") or {}
    value = cfg.get("value")
    use_regex = bool(cfg.get("regex", False))
    step = cfg.get("step")

    if value is None:
        return {"pass": False, "score": 0.0, "reason": "missing required config 'value'"}

    blocks = _reasoning_blocks(context)
    if not blocks:
        return {"pass": False, "score": 0.0, "reason": "no reasoning available"}

    if step is None:
        whole = "\n\n".join(blocks)
        ok = _match(value, whole, use_regex)
        return {
            "pass": ok,
            "score": 1.0 if ok else 0.0,
            "reason": (
                f"value {'matched' if ok else 'not found'} in reasoning "
                f"(regex={use_regex})"
            ),
        }

    if step == "any":
        for i, b in enumerate(blocks):
            if _match(value, b, use_regex):
                return {
                    "pass": True,
                    "score": 1.0,
                    "reason": f"matched block {i} (regex={use_regex})",
                }
        return {
            "pass": False,
            "score": 0.0,
            "reason": f"no block matched (regex={use_regex})",
        }

    # integer step
    try:
        idx = int(step)
    except (TypeError, ValueError):
        return {
            "pass": False,
            "score": 0.0,
            "reason": f"invalid step={step!r}; expected int or 'any'",
        }
    if idx < 0 or idx >= len(blocks):
        return {
            "pass": False,
            "score": 0.0,
            "reason": f"step={idx} out of range (have {len(blocks)} blocks)",
        }
    ok = _match(value, blocks[idx], use_regex)
    return {
        "pass": ok,
        "score": 1.0 if ok else 0.0,
        "reason": f"value {'matched' if ok else 'not found'} in block {idx}",
    }
```

- [ ] **Step 4: Run tests, verify all pass**

```bash
pytest tests/python/test_assert_reasoning_contains.py -v
```

Expected: 15 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/python/test_assert_reasoning_contains.py assertions/assert_reasoning_contains.py
git commit -m "feat: add step-mode (any/int) matching to assert_reasoning_contains"
```

---

## Task 4: Refactor `assert_reasoning_iterations.py`

**Files:**
- Create: `tests/python/test_assert_reasoning_iterations.py`
- Modify: `assertions/assert_reasoning_iterations.py`

Drop the inline `<think>` fallback (Layer 2 will strip those before assertions see `output`). Source priority becomes the same one used by `assert_reasoning_contains.py`.

- [ ] **Step 1: Write failing tests**

Create `tests/python/test_assert_reasoning_iterations.py`:

```python
from assertions.assert_reasoning_iterations import get_assert


def _ctx(response, **cfg):
    return {"response": response, "config": cfg}


def test_paragraphs_in_reasoning_content_count_as_iterations():
    text = "First.\n\nSecond.\n\nThird."
    ctx = _ctx({"reasoning_content": text}, min=3)
    result = get_assert("ok", ctx)
    assert result["pass"] is True


def test_too_few_iterations_fails():
    ctx = _ctx({"reasoning_content": "only one paragraph"}, min=3)
    result = get_assert("ok", ctx)
    assert result["pass"] is False


def test_thinking_field_falls_through_when_reasoning_content_absent():
    text = "1. parse\n2. compute\n3. respond"
    ctx = _ctx({"thinking": text}, min=3)
    assert get_assert("ok", ctx)["pass"] is True


def test_claude_thinking_blocks_each_count_separately():
    response = {
        "content": [
            {"type": "thinking", "thinking": "first"},
            {"type": "thinking", "thinking": "second"},
            {"type": "thinking", "thinking": "third"},
        ]
    }
    ctx = _ctx(response, min=3)
    assert get_assert("ok", ctx)["pass"] is True


def test_no_reasoning_fails_with_reason():
    ctx = _ctx({})
    result = get_assert("ok", ctx)
    assert result["pass"] is False
    assert "no" in result["reason"].lower()


def test_max_bound_enforced():
    text = "\n\n".join(f"step {i}" for i in range(20))
    ctx = _ctx({"reasoning_content": text}, max=5)
    assert get_assert("ok", ctx)["pass"] is False
```

- [ ] **Step 2: Run tests against the current implementation**

```bash
pytest tests/python/test_assert_reasoning_iterations.py -v
```

Expected: most pass (current implementation already handles structured fields), but the Claude blocks test may fail because the existing code returns the first block as a single string rather than counting blocks. Note which fail.

- [ ] **Step 3: Update `assertions/assert_reasoning_iterations.py`**

Replace the entire file with this simplified version that uses the shared block-extraction priority:

```python
"""Assert on the number of reasoning iterations a model surfaced.

Reads reasoning from the same source priority as
`assert_reasoning_contains.py`:
  1. context["response"]["reasoning_content"] — split on blank-line paragraphs
  2. context["response"]["thinking"]          — split on blank-line paragraphs
  3. context["response"]["content"][i] thinking blocks — each block is a step

Counting heuristic per text: max of (numbered/bulleted items, step keywords,
paragraph count). Final iteration count is max across blocks combined OR the
block count when reasoning is structured into blocks.

The Layer 2 transform strips inline <think>...</think> from `output` before
assertions run, so this assertion no longer falls back to parsing `output`.

Config keys:
    min — inclusive lower bound (default: 1)
    max — inclusive upper bound (default: 50)
"""

import re

NUMBERED = re.compile(r"^\s*(?:\d+[.)]|[-*])\s+", re.M)
STEP_KW = re.compile(
    r"\b(step\s*\d+|first(?:ly)?|second(?:ly)?|third(?:ly)?|then|next|finally|therefore)\b",
    re.I,
)


def _split_paragraphs(text):
    parts = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    return parts or [text.strip()]


def _reasoning_blocks(context):
    resp = (context or {}).get("response") or {}

    rc = resp.get("reasoning_content")
    if isinstance(rc, str) and rc.strip():
        return _split_paragraphs(rc)

    t = resp.get("thinking")
    if isinstance(t, str) and t.strip():
        return _split_paragraphs(t)

    blocks = []
    for item in resp.get("content", []) or []:
        if isinstance(item, dict) and item.get("type") == "thinking":
            text = item.get("thinking") or item.get("text") or ""
            if text.strip():
                blocks.append(text)
    return blocks


def _heuristic_count(text):
    bullets = len(NUMBERED.findall(text))
    keywords = len(STEP_KW.findall(text))
    paragraphs = len([p for p in re.split(r"\n\s*\n", text) if p.strip()])
    return max(bullets, keywords, paragraphs, 1)


def get_assert(output, context):
    cfg = (context or {}).get("config") or {}
    minimum = int(cfg.get("min", 1))
    maximum = int(cfg.get("max", 50))

    blocks = _reasoning_blocks(context)
    if not blocks:
        return {"pass": False, "score": 0.0, "reason": "no reasoning available"}

    # If reasoning came as multiple blocks (Claude or paragraph-split), the block
    # count is the natural iteration count. Heuristic count is a floor for
    # single-block cases.
    if len(blocks) > 1:
        iters = len(blocks)
    else:
        iters = _heuristic_count(blocks[0])

    ok = minimum <= iters <= maximum
    return {
        "pass": ok,
        "score": 1.0 if ok else 0.0,
        "reason": f"~{iters} reasoning iterations (allowed {minimum}..{maximum})",
    }
```

- [ ] **Step 4: Run tests, verify all pass**

```bash
pytest tests/python/test_assert_reasoning_iterations.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Run the full Python test suite as a regression check**

```bash
pytest tests/python -v
```

Expected: 21 total passing (15 from Task 3 + 6 here).

- [ ] **Step 6: Commit**

```bash
git add tests/python/test_assert_reasoning_iterations.py assertions/assert_reasoning_iterations.py
git commit -m "refactor: simplify assert_reasoning_iterations to use shared source priority"
```

---

## Task 5: `hooks/normalize_response.js` — strip `<think>...</think>`

**Files:**
- Create: `hooks/normalize_response.test.js`
- Create: `hooks/normalize_response.js`

- [ ] **Step 1: Write failing tests**

Create `hooks/normalize_response.test.js`:

```js
const test = require('node:test');
const assert = require('node:assert');
const normalize = require('./normalize_response');

test('returns unchanged when no reasoning markers', () => {
  assert.strictEqual(normalize('pong', {}), 'pong');
  assert.strictEqual(normalize('hello world', {}), 'hello world');
});

test('strips a complete <think>...</think> pair', () => {
  assert.strictEqual(normalize('<think>reasoning here</think>pong', {}), 'pong');
});

test('strips multi-line <think> block and surrounding whitespace', () => {
  assert.strictEqual(
    normalize('<think>multi\nline\nstuff</think>\n\npong', {}),
    'pong'
  );
});

test('strips <thinking>...</thinking> variant', () => {
  assert.strictEqual(normalize('<thinking>foo</thinking>bar', {}), 'bar');
});

test('leaves unclosed <think> mention alone', () => {
  assert.strictEqual(
    normalize('What does <think> mean in HTML?', {}),
    'What does <think> mean in HTML?'
  );
});

test('handles empty string', () => {
  assert.strictEqual(normalize('', {}), '');
});

test('returns non-string inputs unchanged', () => {
  assert.strictEqual(normalize(null, {}), null);
  assert.strictEqual(normalize(undefined, {}), undefined);
});
```

- [ ] **Step 2: Run tests, verify they fail with module-not-found**

```bash
node --test hooks/normalize_response.test.js
```

Expected: tests fail because `./normalize_response` doesn't exist.

- [ ] **Step 3: Write minimal implementation**

Create `hooks/normalize_response.js`:

```js
// Strips reasoning artifacts from a model response so built-in promptfoo
// assertions see only the final answer. Reasoning content for custom
// assertions is sourced from structured response fields, not from this
// transform — see docs/superpowers/specs/2026-05-07-reasoning-aware-assertions-design.md.

const THINK_PAIR = /<think(?:ing)?>[\s\S]*?<\/think(?:ing)?>/gi;

function normalize(output, context) {
  try {
    if (typeof output !== 'string') return output;
    return output.replace(THINK_PAIR, '').trim();
  } catch (err) {
    process.stderr.write(`normalize_response: ${err.message}\n`);
    return output;
  }
}

module.exports = normalize;
```

- [ ] **Step 4: Run tests, verify they pass**

```bash
node --test hooks/normalize_response.test.js
```

Expected: 7 tests pass.

- [ ] **Step 5: Commit**

```bash
git add hooks/normalize_response.js hooks/normalize_response.test.js
git commit -m "feat: add normalize_response transform (<think> stripping)"
```

---

## Task 6: `hooks/normalize_response.js` — `Thinking:\\n{3,}` artifact + exception fallthrough

**Files:**
- Modify: `hooks/normalize_response.test.js`
- Modify: `hooks/normalize_response.js`

- [ ] **Step 1: Write failing tests**

Append to `hooks/normalize_response.test.js`:

```js
test('strips Thinking:...newlines rendered-template artifact', () => {
  const input = 'Thinking: \nOkay let me consider this.\nMore thinking.\n\n\n\npong';
  assert.strictEqual(normalize(input, {}), 'pong');
});

test('Thinking: prefix without 3+ newline run is left alone', () => {
  // Only two newlines after — not the rendered artifact pattern
  const input = 'Thinking: I should reply\n\npong';
  assert.strictEqual(normalize(input, {}), input.trim());
});

test('Thinking: not at start of string is left alone', () => {
  const input = 'Notice: Thinking: hard about this\n\n\n\nAnswer';
  assert.strictEqual(normalize(input, {}), input.trim());
});

test('combined <think> tag and Thinking: prefix both stripped', () => {
  const input = 'Thinking: foo\n\n\n\n<think>bar</think>pong';
  assert.strictEqual(normalize(input, {}), 'pong');
});
```

- [ ] **Step 2: Run tests, verify the new ones fail**

```bash
node --test hooks/normalize_response.test.js
```

Expected: 7 pass, 4 fail (artifact-stripping not yet implemented).

- [ ] **Step 3: Update implementation**

Replace `hooks/normalize_response.js` body with:

```js
// Strips reasoning artifacts from a model response so built-in promptfoo
// assertions see only the final answer. Reasoning content for custom
// assertions is sourced from structured response fields, not from this
// transform — see docs/superpowers/specs/2026-05-07-reasoning-aware-assertions-design.md.

const THINK_PAIR = /<think(?:ing)?>[\s\S]*?<\/think(?:ing)?>/gi;
const RENDERED_THINKING = /^Thinking:[\s\S]*?\n{3,}/;

function normalize(output, context) {
  try {
    if (typeof output !== 'string') return output;
    let cleaned = output.replace(THINK_PAIR, '');
    cleaned = cleaned.replace(RENDERED_THINKING, '');
    return cleaned.trim();
  } catch (err) {
    process.stderr.write(`normalize_response: ${err.message}\n`);
    return output;
  }
}

module.exports = normalize;
```

- [ ] **Step 4: Run tests, verify all pass**

```bash
node --test hooks/normalize_response.test.js
```

Expected: 11 tests pass.

- [ ] **Step 5: Commit**

```bash
git add hooks/normalize_response.js hooks/normalize_response.test.js
git commit -m "feat: also strip Thinking:\\n{3,} rendered-template artifact"
```

---

## Task 7: Wire transform + provider configs + smoke regex

**Files:**
- Modify: `promptfooconfig.yaml`
- Modify: `providers/ollama_local.yaml`
- Modify: `tests/smoke.yaml`

- [ ] **Step 1: Add transform to `promptfooconfig.yaml`**

Modify `defaultTest.options` block. Current:

```yaml
defaultTest:
  options:
    cache: true
    # TODO: configure an LLM-as-judge for any `llm-rubric` assertions, e.g.
    # provider:
    #   id: anthropic:messages:claude-sonnet-4-5
    #   config:
    #     apiKey: "{{ env.ANTHROPIC_API_KEY }}"
    #     temperature: 0
  assert:
    - type: python
      value: file://assertions/assert_no_censorship.py
```

Replace with:

```yaml
defaultTest:
  options:
    cache: true
    transform: file://hooks/normalize_response.js
    # TODO: configure an LLM-as-judge for any `llm-rubric` assertions, e.g.
    # provider:
    #   id: anthropic:messages:claude-sonnet-4-5
    #   config:
    #     apiKey: "{{ env.ANTHROPIC_API_KEY }}"
    #     temperature: 0
  assert:
    - type: python
      value: file://assertions/assert_no_censorship.py
```

- [ ] **Step 2: Add `think: true` to Ollama provider**

Modify `providers/ollama_local.yaml`. Current:

```yaml
id: openai:chat:llama3.1:8b
label: ollama_local_llama31
config:
  apiBaseUrl: http://localhost:11434/v1
  apiKey: ollama        # placeholder; Ollama ignores it
  temperature: 0.2
```

Replace with:

```yaml
id: openai:chat:llama3.1:8b
label: ollama_local_llama31
config:
  apiBaseUrl: http://localhost:11434/v1
  apiKey: ollama        # placeholder; Ollama ignores it
  temperature: 0.2
  # Ollama returns thinking content in a separate `thinking` field when
  # this is set. Non-thinking models silently ignore it.
  think: true
```

- [ ] **Step 3: Tighten smoke test**

Replace the contents of `tests/smoke.yaml` with:

```yaml
# Liveness ping: every provider should answer this with the single word "pong".
# A trailing period is tolerated; anything else (code blocks, leaked reasoning)
# fails the test. This relies on hooks/normalize_response.js to strip reasoning
# from `output` before this assertion runs.
- description: "smoke ping"
  vars:
    user: "Reply with the single word: pong"
    system: file://system_prompts/concise.txt
  assert:
    - type: regex
      value: '^\s*pong\.?\s*$'
```

- [ ] **Step 4: Quick syntax check on the YAML**

```bash
python3 -c "import yaml; yaml.safe_load(open('promptfooconfig.yaml')); yaml.safe_load(open('providers/ollama_local.yaml')); yaml.safe_load(open('tests/smoke.yaml')); print('OK')"
```

Expected: `OK`.

- [ ] **Step 5: Commit**

```bash
git add promptfooconfig.yaml providers/ollama_local.yaml tests/smoke.yaml
git commit -m "feat: wire normalize_response transform; tighten smoke test"
```

---

## Task 8: `scripts/start_vllm.sh`

**Files:**
- Create: `scripts/start_vllm.sh`

- [ ] **Step 1: Write the script**

Create `scripts/start_vllm.sh`:

```bash
#!/usr/bin/env bash
# Launch vLLM with the right reasoning-parser flags. Reads .env for the
# model, port, and parser. If VLLM_REASONING_PARSER is unset, vLLM is
# launched without --enable-reasoning (use this for non-reasoning models).
#
# Override individual values inline, e.g.:
#   VLLM_REASONING_PARSER=qwen3 ./scripts/start_vllm.sh
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

: "${VLLM_MODEL_ID:?Set VLLM_MODEL_ID in .env or environment}"
PORT="${VLLM_PORT:-8000}"

ARGS=(serve "$VLLM_MODEL_ID" --port "$PORT")

if [[ -n "${VLLM_REASONING_PARSER:-}" ]]; then
  ARGS+=(--enable-reasoning --reasoning-parser "$VLLM_REASONING_PARSER")
fi

if [[ -n "${VLLM_DTYPE:-}" ]]; then
  ARGS+=(--dtype "$VLLM_DTYPE")
fi

if [[ -n "${VLLM_GPU_MEMORY_UTILIZATION:-}" ]]; then
  ARGS+=(--gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION")
fi

echo "Launching: vllm ${ARGS[*]}"
exec vllm "${ARGS[@]}"
```

- [ ] **Step 2: Make it executable**

```bash
chmod +x scripts/start_vllm.sh
```

- [ ] **Step 3: Smoke-check the script with a dry-run check that doesn't call vllm**

```bash
VLLM_MODEL_ID=fake/model VLLM_REASONING_PARSER=deepseek_r1 bash -n scripts/start_vllm.sh && echo "syntax OK"
```

Expected: `syntax OK` (no execution, just a syntax check).

- [ ] **Step 4: Commit**

```bash
git add scripts/start_vllm.sh
git commit -m "feat: add scripts/start_vllm.sh launcher"
```

---

## Task 9: Documentation updates

**Files:**
- Modify: `.env.example`
- Modify: `README.md`

- [ ] **Step 1: Update `.env.example`**

Replace the vLLM section so new variables appear. Find this block:

```
# vLLM (or any OpenAI-compatible) host on your LAN.
VLLM_BASE_URL=http://lab.lan:8000/v1
VLLM_API_KEY=changeme
# Model id served by your vLLM endpoint (see GET /v1/models).
VLLM_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct
```

Replace with:

```
# vLLM (or any OpenAI-compatible) host on your LAN.
VLLM_BASE_URL=http://lab.lan:8000/v1
VLLM_API_KEY=changeme
# Model id served by your vLLM endpoint (see GET /v1/models).
VLLM_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct
# Port the in-repo launcher should bind. Used by ./scripts/start_vllm.sh.
VLLM_PORT=8000
# Reasoning parser to enable for thinking models (e.g. deepseek_r1, qwen3).
# Leave unset for non-reasoning models.
# VLLM_REASONING_PARSER=deepseek_r1
# Optional vLLM tuning flags consumed by the launcher.
# VLLM_DTYPE=bfloat16
# VLLM_GPU_MEMORY_UTILIZATION=0.9
```

- [ ] **Step 2: Update `README.md` — Setup and "What it tests" sections**

Find this block in the "What it tests" list:

```
- **Reasoning iterations** — heuristic step counter against surfaced thinking
  content (DeepSeek `showThinking`, Claude extended thinking, `<think>` tags).
```

Replace with:

```
- **Reasoning iterations and content** — heuristic step counter
  (`assert_reasoning_iterations.py`) plus substring/regex/per-step matching
  against surfaced thinking content (`assert_reasoning_contains.py`). Both read
  from the structured reasoning fields populated by configured providers; see
  `docs/superpowers/specs/2026-05-07-reasoning-aware-assertions-design.md`.
```

Find the "Run" section. Add a new subsection just below it (before "Adding a model"):

```
## Launching vLLM

```bash
./scripts/start_vllm.sh
```

Reads `.env` for `VLLM_MODEL_ID`, `VLLM_PORT`, and (optionally)
`VLLM_REASONING_PARSER`. The reasoning parser is required for thinking models
(`deepseek_r1`, `qwen3`, etc.) — without it, reasoning text is mixed into the
answer field and reasoning assertions cannot read it.

## Running tests for custom assertions

```bash
pytest tests/python                      # Python assertion unit tests
node --test hooks/normalize_response.test.js   # JS transform unit tests
```
```

- [ ] **Step 3: Drop the redundant TODO in `README.md`**

Find at the bottom:

```
## TODO

- Configure an LLM-as-judge (cloud or local) for `llm-rubric` assertions —
  see the comment in `promptfooconfig.yaml`.
- Add a CI workflow if/when this is moved into a git repo.
```

Leave that section as-is — those items remain accurate. No change.

- [ ] **Step 4: Commit**

```bash
git add .env.example README.md
git commit -m "docs: document start_vllm.sh and reasoning-aware assertions"
```

---

## Task 10: End-to-end verification (Plan A)

This task runs the full system against a real reasoning-configured vLLM. No code changes if it passes. **If `reasoning_content` is missing from the response visible to assertions, proceed to Task 11 (Plan B).**

- [ ] **Step 1: Set the reasoning parser in `.env`**

Edit `.env` (or set inline) so `VLLM_REASONING_PARSER` matches your model. For DeepSeek R1 derivatives, use `deepseek_r1`. For Qwen3-thinking, use `qwen3`.

```bash
# Example
echo 'VLLM_REASONING_PARSER=deepseek_r1' >> .env
```

- [ ] **Step 2: Launch vLLM via the new script**

In a separate terminal:

```bash
./scripts/start_vllm.sh
```

Wait until the server reports `Application startup complete`.

- [ ] **Step 3: Run a reasoning test to capture a full response**

```bash
pnpm exec promptfoo eval \
  --config promptfooconfig.yaml \
  --filter-providers vllm_lan \
  --filter-pattern reasoning \
  --output results/verify.json
```

- [ ] **Step 4: Inspect the response object for `reasoning_content`**

```bash
python3 - <<'PY'
import json
data = json.load(open('results/verify.json'))
for r in data['results']['results']:
    resp = (r.get('response') or {})
    print('keys:', sorted(resp.keys()))
    if 'reasoning_content' in resp:
        print('FOUND reasoning_content (len):', len(resp['reasoning_content']))
    elif 'thinking' in resp:
        print('FOUND thinking (len):', len(resp['thinking']))
    else:
        print('NO structured reasoning field at top level')
    break
PY
```

Expected if Plan A works: `FOUND reasoning_content (len): <N>` or `FOUND thinking (len): <N>`.

If output is `NO structured reasoning field at top level`, **stop here and proceed to Task 11.**

- [ ] **Step 5: Run the smoke test against vLLM**

```bash
PROVIDER=vllm_lan ./scripts/smoke.sh
```

Expected: smoke test passes. The tightened `^\s*pong\.?\s*$` regex must hit because `output` is now clean.

- [ ] **Step 6: Run the existing reasoning test as a final check**

```bash
pnpm exec promptfoo eval \
  --config promptfooconfig.yaml \
  --filter-providers vllm_lan \
  --filter-pattern reasoning
```

Expected: the existing `assert_reasoning_iterations` test passes against the refactored code. This proves the source-priority lookup works end-to-end.

- [ ] **Step 7: Commit (verification artifacts only if any)**

If only `results/` got new content, nothing to commit (that directory is gitignored). If anything else changed, commit it. Otherwise skip.

```bash
git status
```

If the working tree is clean (or only `results/` changed and is gitignored), this task is complete. **Do not run Task 11.**

---

## Task 11 (CONTINGENCY — only if Task 10 Step 4 failed)

This task only runs if `promptfoo`'s `openai:chat:` adapter does not pass `reasoning_content` through to the response visible to assertions. We add a custom Python provider wrapper.

**Files:**
- Create: `providers/reasoning_aware_openai.py`
- Modify: `providers/vllm_lan.yaml`
- Modify: `providers/ollama_local.yaml`
- Modify: `requirements.txt` (add `httpx`)
- Create: `tests/python/test_reasoning_aware_openai.py`

- [ ] **Step 1: Add httpx**

Append to `requirements.txt`:

```
httpx>=0.27
```

```bash
pip install -r requirements.txt
```

- [ ] **Step 2: Write failing tests**

Create `tests/python/test_reasoning_aware_openai.py`:

```python
"""Tests the response-shaping logic of reasoning_aware_openai.

The HTTP call itself is mocked; only the response post-processing is under
test.
"""

from providers.reasoning_aware_openai import _shape_response


def test_reasoning_content_lifted_to_top_level():
    raw = {
        "choices": [{
            "message": {
                "content": "pong",
                "reasoning_content": "I should reply with pong.",
            }
        }]
    }
    shaped = _shape_response(raw)
    assert shaped["output"] == "pong"
    assert shaped["reasoning_content"] == "I should reply with pong."


def test_thinking_field_lifted_to_top_level():
    raw = {
        "choices": [{"message": {"content": "ok"}}],
        "thinking": "considered options",
    }
    shaped = _shape_response(raw)
    assert shaped["output"] == "ok"
    assert shaped["thinking"] == "considered options"


def test_no_reasoning_present():
    raw = {"choices": [{"message": {"content": "hi"}}]}
    shaped = _shape_response(raw)
    assert shaped["output"] == "hi"
    assert "reasoning_content" not in shaped
    assert "thinking" not in shaped
```

- [ ] **Step 3: Run tests, verify they fail**

```bash
pytest tests/python/test_reasoning_aware_openai.py -v
```

Expected: ModuleNotFoundError.

- [ ] **Step 4: Write the provider**

Create `providers/reasoning_aware_openai.py`:

```python
"""Custom promptfoo provider that wraps an OpenAI-compatible chat endpoint
and surfaces reasoning_content / thinking at the top level of the response
visible to assertions.

Used when promptfoo's built-in `openai:chat:` adapter strips those fields.
"""

import os
import httpx


def _shape_response(raw):
    msg = (raw.get("choices") or [{}])[0].get("message") or {}
    out = {"output": msg.get("content", "")}
    rc = msg.get("reasoning_content")
    if isinstance(rc, str) and rc.strip():
        out["reasoning_content"] = rc
    t = raw.get("thinking") or msg.get("thinking")
    if isinstance(t, str) and t.strip():
        out["thinking"] = t
    return out


def call_api(prompt, options, context):
    cfg = (options or {}).get("config") or {}
    base = cfg.get("apiBaseUrl") or os.environ.get("OPENAI_BASE_URL")
    key = cfg.get("apiKey") or os.environ.get("OPENAI_API_KEY", "")
    model = cfg.get("model") or options.get("id", "").split(":", 2)[-1]

    payload = {
        "model": model,
        "messages": prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}],
        "temperature": cfg.get("temperature", 0.2),
    }
    if cfg.get("think") is not None:
        payload["think"] = cfg["think"]

    headers = {"Authorization": f"Bearer {key}"} if key else {}

    with httpx.Client(timeout=cfg.get("timeout", 120)) as client:
        resp = client.post(f"{base}/chat/completions", json=payload, headers=headers)
        resp.raise_for_status()
        return _shape_response(resp.json())
```

- [ ] **Step 5: Run tests, verify they pass**

```bash
pytest tests/python/test_reasoning_aware_openai.py -v
```

Expected: 3 passed.

- [ ] **Step 6: Switch providers to use the wrapper**

Replace `providers/vllm_lan.yaml`:

```yaml
id: file://providers/reasoning_aware_openai.py
label: vllm_lan
config:
  apiBaseUrl: "{{ env.VLLM_BASE_URL }}"
  apiKey: "{{ env.VLLM_API_KEY }}"
  model: "{{ env.VLLM_MODEL_ID }}"
  temperature: 0.2
```

Replace `providers/ollama_local.yaml`:

```yaml
id: file://providers/reasoning_aware_openai.py
label: ollama_local_llama31
config:
  apiBaseUrl: http://localhost:11434/v1
  apiKey: ollama
  model: llama3.1:8b
  temperature: 0.2
  think: true
```

- [ ] **Step 7: Re-run Task 10 Step 4**

```bash
pnpm exec promptfoo eval \
  --config promptfooconfig.yaml \
  --filter-providers vllm_lan \
  --filter-pattern reasoning \
  --output results/verify.json

python3 - <<'PY'
import json
data = json.load(open('results/verify.json'))
for r in data['results']['results']:
    resp = r.get('response') or {}
    print('keys:', sorted(resp.keys()))
    print('has reasoning_content:', 'reasoning_content' in resp)
    break
PY
```

Expected: `has reasoning_content: True`.

- [ ] **Step 8: Re-run smoke and reasoning tests**

```bash
PROVIDER=vllm_lan ./scripts/smoke.sh
PROVIDER=ollama_local_llama31 ./scripts/smoke.sh
```

Expected: both pass.

- [ ] **Step 9: Commit**

```bash
git add providers/reasoning_aware_openai.py providers/vllm_lan.yaml providers/ollama_local.yaml requirements.txt tests/python/test_reasoning_aware_openai.py
git commit -m "feat: wrap OpenAI-compat providers to surface reasoning fields"
```
