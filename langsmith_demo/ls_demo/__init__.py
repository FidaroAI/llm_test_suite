"""LangSmith demo package: the LangSmith counterpart of the deep_eval demo.

Same two test cases as deep_eval, expressed the LangSmith way:
  * target.py     — calls the OpenAI-compatible model under test (the function
                    LangSmith evaluates, one call per dataset example).
  * evaluators.py — a deterministic `contains` evaluator and an LLM-as-judge
                    `rubric` evaluator, returning LangSmith {key, score, comment}.
  * datasets.py   — the two server-side datasets (facts + support reply).
"""
