"""DeepEval demo package: a small mirror of the promptfoo regression suite.

Two layers, deliberately separated:
  * model_under_test — calls the OpenAI-compatible LLM whose answers we score.
  * metrics / judge  — score those answers (deterministic ContainsMetric, or the
                       LLM-as-judge GEval rubric).
"""
