"""Helpers for the *live* tests (those that hit a real LLM endpoint).

For this demonstration suite we turn "configured but unreachable / not actually
an LLM endpoint" into a clean pytest skip with a clear reason, rather than a
confusing stack trace. In a real regression CI you'd likely drop this and let an
outage fail loudly — the point here is a tidy out-of-the-box experience.
"""
import pytest
from openai import OpenAIError

from deepeval_demo.model_under_test import generate_answer


def generate_or_skip(question: str, **kwargs) -> str:
    try:
        return generate_answer(question, **kwargs)
    except RuntimeError as e:  # not configured
        pytest.skip(str(e))
    except OpenAIError as e:  # configured but unreachable / wrong endpoint
        pytest.skip(f"model under test unreachable: {e}")
