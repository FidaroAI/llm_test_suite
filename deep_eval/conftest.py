"""Pytest bootstrap: load environment from deep_eval/.env (and fall back to the
parent suite's top-level .env), so tests can read endpoint/key config."""
from pathlib import Path

from dotenv import load_dotenv

_HERE = Path(__file__).resolve().parent

# Local deep_eval/.env wins; then the parent suite's .env fills any gaps
# (VLLM_*, ANTHROPIC_API_KEY) without overriding what we already set.
load_dotenv(_HERE / ".env")
load_dotenv(_HERE.parent / ".env", override=False)
