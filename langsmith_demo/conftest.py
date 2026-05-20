"""Pytest bootstrap: load environment from langsmith_demo/.env, then the parent
suite's .env. The unit tests don't need any of it, but the live path does."""
from pathlib import Path

from dotenv import load_dotenv

_HERE = Path(__file__).resolve().parent
load_dotenv(_HERE / ".env")
load_dotenv(_HERE.parent / ".env", override=False)
