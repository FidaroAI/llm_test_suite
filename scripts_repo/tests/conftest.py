"""Make the project root importable so tests can `from scripts_repo.foo import ...`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
