"""Make project root importable so tests can `from assertions.foo import ...`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
