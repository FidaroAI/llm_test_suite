"""llmeval — a cache-key-centric, decoupled LLM evaluation suite.

Pipeline stages (each independently runnable): generate -> run -> grade ->
compare/pickbest/stats -> report. See DESIGN.md.
"""

import logging

__version__ = "0.1.0"

# Standard library courtesy: emit nothing unless the application configures logging.
# The CLI calls llmeval.logs.configure_logging; an embedder does whatever it likes.
logging.getLogger(__name__).addHandler(logging.NullHandler())
