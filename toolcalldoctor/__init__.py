"""toolcalldoctor -- diagnose and fix tool calling against local model backends.

Three tools, one goal: make local models usable with agent harnesses that
expect strict OpenAI-shaped tool calls.

  probe  Identify which tool-call wire format a backend/model actually emits.
  shim   OpenAI-compatible proxy that deterministically repairs malformed calls.
  score  Measure parse / tool-selection / argument accuracy across models.

Pure standard library. No dependencies.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
