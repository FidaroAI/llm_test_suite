"""Promptfoo output transform: drop everything up to and including the first
``\\n\\n\\n`` so assertions only see the model's final answer.

Useful for reasoning-emitting models (e.g. Qwen3-Thinking) where the response
body is structured as ``<reasoning>\\n\\n\\n<final answer>``. Non-string
outputs (e.g. a dict produced by ``response_format: json_schema`` auto-parse)
are returned unchanged so the transform is safe to wire globally.
"""

DELIMITER = "\n\n\n"


def get_transform(output, context):
    if not isinstance(output, str):
        return output
    idx = output.find(DELIMITER)
    if idx == -1:
        return output
    return output[idx + len(DELIMITER):]
