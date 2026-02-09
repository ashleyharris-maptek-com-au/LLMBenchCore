from __future__ import annotations

# TODO(tech-debt): This module only hosts two tiny helpers. Revisit and inline if
# call sites consolidate or this starts feeling like unnecessary indirection.


def resolve_prompt_prefix(config: dict) -> str | None:
  direct_prefix = config.get("prompt_prefix")
  if isinstance(direct_prefix, str) and direct_prefix.strip():
    return direct_prefix.strip()
  return None


def apply_prompt_prefix(prompt: str, prompt_prefix: str | None) -> str:
  if not prompt_prefix:
    return prompt
  return prompt_prefix + "\n\n" + prompt
