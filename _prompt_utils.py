from __future__ import annotations


def resolve_prompt_prefix(config: dict) -> str | None:
  direct_prefix = config.get("prompt_prefix")
  if isinstance(direct_prefix, str) and direct_prefix.strip():
    return direct_prefix.strip()
  return None


def apply_prompt_prefix(prompt: str, prompt_prefix: str | None) -> str:
  if not prompt_prefix:
    return prompt
  return prompt_prefix + "\n\n" + prompt
