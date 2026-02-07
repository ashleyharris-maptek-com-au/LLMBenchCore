from __future__ import annotations


def _read_field(obj, key: str):
  if obj is None:
    return None
  if isinstance(obj, dict):
    return obj.get(key)
  return getattr(obj, key, None)


def _coerce_int(value):
  if value is None:
    return None
  try:
    return int(value)
  except Exception:
    return None


def _extract_reasoning_tokens(usage):
  output_details = _read_field(usage, "output_tokens_details")
  from_output = _coerce_int(_read_field(output_details, "reasoning_tokens"))
  if from_output is not None:
    return from_output
  completion_details = _read_field(usage, "completion_tokens_details")
  return _coerce_int(_read_field(completion_details, "reasoning_tokens"))


def extract_usage_meta(response_obj, provider: str) -> dict:
  usage = _read_field(response_obj, "usage")
  if usage is None:
    usage_meta = {
      "input_tokens": None,
      "output_tokens": None,
      "reasoning_tokens": None,
    }
  else:
    input_tokens = _coerce_int(_read_field(usage, "input_tokens"))
    if input_tokens is None:
      input_tokens = _coerce_int(_read_field(usage, "prompt_tokens"))
    output_tokens = _coerce_int(_read_field(usage, "output_tokens"))
    if output_tokens is None:
      output_tokens = _coerce_int(_read_field(usage, "completion_tokens"))
    usage_meta = {
      "input_tokens": input_tokens,
      "output_tokens": output_tokens,
      "reasoning_tokens": _extract_reasoning_tokens(usage),
    }

  return {"provider": provider, "usage": usage_meta}
