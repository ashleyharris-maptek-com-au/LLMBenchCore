"""
Z-AI (Zhipu AI) Engine for LLMBenchCore

This module provides an interface to the Z-AI API using the official zai-sdk.

Setup:
1. Install the SDK: pip install zai-sdk
2. Set your API key as an environment variable:
   - Windows: set ZAI_API_KEY=your_api_key_here
   - Linux/Mac: export ZAI_API_KEY=your_api_key_here
   
Get your API key from: https://chat.z.ai/

The SDK documentation can be found at: https://z.ai/model-api

Key differences from other engines:
- Thinking (deep thinking) is ENABLED by default. We explicitly disable it
  when reasoning=False, unlike OpenAI where reasoning is opt-in.
- Structured output uses response_format={"type": "json_object"} with the
  schema embedded in the system prompt (no strict JSON schema mode).
- Function calling follows the OpenAI-compatible format.
- No built-in tools (web_search, code_execution) from the platform; when
  tools=True we provide function-calling tools for web search and local
  Python code execution.
"""

import hashlib
import json
import os
import random
import threading
import time
from typing import Any, Optional

try:
  from . import PromptImageTagging as pit
  from .ToolExecutors import ALL_TOOLS as _BUILTIN_TOOLS
  from .ToolExecutors import dispatch_tool_call as _dispatch_tool_call
except:
  import PromptImageTagging as pit
  from ToolExecutors import ALL_TOOLS as _BUILTIN_TOOLS
  from ToolExecutors import dispatch_tool_call as _dispatch_tool_call

# Maximum number of tool-call round-trips before forcing a final answer
_MAX_TOOL_ROUNDS = 8

# ---------------------------------------------------------------------------
# Adaptive concurrency limiter (dynamic semaphore)
# ---------------------------------------------------------------------------
# Z-AI rate-limits by concurrent request count, not requests-per-minute.
# We keep a module-level limiter that:
#   - blocks a thread if max_slots are already in use
#   - shrinks max_slots on a 429 response (minimum 1)
#   - slowly grows max_slots back on success

_DEFAULT_MAX_SLOTS = 5  # starting guess for max concurrent requests
_ABSOLUTE_MIN_SLOTS = 1


class _ConcurrencyLimiter:
  """Thread-safe adaptive concurrency limiter."""

  def __init__(self, max_slots: int = _DEFAULT_MAX_SLOTS):
    self._lock = threading.Condition(threading.Lock())
    self._active = 0
    self._max_slots = max_slots
    self._successes_since_last_shrink = 0
    # How many consecutive successes before we try raising the limit by 1
    self._grow_after = 200

  @property
  def max_slots(self) -> int:
    with self._lock:
      return self._max_slots

  @property
  def active(self) -> int:
    with self._lock:
      return self._active

  def acquire(self) -> None:
    """Block until a slot is available."""
    with self._lock:
      while self._active >= self._max_slots:
        print(f"[Z-AI limiter] {self._active}/{self._max_slots} slots busy, waiting...", flush=True)
        self._lock.wait(timeout=30)  # re-check periodically
      self._active += 1

  def release_success(self) -> None:
    """Release a slot after a successful request."""
    with self._lock:
      self._active = max(0, self._active - 1)
      self._successes_since_last_shrink += 1
      if self._successes_since_last_shrink >= self._grow_after:
        self._max_slots += 1
        self._successes_since_last_shrink = 0
        print(f"[Z-AI limiter] Raised max concurrent requests to {self._max_slots}", flush=True)
      self._lock.notify_all()

  def release_rate_limited(self) -> None:
    """Release a slot and shrink max_slots because we got a 429."""
    with self._lock:
      self._active = max(0, self._active - 1)
      old = self._max_slots
      self._max_slots = max(_ABSOLUTE_MIN_SLOTS, self._max_slots - 1)
      self._successes_since_last_shrink = 0
      if self._max_slots != old:
        print(
          f"[Z-AI limiter] 429 hit – lowered max concurrent requests from {old} to {self._max_slots}",
          flush=True)
      else:
        print(
          f"[Z-AI limiter] 429 hit – max concurrent requests already at minimum ({self._max_slots})",
          flush=True)
      self._lock.notify_all()

  def release_error(self) -> None:
    """Release a slot after a non-rate-limit error (no capacity change)."""
    with self._lock:
      self._active = max(0, self._active - 1)
      self._lock.notify_all()


_zai_limiter = _ConcurrencyLimiter(_DEFAULT_MAX_SLOTS)

# ---------------------------------------------------------------------------
# Engine class
# ---------------------------------------------------------------------------


class ZaiEngine:
  """
  Z-AI Engine class.
  
  Configuration parameters:
  - model: Model name (e.g., "glm-5", "glm-4.6v")
  - reasoning: Thinking/reasoning mode:
      - False: Explicitly disable deep thinking
      - True or integer 1-10: Enable deep thinking (enabled is the Z-AI default)
  - tools: Tool capabilities:
      - False: No tools available
      - True: Enable function-calling tools (web_search, execute_python)
      - List of function definitions: Enable specific custom tools
  - timeout: Request timeout in seconds
  """

  def __init__(self, model: str, reasoning=False, tools=False, timeout: int = 3600):
    self.model = model
    self.reasoning = reasoning
    self.tools = tools
    self.timeout = timeout
    self.configAndSettingsHash = hashlib.sha256(model.encode() + str(reasoning).encode() +
                                                str(tools).encode() +
                                                str(timeout).encode()).hexdigest()

  def AIHook(self, prompt: str, structure: dict | None) -> tuple:
    """Call the Z-AI API with instance configuration."""
    return _zai_ai_hook(prompt,
                        structure,
                        self.model,
                        self.reasoning,
                        self.tools,
                        timeout_override=self.timeout)


# ---------------------------------------------------------------------------
# Message building helpers
# ---------------------------------------------------------------------------


def _build_zai_messages(prompt: str, structure: dict | None) -> list[dict]:
  """
  Build the messages array for the Z-AI chat completions API.
  Handles image tags in the prompt using the OpenAI-compatible content format.
  """
  prompt_parts = pit.parse_prompt_parts(prompt)
  has_images = any(pt == "image" for pt, _ in prompt_parts)

  messages = []

  # If structured output, add schema guidance in a system message
  if structure is not None:
    schema_json = json.dumps(structure, indent=2, ensure_ascii=False)
    messages.append({
      "role":
      "system",
      "content": ("You MUST respond with valid JSON that conforms to this exact schema:\n"
                  f"{schema_json}\n\n"
                  "Return ONLY the JSON object. No markdown, no code blocks, no explanation.")
    })

  if has_images:
    content: list[dict] = []
    for part_type, part_value in prompt_parts:
      if part_type == "text":
        if part_value:
          content.append({"type": "text", "text": part_value})
      elif part_type == "image":
        if pit.is_url(part_value) or pit.is_data_uri(part_value):
          image_url = part_value
        else:
          image_url = pit.file_to_data_uri(pit.resolve_local_path(part_value))
        content.append({"type": "image_url", "image_url": {"url": image_url}})
    messages.append({"role": "user", "content": content})
  else:
    messages.append({"role": "user", "content": prompt})

  return messages


def build_zai_chat_params(prompt: str,
                          structure: dict | None,
                          model: str,
                          reasoning,
                          tools,
                          stream: bool = True) -> dict:
  """
  Build the parameters for a Z-AI chat completions API call.
  Used by both the sync hook and (potential future) batch submission.
  """
  messages = _build_zai_messages(prompt, structure)

  params: dict[str, Any] = {
    "model": model,
    "messages": messages,
    "max_tokens": 16384,
    "temperature": 1.0,
  }

  if stream:
    params["stream"] = True

  # Structured output: use json_object mode
  if structure is not None:
    params["response_format"] = {"type": "json_object"}

  # Thinking configuration
  # Z-AI enables thinking by default; we must explicitly disable it when
  # reasoning is False/0.
  if reasoning is False or reasoning == 0:
    params["thinking"] = {"type": "disabled"}
  else:
    params["thinking"] = {"type": "enabled"}
    # When thinking is enabled, temperature must be 1.0 (already set above)

  # Tools
  if tools is True:
    params["tools"] = _BUILTIN_TOOLS
    params["tool_choice"] = "auto"
  elif tools and tools is not False:
    if isinstance(tools, list):
      params["tools"] = tools
      params["tool_choice"] = "auto"

  return params


# ---------------------------------------------------------------------------
# Main API hook
# ---------------------------------------------------------------------------


def _zai_ai_hook(prompt: str,
                 structure: dict | None,
                 model: str,
                 reasoning,
                 tools,
                 timeout_override: int | None = None) -> tuple:
  """
  Call the Z-AI API and return (result, chainOfThought).
  
  Handles streaming, thinking content, structured output parsing,
  and multi-round function calling for tools mode.
  """
  from zai import ZaiClient

  _zai_limiter.acquire()
  rate_limited = False
  success = False
  try:
    timeout = timeout_override or 3600
    client = ZaiClient(api_key=os.environ.get("ZAI_API_KEY"), timeout=timeout)

    params = build_zai_chat_params(prompt, structure, model, reasoning, tools)

    # --- Streaming call with tool-call loop ---
    chain_of_thought = ""
    output_text = ""
    all_messages = list(params["messages"])  # mutable copy for tool rounds

    for tool_round in range(_MAX_TOOL_ROUNDS + 1):
      current_params = dict(params)
      current_params["messages"] = all_messages

      response = client.chat.completions.create(**current_params)

      round_thinking = ""
      round_content = ""
      tool_calls_accumulator: dict[int, dict] = {}  # index -> {id, name, arguments_str}

      if params.get("stream"):
        thinking_line_buf = ""

        for chunk in response:
          if not chunk.choices:
            continue
          delta = chunk.choices[0].delta

          # Thinking / reasoning content
          if hasattr(delta, "reasoning_content") and delta.reasoning_content:
            round_thinking += delta.reasoning_content
            thinking_line_buf += delta.reasoning_content
            while "\n" in thinking_line_buf:
              line, thinking_line_buf = thinking_line_buf.split("\n", 1)
              print(f"Thinking: {line}", flush=True)

          # Regular content
          if hasattr(delta, "content") and delta.content:
            round_content += delta.content

          # Tool calls (streamed incrementally)
          if hasattr(delta, "tool_calls") and delta.tool_calls:
            for tc in delta.tool_calls:
              idx = tc.index
              if idx not in tool_calls_accumulator:
                tool_calls_accumulator[idx] = {
                  "id": getattr(tc, "id", None) or "",
                  "name": "",
                  "arguments": ""
                }
              if tc.function:
                if tc.function.name:
                  tool_calls_accumulator[idx]["name"] = tc.function.name
                if tc.function.arguments:
                  tool_calls_accumulator[idx]["arguments"] += tc.function.arguments
              # Capture id if it arrives in a later chunk
              if getattr(tc, "id", None):
                tool_calls_accumulator[idx]["id"] = tc.id

        # Flush remaining thinking
        if thinking_line_buf:
          print(f"Thinking: {thinking_line_buf}", flush=True)
          round_thinking += ""  # already accumulated above
      else:
        # Non-streaming fallback
        msg = response.choices[0].message
        round_content = msg.content or ""
        if hasattr(msg, "reasoning_content") and msg.reasoning_content:
          round_thinking = msg.reasoning_content
          for line in round_thinking.split("\n"):
            print(f"Thinking: {line}", flush=True)
        if hasattr(msg, "tool_calls") and msg.tool_calls:
          for i, tc in enumerate(msg.tool_calls):
            tool_calls_accumulator[i] = {
              "id": tc.id,
              "name": tc.function.name,
              "arguments": tc.function.arguments
            }

      chain_of_thought += round_thinking

      # If no tool calls, we're done
      if not tool_calls_accumulator:
        output_text = round_content
        break

      # --- Process tool calls ---
      # Append the assistant message with tool_calls to conversation
      assistant_msg: dict[str, Any] = {"role": "assistant", "content": round_content or None}
      tc_list = []
      for idx in sorted(tool_calls_accumulator.keys()):
        tc_info = tool_calls_accumulator[idx]
        tc_list.append({
          "id": tc_info["id"],
          "type": "function",
          "function": {
            "name": tc_info["name"],
            "arguments": tc_info["arguments"]
          }
        })
      assistant_msg["tool_calls"] = tc_list
      all_messages.append(assistant_msg)

      # Execute each tool and add results
      for tc_entry in tc_list:
        func_name = tc_entry["function"]["name"]
        try:
          args = json.loads(tc_entry["function"]["arguments"])
        except json.JSONDecodeError:
          args = {}

        print(f"  [Tool] {func_name}({json.dumps(args, ensure_ascii=False)[:200]})")
        tool_result = _dispatch_tool_call(func_name, args)
        print(f"  [Tool] -> {tool_result[:300]}")

        chain_of_thought += f"\n[Tool call: {func_name}({json.dumps(args, ensure_ascii=False)[:100]})]\n"
        chain_of_thought += f"[Tool result: {tool_result[:500]}]\n"

        all_messages.append({
          "role": "tool",
          "content": tool_result,
          "tool_call_id": tc_entry["id"]
        })

      # Continue to next round (model will see tool results)
    else:
      # Exhausted tool rounds - use whatever content we have
      print(f"Warning: Z-AI tool call loop exceeded {_MAX_TOOL_ROUNDS} rounds")
      output_text = round_content

    chain_of_thought = chain_of_thought.strip()

    if chain_of_thought:
      print()  # Blank line after thinking

    success = True

    # Parse structured output
    if structure is not None:
      try:
        # Strip markdown code blocks if present
        parse_text = output_text.strip()
        if parse_text.startswith("```json"):
          parse_text = parse_text[7:]
        if parse_text.startswith("```"):
          parse_text = parse_text[3:]
        if parse_text.endswith("```"):
          parse_text = parse_text[:-3]
        parse_text = parse_text.strip()

        return json.loads(parse_text), chain_of_thought
      except json.JSONDecodeError as e:
        try:
          import json_repair
          repaired = json_repair.repair_json(parse_text)
          return json.loads(repaired), chain_of_thought
        except Exception:
          print(f"Structured parse failed: {e}")
          print("Model returned:\n" + output_text[:500])
          return {}, f"Structured parse failed: {e}"
    else:
      return output_text or "", chain_of_thought

  except Exception as e:
    print(f"Error calling Z-AI API: {e}")

    # Check for content policy violation
    from .ContentViolationHandler import is_content_violation_zai
    if is_content_violation_zai(e):
      print("CONTENT VIOLATION DETECTED (Z-AI)")
      if structure is not None:
        return {"__content_violation__": True, "reason": str(e)}, f"Content violation: {e}"
      else:
        return "__content_violation__", f"Content violation: {e}"

    error_str = str(e).lower()
    if "rate limit" in error_str or "too many requests" in error_str or "429" in error_str:
      rate_limited = True
      # Brief pause before the slot becomes available to others
      time.sleep(random.uniform(2, 8))

    if structure is not None:
      return {}, str(e)
    else:
      return "", str(e)
  finally:
    if rate_limited:
      _zai_limiter.release_rate_limited()
    elif success:
      _zai_limiter.release_success()
    else:
      _zai_limiter.release_error()


if __name__ == "__main__":
  engine = ZaiEngine("glm-5", reasoning=False, tools=False)
  print(engine.AIHook("What's the 7th prime number after 101?", None))

  engine = ZaiEngine("glm-5", reasoning=True, tools=True)
  print(engine.AIHook("What is the closest Australian city to New York?", None))

  print(
    engine.AIHook(
      "What is the furthest Australian city from New York?", {
        "type": "object",
        "properties": {
          "cityName": {
            "type": "string"
          },
          "longitude": {
            "type": "number"
          },
          "latitude": {
            "type": "number"
          }
        },
        "required": ["cityName", "longitude", "latitude"]
      }))
