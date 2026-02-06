"""
Anthropic Claude AI Engine for LLMBenchCore

This module provides an interface to the Anthropic Claude API using the latest anthropic SDK.

Setup:
1. Install the SDK: pip install anthropic
2. Set your API key as an environment variable:
   - Windows: set ANTHROPIC_API_KEY=your_api_key_here
   - Linux/Mac: export ANTHROPIC_API_KEY=your_api_key_here
   
Get your API key from: https://console.anthropic.com/

The SDK documentation can be found at: https://github.com/anthropics/anthropic-sdk-python
"""

import os, json, hashlib, io, base64
from PIL import Image
from . import PromptImageTagging as pit


class ClaudeEngine:
  """
  Anthropic Claude AI Engine class.
  
  Configuration parameters:
  - model: Model name (e.g., "claude-sonnet-4-20250514")
  - reasoning: Extended thinking mode (for supported models):
      - False: No extended thinking (default, faster)
      - integer 1-10: Enable extended thinking (deeper reasoning)
  - tools: Tool capabilities:
      - False: No tools available
      - True: Enable ALL built-in tools (web_search, code_execution)
      - List of tool definitions: Enable specific custom tools
  - prompt_caching: Enable caching for repeated content (default True)
  - timeout: Request timeout in seconds
  """

  def __init__(self, model: str, reasoning=False, tools=False, prompt_caching=True, timeout: int = 3600):
    self.model = model
    self.reasoning = reasoning
    self.tools = tools
    self.prompt_caching = prompt_caching
    self.timeout = timeout
    self.configAndSettingsHash = hashlib.sha256(model.encode() + str(reasoning).encode() +
                                                str(tools).encode() + str(timeout).encode()).hexdigest()

  def AIHook(self, prompt: str, structure: dict | None) -> tuple:
    """Call the Claude API with instance configuration."""
    return _claude_ai_hook(prompt, structure, self.model, self.reasoning, self.tools,
                           self.prompt_caching, timeout_override=self.timeout)


def _build_anthropic_message_content(prompt: str) -> list[dict]:
  prompt_parts = pit.parse_prompt_parts(prompt)
  content_blocks: list[dict] = []
  for part_type, part_value in prompt_parts:
    if part_type == "text":
      if part_value:
        content_blocks.append({"type": "text", "text": part_value})
    elif part_type == "image":
      if pit.is_url(part_value):
        content_blocks.append({"type": "image", "source": {"type": "url", "url": part_value}})
      elif pit.is_data_uri(part_value):
        mime_type, b64 = pit.data_uri_to_base64(part_value)
        content_blocks.append({
          "type": "image",
          "source": {
            "type": "base64",
            "media_type": mime_type,
            "data": b64
          }
        })
      else:
        local_path = pit.resolve_local_path(part_value)
        # Resize if over 8000 pixels on any side
        img = Image.open(local_path)
        max_dim = 7999
        if img.width > max_dim or img.height > max_dim:
          scale = min(max_dim / img.width, max_dim / img.height)
          new_size = (int(img.width * scale), int(img.height * scale))
          print("Resizing image from", img.width, "x", img.height, "to", new_size)
          img = img.resize(new_size, Image.LANCZOS)
        # Convert to base64
        buffer = io.BytesIO()
        fmt = img.format or 'PNG'
        if fmt.upper() == 'JPEG':
          img.save(buffer, format='JPEG', quality=90)
          mime_type = 'image/jpeg'
        else:
          img.save(buffer, format='PNG')
          mime_type = 'image/png'
        b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        content_blocks.append({
          "type": "image",
          "source": {
            "type": "base64",
            "media_type": mime_type,
            "data": b64
          }
        })

  if not content_blocks:
    content_blocks = [{"type": "text", "text": ""}]

  return content_blocks


def _clean_schema_for_anthropic(schema):
  """
  Clean schema properties for Anthropic compatibility.
  - Sets additionalProperties to false for object types (required by Anthropic)
  - Removes unsupported properties
  Modifies the schema in-place.
  """
  if isinstance(schema, dict):
    # For object types, additionalProperties must be explicitly set to false
    if schema.get("type") == "object":
      schema["additionalProperties"] = False
    else:
      schema.pop("additionalProperties", None)
    schema.pop("propertyOrdering", None)
    schema.pop("maximum", None)
    schema.pop("minimum", None)
    schema.pop("minItems", None)
    schema.pop("maxItems", None)
    for value in schema.values():
      _clean_schema_for_anthropic(value)
  elif isinstance(schema, list):
    for item in schema:
      _clean_schema_for_anthropic(item)


def build_anthropic_message_params(prompt: str,
                                   structure: dict | None,
                                   model: str,
                                   reasoning,
                                   tools,
                                   stream: bool = True) -> dict:
  """
  Build the parameters for an Anthropic Messages API call.
  Used by both the sync hook and batch submission.
  """
  import copy

  # Get the model's max tokens
  if "claude-sonnet-4-5" in model:
    max_tokens = 64000
  elif "claude-opus-4-5" in model:
    max_tokens = 64000
  else:
    max_tokens = 6400000

  betas = []
  if tools:
    betas.append("code-execution-2025-08-25")
  if structure:
    betas.append("structured-outputs-2025-11-13")

  content_blocks = _build_anthropic_message_content(prompt)

  # Build message parameters
  message_params = {
    "model": model,
    "max_tokens": max_tokens,
    "messages": [{
      "role": "user",
      "content": content_blocks
    }]
  }

  if stream:
    message_params["stream"] = True

  if len(betas) > 0:
    message_params["betas"] = betas

  # Add tools if specified
  if tools is True:
    # Enable all built-in tools
    message_params["tools"] = [{
      "type": "web_search_20250305",
      "name": "web_search"
    }, {
      "type": "code_execution_20250825",
      "name": "code_execution"
    }]
  elif tools and tools is not False:
    # Custom tools provided
    message_params["tools"] = tools

  # Handle structured output
  if structure is not None:
    # Deep copy to avoid modifying the original schema
    schema_copy = copy.deepcopy(structure)
    _clean_schema_for_anthropic(schema_copy)
    message_params["output_config"] = {"format": {"type": "json_schema", "schema": schema_copy}}

  # Add thinking configuration if enabled (for supported models)
  if reasoning:
    message_params["thinking"] = {"type": "enabled", "budget_tokens": 32768 * reasoning // 10}

  return message_params


def _claude_ai_hook(prompt: str, structure: dict | None, model: str, reasoning, tools,
                    prompt_caching: bool, timeout_override: int | None = None) -> tuple:
  """
    This function is called by the test runner to get the AI's response to a prompt.
    
    Prompt is the question to ask the AI.
    Structure contains the JSON schema for the expected output. If it is None, the output is just a string.
    
    There is no memory between calls to this function, the 'conversation' doesn't persist.
    """
  from anthropic import Anthropic

  # Initialize the client - it will automatically use ANTHROPIC_API_KEY environment variable
  client = Anthropic(timeout=timeout_override or 3600)

  try:
    # Build request parameters using shared helper
    message_params = build_anthropic_message_params(prompt,
                                                    structure,
                                                    model,
                                                    reasoning,
                                                    tools,
                                                    stream=True)

    # Handle prompt caching if enabled
    if prompt_caching:
      # Mark content for caching - last content block is typically cached
      message_params["messages"][0]["content"][-1]["cache_control"] = {"type": "ephemeral"}

    # Make the API call
    responseStream = client.beta.messages.create(**message_params)

    chainOfThought = ""
    textOutput = ""
    thinkingBuffer = ""

    for content_block in responseStream:
      if content_block.type == "content_block_delta":
        if hasattr(content_block.delta, "thinking"):
          chainOfThought += content_block.delta.thinking
          thinkingBuffer += content_block.delta.thinking
          # Print complete lines from the buffer
          while '\n' in thinkingBuffer:
            line, thinkingBuffer = thinkingBuffer.split('\n', 1)
            print("Thinking: " + line)
        elif hasattr(content_block.delta, "text"):
          textOutput += content_block.delta.text

    #print(textOutput)

    if structure is not None:
      try:
        return json.loads(textOutput), chainOfThought
      except Exception as e:
        import json_repair
        try:
          repaired = json_repair.repair_json(textOutput)
          return json.loads(repaired), chainOfThought
        except:
          print(f"Warning: Failed to parse JSON response: {e}")
          print(f"Raw output was: {textOutput[:500]}")
          return {}, f"Warning: Failed to parse JSON response: {e}" + chainOfThought
    else:
      return textOutput, chainOfThought

  except Exception as e:
    print(f"Error calling Claude API: {e}")

    # Check for content policy violation
    from .ContentViolationHandler import is_content_violation_anthropic
    if is_content_violation_anthropic(e):
      print("CONTENT VIOLATION DETECTED (Anthropic)")
      if structure is not None:
        return {"__content_violation__": True, "reason": str(e)}, f"Content violation: {e}"
      else:
        return "__content_violation__", f"Content violation: {e}"

    # Return appropriate empty response based on structure
    if structure is not None:
      return {}, str(e)
    else:
      return "", str(e)


def submit_batch(config: dict, requests: list) -> str | None:
  """
  Submit a batch of requests to Anthropic's Message Batches API.
  
  Args:
    config: Model configuration dict with base_model, reasoning, tools
    requests: List of BatchRequest objects
    
  Returns:
    Batch ID if successful, None otherwise
  """
  from anthropic import Anthropic

  client = Anthropic()
  model = config.get("base_model", "claude-sonnet-4-5")
  reasoning = config.get("reasoning", False)
  tools = config.get("tools", False)

  # Build batch requests using the shared helper
  batch_requests = []
  batch_betas = set()  # Collect betas for batch-level param

  for req in requests:
    # Use the shared helper to build params (includes tools, reasoning, structure, schema cleaning)
    params = build_anthropic_message_params(req.prompt,
                                            req.structure,
                                            model,
                                            reasoning,
                                            tools,
                                            stream=False)
    # Extract betas from params - must be passed at batch level, not per-request
    if "betas" in params:
      batch_betas.update(params.pop("betas"))
    batch_requests.append({"custom_id": req.custom_id, "params": params})

  # Create batch - pass betas at batch level if any are needed
  if batch_betas:
    batch = client.beta.messages.batches.create(betas=list(batch_betas), requests=batch_requests)
  else:
    batch = client.messages.batches.create(requests=batch_requests)
  return batch.id


def poll_batch(batch_id: str, requests: list) -> tuple:
  """
  Poll an Anthropic batch for status and results.
  
  Args:
    batch_id: The batch ID to poll
    requests: List of original BatchRequest objects (for parsing results)
    
  Returns:
    Tuple of (status_string, list of result dicts)
    status_string is one of: "completed", "failed", "processing"
  """
  from anthropic import Anthropic
  import json

  client = Anthropic()
  batch_status = client.messages.batches.retrieve(batch_id)

  results = []

  if batch_status.processing_status == "ended":
    # Download results
    req_map = {r.custom_id: r for r in requests}

    for result in client.messages.batches.results(batch_id):
      custom_id = result.custom_id

      if result.result.type == "succeeded":
        message = result.result.message
        text_output = ""
        cot = ""

        for block in message.content:
          if block.type == "text":
            text_output += block.text
          elif block.type == "thinking":
            cot += block.thinking + "\n"

        # Parse JSON if structured
        result_data = text_output
        req = req_map.get(custom_id)
        if req and req.structure:
          try:
            json_text = text_output.strip()
            if json_text.startswith("```json"):
              json_text = json_text[7:]
            if json_text.startswith("```"):
              json_text = json_text[3:]
            if json_text.endswith("```"):
              json_text = json_text[:-3]
            result_data = json.loads(json_text.strip())
          except:
            pass

        results.append({
          "custom_id": custom_id,
          "success": True,
          "result": result_data,
          "chain_of_thought": cot.strip(),
          "error": None
        })
      else:
        results.append({
          "custom_id": custom_id,
          "success": False,
          "result": None,
          "chain_of_thought": "",
          "error": str(result.result)
        })

    return "completed", results

  elif batch_status.processing_status == "canceling":
    return "failed", results

  else:
    # Still processing
    counts = batch_status.request_counts
    print(
      f"[Batch] Anthropic batch {batch_id}: {batch_status.processing_status} "
      f"({counts.succeeded + counts.errored}/{counts.processing + counts.succeeded + counts.errored})"
    )
    return "processing", results


if __name__ == "__main__":
  engine = ClaudeEngine("claude-sonnet-4-5-20250929", False, False)
  print(engine.AIHook("What's the 7th prime number after 101?", None))

  engine = ClaudeEngine("claude-sonnet-4-5-20250929", True, True)
  print(engine.AIHook("What is the closest Australian city to New York?", None))

  print(
    engine.AIHook(
      "What is the furtherest Australian city from New York?", {
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
        "required": ["cityName", "longitude", "latitude"],
        "additionalProperties": False
      }))
