"""
xAI Grok AI Engine for LLMBenchCore

This module provides an interface to the xAI Grok API using the official xai-sdk.

Setup:
1. Install the SDK: pip install xai-sdk
2. Set your API key as an environment variable:
   - Windows: set XAI_API_KEY=your_api_key_here
   - Linux/Mac: export XAI_API_KEY=your_api_key_here
   
Get your API key from: https://console.x.ai/

The SDK documentation can be found at: https://docs.x.ai/
"""

import hashlib
import os
import json
import random
import time
from . import PromptImageTagging as pit
from typing import Any, List, Optional
from pydantic import BaseModel, create_model


class GrokEngine:
  """
  xAI Grok AI Engine class.
  
  Configuration parameters:
  - model: Model name (e.g., "grok-3")
  - reasoning: Reasoning mode:
      - False or 0: No special reasoning (standard mode)
      - "o1-preview": Use o1-preview model with extended reasoning
      - "o1-mini": Use o1-mini model (faster reasoning)
      - Integer (1-10): Reasoning effort level (for o1 models)
  - tools: Tool capabilities:
      - False: No tools available
      - True: Enable ALL built-in tools (web_search, code_interpreter)
      - List of function definitions: Enable specific custom tools
  - timeout: Request timeout in seconds
  """

  def __init__(self, model: str, reasoning=False, tools=False, timeout: int = 3600):
    self.model = model
    self.reasoning = reasoning
    self.tools = tools
    self.timeout = timeout
    self.configAndSettingsHash = hashlib.sha256(model.encode() + str(reasoning).encode() + str(tools).encode() + str(timeout).encode()).hexdigest()

  def AIHook(self, prompt: str, structure: dict | None) -> tuple:
    """Call the Grok API with instance configuration."""
    return _grok_ai_hook(prompt, structure, self.model, self.reasoning, self.tools, timeout_override=self.timeout)


def json_schema_to_pydantic(schema: dict, name: str = "DynamicModel") -> type[BaseModel]:
  """
    Convert a JSON schema dict to a Pydantic model class.
    Supports basic types: string, number, integer, boolean, array, object.
    """

  def get_python_type(prop_schema: dict) -> Any:
    """Convert JSON schema type to Python/Pydantic type."""
    json_type = prop_schema.get("type", "string")

    if json_type == "string":
      return str
    elif json_type == "number":
      return float
    elif json_type == "integer":
      return int
    elif json_type == "boolean":
      return bool
    elif json_type == "array":
      items_schema = prop_schema.get("items", {})
      item_type = get_python_type(items_schema)
      return List[item_type]
    elif json_type == "object":
      # Nested object - create a nested model
      nested_props = prop_schema.get("properties", {})
      if nested_props:
        return json_schema_to_pydantic(prop_schema, name + "Nested")
      return dict
    else:
      return Any

  properties = schema.get("properties", {})
  required = set(schema.get("required", []))

  # Build field definitions for create_model
  field_definitions = {}
  for prop_name, prop_schema in properties.items():
    python_type = get_python_type(prop_schema)
    if prop_name in required:
      field_definitions[prop_name] = (python_type, ...)
    else:
      field_definitions[prop_name] = (Optional[python_type], None)

  return create_model(name, **field_definitions)


def _build_xai_user_args(prompt: str, structure: dict | None) -> list[Any]:
  from xai_sdk.chat import image
  from PIL import Image
  import io
  import base64

  prompt_parts = pit.parse_prompt_parts(prompt)
  user_args: list[Any] = []
  for part_type, part_value in prompt_parts:
    if part_type == "text":
      if part_value:
        user_args.append(part_value)
    elif part_type == "image":
      if pit.is_url(part_value):
        user_args.append(image(part_value))
      elif pit.is_data_uri(part_value):
        # Keep full data URI format - xAI SDK requires it
        user_args.append(image(part_value))
      else:
        # Convert local file to data URI format, resizing if needed
        local_path = pit.resolve_local_path(part_value)
        # Resize if over 8000 pixels on any side (xAI has stricter limits)
        img = Image.open(local_path)
        max_dim = 8000
        if img.width > max_dim or img.height > max_dim:
          scale = min(max_dim / img.width, max_dim / img.height)
          new_size = (int(img.width * scale), int(img.height * scale))
          print(
            f"Resizing image from {img.width}x{img.height} to {new_size[0]}x{new_size[1]} for xAI")
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
          data_uri = f"data:{mime_type};base64,{b64}"
        else:
          data_uri = pit.file_to_data_uri(local_path)
        user_args.append(image(data_uri))

  if structure is not None:
    schema_json = json.dumps(structure, indent=2)
    user_args.append(f"""

You MUST respond with valid JSON that matches this exact schema:
{schema_json}

Return ONLY the JSON object, no markdown formatting, no code blocks, no explanation.""")

  if not user_args:
    user_args = [""]

  return user_args


def build_xai_chat_params(model: str, tools) -> dict:
  """
  Build the chat creation parameters for xAI.
  Used by both the sync hook and batch submission.
  """
  chat_params = {"model": model}

  # Add tools if specified
  if tools is True:
    from xai_sdk.tools import code_execution as xai_code_execution
    from xai_sdk.tools import web_search as xai_web_search
    from xai_sdk.tools import x_search as xai_x_search
    chat_params["tools"] = [
      xai_web_search(),
      xai_x_search(),
      xai_code_execution(),
    ]
  elif tools and tools is not False:
    if isinstance(tools, list):
      chat_params["tools"] = tools

  return chat_params


def _grok_ai_hook(prompt: str, structure: dict | None, model: str, reasoning, tools, timeout_override: int | None = None) -> tuple:
  """
    This function is called by the test runner to get the AI's response to a prompt.
    
    Prompt is the question to ask the AI.
    Structure contains the JSON schema for the expected output. If it is None, the output is just a string.
    
    There is no memory between calls to this function, the 'conversation' doesn't persist.
    
    Returns tuple of (result, chainOfThought).
    """
  from xai_sdk import Client
  from xai_sdk.chat import user

  try:
    # Initialize the client - uses XAI_API_KEY environment variable
    client = Client(timeout=timeout_override or 3600)

    # Build chat creation parameters using shared helper
    chat_params = build_xai_chat_params(model, tools)

    # Convert JSON schema to Pydantic model if provided
    pydantic_model = None
    if structure is not None:
      try:
        pydantic_model = json_schema_to_pydantic(structure, "ResponseModel")
      except Exception as e:
        print(f"Failed to convert schema to Pydantic: {e}")
        pydantic_model = None

    # Create chat and add user message
    chat = client.chat.create(**chat_params)

    user_args = _build_xai_user_args(prompt, structure if pydantic_model is not None else None)
    chat.append(user(*user_args))

    # Stream response and accumulate (works for both structured and unstructured)
    chainOfThought = ""
    output_text = ""
    current_thinking_line = ""

    for response, chunk in chat.stream():
      # Check if this chunk contains reasoning/thinking content
      if hasattr(chunk, 'reasoning_content') and chunk.reasoning_content:
        current_thinking_line += chunk.reasoning_content
        while "\n" in current_thinking_line:
          line, current_thinking_line = current_thinking_line.split("\n", 1)
          print(f"Thinking: {line}", flush=True)
          chainOfThought += line + "\n"

      # Regular content
      if hasattr(chunk, 'content') and chunk.content:
        output_text += chunk.content

    # Flush any remaining thinking content
    if current_thinking_line:
      print(f"Thinking: {current_thinking_line}", flush=True)
      chainOfThought += current_thinking_line

    chainOfThought = chainOfThought.rstrip("\n")

    # Also check final response for reasoning content if not captured during streaming
    if not chainOfThought and hasattr(response, 'reasoning_content') and response.reasoning_content:
      chainOfThought = response.reasoning_content
      for line in chainOfThought.split("\n"):
        print(f"Thinking: {line}", flush=True)

    # Get final content from response if streaming didn't capture it
    if not output_text and hasattr(response, 'content'):
      output_text = response.content or ""

    if chainOfThought:
      print()  # Blank line after thinking

    # Parse structured output if we have a Pydantic model
    if pydantic_model is not None:
      try:
        # Strip markdown code blocks if present
        parse_text = output_text
        if "```json" in parse_text:
          parse_text = parse_text.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in parse_text:
          parse_text = parse_text.split("```", 1)[1].split("```", 1)[0]

        parse_text = parse_text.strip()

        # Parse and validate with Pydantic
        parsed_obj = pydantic_model.model_validate_json(parse_text)
        result_dict = parsed_obj.model_dump()
        return result_dict, chainOfThought
      except Exception as e:
        try:
          import json_repair
          repaired = json_repair.repair_json(parse_text)
          parsed_obj = pydantic_model.model_validate_json(repaired)
          result_dict = parsed_obj.model_dump()
          return result_dict, chainOfThought
        except:
          print(f"Structured parse failed: {e}")
          print("Model returned:\n" + output_text)
          return {}, f"Structured parse failed: {e}"
    else:
      # Non-structured output - just return the text
      return output_text or "", chainOfThought

  except Exception as e:
    print(f"Error calling xAI Grok API: {e}")

    # Check for content policy violation
    from .ContentViolationHandler import is_content_violation_xai
    if is_content_violation_xai(e):
      print("CONTENT VIOLATION DETECTED (xAI Grok)")
      if structure is not None:
        return {"__content_violation__": True, "reason": str(e)}, f"Content violation: {e}"
      else:
        return "__content_violation__", f"Content violation: {e}"

    if "rate limit" in str(e) or "StatusCode.DEADLINE_EXCEEDED" in str(
        e) or "StatusCode.DEADLINE_EXCEEDED" in str(e):
      print("Hit rate limit - pausing for a random time between 5 and 30 minutes.")
      time.sleep(random.randint(300, 1800))

    return None


def _xai_rest_headers() -> dict:
  """Return Authorization + Content-Type headers for xAI REST API."""
  api_key = os.environ.get("XAI_API_KEY", "")
  return {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
  }


_XAI_BASE = "https://api.x.ai/v1"


def _build_rest_user_content(prompt: str, structure: dict | None) -> str | list:
  """
  Convert a prompt (possibly with [[image:...]] tags) into OpenAI-compatible
  message content for the xAI REST API.
  Returns a plain string for text-only prompts, or a list of content parts
  for multimodal prompts.
  """
  from PIL import Image
  import io
  import base64

  prompt_parts = pit.parse_prompt_parts(prompt)

  # Append structured-output instruction if needed
  if structure is not None:
    schema_json = json.dumps(structure, indent=2)
    prompt_parts.append(("text", f"""

You MUST respond with valid JSON that matches this exact schema:
{schema_json}

Return ONLY the JSON object, no markdown formatting, no code blocks, no explanation."""))

  has_images = any(t == "image" for t, _ in prompt_parts)
  if not has_images:
    # Pure text – just concatenate
    return "".join(v for _, v in prompt_parts)

  # Multimodal: build content-parts list
  content_parts: list[dict] = []
  for part_type, part_value in prompt_parts:
    if part_type == "text":
      if part_value:
        content_parts.append({"type": "text", "text": part_value})
    elif part_type == "image":
      if pit.is_url(part_value):
        url = part_value
      elif pit.is_data_uri(part_value):
        url = part_value
      else:
        local_path = pit.resolve_local_path(part_value)
        img = Image.open(local_path)
        max_dim = 8000
        if img.width > max_dim or img.height > max_dim:
          scale = min(max_dim / img.width, max_dim / img.height)
          new_size = (int(img.width * scale), int(img.height * scale))
          img = img.resize(new_size, Image.LANCZOS)
          buffer = io.BytesIO()
          fmt = img.format or 'PNG'
          if fmt.upper() == 'JPEG':
            img.save(buffer, format='JPEG', quality=90)
            mime_type = 'image/jpeg'
          else:
            img.save(buffer, format='PNG')
            mime_type = 'image/png'
          b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
          url = f"data:{mime_type};base64,{b64}"
        else:
          url = pit.file_to_data_uri(local_path)
      content_parts.append({
        "type": "image_url",
        "image_url": {"url": url}
      })

  return content_parts


def submit_batch(config: dict, requests: list) -> str | None:
  """
  Submit a batch of requests to xAI's Batch API via REST.

  The xAI Python SDK's gRPC batch endpoint is unreliable (returns
  "Stream removed" from Cloudflare), so we use the REST API directly.

  Args:
    config: Model configuration dict with base_model, reasoning, tools
    requests: List of BatchRequest objects

  Returns:
    Batch ID if successful, None otherwise
  """
  import requests as http

  headers = _xai_rest_headers()
  model = config.get("base_model", "grok-3")
  tools_cfg = config.get("tools", False)

  # Step 1: Create batch
  print(f"[Batch-xAI] Creating batch (model={model})...")
  resp = http.post(f"{_XAI_BASE}/batches", headers=headers,
                   json={"name": f"LLMBenchCore_{config.get('name', 'batch')}"})
  resp.raise_for_status()
  batch_data = resp.json()
  batch_id = batch_data["batch_id"]
  print(f"[Batch-xAI] Batch created: {batch_id}")

  # Step 2: Build and add requests in chunks
  # xAI allows up to 100 add-batch-requests calls per 30s per team.
  CHUNK_SIZE = 20
  for chunk_start in range(0, len(requests), CHUNK_SIZE):
    chunk = requests[chunk_start:chunk_start + CHUNK_SIZE]
    chunk_num = chunk_start // CHUNK_SIZE + 1
    total_chunks = (len(requests) + CHUNK_SIZE - 1) // CHUNK_SIZE

    batch_reqs = []
    for req in chunk:
      user_content = _build_rest_user_content(req.prompt, req.structure)
      messages = [{"role": "user", "content": user_content}]

      completion_body: dict = {"messages": messages, "model": model}

      # Add tools if configured
      if tools_cfg is True:
        completion_body["tools"] = [
          {"type": "function", "function": {"name": "web_search"}},
          {"type": "function", "function": {"name": "code_execution"}},
        ]

      batch_reqs.append({
        "batch_request_id": req.custom_id,
        "batch_request": {
          "chat_get_completion": completion_body
        }
      })

    print(f"[Batch-xAI] Adding chunk {chunk_num}/{total_chunks} ({len(chunk)} requests)...")
    resp = http.post(f"{_XAI_BASE}/batches/{batch_id}/requests",
                     headers=headers,
                     json={"batch_requests": batch_reqs})
    resp.raise_for_status()
    print(f"[Batch-xAI] Chunk {chunk_num} added OK")

  return batch_id


def poll_batch(batch_id: str, requests: list) -> tuple:
  """
  Poll an xAI batch for status and results via REST.
  
  Args:
    batch_id: The batch ID to poll
    requests: List of original BatchRequest objects (for parsing results)
    
  Returns:
    Tuple of (status_string, list of result dicts)
    status_string is one of: "completed", "failed", "processing"
  """
  import requests as http

  headers = _xai_rest_headers()

  # Get batch status
  resp = http.get(f"{_XAI_BASE}/batches/{batch_id}", headers=headers)
  resp.raise_for_status()
  batch_data = resp.json()

  state = batch_data.get("state", {})
  num_pending = state.get("num_pending", 0)
  num_success = state.get("num_success", 0)
  num_error = state.get("num_error", 0)
  num_requests = state.get("num_requests", 0)

  results = []

  if num_pending == 0 and (num_success > 0 or num_error > 0):
    # Retrieve all results with pagination
    req_map = {r.custom_id: r for r in requests}
    pagination_token = None

    while True:
      params: dict = {"page_size": 100}
      if pagination_token:
        params["pagination_token"] = pagination_token

      resp = http.get(f"{_XAI_BASE}/batches/{batch_id}/results",
                      headers=headers, params=params)
      resp.raise_for_status()
      page = resp.json()

      # REST API returns: {"results": [...], "pagination_token": ...}
      # Each result: {"batch_request_id": "...", "batch_result": {"response": {"chat_get_completion": {...}}}}
      for result in page.get("results", []):
        custom_id = result.get("batch_request_id", "")
        batch_result = result.get("batch_result", {})

        # Check for error results
        error = batch_result.get("error")
        if error:
          results.append({
            "custom_id": custom_id,
            "success": False,
            "result": None,
            "chain_of_thought": "",
            "error": str(error)
          })
          continue

        # Extract content from nested response structure
        chat_completion = batch_result.get("response", {}).get("chat_get_completion", {})
        choices = chat_completion.get("choices", [])
        message = choices[0].get("message", {}) if choices else {}

        text_content = message.get("content", "") or ""
        cot = message.get("reasoning_content", "") or ""

        # Parse JSON if structured
        result_data = text_content
        req = req_map.get(custom_id)
        if req and req.structure:
          try:
            json_text = text_content.strip()
            if "```json" in json_text:
              json_text = json_text.split("```json", 1)[1].split("```", 1)[0]
            elif "```" in json_text:
              json_text = json_text.split("```", 1)[1].split("```", 1)[0]
            result_data = json.loads(json_text.strip())
          except:
            pass

        results.append({
          "custom_id": custom_id,
          "success": True,
          "result": result_data,
          "chain_of_thought": cot,
          "error": None
        })

      pagination_token = page.get("pagination_token")
      if not pagination_token:
        break

    return "completed", results

  elif num_pending == 0 and num_success == 0 and num_error == 0:
    return "failed", results

  else:
    completed = num_success + num_error
    print(f"[Batch] xAI batch {batch_id}: {completed}/{num_requests} complete, {num_pending} pending")
    return "processing", results
