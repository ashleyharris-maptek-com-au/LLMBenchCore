"""
OpenAI ChatGPT AI Engine for LLMBenchCore

This module provides an interface to the OpenAI API using the Responses API.

Setup:
1. Install the SDK: pip install openai
2. Set your API key as an environment variable:
   - Windows: set OPENAI_API_KEY=your_api_key_here
   - Linux/Mac: export OPENAI_API_KEY=your_api_key_here
   
Get your API key from: https://platform.openai.com/api-keys

The SDK documentation can be found at: https://platform.openai.com/docs
Responses API reference: https://platform.openai.com/docs/api-reference/responses
"""

import hashlib
import os
import json
import random
import time
from . import PromptImageTagging as pit


class OpenAIEngine:
  """
  OpenAI ChatGPT AI Engine class.
  
  Configuration parameters:
  - model: Model name (e.g., "gpt-5-nano")
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
    self.forcedFailure = False
    self.configAndSettingsHash = hashlib.sha256(model.encode() + str(reasoning).encode() + str(tools).encode() + str(timeout).encode()).hexdigest()

  def AIHook(self, prompt: str, structure: dict | None) -> tuple:
    """Call the OpenAI API with instance configuration."""
    result = _openai_ai_hook(prompt, structure, self.model, self.reasoning, self.tools, self, timeout_override=self.timeout)
    return result


def build_openai_input(prompt: str):
  prompt_parts = pit.parse_prompt_parts(prompt)
  has_images = any(part_type == "image" for part_type, _ in prompt_parts)
  if not has_images:
    return prompt

  content: list[dict] = []
  for part_type, part_value in prompt_parts:
    if part_type == "text":
      if part_value:
        content.append({"type": "input_text", "text": part_value})
    elif part_type == "image":
      if pit.is_url(part_value) or pit.is_data_uri(part_value):
        image_url = part_value
      else:
        image_url = pit.file_to_data_uri(pit.resolve_local_path(part_value))
      content.append({"type": "input_image", "image_url": image_url, "detail": "high"})

  return [{"role": "user", "content": content}]


def build_openai_response_params(prompt: str,
                                 structure: dict | None,
                                 model: str,
                                 reasoning,
                                 tools,
                                 flex_banned=True) -> dict:
  """
  Build the parameters for an OpenAI Responses API call.
  Used by both the sync hook and batch submission.
  """
  # Determine model to use
  model_to_use = model

  # Override model if reasoning specifies an o1 model
  if isinstance(reasoning, str) and reasoning in ["o1-preview", "o1-mini"]:
    model_to_use = reasoning

  # Build Responses API parameters
  input_value = build_openai_input(prompt)

  response_params = {"model": model_to_use, "input": input_value, "service_tier": "flex"}

  if flex_banned or "5.2-pro" in model_to_use:
    # Flex isn't supported by 5.2-pro
    del response_params["service_tier"]

  # Add reasoning effort
  if isinstance(reasoning, int) and reasoning > 0:
    # Map 1-10 scale to low/medium/high
    if reasoning <= 3:
      response_params["reasoning"] = {"effort": "low"}
    elif reasoning <= 7:
      response_params["reasoning"] = {"effort": "medium"}
    elif reasoning == 10 and "gpt-5.2" in model_to_use:
      response_params["reasoning"] = {"effort": "xhigh"}
    else:
      response_params["reasoning"] = {"effort": "high"}

    response_params["reasoning"]["summary"] = "auto"

  # Handle structured output using the text.format parameter
  if structure is not None:
    response_params["text"] = {
      "format": {
        "type": "json_schema",
        "name": "structured_response",
        "schema": structure,
        "strict": True
      }
    }

  # Add tools if specified
  if tools is True:
    # Enable built-in hosted tools
    # Note: file_search requires a vector_store, so it's excluded
    response_params["tools"] = [{"type": "web_search"}]

    # 5.2 pro doesn't support code execution.
    if "5.2-pro" not in model_to_use:
      response_params["tools"].append({"type": "code_interpreter", "container": {"type": "auto"}})

  elif tools and tools is not False:
    # Convert function list to OpenAI tool format if needed
    tools_list = []
    for tool in (tools if isinstance(tools, list) else [tools]):
      if isinstance(tool, dict):
        # Already in correct format
        tools_list.append(tool)
      elif callable(tool):
        # Convert Python function to tool definition
        import inspect
        sig = inspect.signature(tool)
        doc = inspect.getdoc(tool) or "No description"

        properties = {}
        required = []
        for param_name, param in sig.parameters.items():
          param_type = "string"  # Default type
          if param.annotation != inspect.Parameter.empty:
            if param.annotation == int:
              param_type = "integer"
            elif param.annotation == float:
              param_type = "number"
            elif param.annotation == bool:
              param_type = "boolean"

          properties[param_name] = {"type": param_type}
          if param.default == inspect.Parameter.empty:
            required.append(param_name)

        tools_list.append({
          "type": "function",
          "function": {
            "name": tool.__name__,
            "description": doc,
            "parameters": {
              "type": "object",
              "properties": properties,
              "required": required
            }
          }
        })

    if tools_list:
      response_params["tools"] = tools_list

  return response_params


def _openai_ai_hook(prompt: str, structure: dict | None, model: str, reasoning,
                    tools, engine_instance, timeout_override: int | None = None) -> tuple:
  """
    This function is called by the test runner to get the AI's response to a prompt.
    
    Prompt is the question to ask the AI.
    Structure contains the JSON schema for the expected output. If it is None, the output is just a string.
    
    There is no memory between calls to this function, the 'conversation' doesn't persist.
    
    Uses the OpenAI Responses API.
    """
  if engine_instance.forcedFailure:
    return {"error": "Forced failure"}, "Forced failure due to API instability"
  from openai import OpenAI

  try:
    # Initialize the client - it will automatically use OPENAI_API_KEY environment variable
    timeout = timeout_override or 3600
    client = OpenAI(timeout=timeout)

    # Build request parameters using shared helper
    response_params = build_openai_response_params(prompt, structure, model, reasoning, tools)

    # Make the API call using Responses API with streaming
    stream = client.responses.create(stream=True, timeout=timeout, **response_params)

    chainOfThought = ""
    output_text = ""
    current_reasoning_line = ""

    # Process streaming events
    for event in stream:
      event_type = event.type

      # Handle reasoning summary deltas - print line by line as they arrive
      if event_type == "response.reasoning_summary_text.delta":
        delta = event.delta
        current_reasoning_line += delta
        # Print complete lines as they arrive
        while "\n" in current_reasoning_line:
          line, current_reasoning_line = current_reasoning_line.split("\n", 1)
          print(f"Thinking: {line}", flush=True)
          chainOfThought += line + "\n"

      # Handle reasoning summary done - flush any remaining text
      elif event_type == "response.reasoning_summary_text.done":
        if current_reasoning_line:
          print(f"Thinking: {current_reasoning_line}", flush=True)
          chainOfThought += current_reasoning_line
          current_reasoning_line = ""

      # Handle output text deltas - accumulate silently
      elif event_type == "response.output_text.delta":
        output_text += event.delta

      # Handle completion
      elif event_type == "response.completed":
        # Final response is available if needed
        pass

    # Strip trailing newline from chain of thought if present
    chainOfThought = chainOfThought.rstrip("\n")

    #print(output_text)

    # Extract content
    if structure is not None:
      # Parse JSON response
      if output_text:
        return json.loads(output_text), chainOfThought
      return {}, chainOfThought
    else:
      # Return text response
      return output_text or "", chainOfThought

  except json.JSONDecodeError:
    print(
      "Error decoding JSON response. OpenAI has schema validation that's failing. Consider the whole service down when this is encountered."
    )
    engine_instance.forcedFailure = True
    return {"unacceptableFailure": True}, ""  # to ensure we don't retry.
  except Exception as e:
    print(f"Error calling OpenAI API: {e}")

    # Check for content policy violation
    from .ContentViolationHandler import is_content_violation_openai
    if is_content_violation_openai(e):
      print("CONTENT VIOLATION DETECTED (OpenAI)")
      if structure is not None:
        return {"__content_violation__": True, "reason": str(e)}, f"Content violation: {e}"
      else:
        return "__content_violation__", f"Content violation: {e}"

    if "You exceeded your current quota," in str(e):
      print("QUOTA EXCEEDED. Waiting 15 minutes to an hour.")
      time.sleep(random.randint(900, 3600))
      return "", ""

    # Return appropriate empty response based on structure
    if structure is not None:
      return {}, ""
    else:
      return "", ""


def submit_batch(config: dict, requests: list, timeout_override: int | None = None) -> str | None:
  """
  Submit a batch of requests to OpenAI's Batch API.
  
  Args:
    config: Model configuration dict with base_model, reasoning, tools
    requests: List of BatchRequest objects
    
  Returns:
    Batch ID if successful, None otherwise
  """
  from openai import OpenAI
  import tempfile

  client = OpenAI(timeout=timeout_override or 3600)
  model = config.get("base_model", "gpt-4o")
  reasoning = config.get("reasoning", False)
  tools = config.get("tools", False)

  # Build JSONL content using the shared helper
  jsonl_lines = []
  for req in requests:
    # Use the shared helper to build request params (includes tools, reasoning, structure)
    body = build_openai_response_params(req.prompt,
                                        req.structure,
                                        model,
                                        reasoning,
                                        tools,
                                        flex_banned=True)

    line = {"custom_id": req.custom_id, "method": "POST", "url": "/v1/responses", "body": body}
    jsonl_lines.append(json.dumps(line))

  # Write to temp file
  batch_file_path = os.path.join(tempfile.gettempdir(), f"batch_openai_{int(time.time())}.jsonl")
  with open(batch_file_path, "w", encoding="utf-8") as f:
    f.write("\n".join(jsonl_lines))

  # Upload file
  with open(batch_file_path, "rb") as f:
    file_obj = client.files.create(file=f, purpose="batch")

  # Create batch
  batch = client.batches.create(
    input_file_id=file_obj.id,
    endpoint="/v1/responses",
    completion_window="24h",
    metadata={"description": f"LLMBenchCore batch for {config['name']}"})

  return batch.id


def poll_batch(batch_id: str, requests: list, timeout_override: int | None = None) -> tuple:
  """
  Poll an OpenAI batch for status and results.
  
  Args:
    batch_id: The batch ID to poll
    requests: List of original BatchRequest objects (for parsing results)
    
  Returns:
    Tuple of (status_string, list of result dicts)
    status_string is one of: "completed", "failed", "processing"
    result dicts have: custom_id, success, result, chain_of_thought, error
  """
  from openai import OpenAI

  client = OpenAI(timeout=timeout_override or 3600)
  batch_status = client.batches.retrieve(batch_id)

  results = []

  if batch_status.status == "completed":
    # Check for error file first
    error_file_id = batch_status.error_file_id
    if error_file_id:
      error_content = client.files.content(error_file_id)
      print(f"[Batch] OpenAI error file contents:\n{error_content.text[:2000]}")

    # Download results
    output_file_id = batch_status.output_file_id
    if output_file_id:
      content = client.files.content(output_file_id)
      lines = content.text.strip().split("\n")

      # Build request lookup
      req_map = {r.custom_id: r for r in requests}

      print(f"[Batch] OpenAI: Processing {len(lines)} result lines")
      # Debug: show first line structure
      if lines and lines[0].strip():
        first_obj = json.loads(lines[0])
        print(f"[Batch] OpenAI: First result structure: {list(first_obj.keys())}")
        if "response" in first_obj:
          print(f"[Batch] OpenAI: response keys: {list(first_obj['response'].keys())}")
          if "body" in first_obj["response"]:
            print(f"[Batch] OpenAI: body keys: {list(first_obj['response']['body'].keys())}")
      for line in lines:
        if not line.strip():
          continue
        result_obj = json.loads(line)
        custom_id = result_obj.get("custom_id")
        response = result_obj.get("response", {})
        body = response.get("body", {})

        # Check for errors in the response
        if result_obj.get("error"):
          print(f"[Batch] OpenAI error for {custom_id}: {result_obj.get('error')}")
        if response.get("status_code") and response.get("status_code") != 200:
          print(f"[Batch] OpenAI non-200 for {custom_id}: {response.get('status_code')}")

        # Extract text output
        # Note: The Responses API returns content with type "text" (not "output_text")
        text_output = ""
        cot = ""
        if "output" in body:
          for item in body.get("output", []):
            if item.get("type") == "message":
              for content_item in item.get("content", []):
                # Content type is "text" in the response object (streaming uses "output_text" events)
                if content_item.get("type") in ("text", "output_text"):
                  text_output = content_item.get("text", "")
            elif item.get("type") == "reasoning":
              for summary in item.get("summary", []):
                if summary.get("type") == "summary_text":
                  cot += summary.get("text", "") + "\n"

        # Log if text_output is empty (for debugging)
        if not text_output:
          print(f"[Batch] OpenAI: Empty text_output for {custom_id}")
          # Debug: show what content types we got
          if "output" in body:
            for item in body.get("output", []):
              if item.get("type") == "message":
                types = [c.get("type") for c in item.get("content", [])]
                print(f"[Batch] OpenAI: content types for {custom_id}: {types}")

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

    return "completed", results

  elif batch_status.status in ["failed", "cancelled", "expired"]:
    return "failed", results

  else:
    # Still processing
    completed = batch_status.request_counts.completed if batch_status.request_counts else 0
    total = batch_status.request_counts.total if batch_status.request_counts else 0
    print(f"[Batch] OpenAI batch {batch_id}: {batch_status.status} ({completed}/{total})")
    return "processing", results
