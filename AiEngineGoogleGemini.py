"""
Google Gemini AI Engine for LLMBenchCore

This module provides an interface to the Google Gemini API using the latest google-genai SDK.

Setup:
1. Install the SDK: pip install google-genai
2. Set your API key as an environment variable:
   - Windows: set GEMINI_API_KEY=your_api_key_here
   - Linux/Mac: export GEMINI_API_KEY=your_api_key_here
   
Get your API key from: https://ai.google.dev/

The SDK documentation can be found at: https://googleapis.github.io/python-genai/
"""

import hashlib
import os
import json
import random
from . import PromptImageTagging as pit
import time
import threading
import queue
from urllib.request import Request, urlopen
from typing import Any, List, Optional
from google import genai
from google.genai import types
from pydantic import BaseModel, create_model

DEFAULT_TIMEOUT_SECONDS = 3600 * 3  # 3 hour timeout


class GeminiEngine:
  """
  Google Gemini AI Engine class.
  
  Configuration parameters:
  - model: Model name (e.g., "gemini-2.5-flash")
  - reasoning: Reasoning effort on a 0-10 scale:
      - 0 or False: Thinking disabled (fastest, no reasoning)
      - 1-3: Low reasoning (maps to ~128-1024 token budget)
      - 4-7: Medium reasoning (maps to ~2048-8192 token budget)
      - 8-10: High reasoning (maps to ~12288-24576 token budget)
  - tools: Tool capabilities:
      - False: No tools available
      - "google_search": Enable Google Search grounding
      - "code_execution": Enable Python code execution
      - List of functions: Enable custom function calling
      - List of strings/functions: Mix built-in and custom tools
  - timeout: Request timeout in seconds
  """

  def __init__(self, model: str, reasoning=False, tools=False, timeout: int = DEFAULT_TIMEOUT_SECONDS):
    self.model = model
    self.reasoning = reasoning
    self.tools = tools
    self.timeout = timeout
    self.configAndSettingsHash = hashlib.sha256(model.encode() + str(reasoning).encode() +
                                                str(tools).encode() + str(timeout).encode()).hexdigest()

  def AIHook(self, prompt: str, structure: dict | None) -> tuple:
    """Call the Gemini API with instance configuration."""
    return _gemini_ai_hook(prompt, structure, self.model, self.reasoning, self.tools,
                          timeout_override=self.timeout)


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


def _build_gemini_contents(prompt: str, structure: dict | None, tools) -> list[Any]:
  prompt_parts = pit.parse_prompt_parts(prompt)
  contents: list[Any] = []
  for part_type, part_value in prompt_parts:
    if part_type == "text":
      if part_value:
        contents.append(part_value)
    elif part_type == "image":
      if pit.is_url(part_value):
        req = Request(part_value, headers={"User-Agent": "LLMBenchCore/1.0"})
        with urlopen(req, timeout=30) as resp:
          content_type = resp.headers.get("Content-Type")
          image_bytes = resp.read()

        mime_type = None
        if content_type:
          mime_type = content_type.split(";", 1)[0].strip().lower()
        if not mime_type:
          mime_type = pit.guess_image_mime_type_from_ref(part_value)

        contents.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))
      elif pit.is_data_uri(part_value):
        mime_type, image_bytes = pit.decode_data_uri(part_value)
        contents.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))
      else:
        local_path = pit.resolve_local_path(part_value)
        mime_type = pit.guess_image_mime_type_from_path(local_path)
        image_bytes = pit.read_file_bytes(local_path)
        contents.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))

  if structure is not None and tools:
    schema_json = json.dumps(structure, indent=2)
    contents.append(f"""

You MUST respond with valid JSON that matches this exact schema:
{schema_json}

Return ONLY the JSON object, no markdown formatting, no code blocks, no explanation.""")

  if not contents:
    contents = [""]

  return contents


def _clean_schema_for_gemini(schema):
  """
  Remove schema properties that Gemini doesn't support.
  Returns a cleaned copy of the schema.
  """
  import copy
  cleaned = copy.deepcopy(schema)

  def remove_additional_properties(s):
    if isinstance(s, dict):
      s.pop("additionalProperties", None)
      for key, value in s.items():
        if isinstance(value, dict):
          remove_additional_properties(value)
        elif isinstance(value, list) and value and isinstance(value[0], dict):
          for item in value:
            if isinstance(item, dict):
              remove_additional_properties(item)
    return s

  return remove_additional_properties(cleaned)


def build_gemini_contents(prompt: str, structure: dict | None, tools) -> list:
  """
  Build the contents list for a Gemini API call.
  Used by both the sync hook and batch submission.
  
  Args:
    prompt: The prompt text (may contain image tags)
    structure: JSON schema for structured output (or None)
    tools: Tool configuration
    
  Returns:
    List of content parts for the Gemini API
  """
  prompt_parts = pit.parse_prompt_parts(prompt)
  contents: list[Any] = []

  for part_type, part_value in prompt_parts:
    if part_type == "text":
      if part_value:
        contents.append(part_value)
    elif part_type == "image":
      if pit.is_url(part_value):
        req = Request(part_value, headers={"User-Agent": "LLMBenchCore/1.0"})
        with urlopen(req, timeout=30) as resp:
          content_type = resp.headers.get("Content-Type")
          image_bytes = resp.read()

        mime_type = None
        if content_type:
          mime_type = content_type.split(";", 1)[0].strip().lower()
        if not mime_type:
          mime_type = pit.guess_image_mime_type_from_ref(part_value)

        contents.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))
      elif pit.is_data_uri(part_value):
        mime_type, image_bytes = pit.decode_data_uri(part_value)
        contents.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))
      else:
        local_path = pit.resolve_local_path(part_value)
        mime_type = pit.guess_image_mime_type_from_path(local_path)
        image_bytes = pit.read_file_bytes(local_path)
        contents.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))

  # For two-pass approach (tools + structured output), add schema hint
  needs_two_pass = tools and tools is not False and structure is not None
  if needs_two_pass:
    schema_json = json.dumps(structure, indent=2)
    contents.append(
      f"""\n\nProvide your answer with all relevant details. Your response will be parsed into this JSON structure:
{schema_json}

Make sure to include all the information needed to populate these fields.""")

  if not contents:
    contents = [""]

  return contents


def build_gemini_config(structure: dict | None, reasoning, tools) -> dict:
  """
  Build the configuration parameters for a Gemini API call.
  Used by both the sync hook and batch submission.
  
  Returns a dict of config params (may be empty).
  """
  config_params = {}

  # Add structured output if schema provided (but not when tools are enabled - Gemini doesn't support both)
  if structure is not None and not tools:
    cleaned_structure = _clean_schema_for_gemini(structure)
    config_params['response_mime_type'] = 'application/json'
    config_params['response_schema'] = cleaned_structure

  # Add thinking/reasoning configuration if reasoning is set
  # Map 0-10 scale to Gemini's thinking_budget (0-24576)
  if reasoning and reasoning != 0:
    if isinstance(reasoning, int) and reasoning > 0:
      if reasoning <= 3:
        thinking_budget = 128 * (2**(reasoning - 1))  # 128, 256, 512
      elif reasoning <= 7:
        thinking_budget = 1024 * (2**(reasoning - 4))  # 1024, 2048, 4096, 8192
      else:
        thinking_budget = 8192 * (2**(reasoning - 7))  # 8192, 16384, 24576 (capped)
      thinking_budget = min(thinking_budget, 24576)  # Cap at max
    else:
      thinking_budget = 1024  # Default for truthy non-int values

    config_params['thinking_config'] = types.ThinkingConfig(thinking_budget=thinking_budget)

  # Add tools if specified (supports built-in and custom tools)
  if tools and tools is not False:
    tools_list = []
    tools_to_process = tools if isinstance(tools, list) else [tools]

    for tool in tools_to_process:
      if isinstance(tool, str):
        if tool == "google_search":
          tools_list.append(types.Tool(google_search=types.GoogleSearch()))
        elif tool == "code_execution":
          tools_list.append(types.Tool(code_execution=types.ToolCodeExecution()))
        else:
          print(f"Warning: Unknown built-in tool '{tool}', ignoring")
      elif tool is True:
        tools_list.append(types.Tool(code_execution=types.ToolCodeExecution()))
        tools_list.append(types.Tool(google_search=types.GoogleSearch()))
      elif callable(tool):
        tools_list.append(tool)
      else:
        print(f"Warning: Invalid tool type {type(tool)}, ignoring")

    if tools_list:
      config_params['tools'] = tools_list

  return config_params


def _gemini_ai_hook(prompt: str, structure: dict | None, model: str, reasoning,
                    tools, timeout_override: int | None = None) -> dict | str:
  """
    This function is called by the test runner to get the AI's response to a prompt.
    
    Prompt is the question to ask the AI.
    Structure contains the JSON schema for the expected output. If it is None, the output is just a string.
    
    There is no memory between calls to this function, the 'conversation' doesn't persist.
    
    Returns tuple of (result, chainOfThought).
    """
  # Initialize the client - it will automatically use the GEMINI_API_KEY or GOOGLE_API_KEY
  # environment variable
  client = genai.Client()

  try:
    # Build configuration and contents using shared helpers
    config_params = build_gemini_config(structure, reasoning, tools)
    config = types.GenerateContentConfig(**config_params) if config_params else None
    contents = build_gemini_contents(prompt, structure, tools)

    # Track if we need a two-pass approach (tools + structured output)
    needs_two_pass = tools and tools is not False and structure is not None

    # Generate content with streaming to capture thinking in real-time
    chainOfThought = ""
    output_text = ""
    current_thinking_line = ""

    stream = client.models.generate_content_stream(model=model, contents=contents, config=config)

    # Use a thread + queue to enable hard timeout on frozen streams
    chunk_queue = queue.Queue()

    def stream_reader():
      """Read chunks from stream and put them in queue."""
      try:
        for chunk in stream:
          chunk_queue.put(("chunk", chunk))
        chunk_queue.put(("done", None))
      except Exception as e:
        chunk_queue.put(("error", e))

    reader_thread = threading.Thread(target=stream_reader, daemon=True)
    reader_thread.start()

    start_time = time.time()
    timed_out = False
    timeout_seconds = timeout_override or DEFAULT_TIMEOUT_SECONDS

    while True:
      # Check total elapsed time
      elapsed = time.time() - start_time
      if elapsed > timeout_seconds:
        print(f"Timeout: Gemini API call exceeded {timeout_seconds} seconds")
        timed_out = True
        break

      # Wait for next chunk with timeout (check every 30 seconds)
      remaining = timeout_seconds - elapsed
      try:
        msg_type, payload = chunk_queue.get(timeout=min(30, remaining))
      except queue.Empty:
        continue  # No chunk yet, loop back to check timeout

      if msg_type == "done":
        break
      elif msg_type == "error":
        print(f"Stream error: {payload}")

        if "RESOURCE_EXHAUSTED" in payload or "exceeded your current quota" in payload:
          print(
            "Pausing for a random period of time to help sooth quota issues (15 minutes - 1 hour)")
          time.sleep(random.randint(900, 3600))
        break
      elif msg_type == "chunk":
        chunk = payload
        # Process each candidate in the chunk
        for candidate in chunk.candidates:
          if candidate.content and candidate.content.parts:
            for part in candidate.content.parts:
              # Check if this is a thinking part
              if hasattr(part, 'thought') and part.thought:
                # This is thinking content
                thought_text = part.text if hasattr(part, 'text') and part.text else ""
                current_thinking_line += thought_text
                # Print complete lines as they arrive
                while "\n" in current_thinking_line:
                  line, current_thinking_line = current_thinking_line.split("\n", 1)
                  print(f"Thinking: {line}", flush=True)
                  chainOfThought += line + "\n"
              elif hasattr(part, 'text') and part.text:
                # This is regular output content
                output_text += part.text
              elif hasattr(part, 'executable_code') and part.executable_code:
                # Gemini is running code:
                print("Executing the following code:")
                print("\n> " + "\n> ".join(part.executable_code.code.split("\n")))
              elif hasattr(part, 'code_execution_result') and part.code_execution_result:
                # Gemini is running code:
                print("Code execution returned:")
                print("\n< " + "\n< ".join(part.code_execution_result.output.split("\n")))

    # Flush any remaining thinking content
    if current_thinking_line:
      print(f"Thinking: {current_thinking_line}", flush=True)
      chainOfThought += current_thinking_line

    # Strip trailing newline from chain of thought
    chainOfThought = chainOfThought.rstrip("\n")

    if chainOfThought:
      print()  # Blank line after thinking

    # Parse and return response
    if structure is not None:
      # Two-pass approach: use a second call to convert text to structured output
      if needs_two_pass and output_text:
        print(f"Two-pass: Converting text response to structured output...")
        print(output_text)
        try:
          # Second pass: use structured output to parse the text answer
          def remove_additional_properties(schema):
            if isinstance(schema, dict):
              schema.pop("additionalProperties", None)
              for key, value in schema.items():
                if isinstance(value, dict):
                  remove_additional_properties(value)
                elif isinstance(value, list) and value and isinstance(value[0], dict):
                  for item in value:
                    if isinstance(item, dict):
                      remove_additional_properties(item)
            return schema

          cleaned_structure = remove_additional_properties(structure.copy())
          parse_config = types.GenerateContentConfig(response_mime_type='application/json',
                                                     response_schema=cleaned_structure)
          parse_prompt = f"""Extract the answer from the following text and return it as JSON.

Text to parse:
{output_text}

Return ONLY the JSON object matching the schema."""

          parse_response = client.models.generate_content(model=model,
                                                          contents=[parse_prompt],
                                                          config=parse_config)

          if parse_response.text:
            parsed = json.loads(parse_response.text)
            return parsed, chainOfThought
        except Exception as e:
          print(f"Warning: Two-pass structured output failed: {e}")
          # Fall through to try direct text parsing

      if output_text:
        # Try to extract JSON from output (model may wrap it in markdown)
        json_text = output_text.strip()
        if json_text.startswith("```json"):
          json_text = json_text[7:]
        if json_text.startswith("```"):
          json_text = json_text[3:]
        if json_text.endswith("```"):
          json_text = json_text[:-3]
        json_text = json_text.strip()

        try:
          # When tools are enabled, use Pydantic for full schema validation
          if tools and structure:
            try:
              pydantic_model = json_schema_to_pydantic(structure, "ResponseModel")
              parsed_obj = pydantic_model.model_validate_json(json_text)
              return parsed_obj.model_dump(), chainOfThought
            except Exception as e:
              try:
                import json_repair
                repaired = json_repair.repair_json(json_text)
                parsed_obj = pydantic_model.model_validate_json(repaired)
                return parsed_obj.model_dump(), chainOfThought
              except:
                print(f"Warning: Failed to parse JSON response: {e}")
                print(f"Raw output was: {output_text[:500]}")
                return {}, chainOfThought + "\n\nFailed to parse JSON.\n" + output_text
          else:
            parsed = json.loads(json_text)
            return parsed, chainOfThought
        except json.JSONDecodeError as e:
          try:
            import json_repair
            repaired = json_repair.repair_json(json_text)
            return json.loads(repaired), chainOfThought
          except:
            print(f"Warning: Failed to parse JSON response: {e}")
            print(f"Raw output was: {output_text[:500]}")
            return {}, chainOfThought + "\n\nDid not output valid JSON.\n" + output_text
      return {}, chainOfThought
    else:
      return output_text or "", chainOfThought

  except Exception as e:
    print(f"Error calling Gemini API: {e}")

    # Check for content policy violation
    from .ContentViolationHandler import is_content_violation_gemini
    if is_content_violation_gemini(error=e):
      print("CONTENT VIOLATION DETECTED (Gemini)")
      if structure is not None:
        return {"__content_violation__": True, "reason": str(e)}, f"Content violation: {e}"
      else:
        return "__content_violation__", f"Content violation: {e}"

    # Return appropriate empty response based on structure
    if structure is not None:
      return {}, ""
    else:
      return "", ""


def submit_batch(config: dict, requests: list) -> str | None:
  """
  Submit a batch of requests to Gemini's Batch API.
  
  Args:
    config: Model configuration dict with base_model, reasoning, tools
    requests: List of BatchRequest objects
    
  Returns:
    Batch job name if successful, None otherwise
  """
  client = genai.Client()
  model = config.get("base_model", "gemini-2.0-flash")
  reasoning = config.get("reasoning", False)
  tools = config.get("tools", False)

  # Build inline requests using shared helpers
  # Format per Gemini Batch API: each request has 'contents' with parts and role
  inline_requests = []
  for req in requests:
    # Build content parts using existing helper
    content_parts = build_gemini_contents(req.prompt, req.structure, tools)

    # Convert to batch API format: contents must have parts and role
    # content_parts is a list of strings and Part objects
    parts = []
    for part in content_parts:
      if isinstance(part, str):
        parts.append({'text': part})
      else:
        # Already a Part object, keep as-is
        parts.append(part)

    request_obj = {'contents': [{'parts': parts, 'role': 'user'}]}

    # Build config using shared helper
    config_params = build_gemini_config(req.structure, reasoning, tools)
    if config_params:
      # Batch API config is flat (not wrapped in generation_config)
      batch_config = {}
      if 'response_mime_type' in config_params:
        batch_config['response_mime_type'] = config_params['response_mime_type']
      if 'response_schema' in config_params:
        batch_config['response_schema'] = config_params['response_schema']
      if batch_config:
        request_obj["config"] = batch_config

    inline_requests.append(request_obj)

  # Create batch - use 'src' parameter (not 'requests')
  batch_job = client.batches.create(
    model=f"models/{model}",
    src=inline_requests,
    config={"display_name": f"LLMBenchCore batch for {config['name']}"})

  return batch_job.name


def poll_batch(batch_id: str, requests: list) -> tuple:
  """
  Poll a Gemini batch for status and results.
  
  Args:
    batch_id: The batch job name to poll
    requests: List of original BatchRequest objects (for parsing results)
    
  Returns:
    Tuple of (status_string, list of result dicts)
    status_string is one of: "completed", "failed", "processing"
  """
  from google import genai
  import json

  client = genai.Client()
  batch_job = client.batches.get(name=batch_id)

  results = []
  state = batch_job.state.name if hasattr(batch_job.state, 'name') else str(batch_job.state)

  if state == "JOB_STATE_SUCCEEDED":
    # Get results from dest.inlined_responses (for inline batch requests)
    inlined_responses = []
    if batch_job.dest and hasattr(batch_job.dest,
                                  'inlined_responses') and batch_job.dest.inlined_responses:
      inlined_responses = batch_job.dest.inlined_responses

    for i, inline_response in enumerate(inlined_responses):
      if i >= len(requests):
        break

      req = requests[i]
      custom_id = req.custom_id

      if inline_response.error:
        results.append({
          "custom_id": custom_id,
          "success": False,
          "result": None,
          "chain_of_thought": "",
          "error": str(inline_response.error)
        })
        continue

      # Extract text from response
      text_content = ""
      cot = ""

      if inline_response.response:
        # Try to get text directly
        try:
          text_content = inline_response.response.text
        except AttributeError:
          # Fallback: try candidates structure
          if hasattr(inline_response.response,
                     'candidates') and inline_response.response.candidates:
            candidate = inline_response.response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
              for part in candidate.content.parts:
                if hasattr(part, 'thought') and part.thought:
                  cot += part.text + "\n"
                elif hasattr(part, 'text'):
                  text_content += part.text

      # Parse JSON if structured
      result_data = text_content
      if req.structure:
        try:
          json_text = text_content.strip()
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

  elif state in ["JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"]:
    return "failed", results

  else:
    # Still processing
    print(f"[Batch] Gemini batch {batch_id}: {state}")
    # SDK shape varies; avoid assuming batch_stats exists
    stats = getattr(batch_job, "batch_stats", None)
    if stats is None:
      stats = getattr(batch_job, "stats", None)
    if stats is None:
      stats = getattr(batch_job, "batchStats", None)
    if stats is None:
      stats = getattr(batch_job, "completion_stats", None)
    if stats is not None:
      print(f"[Batch] Gemini batch {batch_id}: stats={stats}")
    else:
      print(batch_job)
    return "processing", results
