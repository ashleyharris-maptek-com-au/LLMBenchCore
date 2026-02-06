"""
Azure OpenAI Responses API engine for LLMBenchCore.

This mirrors the OpenAI engine but uses AzureOpenAI with api_version.

Required env vars:
- AZURE_OPENAI_API_KEY
- AZURE_OPENAI_ENDPOINT (can be the base resource URL or a full /openai/responses?api-version=... URL)
Optional:
- AZURE_OPENAI_API_VERSION (defaults to api-version parsed from endpoint or 2025-04-01-preview)
"""

import hashlib
import json
import os
import random
import time
from urllib.parse import parse_qs, urlparse

from . import PromptImageTagging as pit


def _normalize_azure_endpoint(endpoint: str, api_version: str | None) -> tuple[str, str | None]:
  endpoint = (endpoint or "").strip()
  if not endpoint:
    return endpoint, api_version

  parsed = urlparse(endpoint)
  if parsed.scheme and parsed.netloc:
    base = f"{parsed.scheme}://{parsed.netloc}"
    if not api_version:
      query = parse_qs(parsed.query)
      if "api-version" in query and query["api-version"]:
        api_version = query["api-version"][0]
    return base, api_version

  return endpoint.rstrip("/"), api_version


def _supports_responses(api_version: str | None) -> bool:
  if not api_version:
    return True
  try:
    parts = api_version.split("-")
    if len(parts) < 3:
      return True
    year, month, day = (int(parts[0]), int(parts[1]), int(parts[2]))
    return (year, month, day) >= (2025, 3, 1)
  except Exception:
    return True


def _sanitize_schema_for_azure(schema):
  if isinstance(schema, dict):
    return {
      key: _sanitize_schema_for_azure(value)
      for key, value in schema.items()
      if key != "uniqueItems"
    }
  if isinstance(schema, list):
    return [_sanitize_schema_for_azure(value) for value in schema]
  return schema


class AzureOpenAIEngine:
  """
  Azure OpenAI Responses API engine.

  Configuration parameters:
  - model: Azure deployment name
  - reasoning: False/0 or effort level (1-10)
  - tools: False/True/custom tools list
  - endpoint: Azure OpenAI resource endpoint (or full responses URL)
  - api_version: Azure OpenAI api-version string
  """

  def __init__(self, model: str, reasoning=False, tools=False, endpoint: str | None = None,
               api_version: str | None = None, timeout: int = 3600):
    self.model = model
    self.reasoning = reasoning
    self.tools = tools
    self.endpoint = endpoint
    self.api_version = api_version
    self.timeout = timeout
    self.forcedFailure = False
    self.configAndSettingsHash = hashlib.sha256(
      model.encode() + str(reasoning).encode() + str(tools).encode() +
      str(endpoint).encode() + str(api_version).encode() + str(timeout).encode()
    ).hexdigest()

  def AIHook(self, prompt: str, structure: dict | None) -> tuple:
    result = _azure_openai_ai_hook(prompt, structure, self.model, self.reasoning, self.tools,
                                   self.endpoint, self.api_version, self, timeout_override=self.timeout)
    return result


def _build_openai_input(prompt: str):
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


def _build_chat_messages(prompt: str):
  input_value = _build_openai_input(prompt)
  if isinstance(input_value, str):
    return [{"role": "user", "content": input_value}]
  if isinstance(input_value, list) and input_value:
    msg = input_value[0]
    content = []
    for item in msg.get("content", []):
      if item.get("type") == "input_text":
        content.append({"type": "text", "text": item.get("text", "")})
      elif item.get("type") == "input_image":
        image_url = item.get("image_url")
        if isinstance(image_url, str):
          content.append({"type": "image_url", "image_url": {"url": image_url}})
        elif image_url:
          content.append({"type": "image_url", "image_url": image_url})
    return [{"role": "user", "content": content}]
  return [{"role": "user", "content": prompt}]


def _convert_tools(tools):
  if not tools or tools is False:
    return None
  if tools is True:
    return True

  tools_list = []
  for tool in (tools if isinstance(tools, list) else [tools]):
    if isinstance(tool, dict):
      tools_list.append(tool)
    elif callable(tool):
      import inspect
      sig = inspect.signature(tool)
      doc = inspect.getdoc(tool) or "No description"

      properties = {}
      required = []
      for param_name, param in sig.parameters.items():
        param_type = "string"
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

      tool_def = {
        "type": "function",
        "name": tool.__name__,
        "description": doc,
        "parameters": {
          "type": "object",
          "properties": properties,
          "required": required
        }
      }
      tools_list.append(tool_def)

  return tools_list or None


def _azure_openai_ai_hook(prompt: str, structure: dict | None, model: str, reasoning, tools,
                          endpoint: str | None, api_version: str | None,
                          engine_instance, timeout_override: int | None = None) -> tuple:
  if engine_instance.forcedFailure:
    return {"error": "Forced failure"}, "Forced failure due to API instability"

  from openai import AzureOpenAI

  last_output_text = ""
  try:
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    if not api_key:
      raise RuntimeError("AZURE_OPENAI_API_KEY is not set")

    endpoint = endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    endpoint, api_version = _normalize_azure_endpoint(endpoint, api_version)
    if not endpoint:
      raise RuntimeError("AZURE_OPENAI_ENDPOINT is not set")

    if not api_version:
      api_version = os.environ.get("AZURE_OPENAI_API_VERSION") or "2025-04-01-preview"

    client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version,
                         timeout=timeout_override or engine_instance.timeout)

    # Azure uses deployment name for model
    model_to_use = model
    if isinstance(reasoning, str) and reasoning:
      # Allow override by deployment name if supplied as a string.
      model_to_use = reasoning

    tools_converted = _convert_tools(tools)

    if not _supports_responses(api_version):
      messages = _build_chat_messages(prompt)
      chat_params = {"model": model_to_use, "messages": messages}

      if structure is not None:
        schema = _sanitize_schema_for_azure(structure)
        chat_params["response_format"] = {
          "type": "json_schema",
          "json_schema": {
            "name": "structured_response",
            "schema": schema,
            "strict": True
          }
        }

      if tools_converted and tools_converted is not True:
        chat_params["tools"] = tools_converted

      response = client.chat.completions.create(**chat_params)
      output_text = response.choices[0].message.content if response.choices else ""
      last_output_text = output_text

      if structure is not None:
        if output_text:
          try:
            return json.loads(output_text), ""
          except json.JSONDecodeError:
            print("Error decoding JSON response. Returning raw text for parsing.")
            return output_text, ""
        return {}, ""
      else:
        return output_text or "", ""

    input_value = _build_openai_input(prompt)

    response_params = {"model": model_to_use, "input": input_value}

    if isinstance(reasoning, int) and reasoning > 0:
      response_params["reasoning"] = {"effort": reasoning, "summary": "auto"}

    if structure is not None:
      schema = _sanitize_schema_for_azure(structure)
      response_params["text"] = {
        "format": {
          "type": "json_schema",
          "name": "structured_response",
          "schema": schema,
          "strict": True
        }
      }

    if tools_converted is True:
      # Azure Responses does not currently support web_search; keep tool set minimal.
      if "5.2-pro" not in model_to_use:
        response_params["tools"] = [{"type": "code_interpreter", "container": {"type": "auto"}}]
    elif tools_converted:
      response_params["tools"] = tools_converted

    stream = client.responses.create(stream=True, timeout=timeout_override or 3600, **response_params)

    chain_of_thought = ""
    output_text = ""
    current_reasoning_line = ""

    for event in stream:
      event_type = event.type

      if event_type == "response.reasoning_summary_text.delta":
        delta = event.delta
        current_reasoning_line += delta
        while "\n" in current_reasoning_line:
          line, current_reasoning_line = current_reasoning_line.split("\n", 1)
          print(f"Thinking: {line}", flush=True)
          chain_of_thought += line + "\n"

      elif event_type == "response.reasoning_summary_text.done":
        if current_reasoning_line:
          print(f"Thinking: {current_reasoning_line}", flush=True)
          chain_of_thought += current_reasoning_line
          current_reasoning_line = ""

      elif event_type == "response.output_text.delta":
        output_text += event.delta

      elif event_type == "response.completed":
        pass

    chain_of_thought = chain_of_thought.rstrip("\n")
    last_output_text = output_text

    if structure is not None:
      if output_text:
        try:
          return json.loads(output_text), chain_of_thought
        except json.JSONDecodeError:
          print("Error decoding JSON response. Returning raw text for parsing.")
          return output_text, chain_of_thought
      return {}, chain_of_thought
    else:
      return output_text or "", chain_of_thought

  except json.JSONDecodeError:
    print("Error decoding JSON response. Returning raw text for parsing.")
    return last_output_text or {"unacceptableFailure": True}, ""
  except Exception as e:
    print(f"Error calling Azure OpenAI API: {e}")

    from .ContentViolationHandler import is_content_violation_openai
    if is_content_violation_openai(e):
      print("CONTENT VIOLATION DETECTED (Azure OpenAI)")
      if structure is not None:
        return {"__content_violation__": True, "reason": str(e)}, f"Content violation: {e}"
      else:
        return "__content_violation__", f"Content violation: {e}"

    if "You exceeded your current quota," in str(e):
      print("QUOTA EXCEEDED. Waiting 15 minutes to an hour.")
      time.sleep(random.randint(900, 3600))
      return "", ""

    if structure is not None:
      return {}, ""
    else:
      return "", ""
