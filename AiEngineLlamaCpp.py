"""
llama.cpp Server AI Engine for LLMBenchCore

This module provides an interface to a llama.cpp server running locally or remotely.
llama.cpp server provides an OpenAI-compatible API.

Setup:
1. Build and run llama.cpp server:
   ./llama-server -m your_model.gguf --port 8080
   
   For vision models:
   ./llama-server -m your_vision_model.gguf --mmproj mmproj.gguf --port 8080

2. Set the server URL as an environment variable (optional, defaults to http://localhost:8080):
   - Windows: set LLAMACPP_BASE_URL=http://localhost:8080
   - Linux/Mac: export LLAMACPP_BASE_URL=http://localhost:8080

The llama.cpp server documentation can be found at: https://github.com/ggerganov/llama.cpp/tree/master/examples/server
"""

import hashlib
import os
import json
import threading
import requests
import uuid
from . import PromptImageTagging as pit
from .ToolExecutors import ALL_TOOLS, dispatch_tool_call


class LlamaCppEngine:
  """
  llama.cpp Server AI Engine class.
  
  Configuration parameters:
  - model: Model name/identifier (used for cache key, not sent to server unless needed)
  - base_url: Server URL (defaults to LLAMACPP_BASE_URL env var or http://localhost:8080)
  - timeout: Request timeout in seconds (default: 3600)
  - tools: Tool capabilities:
      - False: No tools available
      - True: Enable Python code execution tool
  """

  def __init__(self, model: str, base_url: str | None = None, timeout: int = 3600, tools=False):
    self.model = model
    self.base_url = base_url or os.environ.get("LLAMACPP_BASE_URL", "http://localhost:8080")
    self.timeout = timeout
    self.tools = tools
    self.configAndSettingsHash = hashlib.sha256(
      model.encode() + self.base_url.encode() + str(tools).encode()
    ).hexdigest()

  def AIHook(self, prompt: str, structure: dict | None) -> tuple:
    """Call the llama.cpp server with instance configuration."""
    return _llamacpp_ai_hook(prompt, structure, self.model, self.base_url, self.timeout, self.tools)


def build_llamacpp_messages(prompt: str) -> list[dict]:
  """
  Build messages array for llama.cpp server chat completions API.
  Handles text and images, converting images to base64 data URIs.
  """
  prompt_parts = pit.parse_prompt_parts(prompt)
  has_images = any(part_type == "image" for part_type, _ in prompt_parts)
  
  if not has_images:
    # Simple text-only message
    return [{"role": "user", "content": prompt}]
  
  # Build multimodal content array
  content: list[dict] = []
  for part_type, part_value in prompt_parts:
    if part_type == "text":
      if part_value:
        content.append({"type": "text", "text": part_value})
    elif part_type == "image":
      # Convert to base64 data URI format
      if pit.is_url(part_value):
        # For URLs, download and convert to base64
        image_url = part_value
        content.append({
          "type": "image_url",
          "image_url": {"url": image_url}
        })
      elif pit.is_data_uri(part_value):
        content.append({
          "type": "image_url",
          "image_url": {"url": part_value}
        })
      else:
        # Local file - convert to data URI
        local_path = pit.resolve_local_path(part_value)
        data_uri = pit.file_to_data_uri(local_path)
        content.append({
          "type": "image_url",
          "image_url": {"url": data_uri}
        })
  
  return [{"role": "user", "content": content}]


mutex = threading.Lock()

def _llamacpp_ai_hook(prompt: str, structure: dict | None, model: str, base_url: str, 
                      timeout: int, tools=False) -> tuple:
 """
  This function is called by the test runner to get the AI's response to a prompt.
  
  Prompt is the question to ask the AI.
  Structure contains the JSON schema for the expected output. If it is None, the output is just a string.
  
  There is no memory between calls to this function, the 'conversation' doesn't persist.
  
  Returns tuple of (result, chainOfThought).
  """
 with mutex: # No point spamming my low-end consumer GPU with multiple requests
  try:
    # Build the API endpoint URL
    api_url = f"{base_url.rstrip('/')}/v1/chat/completions"
    
    # Build messages
    messages = build_llamacpp_messages(prompt)
    
    # Build request payload
    payload = {
      "messages": messages,
      "stream": False,
    }
    
    # Add model if specified (some llama.cpp setups ignore this)
    if model:
      payload["model"] = model
    
    # Add tools if enabled
    if tools is True:
      payload["tools"] = ALL_TOOLS
      payload["tool_choice"] = "auto"
    
    # Handle structured output using JSON schema (only if no tools, as they can conflict)
    if structure is not None and not tools:
      # llama.cpp supports JSON schema via response_format
      payload["response_format"] = {
        "type": "json_schema",
        "json_schema": {
          "name": "response",
          "strict": True,
          "schema": structure
        }
      }
    
    # Make the API call
    headers = {"Content-Type": "application/json"}
    
    # Add API key if set (some deployments require it)
    api_key = os.environ.get("LLAMACPP_API_KEY")
    if api_key:
      headers["Authorization"] = f"Bearer {api_key}"
    
    # Tool calling loop - continue until we get a final response (no tool calls)
    chainOfThought = ""
    max_tool_iterations = 10
    iteration = 0
    
    while iteration < max_tool_iterations:
      iteration += 1
      
      response = requests.post(api_url, json=payload, headers=headers, timeout=timeout)
      response.raise_for_status()
      
      result_json = response.json()
      
      # Extract the response content
      choices = result_json.get("choices", [])
      if not choices:
        print("No choices in llama.cpp response")
        if structure is not None:
          return {}, ""
        return "", ""
      
      message = choices[0].get("message", {})
      finish_reason = choices[0].get("finish_reason", "")
      
      # Check if the model wants to call tools
      tool_calls = message.get("tool_calls", [])
      
      if tool_calls and finish_reason == "tool_calls":
        # Process each tool call
        # First, add the assistant's message with tool calls to conversation
        payload["messages"].append(message)
        
        for tool_call in tool_calls:
          tool_id = tool_call.get("id", f"call_{uuid.uuid4().hex[:8]}")
          function = tool_call.get("function", {})
          func_name = function.get("name", "")
          func_args_str = function.get("arguments", "{}")
          
          try:
            func_args = json.loads(func_args_str)
          except json.JSONDecodeError:
            func_args = {}
          
          # Execute the tool
          tool_result = dispatch_tool_call(func_name, func_args)
          chainOfThought += f"\n[Tool: {func_name}]\n{json.dumps(func_args)[:200]}\n[Output]\n{tool_result}\n"
          
          # Add tool result to messages
          payload["messages"].append({
            "role": "tool",
            "tool_call_id": tool_id,
            "content": tool_result
          })
        
        # Continue the loop to get the next response
        continue
      
      # No tool calls - this is the final response
      output_text = message.get("content", "") or ""
      
      # Parse structured output if requested
      if structure is not None:
        # If tools were enabled, we need to ask the model to format as JSON
        if tools:
          # Add instruction for JSON output and make another call
          schema_json = json.dumps(structure, indent=2)
          payload["messages"].append({
            "role": "user",
            "content": f"Now provide your final answer as JSON matching this schema:\n{schema_json}\n\nReturn ONLY the JSON object, no explanation."
          })
          # Remove tools for final formatting call
          payload.pop("tools", None)
          payload.pop("tool_choice", None)
          payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
              "name": "response",
              "strict": True,
              "schema": structure
            }
          }
          
          format_response = requests.post(api_url, json=payload, headers=headers, timeout=timeout)
          format_response.raise_for_status()
          format_result = format_response.json()
          
          format_choices = format_result.get("choices", [])
          if format_choices:
            output_text = format_choices[0].get("message", {}).get("content", "")
        
        try:
          # Strip markdown code blocks if present
          parse_text = output_text
          if "```json" in parse_text:
            parse_text = parse_text.split("```json", 1)[1].split("```", 1)[0]
          elif "```" in parse_text:
            parse_text = parse_text.split("```", 1)[1].split("```", 1)[0]
          
          parse_text = parse_text.strip()
          return json.loads(parse_text), chainOfThought
        except json.JSONDecodeError as e:
          import json_repair
          try:
            repaired = json_repair.repair_json(output_text)
            return json.loads(repaired), chainOfThought
          except:
            print(f"Failed to parse JSON response: {e}")
            print(f"Raw response: {output_text}")
            return {}, chainOfThought
      else:
        return output_text or "", chainOfThought
    
    # Max iterations reached
    print(f"Warning: Max tool iterations ({max_tool_iterations}) reached")
    if structure is not None:
      return {}, chainOfThought
    return "", chainOfThought
    
  except requests.exceptions.Timeout:
    print(f"Timeout calling llama.cpp server at {base_url}")
    if structure is not None:
      return {}, ""
    return "", ""
    
  except requests.exceptions.ConnectionError as e:
    print(f"Connection error to llama.cpp server at {base_url}: {e}")
    if structure is not None:
      return {}, ""
    return "", ""
    
  except requests.exceptions.HTTPError as e:
    print(f"HTTP error from llama.cpp server: {e}")
    if e.response is not None:
      print(f"Response body: {e.response.text}")
    if structure is not None:
      return {}, ""
    return "", ""
    
  except Exception as e:
    print(f"Error calling llama.cpp server: {e}")
    if structure is not None:
      return {}, ""
    return "", ""
