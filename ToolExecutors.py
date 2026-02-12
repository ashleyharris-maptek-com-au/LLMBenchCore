"""
Shared tool definitions and executors for AI engines that lack
built-in tools (web search, code execution).

Any engine can import the tool JSON schemas and dispatch function
to add these capabilities via function-calling.
"""

import json
import os
import subprocess
import sys
import tempfile
import uuid

# ---------------------------------------------------------------------------
# Tool JSON schemas (OpenAI-compatible function-calling format)
# ---------------------------------------------------------------------------

WEB_SEARCH_TOOL = {
  "type": "function",
  "function": {
    "name":
    "web_search",
    "description":
    "Search the web for current information. Returns a list of relevant "
    "result snippets. Use this when you need up-to-date facts, data, or "
    "information not in your training data.",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {
          "type": "string",
          "description": "The search query string"
        }
      },
      "required": ["query"]
    }
  }
}

PYTHON_EXECUTE_TOOL = {
  "type": "function",
  "function": {
    "name":
    "execute_python",
    "description":
    "Execute Python code in a sandboxed environment and return stdout/stderr. "
    "The environment has numpy, scipy, and the Python standard library. "
    "Use this for calculations, data processing, or code verification.",
    "parameters": {
      "type": "object",
      "properties": {
        "code": {
          "type": "string",
          "description": "Python code to execute. Use print() to output results."
        }
      },
      "required": ["code"]
    }
  }
}

ALL_TOOLS = [WEB_SEARCH_TOOL, PYTHON_EXECUTE_TOOL]

# ---------------------------------------------------------------------------
# Tool executors
# ---------------------------------------------------------------------------


def execute_web_search(query: str) -> str:
  """Perform a web search using DuckDuckGo and return results as a string."""
  try:
    from duckduckgo_search import DDGS
    with DDGS() as ddgs:
      results = list(ddgs.text(query, max_results=5))
    if not results:
      return "No results found."
    lines = []
    for r in results:
      lines.append(f"Title: {r.get('title', '')}")
      lines.append(f"URL: {r.get('href', '')}")
      lines.append(f"Snippet: {r.get('body', '')}")
      lines.append("")
    return "\n".join(lines)
  except ImportError:
    return ("web_search unavailable: install duckduckgo-search package "
            "(pip install duckduckgo-search) for web search support.")
  except Exception as e:
    return f"web_search error: {e}"


def execute_python(code: str, timeout: int = 120) -> str:
  """Execute Python code in a subprocess and return combined stdout+stderr."""
  work_dir = os.path.join(tempfile.gettempdir(), "llmbench_code_exec")
  os.makedirs(work_dir, exist_ok=True)
  script_name = f"exec_{uuid.uuid4().hex[:8]}.py"
  script_path = os.path.join(work_dir, script_name)

  try:
    with open(script_path, "w", encoding="utf-8") as f:
      f.write(code)

    print(f"Executing Python code:")
    print("\n> " + "\n> ".join(code.split("\n")))

    result = subprocess.run([sys.executable, script_path],
                            capture_output=True,
                            text=True,
                            timeout=timeout,
                            cwd=work_dir)
    output = ""
    if result.stdout:
      output += result.stdout
    if result.stderr:
      if output:
        output += "\n"
      output += f"STDERR:\n{result.stderr}"
    if result.returncode != 0:
      output += f"\n[exit code {result.returncode}]"

    print(f"Code execution returned:")
    print("\n< " + "\n< ".join(output.split("\n")))

    return output.strip() or "(no output)"
  except subprocess.TimeoutExpired:
    return f"Execution timed out after {timeout} seconds."
  except Exception as e:
    return f"Execution error: {e}"
  finally:
    try:
      if os.path.exists(script_path):
        os.remove(script_path)
    except:
      pass


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def dispatch_tool_call(function_name: str, arguments: dict) -> str:
  """Route a tool call to the appropriate executor.

  Recognises aliases so engines that used different names still work:
    - "web_search"                       -> execute_web_search
    - "execute_python" / "run_python_code" -> execute_python
  """
  if function_name == "web_search":
    return execute_web_search(arguments.get("query", ""))
  elif function_name in ("execute_python", "run_python_code"):
    return execute_python(arguments.get("code", ""))
  else:
    return f"Unknown tool: {function_name}"
