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
from html import unescape
from html.parser import HTMLParser
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

# Set to True at edit time to allow fetch_url() for any http/https URL.
ALLOW_FETCH_ANY_URL = False

_MAX_FETCH_BYTES = 512 * 1024
_MAX_FETCH_TEXT_CHARS = 12000
_MAX_TRACKED_SEARCH_URLS = 200
_WEB_SEARCH_RESULT_URLS: set[str] = set()

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

FETCH_URL_TOOL = {
  "type": "function",
  "function": {
    "name":
    "fetch_url",
    "description":
    "Fetch the contents of a URL. By default this is limited to URLs returned "
    "by prior web_search calls, unless ALLOW_FETCH_ANY_URL is enabled at "
    "compile time.",
    "parameters": {
      "type": "object",
      "properties": {
        "url": {
          "type": "string",
          "description": "The URL to fetch"
        }
      },
      "required": ["url"]
    }
  }
}

ALL_TOOLS = [WEB_SEARCH_TOOL, FETCH_URL_TOOL, PYTHON_EXECUTE_TOOL]


class _HTMLTextExtractor(HTMLParser):
  """Extract a readable text approximation from HTML."""

  def __init__(self):
    super().__init__()
    self._skip_depth = 0
    self._title_parts: list[str] = []
    self._text_parts: list[str] = []
    self._in_title = False

  def handle_starttag(self, tag, attrs):
    if tag in {"script", "style", "noscript"}:
      self._skip_depth += 1
    elif tag == "title":
      self._in_title = True
    elif tag in {"p", "div", "br", "li", "tr", "section", "article", "h1", "h2", "h3", "h4"}:
      self._text_parts.append("\n")

  def handle_endtag(self, tag):
    if tag in {"script", "style", "noscript"} and self._skip_depth > 0:
      self._skip_depth -= 1
    elif tag == "title":
      self._in_title = False
    elif tag in {"p", "div", "br", "li", "tr", "section", "article", "h1", "h2", "h3", "h4"}:
      self._text_parts.append("\n")

  def handle_data(self, data):
    if self._skip_depth > 0:
      return
    text = unescape(data or "")
    if not text.strip():
      return
    if self._in_title:
      self._title_parts.append(text.strip())
    self._text_parts.append(text)

  def result(self) -> tuple[str, str]:
    title = " ".join(part for part in self._title_parts if part).strip()
    body = "".join(self._text_parts)
    lines = [line.strip() for line in body.splitlines()]
    body = "\n".join(line for line in lines if line)
    return title, body


def _normalize_url(url: str) -> str:
  parts = urlsplit((url or "").strip())
  if parts.scheme not in {"http", "https"} or not parts.netloc:
    return ""
  path = parts.path or "/"
  if path != "/" and path.endswith("/"):
    path = path[:-1]
  return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, parts.query, ""))


def _remember_search_result_url(url: str) -> None:
  normalized = _normalize_url(url)
  if not normalized:
    return
  _WEB_SEARCH_RESULT_URLS.add(normalized)
  if len(_WEB_SEARCH_RESULT_URLS) > _MAX_TRACKED_SEARCH_URLS:
    while len(_WEB_SEARCH_RESULT_URLS) > _MAX_TRACKED_SEARCH_URLS:
      _WEB_SEARCH_RESULT_URLS.pop()


def _fetch_url_is_allowed(url: str) -> bool:
  normalized = _normalize_url(url)
  if not normalized:
    return False
  if ALLOW_FETCH_ANY_URL:
    return True
  return normalized in _WEB_SEARCH_RESULT_URLS


# ---------------------------------------------------------------------------
# Tool executors
# ---------------------------------------------------------------------------


def execute_web_search(query: str) -> str:
  """Perform a web search using DuckDuckGo and return results as a string."""
  try:
    try:
      from ddgs import DDGS
    except ImportError:
      from duckduckgo_search import DDGS
    with DDGS() as ddgs:
      print("> Searching the web for:", query)
      results = list(ddgs.text(query, max_results=5))
    if not results:
      return "No results found."
    lines = []
    for r in results:
      href = r.get('href', '')
      _remember_search_result_url(href)
      lines.append(f"Title: {r.get('title', '')}")
      lines.append(f"URL: {href}")
      lines.append(f"Snippet: {r.get('body', '')}")
      lines.append("")
    return "\n".join(lines)
  except ImportError:
    return ("web_search unavailable: install the ddgs package "
            "(pip install ddgs) for web search support.")
  except Exception as e:
    return f"web_search error: {e}"


def execute_fetch_url(url: str) -> str:
  """Fetch a URL and return a readable text extract."""
  normalized = _normalize_url(url)
  if not normalized:
    return "fetch_url error: only absolute http/https URLs are supported."
  if not _fetch_url_is_allowed(normalized):
    return ("fetch_url blocked: URL was not returned by a prior web_search call. "
            "Set ALLOW_FETCH_ANY_URL = True to allow arbitrary URLs.")

  try:
    print("> Fetching URL:", normalized)
    request = Request(normalized,
                      headers={"User-Agent": ("Mozilla/5.0 (compatible; LLMBenchCore/1.0;")})
    with urlopen(request, timeout=30) as response:
      content_type = response.headers.get("Content-Type", "")
      raw_bytes = response.read(_MAX_FETCH_BYTES + 1)

    truncated = len(raw_bytes) > _MAX_FETCH_BYTES
    raw_bytes = raw_bytes[:_MAX_FETCH_BYTES]
    encoding = "utf-8"
    if "charset=" in content_type:
      encoding = content_type.split("charset=", 1)[1].split(";", 1)[0].strip() or "utf-8"
    text = raw_bytes.decode(encoding, errors="replace")

    if "html" in content_type.lower() or "<html" in text[:1000].lower():
      parser = _HTMLTextExtractor()
      parser.feed(text)
      title, body = parser.result()
      lines = []
      if title:
        lines.append(f"Title: {title}")
      lines.append(f"URL: {normalized}")
      lines.append(f"Content-Type: {content_type or 'text/html'}")
      lines.append("")
      lines.append(body[:_MAX_FETCH_TEXT_CHARS] or "(no readable text found)")
      if truncated or len(body) > _MAX_FETCH_TEXT_CHARS:
        lines.append("\n[truncated]")
      return "\n".join(lines).strip()

    output = text[:_MAX_FETCH_TEXT_CHARS]
    result = (f"URL: {normalized}\n"
              f"Content-Type: {content_type or 'text/plain'}\n\n"
              f"{output}")
    if truncated or len(text) > _MAX_FETCH_TEXT_CHARS:
      result += "\n\n[truncated]"
    return result
  except Exception as e:
    return f"fetch_url error: {e}"


def execute_python(code: str, timeout: int = 300) -> str:
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
    - "fetch_url" / "read_url"           -> execute_fetch_url
    - "execute_python" / "run_python_code" -> execute_python
  """
  if function_name == "web_search":
    return execute_web_search(arguments.get("query", ""))
  elif function_name in ("fetch_url", "read_url"):
    return execute_fetch_url(arguments.get("url", ""))
  elif function_name in ("execute_python", "run_python_code"):
    return execute_python(arguments.get("code", ""))
  else:
    return f"Unknown tool: {function_name}"
