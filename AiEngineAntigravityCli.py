import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path

from filelock import FileLock

from .AiEngineCliWorkspace import (create_workspace_dir, largest_new_file, read_text_file_if_exists,
                                   remove_workspace_dir, result_file_types_from_context,
                                   snapshot_workspace_files, write_prompt_workspace)

_ANTIGRAVITY_CREDENTIAL_TARGET = "gemini:antigravity"
_ANTIGRAVITY_CREDENTIAL_USER = "antigravity"


def _reasoning_label(reasoning) -> str:
  if isinstance(reasoning, int) and reasoning > 0:
    if reasoning <= 3:
      return "low"
    if reasoning >= 9:
      return "max"
    if reasoning > 6:
      return "high"
    return "medium"
  if reasoning is True:
    return "medium"
  return "none"


def _find_antigravity_cli() -> str | None:
  explicit = os.environ.get("ANTIGRAVITY_CLI_PATH")
  if explicit and os.path.exists(explicit):
    return explicit

  for executable in ("agy", "agy.exe", "agy.cmd", "antigravity", "antigravity.exe",
                     "antigravity.cmd"):
    found = shutil.which(executable)
    if found:
      return found

  local_app_data = os.environ.get("LOCALAPPDATA")
  if local_app_data:
    default_path = os.path.join(local_app_data, "agy", "bin", "agy.exe")
    if os.path.exists(default_path):
      return default_path

  return None


def _duration_arg(seconds: int | None) -> str:
  seconds = max(1, int(seconds or 3600 * 3))
  return f"{seconds}s"


def _build_prompt(workspace_paths: dict[str, object], structure: dict | None, reasoning, tools,
                  run_id: str) -> str:
  instructions = [
    "You are answering a benchmark prompt.",
    "Work only in the current directory.",
    "A copy of the benchmark question is available in question.txt.",
    f"Internal benchmark run id: {run_id}. Do not include this id in the answer.",
  ]

  image_records = workspace_paths.get("image_records", [])
  if image_records:
    instructions.append("The prompt has these attached image files:")
    for record in image_records:
      if isinstance(record, dict) and record.get("local_ref"):
        instructions.append(f"- {record['local_ref']}")

  reasoning_label = _reasoning_label(reasoning)
  if reasoning_label != "none":
    instructions.append(
      f"Use {reasoning_label} reasoning effort before producing the final answer.")

  if structure is not None:
    instructions.append(
      "Read the required JSON schema from structure.json and write the final valid JSON "
      "answer to answer.json. Also make your final response exactly that JSON."
    )
  else:
    instructions.append("Keep your answer in a single file, preferably answer.txt.")

  if tools:
    instructions.append(
      "Tool access is allowed. Use tools when they materially improve the answer.")
  else:
    instructions.append(
      "Use only the current sandboxed workspace and avoid external commands unless needed "
      "to write the final answer."
    )

  instructions.append("Benchmark question:")
  instructions.append(str(workspace_paths.get("question_text", "")).strip())
  return "\n\n".join(instructions)


def _app_data_dir() -> str:
  return os.path.join(os.path.expanduser("~"), ".gemini", "antigravity-cli")


def _settings_path(app_data_dir: str) -> str:
  return os.path.join(app_data_dir, "settings.json")


def _settings_lock_path(app_data_dir: str) -> str:
  return os.environ.get("ANTIGRAVITY_CLI_SETTINGS_LOCK",
                        os.path.join(app_data_dir, "settings.json.llmbench.lock"))


def _settings_lock_timeout() -> float:
  raw_timeout = os.environ.get("ANTIGRAVITY_CLI_SETTINGS_LOCK_TIMEOUT", "3600")
  try:
    timeout = float(raw_timeout)
  except ValueError:
    return 3600.0
  if timeout < 0:
    return 3600.0
  return min(timeout, 86400.0)


_ANTIGRAVITY_CLI_ALIAS_MODELS = {
  "",
  "antigravity",
  "antigravity-cli",
  "antigravity_cli",
}
_ANTIGRAVITY_CLI_MODEL_LABELS = {
  "gemini-3.5-flash": "Gemini 3.5 Flash",
  "gemini-3-pro": "Gemini 3 Pro",
  "gemini-3-pro-preview": "Gemini 3 Pro",
  "gemini-3.1-pro": "Gemini 3.1 Pro",
  "gemini-3.1-pro-preview": "Gemini 3.1 Pro",
}


def _antigravity_cli_reasoning_suffix(reasoning) -> str:
  reasoning_label = _reasoning_label(reasoning)
  if reasoning_label in ("none", "low"):
    return "Low"
  if reasoning_label == "medium":
    return "Medium"
  return "High"


def _antigravity_cli_model_label(model: str | None, reasoning) -> str:
  requested = str(model or "").strip()
  requested_key = requested.lower()
  if requested_key in _ANTIGRAVITY_CLI_ALIAS_MODELS:
    requested_key = "gemini-3.5-flash"
  label = _ANTIGRAVITY_CLI_MODEL_LABELS.get(requested_key)
  if label:
    return f"{label} ({_antigravity_cli_reasoning_suffix(reasoning)})"
  return requested


def _antigravity_cli_supports_model(model: str | None) -> bool:
  requested = str(model or "").strip().lower()
  return requested in _ANTIGRAVITY_CLI_ALIAS_MODELS or requested in _ANTIGRAVITY_CLI_MODEL_LABELS


def _load_settings(settings_path: str) -> dict:
  if not os.path.exists(settings_path):
    return {}
  try:
    with open(settings_path, "r", encoding="utf-8") as f:
      data = json.load(f)
  except json.JSONDecodeError as e:
    raise RuntimeError(f"Cannot update Antigravity settings; {settings_path} is invalid JSON") from e
  if not isinstance(data, dict):
    raise RuntimeError(f"Cannot update Antigravity settings; {settings_path} is not a JSON object")
  return data


def _write_settings(settings_path: str, settings: dict) -> None:
  os.makedirs(os.path.dirname(settings_path), exist_ok=True)
  temp_path = f"{settings_path}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
  try:
    with open(temp_path, "w", encoding="utf-8") as f:
      json.dump(settings, f, indent=2)
      f.write("\n")
    os.replace(temp_path, settings_path)
  finally:
    if os.path.exists(temp_path):
      os.remove(temp_path)


def _set_antigravity_model_setting(app_data_dir: str, model: str | None, reasoning) -> tuple[str, str]:
  settings_path = _settings_path(app_data_dir)
  cli_model = _antigravity_cli_model_label(model, reasoning)
  settings = _load_settings(settings_path)
  if settings.get("model") != cli_model:
    settings["model"] = cli_model
    _write_settings(settings_path, settings)
  return settings_path, cli_model


def _conversation_dirs(app_data_dir: str) -> set[str]:
  brain_dir = Path(app_data_dir) / "brain"
  if not brain_dir.exists():
    return set()
  return {path.name for path in brain_dir.iterdir() if path.is_dir()}


def _read_last_conversation_id(app_data_dir: str, workspace_dir: str) -> str | None:
  path = Path(app_data_dir) / "cache" / "last_conversations.json"
  if not path.exists():
    return None
  try:
    data = json.loads(path.read_text(encoding="utf-8"))
  except Exception:
    return None
  if not isinstance(data, dict):
    return None
  return data.get(workspace_dir) or data.get(os.path.abspath(workspace_dir))


def _transcript_paths(app_data_dir: str, conversation_id: str) -> list[Path]:
  logs_dir = Path(app_data_dir) / "brain" / conversation_id / ".system_generated" / "logs"
  return [
    logs_dir / "transcript_full.jsonl",
    logs_dir / "transcript.jsonl",
  ]


def _extract_model_text_from_transcript(transcript_text: str) -> str:
  model_texts: list[str] = []
  for line in transcript_text.splitlines():
    line = line.strip()
    if not line:
      continue
    try:
      item = json.loads(line)
    except json.JSONDecodeError:
      continue
    if not isinstance(item, dict):
      continue
    if item.get("source") != "MODEL":
      continue
    if item.get("status") not in (None, "DONE"):
      continue
    content = item.get("content")
    if isinstance(content, str) and content.strip():
      model_texts.append(content.strip())
  return model_texts[-1] if model_texts else ""


def _read_antigravity_transcript(app_data_dir: str, workspace_dir: str, run_id: str,
                                 previous_conversations: set[str]) -> tuple[str, str | None, str]:
  candidates: list[str] = []
  last_conversation_id = _read_last_conversation_id(app_data_dir, workspace_dir)
  if last_conversation_id:
    candidates.append(last_conversation_id)

  brain_dir = Path(app_data_dir) / "brain"
  if brain_dir.exists():
    current = [path for path in brain_dir.iterdir() if path.is_dir()]
    current.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    for path in current:
      if path.name not in candidates and path.name not in previous_conversations:
        candidates.append(path.name)
    for path in current:
      if path.name not in candidates:
        candidates.append(path.name)

  fallback: tuple[str, str | None, str] = ("", None, "")
  for conversation_id in candidates:
    for transcript_path in _transcript_paths(app_data_dir, conversation_id):
      if not transcript_path.exists():
        continue
      transcript_text = transcript_path.read_text(encoding="utf-8", errors="replace")
      extracted = _extract_model_text_from_transcript(transcript_text)
      if fallback[1] is None and extracted:
        fallback = (extracted, conversation_id, transcript_text)
      if run_id in transcript_text:
        return extracted, conversation_id, transcript_text
  return fallback


def _extract_stdout_text(stdout: str) -> str:
  return (stdout or "").strip()


def _split_env_paths(value: str) -> list[str]:
  paths: list[str] = []
  for piece in re.split(r"[;,]", value):
    piece = piece.strip().strip('"')
    if piece:
      paths.append(os.path.expanduser(piece))
  return paths


def _antigravity_token_files() -> list[str]:
  files: list[str] = []
  explicit_files = os.environ.get("ANTIGRAVITY_CLI_TOKEN_FILES")
  if explicit_files:
    files.extend(_split_env_paths(explicit_files))

  candidate_dirs = []
  explicit_dir = os.environ.get("ANTIGRAVITY_CLI_TOKEN_DIR")
  if explicit_dir:
    candidate_dirs.append(os.path.expanduser(explicit_dir))

  home = os.path.expanduser("~")
  candidate_dirs.extend([
    os.path.join(home, ".gemini", "antigravity-cli", "alt"),
    os.path.join(home, ".gemini", "antigravity-alt"),
  ])
  if os.name == "nt":
    candidate_dirs.append(r"C:\creds")

  for directory in candidate_dirs:
    if not os.path.isdir(directory):
      continue
    for pattern in ("*.txt", "*.token", "*.cred", "*.credential"):
      files.extend(str(path) for path in Path(directory).glob(pattern))
    for path in Path(directory).glob("*"):
      if path.is_dir():
        for name in ("antigravity_token.txt", "token.txt", "credential.txt"):
          token_path = path / name
          if token_path.exists():
            files.append(str(token_path))

  existing = []
  seen = set()
  for path in files:
    normalized = os.path.abspath(path)
    if normalized in seen or not os.path.exists(normalized):
      continue
    seen.add(normalized)
    existing.append(normalized)
  return existing


def _antigravity_state_path() -> str:
  return os.environ.get(
    "ANTIGRAVITY_CLI_ROLLOVER_STATE",
    os.path.join(_app_data_dir(), "llmbench_account_state.json"),
  )


def _load_rollover_state() -> dict:
  state_path = _antigravity_state_path()
  if not os.path.exists(state_path):
    return {}
  try:
    with open(state_path, "r", encoding="utf-8") as f:
      data = json.load(f)
    return data if isinstance(data, dict) else {}
  except Exception:
    return {}


def _save_rollover_state(state: dict) -> None:
  state_path = _antigravity_state_path()
  os.makedirs(os.path.dirname(state_path), exist_ok=True)
  with open(state_path, "w", encoding="utf-8") as f:
    json.dump(state, f, indent=2)


def _choose_next_token_file(token_files: list[str], state: dict) -> str | None:
  if not token_files:
    return None
  active = state.get("active_token_file")
  if active in token_files and len(token_files) > 1:
    index = token_files.index(active)
    return token_files[(index + 1) % len(token_files)]
  return random.choice(token_files)


def _write_windows_credential_from_token_file(token_file: str) -> None:
  if os.name != "nt":
    raise RuntimeError("Antigravity credential rollover is currently implemented only on Windows")

  script = r"""
param(
  [Parameter(Mandatory=$true)][string]$TokenFile,
  [Parameter(Mandatory=$true)][string]$VaultTarget,
  [Parameter(Mandatory=$true)][string]$VaultUser
)

$encrypted = Get-Content -LiteralPath $TokenFile -Raw | ConvertTo-SecureString
$bstr = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($encrypted)
try {
  $plainToken = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto($bstr)
  cmdkey /delete:$VaultTarget | Out-Null
  cmdkey /generic:$VaultTarget /user:$VaultUser /pass:$plainToken | Out-Null
  if ($LASTEXITCODE -ne 0) {
    throw "cmdkey failed with exit code $LASTEXITCODE"
  }
}
finally {
  if ($bstr -ne [IntPtr]::Zero) {
    [System.Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
  }
}
"""
  completed = subprocess.run([
    "powershell",
    "-NoProfile",
    "-ExecutionPolicy",
    "Bypass",
    "-Command",
    script,
    "-TokenFile",
    token_file,
    "-VaultTarget",
    _ANTIGRAVITY_CREDENTIAL_TARGET,
    "-VaultUser",
    _ANTIGRAVITY_CREDENTIAL_USER,
  ],
                             capture_output=True,
                             text=True,
                             timeout=60,
                             encoding="utf-8",
                             errors="replace")
  if completed.returncode != 0:
    raise RuntimeError(
      f"Failed to update Antigravity credential: {(completed.stderr or completed.stdout).strip()}"
    )


def _switch_antigravity_account() -> bool:
  token_files = _antigravity_token_files()
  if not token_files:
    return False

  state = _load_rollover_state()
  min_switch_seconds = int(os.environ.get("ANTIGRAVITY_CLI_MIN_SWITCH_SECONDS", "3600"))
  last_switch_time = float(state.get("last_switch_time") or 0)
  if state.get("active_token_file") and time.time() - last_switch_time < min_switch_seconds:
    print("Antigravity quota exhausted shortly after an account switch; not switching again yet.")
    return False

  token_file = _choose_next_token_file(token_files, state)
  if not token_file:
    return False

  print(f"Switching Antigravity credential to token file: {token_file}")
  _write_windows_credential_from_token_file(token_file)
  state["active_token_file"] = token_file
  state["last_switch_time"] = time.time()
  _save_rollover_state(state)
  return True


def _quota_sleep_seconds(text: str) -> int:
  match = re.search(r"(\d+)\s*h(?:ours?)?\s*(\d+)?\s*m?(?:in(?:utes?)?)?\s*(\d+)?\s*s?",
                    text,
                    re.IGNORECASE)
  if match:
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    return max(1, hours * 3600 + minutes * 60 + seconds)
  return 3600 * 1


def _looks_like_quota_exhausted(text: str) -> bool:
  lowered = text.lower()
  if "quota available" in lowered and not any(pattern in lowered for pattern in (
      "quota exhausted",
      "resource_exhausted",
      "resource exhausted",
      "quota exceeded",
      "exceeded your quota",
      "quota has been exceeded",
  )):
    return False
  patterns = [
    "quota exhausted",
    "resource_exhausted",
    "resource exhausted",
    "rate limit",
    "rate_limit",
    "too many requests",
    "429",
    "quota exceeded",
    "exceeded your quota",
    "quota has been exceeded",
    "free quota exhausted",
    "free quota exceeded",
    "usage limit exceeded",
    "daily limit exceeded",
    "limit reached",
  ]
  return any(pattern in lowered for pattern in patterns)


class AntigravityCliEngine:

  def __init__(self,
               model: str = "antigravity-cli",
               reasoning=False,
               tools=False,
               timeout: int = 3600 * 3,
               emit_meta: bool = False):
    self.model = model or "antigravity-cli"
    self.reasoning = reasoning
    self.tools = tools
    self.timeout = timeout
    self.emit_meta = emit_meta
    prompt_contract_version = "antigravity-cli-workspace-v3-result-file-types"
    self.configAndSettingsHash = hashlib.sha256(
      (self.model + "|" + _reasoning_label(reasoning) + "|" + str(tools) + "|" +
       prompt_contract_version).encode("utf-8")).hexdigest()

  def Available(self):
    return _find_antigravity_cli() is not None and _antigravity_cli_supports_model(self.model)

  def AIHook(self, prompt: str, structure: dict | None, context: dict | None = None):
    result = _antigravity_cli_ai_hook(prompt,
                                      structure,
                                      self.model,
                                      self.reasoning,
                                      self.tools,
                                      timeout_override=self.timeout,
                                      context=context)
    if not self.emit_meta and isinstance(result, tuple) and len(result) >= 2:
      return result[0], result[1]
    return result


def _antigravity_cli_ai_hook(prompt: str,
                             structure: dict | None,
                             model: str,
                             reasoning,
                             tools,
                             timeout_override: int | None = None,
                             context: dict | None = None):
  antigravity_path = _find_antigravity_cli()
  if not antigravity_path:
    raise RuntimeError("Antigravity CLI is not installed or not on PATH")

  app_data_dir = _app_data_dir()
  previous_conversations: set[str] = set()
  workspace_dir = create_workspace_dir("llmbench_antigravity")
  run_id = uuid.uuid4().hex
  try:
    workspace_paths = write_prompt_workspace(prompt, structure, workspace_dir)
    initial_files = snapshot_workspace_files(workspace_dir)
    prompt_input = _build_prompt(workspace_paths, structure, reasoning, tools, run_id)

    command = [
      antigravity_path,
      "--print",
      prompt_input,
      "--print-timeout",
      _duration_arg(timeout_override or 3600 * 3),
      "--add-dir",
      workspace_dir,
    ]
    if tools:
      command.append("--dangerously-skip-permissions")
    else:
      command.append("--sandbox")

    env = os.environ.copy()
    env.pop("GEMINI_API_KEY", None)
    env.pop("GOOGLE_API_KEY", None)

    os.makedirs(app_data_dir, exist_ok=True)
    lock_path = os.path.abspath(_settings_lock_path(app_data_dir))
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    with FileLock(lock_path, timeout=_settings_lock_timeout()):
      settings_path, cli_model = _set_antigravity_model_setting(app_data_dir, model, reasoning)
      previous_conversations = _conversation_dirs(app_data_dir)
      completed = subprocess.run(command,
                                 cwd=workspace_dir,
                                 capture_output=True,
                                 text=True,
                                 timeout=timeout_override or 3600 * 3,
                                 encoding="utf-8",
                                 errors="replace",
                                 env=env)

    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    transcript_text, conversation_id, raw_transcript = _read_antigravity_transcript(
      app_data_dir, workspace_dir, run_id, previous_conversations)
    cli_output_text = "\n".join(filter(None, (stdout, stderr, transcript_text))).strip()
    with open(str(workspace_paths["cli_output"]), "w", encoding="utf-8") as f:
      f.write(cli_output_text)

    if completed.returncode != 0:
      combined_error_text = "\n".join(filter(None, (stdout, stderr, transcript_text, raw_transcript)))
      if _looks_like_quota_exhausted(combined_error_text):
        if _switch_antigravity_account():
          return None
        sleep_seconds = _quota_sleep_seconds(combined_error_text)
        print(f"Antigravity quota exhausted, waiting {sleep_seconds} seconds...")
        time.sleep(sleep_seconds)
        return None
      print("Antigravity CLI command failed: ", command[:1] + ["--print", "<prompt>", *command[3:]])
      return "", (stderr or stdout or transcript_text or "antigravity CLI failed").strip()

    answer_json_text = read_text_file_if_exists(str(workspace_paths["answer_json"]))
    answer_txt_text = read_text_file_if_exists(str(workspace_paths["answer_txt"]))
    stdout_text = _extract_stdout_text(stdout)
    result_file_types = result_file_types_from_context(context)
    largest_answer_path = largest_new_file(
      workspace_dir,
      initial_files,
      exclude_paths=[str(workspace_paths["cli_output"])],
      output_text="\n".join(filter(None, (stdout, stderr, cli_output_text, transcript_text))),
      result_file_types=result_file_types)
    largest_answer_text = read_text_file_if_exists(
      largest_answer_path) if largest_answer_path else ""

    meta = {
      "backend": "antigravity-cli",
      "model": model,
      "cli_model": cli_model,
      "reasoning": reasoning,
      "tools": bool(tools),
      "settings_path": settings_path,
      "workspace_contract": "question-in-prompt + selected-created-file",
      "answer_file": "answer.json" if structure is not None else largest_answer_path,
      "result_file_types": result_file_types,
      "conversation_id": conversation_id,
      "stdout": stdout[-4000:],
      "stderr": stderr[-4000:],
      "cli_output": cli_output_text[-4000:],
      "transcript": transcript_text[-4000:],
    }

    if structure is not None:
      output_text = (answer_json_text or answer_txt_text or largest_answer_text or
                     transcript_text or stdout_text)
      try:
        return json.loads(output_text), "", meta
      except json.JSONDecodeError as e:
        raise RuntimeError(
          f"antigravity CLI returned invalid JSON: {e}: {output_text[:500]}") from e

    output_text = (
      largest_answer_text or answer_txt_text or answer_json_text or transcript_text or stdout_text)
    return output_text, "", meta
  finally:
    remove_workspace_dir(workspace_dir)
