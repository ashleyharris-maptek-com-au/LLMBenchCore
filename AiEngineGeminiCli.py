import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import random
from typing import Any

from .AiEngineCliWorkspace import (create_workspace_dir, largest_new_file, read_text_file_if_exists,
                                   remove_workspace_dir, snapshot_workspace_files,
                                   write_prompt_workspace)


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


def _gemini_thinking_budget(reasoning) -> int:
  if isinstance(reasoning, int) and reasoning > 0:
    if reasoning <= 3:
      return 128 * (2**(reasoning - 1))
    if reasoning <= 7:
      return 1024 * (2**(reasoning - 4))
    return min(8192 * (2**(reasoning - 7)), 24576)
  if reasoning is True:
    return 1024
  return 0


def _gemini_uses_thinking_level(model: str) -> bool:
  return model.startswith("gemini-3")


def _gemini_thinking_level(reasoning) -> str | None:
  if not reasoning:
    return None
  if isinstance(reasoning, int) and reasoning > 0 and reasoning <= 3:
    return "LOW"
  return "HIGH"


def _sanitize_alias_component(value: str) -> str:
  return re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()


def _gemini_alias_name(model: str, reasoning) -> str:
  return f"llmbench-{_sanitize_alias_component(model)}-{_reasoning_label(reasoning)}"


def _build_gemini_alias(model: str, reasoning) -> dict:
  thinking_config: dict[str, object] = {}
  if _gemini_uses_thinking_level(model):
    thinking_level = _gemini_thinking_level(reasoning)
    if thinking_level:
      thinking_config["thinkingLevel"] = thinking_level
      thinking_config["includeThoughts"] = True
  else:
    thinking_config["thinkingBudget"] = _gemini_thinking_budget(reasoning)
    if reasoning:
      thinking_config["includeThoughts"] = True

  generate_content_config = {
    "temperature": 0,
    "topP": 1,
  }
  if thinking_config:
    generate_content_config["thinkingConfig"] = thinking_config

  return {
    "modelConfig": {
      "model": model,
      "generateContentConfig": generate_content_config,
    }
  }


def _write_gemini_reasoning_alias(settings_path: str, alias_name: str, alias_config: dict) -> str:
  os.makedirs(os.path.dirname(settings_path), exist_ok=True)
  settings: dict[str, Any] = {}
  if os.path.exists(settings_path):
    with open(settings_path, "r", encoding="utf-8") as f:
      settings = json.load(f)

  model_configs = settings.setdefault("modelConfigs", {})
  custom_aliases = model_configs.setdefault("customAliases", {})
  if custom_aliases.get(alias_name) != alias_config:
    custom_aliases[alias_name] = alias_config
    with open(settings_path, "w", encoding="utf-8") as f:
      json.dump(settings, f, indent=2)

  return settings_path


def _ensure_gemini_reasoning_alias(model: str,
                                   reasoning,
                                   workspace_dir: str | None = None) -> tuple[str, str]:
  alias_name = _gemini_alias_name(model, reasoning)
  alias_config = _build_gemini_alias(model, reasoning)
  settings_dir = os.path.join(os.path.expanduser("~"), ".gemini")
  settings_path = os.path.join(settings_dir, "settings.json")

  try:
    return alias_name, _write_gemini_reasoning_alias(settings_path, alias_name, alias_config)
  except PermissionError:
    if not workspace_dir:
      raise

  workspace_settings_path = os.path.join(workspace_dir, ".gemini", "settings.json")
  return alias_name, _write_gemini_reasoning_alias(workspace_settings_path, alias_name,
                                                   alias_config)


def _build_prompt(workspace_paths: dict[str, object], structure: dict | None, tools) -> str:

  instructions = [str(workspace_paths.get("question_text", "")).strip()]

  image_records = workspace_paths.get("image_records", [])
  if image_records:
    instructions.append("See the following image files:")
    for record in image_records:
      if isinstance(record, dict) and record.get("local_ref"):
        instructions.append(f"@{record['local_ref']}")
  if structure is not None:
    instructions.append(
      "See the JSON schema in structure.json. Once you're happy with the final answer, put it in answer.json"
    )
  else:
    instructions.append("Keep the answer to a single file.")

  if tools:
    instructions.append(
      "You have access to all the tools on this machine and can compile and execute code.")

  return "\n\n".join(instructions)


def _extract_json_cli_text(value) -> str:
  if isinstance(value, str):
    return value
  if isinstance(value, dict):
    for key in ("response", "result", "text", "content", "message"):
      if key in value:
        extracted = _extract_json_cli_text(value[key])
        if extracted:
          return extracted
    return ""
  if isinstance(value, list):
    return "\n".join(filter(None, (_extract_json_cli_text(item) for item in value)))
  return ""


def _extract_stdout_text(stdout: str) -> str:
  text = stdout.strip()
  if not text:
    return ""
  try:
    return _extract_json_cli_text(json.loads(text)).strip()
  except Exception:
    return text


class GeminiCliEngine:

  @staticmethod
  def Available():
    return shutil.which("gemini") is not None

  def __init__(self,
               model: str,
               reasoning=False,
               tools=False,
               timeout: int = 3600 * 3,
               emit_meta: bool = False):
    self.model = model
    self.reasoning = reasoning
    self.tools = tools
    self.timeout = timeout
    self.emit_meta = emit_meta
    prompt_contract_version = "gemini-cli-workspace-v1"
    self.configAndSettingsHash = hashlib.sha256(
      (model + "|" + _reasoning_label(reasoning) + "|" + str(tools) + "|" +
       prompt_contract_version).encode("utf-8")).hexdigest()

  def AIHook(self, prompt: str, structure: dict | None):
    result = _gemini_cli_ai_hook(prompt,
                                 structure,
                                 self.model,
                                 self.reasoning,
                                 self.tools,
                                 timeout_override=self.timeout)
    if not self.emit_meta and isinstance(result, tuple) and len(result) >= 2:
      return result[0], result[1]
    return result


def _gemini_cli_ai_hook(prompt: str,
                        structure: dict | None,
                        model: str,
                        reasoning,
                        tools,
                        timeout_override: int | None = None):
  gemini_path = shutil.which("gemini")
  if not gemini_path:
    raise RuntimeError("gemini CLI is not installed or not on PATH")

  workspace_dir = create_workspace_dir("llmbench_gemini")
  try:
    workspace_paths = write_prompt_workspace(prompt, structure, workspace_dir)
    initial_files = snapshot_workspace_files(workspace_dir)
    cli_model, settings_path = _ensure_gemini_reasoning_alias(model, reasoning, workspace_dir)
    prompt_input = _build_prompt(workspace_paths, structure, tools)
    approval_mode = "yolo" if tools else "auto_edit"
    command = [
      gemini_path,
      "-m",
      cli_model,
      "--skip-trust",
      "--approval-mode",
      approval_mode,
      "--output-format",
      "json",
      "-p",
      prompt_input,
    ]

    env = os.environ.copy()
    env.pop("GEMINI_API_KEY", None)
    env.pop("GOOGLE_API_KEY", None)

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
    with open(str(workspace_paths["cli_output"]), "w", encoding="utf-8") as f:
      f.write(stdout)

    if completed.returncode != 0:
      print("Gemini CLI command failed: ", command)

      if "MODEL_CAPACITY_EXHAUSTED" in completed.stderr:
        print("Model capacity exhausted, waiting 5 minutes...")
        time.sleep(300)
        return None

      if "ModelNotFoundError" in completed.stderr:
        raise LookupError("Model is not supported when using Gemini via CLI")

      if "QUOTA_EXHAUSTED" in completed.stderr:

        googleAccountFile = os.path.expanduser("~/.gemini/google_accounts.json")
        oauthCredsFile = os.path.expanduser("~/.gemini/oauth_creds.json")

        altFolder = os.path.expanduser("~/.gemini/alt")

        gaLastModified = os.path.getmtime(googleAccountFile)

        if os.path.exists(altFolder):
          if gaLastModified < time.time() - 3600:
            print("Google account file is older than 1 hour, switching to alt account...")

            with open(googleAccountFile) as f:
              gaJson = json.load(f)

            activeAlt = gaJson["active"].replace("@gmail.com", "")
            altAccounts = os.listdir(altFolder)
            altAccounts.remove(activeAlt)
            altAccount = random.choice(altAccounts)
            print(f"Switching to alt account: {altAccount}")

            gaJson["active"] = altAccount + "@gmail.com"
            with open(googleAccountFile, "w") as f:
              json.dump(gaJson, f)

            # Copy new creds
            shutil.copy(os.path.join(altFolder, altAccount, "oauth_creds.json"), oauthCredsFile)

            print("Switched to alt account")
            return None

          print("Quota exhausted after switched to an alt, waiting 3 hours...")
          time.sleep(3600 * 3)
          return None

        # delay format is like 18h47m33s

        timeToWaitString = re.search(r'(\d+h)(\d+m)(\d+s)', completed.stderr)

        timeToWaitSeconds = 0
        if timeToWaitString:
          timeToWaitSeconds += int(timeToWaitString.group(1)[:-1]) * 3600
          timeToWaitSeconds += int(timeToWaitString.group(2)[:-1]) * 60
          timeToWaitSeconds += int(timeToWaitString.group(3)[:-1])
        else:
          # Gemini quota rolls over after 24 hours, so wait 5 hours? worst
          # case that means we have to repeat 5 tests.
          timeToWaitSeconds = 3600 * 5

        print(f"Quota exhausted, waiting {timeToWaitSeconds} seconds...")
        time.sleep(timeToWaitSeconds)
        return None

      return "", (stderr or stdout or "gemini CLI failed").strip()

    cli_output_text = read_text_file_if_exists(str(workspace_paths["cli_output"]))
    answer_json_text = read_text_file_if_exists(str(workspace_paths["answer_json"]))
    answer_txt_text = read_text_file_if_exists(str(workspace_paths["answer_txt"]))
    stdout_text = _extract_stdout_text(stdout)
    largest_answer_path = largest_new_file(workspace_dir,
                                           initial_files,
                                           exclude_paths=[str(workspace_paths["cli_output"])])
    largest_answer_text = read_text_file_if_exists(
      largest_answer_path) if largest_answer_path else ""

    meta = {
      "backend": "gemini-cli",
      "model": model,
      "cli_model": cli_model,
      "settings_path": settings_path,
      "reasoning": reasoning,
      "tools": bool(tools),
      "workspace_contract": "question-in-prompt + largest-created-file",
      "answer_file": "answer.json" if structure is not None else largest_answer_path,
      "stdout": stdout[-4000:],
      "stderr": stderr[-4000:],
      "cli_output": cli_output_text[-4000:],
    }

    if structure is not None:
      output_text = answer_json_text or answer_txt_text or largest_answer_text or stdout_text
      try:
        return json.loads(output_text), "", meta
      except json.JSONDecodeError as e:
        raise RuntimeError(f"gemini CLI returned invalid JSON: {e}: {output_text[:500]}") from e

    output_text = largest_answer_text or answer_txt_text or answer_json_text or stdout_text
    return output_text, "", meta
  finally:
    remove_workspace_dir(workspace_dir)
