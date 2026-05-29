import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timedelta

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


def _build_prompt(workspace_paths: dict[str, object], structure: dict | None, tools) -> str:
  instructions = [
    "You are answering a benchmark prompt.",
    "Work only in the current directory.",
    "A copy of the benchmark question is available in @question.txt.",
  ]

  image_records = workspace_paths.get("image_records", [])
  if image_records:
    instructions.append("The prompt has these attached image files:")
    for record in image_records:
      if isinstance(record, dict) and record.get("local_ref"):
        instructions.append(f"@{record['local_ref']}")

  if structure is not None:
    instructions.append(
      "Read the required JSON schema from @structure.json and write the final valid JSON answer to answer.json."
    )
  else:
    instructions.append("Keep your answer in a single file.")

  if tools:
    instructions.append(
      "Tool access is allowed. Use tools when they materially improve the answer.")

  instructions.append("Benchmark question:")
  instructions.append(str(workspace_paths.get("question_text", "")).strip())
  return "\n\n".join(instructions)


def _extract_json_cli_text(value):
  if isinstance(value, str):
    return value
  if isinstance(value, dict):
    if isinstance(value.get("result"), dict):
      return json.dumps(value["result"])
    for key in ("result", "response", "text", "content", "message"):
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


def _claude_rate_limit_sleep_seconds(stdout: str) -> tuple[float | None, str | None]:
  reset_marker = "resets"
  reset_index = stdout.find(reset_marker)
  if reset_index < 0:
    return None, None
  remainder = stdout[reset_index + len(reset_marker):]
  open_paren_index = remainder.find("(")
  if open_paren_index < 0:
    return None, None
  reset_text = remainder[:open_paren_index].strip(" .·\n\r\t")
  if not reset_text:
    return None, None
  normalized = " ".join(reset_text.lower().split())
  for fmt in ("%I:%M%p", "%I%p", "%H:%M", "%H"):
    try:
      parsed_time = datetime.strptime(normalized, fmt).time()
      now = datetime.now().astimezone()
      target = now.replace(hour=parsed_time.hour,
                           minute=parsed_time.minute,
                           second=0,
                           microsecond=0)
      if target <= now:
        target += timedelta(days=1)
      target += timedelta(minutes=5)
      return max(0.0, (target - now).total_seconds()), reset_text
    except ValueError:
      continue
  return None, reset_text


class ClaudeCliEngine:

  @staticmethod
  def Available():
    return shutil.which("claude") is not None

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
    prompt_contract_version = "claude-cli-workspace-v1"
    self.configAndSettingsHash = hashlib.sha256(
      (model + "|" + _reasoning_label(reasoning) + "|" + str(tools) + "|" +
       prompt_contract_version).encode("utf-8")).hexdigest()

  def AIHook(self, prompt: str, structure: dict | None):
    result = _claude_cli_ai_hook(prompt,
                                 structure,
                                 self.model,
                                 self.reasoning,
                                 self.tools,
                                 timeout_override=self.timeout)
    if not self.emit_meta and isinstance(result, tuple) and len(result) >= 2:
      return result[0], result[1]
    return result


def _claude_cli_ai_hook(prompt: str,
                        structure: dict | None,
                        model: str,
                        reasoning,
                        tools,
                        timeout_override: int | None = None):
  claude_path = shutil.which("claude")
  if not claude_path:
    raise RuntimeError("claude CLI is not installed or not on PATH")

  workspace_dir = create_workspace_dir("llmbench_claude")
  try:
    workspace_paths = write_prompt_workspace(prompt, structure, workspace_dir)
    initial_files = snapshot_workspace_files(workspace_dir)
    prompt_input = _build_prompt(workspace_paths, structure, tools)
    permission_mode = "bypassPermissions" if tools else "acceptEdits"
    command = [
      claude_path,
      "--print",
      "--model",
      model,
      "--permission-mode",
      permission_mode,
      "--no-session-persistence",
      "--output-format",
      "json",
    ]

    if tools:
      command.append("--dangerously-skip-permissions")

    reasoning_label = _reasoning_label(reasoning)
    if reasoning_label != "none":
      command.extend(["--effort", reasoning_label])

    command.append(prompt_input)

    env = os.environ.copy()
    env.pop("ANTHROPIC_API_KEY", None)  # Don't pay per call.
    env["CLAUDE_CODE_SKIP_PROMPT_HISTORY"] = "1"
    env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = "64000"

    completed = subprocess.run(command,
                               cwd=workspace_dir,
                               capture_output=True,
                               text=True,
                               input=prompt_input,
                               timeout=timeout_override or 3600 * 3,
                               encoding="utf-8",
                               errors="replace",
                               env=env)

    stdout = completed.stdout or ""
    stderr = completed.stderr or ""

    if completed.returncode == 1 and stderr == "":
      print("Calude CLI command failed: ", command)
      if '"api_error_status":429' in stdout or "You've hit your limit" in stdout:
        sleep_seconds, reset_text = _claude_rate_limit_sleep_seconds(stdout)
        if sleep_seconds is not None:
          print(
            f"Rate limited, waiting until 5 minutes after local reset time {reset_text} ({sleep_seconds / 60:.1f} minutes)..."
          )
          time.sleep(sleep_seconds)
        else:
          print("Rate limited, waiting 5 hours...")
          time.sleep(3600 * 5)
        print("End of rate limited wait!")
        return None
      if "API Error: 529 Overloaded" in stdout:
        print("Overloaded, waiting 10 minutes...")
        time.sleep(600)
        return None

    with open(str(workspace_paths["cli_output"]), "w", encoding="utf-8") as f:
      f.write(stdout)

    if completed.returncode != 0:
      return "", (stderr or stdout or "claude CLI failed").strip()

    cli_output_text = read_text_file_if_exists(str(workspace_paths["cli_output"]))
    answer_json_text = read_text_file_if_exists(str(workspace_paths["answer_json"]))
    answer_txt_text = read_text_file_if_exists(str(workspace_paths["answer_txt"]))
    stdout_text = _extract_stdout_text(stdout)
    largest_answer_path = largest_new_file(
      workspace_dir,
      initial_files,
      exclude_paths=[str(workspace_paths["cli_output"])],
      output_text="\n".join(filter(None, (stdout, stderr, cli_output_text, stdout_text))))
    largest_answer_text = read_text_file_if_exists(
      largest_answer_path) if largest_answer_path else ""

    meta = {
      "backend": "claude-cli",
      "model": model,
      "reasoning": reasoning,
      "tools": bool(tools),
      "workspace_contract": "question-in-prompt + selected-created-file",
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
        raise RuntimeError(f"claude CLI returned invalid JSON: {e}: {output_text[:500]}") from e

    output_text = largest_answer_text or answer_txt_text or answer_json_text or stdout_text
    return output_text, "", meta
  finally:
    remove_workspace_dir(workspace_dir)
