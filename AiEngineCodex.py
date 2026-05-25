import hashlib
import json
import os
import shutil
import subprocess
import uuid
from urllib.request import Request, urlopen

from . import PromptImageTagging as pit
from .AiEngineCliWorkspace import largest_new_file, snapshot_workspace_files

_IMAGE_EXTENSIONS_BY_MIME_TYPE = {
  "image/jpeg": ".jpg",
  "image/png": ".png",
  "image/gif": ".gif",
  "image/webp": ".webp",
}


def _reasoning_label(reasoning) -> str:
  if isinstance(reasoning, int) and reasoning > 0:
    if reasoning <= 3:
      return "low"
    if reasoning >= 9:
      return "xhigh"
    if reasoning > 6:
      return "high"
    return "medium"
  if reasoning is True:
    return "medium"
  return "none"


def _build_prompt(workspace_dir: str, structure: dict | None, reasoning, tools) -> str:
  instructions = [
    "You are answering a benchmark prompt.",
    f"Work only in this directory: {workspace_dir}",
    "A copy of the benchmark question is available in question.txt.",
    "If an images directory exists, use the image files there as the prompt's attached images. The image references in question.txt point to those local files.",
  ]

  if structure is not None:
    instructions.append(
      "The required JSON schema is in structure.json. Once you're happy with the final answer, put it in answer.json"
    )
  else:
    instructions.append("Once you've developed your answer, keep your answer in a single file.")

  if tools:
    instructions.append(
      "You have access to all the tools on this machine and can compile and execute code.")

  instructions.append("Benchmark question:")
  instructions.append(read_question_text(workspace_dir))
  return "\n\n".join(instructions)


def _image_extension_for_mime_type(mime_type: str) -> str:
  try:
    return _IMAGE_EXTENSIONS_BY_MIME_TYPE[mime_type.lower()]
  except KeyError as e:
    raise ValueError(f"Unsupported image MIME type: {mime_type}") from e


def _write_image_bytes(image_dir: str, image_index: int, image_bytes: bytes,
                       mime_type: str) -> tuple[str, str]:
  extension = _image_extension_for_mime_type(mime_type)
  filename = f"image_{image_index:02d}{extension}"
  target_path = os.path.join(image_dir, filename)
  with open(target_path, "wb") as f:
    f.write(image_bytes)
  return f"images/{filename}", target_path


def _copy_local_prompt_image(ref: str, image_dir: str, image_index: int) -> tuple[str, str, str]:
  source_path = pit.resolve_local_path(ref)
  mime_type = pit.guess_image_mime_type_from_path(source_path)
  extension = _image_extension_for_mime_type(mime_type)
  filename = f"image_{image_index:02d}{extension}"
  target_path = os.path.join(image_dir, filename)
  shutil.copyfile(source_path, target_path)
  return f"images/{filename}", target_path, mime_type


def _download_prompt_image(ref: str, image_dir: str, image_index: int) -> tuple[str, str, str]:
  request = Request(ref, headers={"User-Agent": "LLMBenchCore/1.0"})
  with urlopen(request, timeout=30) as response:
    content_type = response.headers.get("Content-Type")
    image_bytes = response.read()

  mime_type = None
  if content_type:
    mime_type = content_type.split(";", 1)[0].strip().lower()
  if mime_type not in _IMAGE_EXTENSIONS_BY_MIME_TYPE:
    mime_type = pit.guess_image_mime_type_from_ref(ref)

  local_ref, target_path = _write_image_bytes(image_dir, image_index, image_bytes, mime_type)
  return local_ref, target_path, mime_type


def _materialize_prompt_image(ref: str, image_dir: str, image_index: int) -> tuple[str, str, str]:
  if pit.is_data_uri(ref):
    mime_type, image_bytes = pit.decode_data_uri(ref)
    local_ref, target_path = _write_image_bytes(image_dir, image_index, image_bytes, mime_type)
    return local_ref, target_path, mime_type
  if pit.is_url(ref):
    return _download_prompt_image(ref, image_dir, image_index)
  return _copy_local_prompt_image(ref, image_dir, image_index)


def _write_prompt_workspace(prompt: str, structure: dict | None,
                            workspace_dir: str) -> dict[str, object]:
  prompt_parts = pit.parse_prompt_parts(prompt)
  has_images = any(part_type == "image" for part_type, _ in prompt_parts)
  image_records = []
  question_parts = []

  if has_images:
    image_dir = os.path.join(workspace_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    image_index = 1

    for part_type, part_value in prompt_parts:
      if part_type == "text":
        question_parts.append(part_value)
      elif part_type == "image":
        local_ref, target_path, mime_type = _materialize_prompt_image(part_value, image_dir,
                                                                      image_index)
        question_parts.append(f"[[image:{local_ref}]]")
        image_records.append({
          "original_ref": part_value,
          "local_ref": local_ref,
          "path": os.path.abspath(target_path),
          "mime_type": mime_type,
        })
        image_index += 1
  else:
    question_parts.append(prompt)

  question_text = "".join(question_parts).strip()

  paths = {
    "question": os.path.join(workspace_dir, "question.txt"),
    "answer_json": os.path.join(workspace_dir, "answer.json"),
    "answer_txt": os.path.join(workspace_dir, "answer.txt"),
    "codex_output": os.path.join(workspace_dir, "codex_output.txt"),
    "image_paths": [],
    "question_text": question_text,
  }

  with open(paths["question"], "w", encoding="utf-8") as f:
    f.write(question_text)

  if structure is not None:
    paths["structure"] = os.path.join(workspace_dir, "structure.json")
    with open(paths["structure"], "w", encoding="utf-8") as f:
      json.dump(structure, f, indent=2, sort_keys=True)

  if structure:
    with open(paths["answer_json"], "w", encoding="utf-8"):
      pass
  else:
    with open(paths["answer_txt"], "w", encoding="utf-8"):
      pass

  if image_records:
    paths["image_paths"] = [record["path"] for record in image_records]
    paths["images"] = os.path.join(workspace_dir, "images.json")
    with open(paths["images"], "w", encoding="utf-8") as f:
      json.dump(image_records, f, indent=2, sort_keys=True)

  return paths


def _read_text_file_if_exists(path: str) -> str:
  if not os.path.exists(path):
    return ""
  with open(path, "r", encoding="utf-8") as f:
    return f.read().strip()


def read_question_text(workspace_dir: str) -> str:
  return _read_text_file_if_exists(os.path.join(workspace_dir, "question.txt"))


def _create_codex_workspace_dir() -> str:
  parent_dir = os.getcwd()
  for _ in range(100):
    workspace_dir = os.path.join(parent_dir, f"llmbench_codex_{uuid.uuid4().hex}")
    try:
      os.mkdir(workspace_dir)
      return os.path.abspath(workspace_dir)
    except FileExistsError:
      continue
  raise RuntimeError("Failed to create a unique Codex workspace directory")


class OpenAIEngineCodex:

  @staticmethod
  def Available():
    return shutil_which("codex") is not None

  def __init__(self,
               model: str,
               reasoning=False,
               tools=False,
               timeout: int = 3600,
               max_output_tokens: int | None = None,
               temperature: float | None = None,
               emit_meta: bool = False):
    self.model = model
    self.reasoning = reasoning
    self.tools = tools
    self.timeout = timeout
    self.max_output_tokens = max_output_tokens
    self.temperature = temperature
    self.emit_meta = emit_meta
    effective_reasoning_label = _reasoning_label(reasoning)
    prompt_contract_version = "codex-cli-v5-temp-workspace"
    self.configAndSettingsHash = hashlib.sha256(
      (model + "|" + effective_reasoning_label + "|" + str(tools) + "|" +
       prompt_contract_version).encode("utf-8")).hexdigest()

  def AIHook(self, prompt: str, structure: dict | None):
    result = _codex_ai_hook(prompt,
                            structure,
                            self.model,
                            self.reasoning,
                            self.tools,
                            timeout_override=self.timeout)
    if not self.emit_meta and isinstance(result, tuple) and len(result) >= 2:
      return result[0], result[1]
    return result


def shutil_which(executable: str):
  import shutil
  return shutil.which(executable)


def _codex_ai_hook(prompt: str,
                   structure: dict | None,
                   model: str,
                   reasoning,
                   tools,
                   timeout_override: int | None = None):
  codex_path = shutil_which("codex")
  if not codex_path:
    raise RuntimeError("codex CLI is not installed or not on PATH")

  workspace_dir = _create_codex_workspace_dir()
  try:
    workspace_paths = _write_prompt_workspace(prompt, structure, workspace_dir)
    initial_files = snapshot_workspace_files(workspace_dir)
    prompt_input = _build_prompt(workspace_dir, structure, reasoning, tools)

    sandbox_mode = "workspace-write" if not tools else "danger-full-access"
    image_args = []
    for image_path in workspace_paths.get("image_paths", []):
      image_args.extend(["-i", str(image_path)])

    command = [
      codex_path,
      "exec",
      "-m",
      model,
      *image_args,
      "-c",
      "model_reasoning_effort=" + str(_reasoning_label(reasoning)),
      "--disable",
      "fast_mode",
      "-C",
      workspace_dir,
      "-s",
      sandbox_mode,
      "--ephemeral",
      "--skip-git-repo-check",
      "-o",
      str(workspace_paths["codex_output"]),
      "-",
    ]

    # Don't pass the API key into codex, as we don't want it to run at full API
    # costs while pretending to be a subscription service.
    e = os.environ.copy()
    e.pop("OPENAI_API_KEY", None)

    completed = subprocess.run(command,
                               input=prompt_input,
                               capture_output=True,
                               text=True,
                               timeout=timeout_override or 3600 * 3,
                               encoding="utf-8",
                               errors="replace",
                               env=e)

    stdout = completed.stdout or ""
    stderr = completed.stderr or ""

    if "model is not supported when using Codex with a ChatGPT account" in stdout + stderr:
      raise LookupError("Model is not supported when using Codex with a ChatGPT account")

    if completed.returncode != 0:
      print("Codex CLI command failed: ", command)
      raise RuntimeError((stderr or stdout or "codex exec failed").strip())

    codex_output_text = _read_text_file_if_exists(workspace_paths["codex_output"])
    answer_json_text = _read_text_file_if_exists(workspace_paths["answer_json"])
    answer_txt_text = _read_text_file_if_exists(workspace_paths["answer_txt"])
    largest_answer_path = largest_new_file(
      workspace_dir,
      initial_files,
      exclude_paths=[str(workspace_paths["codex_output"])]
    )
    largest_answer_text = _read_text_file_if_exists(largest_answer_path) if largest_answer_path else ""

    meta = {
      "backend": "codex-cli",
      "model": model,
      "reasoning": reasoning,
      "tools": bool(tools),
      "workspace_contract": "question-in-prompt + largest-created-file",
      "answer_file": "answer.json" if structure is not None else largest_answer_path,
      "stdout": stdout[-4000:],
      "stderr": stderr[-4000:],
      "codex_output": codex_output_text[-4000:],
    }

    if structure is not None:
      output_text = answer_json_text or answer_txt_text or largest_answer_text or codex_output_text or stdout.strip()
      try:
        return json.loads(output_text), "", meta
      except json.JSONDecodeError as e:
        raise RuntimeError(f"codex returned invalid JSON: {e}: {output_text[:500]}") from e

    output_text = largest_answer_text or answer_txt_text or answer_json_text or codex_output_text or stdout.strip()
    return output_text, "", meta
  finally:
    shutil.rmtree(workspace_dir, ignore_errors=True)
