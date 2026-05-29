import json
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

from . import PromptImageTagging as pit

_IMAGE_EXTENSIONS_BY_MIME_TYPE = {
  "image/jpeg": ".jpg",
  "image/png": ".png",
  "image/gif": ".gif",
  "image/webp": ".webp",
}

_BINARY_SAMPLE_BYTES = 8192
_BINARY_FALLBACK_MAX_BYTES = 5 * 1024 * 1024
_EXECUTABLE_EXTENSIONS = {
  ".a",
  ".dll",
  ".dylib",
  ".exe",
  ".lib",
  ".o",
  ".obj",
  ".out",
  ".so",
  ".wasm",
}
_EXECUTABLE_MAGIC_PREFIXES = (
  b"MZ",
  b"\x7fELF",
  b"\xcf\xfa\xed\xfe",
  b"\xce\xfa\xed\xfe",
  b"\xfe\xed\xfa\xcf",
  b"\xfe\xed\xfa\xce",
  b"\x00asm",
)
_TEXT_BOM_PREFIXES = (
  b"\xef\xbb\xbf",
  b"\xff\xfe",
  b"\xfe\xff",
  b"\xff\xfe\x00\x00",
  b"\x00\x00\xfe\xff",
)
_INPUT_DATA_PARTS = {
  "data",
  "dataset",
  "datasets",
  "fixture",
  "fixtures",
  "image",
  "images",
  "input",
  "inputs",
  "reference",
  "reference_images",
  "test_data",
  "test_inputs",
  "testcases",
  "test_cases",
}
_INPUT_DATA_STEMS = {
  "data",
  "dataset",
  "expected",
  "fixture",
  "fixtures",
  "input",
  "inputs",
  "prompt",
  "question",
  "reference",
  "sample_input",
  "stdin",
  "test_input",
  "testcase",
  "testcases",
}


@dataclass(frozen=True)
class _WorkspaceFileCandidate:
  size: int
  resolved: str
  relative: str
  name: str
  is_binary: bool
  is_executable: bool
  is_input_like: bool
  is_mentioned: bool


def create_workspace_dir(prefix: str) -> str:
  parent_dir = os.getcwd()
  for _ in range(100):
    workspace_dir = os.path.join(parent_dir, f"{prefix}_{uuid.uuid4().hex}")
    try:
      os.mkdir(workspace_dir)
      return os.path.abspath(workspace_dir)
    except FileExistsError:
      continue
  raise RuntimeError(f"Failed to create a unique {prefix} workspace directory")


def remove_workspace_dir(workspace_dir: str) -> None:
  shutil.rmtree(workspace_dir, ignore_errors=True)


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


def write_prompt_workspace(prompt: str, structure: dict | None,
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
    "cli_output": os.path.join(workspace_dir, "cli_output.txt"),
    "image_paths": [],
    "image_records": image_records,
    "question_text": question_text,
  }

  with open(paths["question"], "w", encoding="utf-8") as f:
    f.write(question_text + "\n")

  if structure is not None:
    paths["structure"] = os.path.join(workspace_dir, "structure.json")
    with open(paths["structure"], "w", encoding="utf-8") as f:
      json.dump(structure, f, indent=2, sort_keys=True)

  with open(paths["answer_json"], "w", encoding="utf-8"):
    pass
  with open(paths["answer_txt"], "w", encoding="utf-8"):
    pass

  if image_records:
    paths["image_paths"] = [record["path"] for record in image_records]
    paths["images"] = os.path.join(workspace_dir, "images.json")
    with open(paths["images"], "w", encoding="utf-8") as f:
      json.dump(image_records, f, indent=2, sort_keys=True)

  return paths


def read_text_file_if_exists(path: str) -> str:
  if not os.path.exists(path):
    return ""
  with open(path, "r", encoding="utf-8", errors="ignore") as f:
    return f.read().strip()


def snapshot_workspace_files(workspace_dir: str) -> set[str]:
  return {str(path.resolve()) for path in Path(workspace_dir).rglob("*") if path.is_file()}


def _normalized_name(value: str) -> str:
  return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _looks_like_executable_machine_code(path: Path, sample: bytes) -> bool:
  if path.suffix.lower() in _EXECUTABLE_EXTENSIONS:
    return True
  return any(sample.startswith(prefix) for prefix in _EXECUTABLE_MAGIC_PREFIXES)


def _classify_file(path: Path) -> tuple[bool, bool]:
  try:
    with open(path, "rb") as f:
      sample = f.read(_BINARY_SAMPLE_BYTES)
  except OSError:
    return True, False

  if not sample:
    return False, False

  is_executable = _looks_like_executable_machine_code(path, sample)
  if is_executable:
    return True, True

  if any(sample.startswith(prefix) for prefix in _TEXT_BOM_PREFIXES):
    return False, False

  if b"\x00" in sample:
    return True, False

  try:
    sample.decode("utf-8")
  except UnicodeDecodeError:
    return True, False

  control_count = sum(1 for byte in sample if byte < 32 and byte not in b"\t\n\r\f\b")
  return control_count / len(sample) > 0.30, False


def _looks_like_input_data(workspace_root: Path, path: Path) -> bool:
  try:
    relative_path = path.relative_to(workspace_root)
  except ValueError:
    relative_path = path

  parts = [_normalized_name(part) for part in relative_path.parts]
  if any(part in _INPUT_DATA_PARTS for part in parts[:-1]):
    return True

  stem = _normalized_name(path.stem)
  if stem in _INPUT_DATA_STEMS:
    return True

  return any(
    stem.endswith(suffix)
    for suffix in (
      "_data",
      "_dataset",
      "_expected",
      "_fixture",
      "_fixtures",
      "_input",
      "_inputs",
      "_reference",
      "_testcase",
      "_testcases",
    )
  )


def _text_mentions_file(output_text: str, workspace_root: Path, path: Path) -> bool:
  if not output_text:
    return False

  haystack = output_text.lower().replace("\\", "/")
  terms = {path.name.lower()}

  try:
    terms.add(path.relative_to(workspace_root).as_posix().lower())
  except ValueError:
    pass

  terms.add(str(path.resolve()).lower().replace("\\", "/"))

  for term in sorted((term for term in terms if len(term) >= 3), key=len, reverse=True):
    if "/" in term:
      if term in haystack:
        return True
      continue

    if re.search(r"(?<![a-z0-9_.-])" + re.escape(term) + r"(?![a-z0-9_.-])", haystack):
      return True

  return False


def _largest_candidate(candidates: list[_WorkspaceFileCandidate]) -> _WorkspaceFileCandidate:
  return max(candidates, key=lambda candidate: (candidate.size, candidate.resolved))


def largest_new_file(workspace_dir: str,
                     previous_files: set[str],
                     exclude_paths: list[str] | None = None,
                     output_text: str = "",
                     binary_size_cap: int = _BINARY_FALLBACK_MAX_BYTES) -> str | None:
  previous = {str(Path(path).resolve()) for path in previous_files}
  excluded = {str(Path(path).resolve()) for path in (exclude_paths or [])}
  workspace_root = Path(workspace_dir).resolve()
  candidates: list[_WorkspaceFileCandidate] = []

  for path in workspace_root.rglob("*"):
    if not path.is_file():
      continue

    resolved = str(path.resolve())
    if resolved in previous or resolved in excluded:
      continue

    try:
      size = path.stat().st_size
    except OSError:
      continue

    is_binary, is_executable = _classify_file(path)
    try:
      relative = path.relative_to(workspace_root).as_posix()
    except ValueError:
      relative = path.name
    candidates.append(
      _WorkspaceFileCandidate(size=size,
                              resolved=resolved,
                              relative=relative,
                              name=path.name,
                              is_binary=is_binary,
                              is_executable=is_executable,
                              is_input_like=_looks_like_input_data(workspace_root, path),
                              is_mentioned=_text_mentions_file(output_text, workspace_root, path)))

  if not candidates:
    return None

  mentioned = [
    candidate for candidate in candidates
    if candidate.is_mentioned and not candidate.is_input_like and not candidate.is_executable
  ]
  if len(mentioned) == 1 and (not mentioned[0].is_binary
                              or mentioned[0].size <= binary_size_cap):
    return mentioned[0].resolved

  mentioned_text = [candidate for candidate in mentioned if not candidate.is_binary]
  if mentioned_text:
    return _largest_candidate(mentioned_text).resolved

  text_candidates = [
    candidate for candidate in candidates
    if not candidate.is_binary and not candidate.is_input_like
  ]
  if text_candidates:
    return _largest_candidate(text_candidates).resolved

  binary_candidates = [
    candidate for candidate in candidates
    if candidate.is_binary and not candidate.is_executable and candidate.size <= binary_size_cap
  ]
  if binary_candidates:
    return _largest_candidate(binary_candidates).resolved

  return None
