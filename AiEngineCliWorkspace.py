import json
import os
import shutil
import uuid
from urllib.request import Request, urlopen

from . import PromptImageTagging as pit


_IMAGE_EXTENSIONS_BY_MIME_TYPE = {
  "image/jpeg": ".jpg",
  "image/png": ".png",
  "image/gif": ".gif",
  "image/webp": ".webp",
}


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

  paths = {
    "question": os.path.join(workspace_dir, "question.txt"),
    "answer_json": os.path.join(workspace_dir, "answer.json"),
    "answer_txt": os.path.join(workspace_dir, "answer.txt"),
    "cli_output": os.path.join(workspace_dir, "cli_output.txt"),
    "image_paths": [],
    "image_records": image_records,
  }

  with open(paths["question"], "w", encoding="utf-8") as f:
    f.write("".join(question_parts))

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
  with open(path, "r", encoding="utf-8") as f:
    return f.read().strip()
