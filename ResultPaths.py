import contextvars
import os
from urllib.parse import quote

RESULTS_ROOT = "results"
MODELS_DIR = os.path.join(RESULTS_ROOT, "models")
RESULTS_RAW_DIR = os.path.join(RESULTS_ROOT, "raw")
RESULTS_PROMPTS_DIR = os.path.join(RESULTS_ROOT, "prompts")
RESULTS_COT_DIR = os.path.join(RESULTS_ROOT, "cot")

_CURRENT_MODEL_NAME = contextvars.ContextVar("llmbenchcore_current_model_name", default=None)


def _sanitize_model_name(model_name: str) -> str:
  name = str(model_name).replace("\0", "")
  name = name.replace("/", "_").replace("\\", "_")
  return name


def model_dir_name(model_name: str) -> str:
  return _sanitize_model_name(model_name)


def model_root(model_name: str) -> str:
  return os.path.join(MODELS_DIR, model_dir_name(model_name))


def model_raw_dir(model_name: str) -> str:
  return os.path.join(model_root(model_name), "raw")


def model_prompt_dir(model_name: str) -> str:
  return os.path.join(model_root(model_name), "prompts")


def model_cot_dir(model_name: str) -> str:
  return os.path.join(model_root(model_name), "cot")


def model_artifact_dir(model_name: str | None = None) -> str:
  resolved = model_name if model_name is not None else get_current_model()
  if resolved is None:
    return RESULTS_ROOT
  return os.path.join(model_root(resolved), "artifacts")


def model_report_path(model_name: str) -> str:
  return os.path.join(model_root(model_name), "report.html")


def model_report_href(model_name: str) -> str:
  return "models/" + quote(model_dir_name(model_name), safe="") + "/report.html"


def model_report_href_with_anchor(model_name: str, anchor: str) -> str:
  return model_report_href(model_name) + "#" + anchor


def prompt_cache_filename(model_name: str, question_index: int, subpass: int) -> str:
  return f"{model_name}_{question_index}_{subpass}.txt"


def model_raw_path(model_name: str, question_index: int, subpass: int) -> str:
  return os.path.join(model_raw_dir(model_name), prompt_cache_filename(model_name, question_index,
                                                                       subpass))


def model_prompt_path(model_name: str, question_index: int, subpass: int) -> str:
  return os.path.join(model_prompt_dir(model_name),
                      prompt_cache_filename(model_name, question_index, subpass))


def model_cot_path(model_name: str, question_index: int, subpass: int) -> str:
  return os.path.join(model_cot_dir(model_name), prompt_cache_filename(model_name, question_index,
                                                                       subpass))


def legacy_raw_path(model_name: str, question_index: int, subpass: int) -> str:
  return os.path.join(RESULTS_RAW_DIR, prompt_cache_filename(model_name, question_index, subpass))


def legacy_prompt_path(model_name: str, question_index: int, subpass: int) -> str:
  return os.path.join(RESULTS_PROMPTS_DIR, prompt_cache_filename(model_name, question_index,
                                                                 subpass))


def legacy_cot_path(model_name: str, question_index: int, subpass: int) -> str:
  return os.path.join(RESULTS_COT_DIR, prompt_cache_filename(model_name, question_index, subpass))


def ensure_model_dirs(model_name: str) -> None:
  os.makedirs(model_raw_dir(model_name), exist_ok=True)
  os.makedirs(model_prompt_dir(model_name), exist_ok=True)
  os.makedirs(model_cot_dir(model_name), exist_ok=True)
  os.makedirs(model_artifact_dir(model_name), exist_ok=True)


def ensure_global_result_dirs() -> None:
  os.makedirs(RESULTS_ROOT, exist_ok=True)
  os.makedirs(RESULTS_RAW_DIR, exist_ok=True)
  os.makedirs(RESULTS_PROMPTS_DIR, exist_ok=True)
  os.makedirs(RESULTS_COT_DIR, exist_ok=True)
  os.makedirs(MODELS_DIR, exist_ok=True)


def set_current_model(model_name: str):
  return _CURRENT_MODEL_NAME.set(model_name)


def reset_current_model(token) -> None:
  _CURRENT_MODEL_NAME.reset(token)


def get_current_model() -> str | None:
  return _CURRENT_MODEL_NAME.get()


def model_artifact_path(filename: str, model_name: str | None = None) -> str:
  rel = str(filename).replace("\\", "/")
  if os.path.isabs(rel):
    return rel
  if rel.startswith("results/models/"):
    return rel
  if rel.startswith("results/"):
    rel = rel[len("results/"):]
  rel = rel.lstrip("/")
  path = os.path.join(model_artifact_dir(model_name), rel)
  parent = os.path.dirname(path)
  if parent:
    os.makedirs(parent, exist_ok=True)
  return path


def result_path(filename: str, model_name: str | None = None) -> str:
  """Task-facing alias for model artifact pathing."""
  return model_artifact_path(filename, model_name)


def relative_path_from_model_root(path: str, model_name: str) -> str:
  if not path:
    return path
  abs_path = os.path.abspath(path)
  abs_model_root = os.path.abspath(model_root(model_name))
  return os.path.relpath(abs_path, abs_model_root).replace("\\", "/")


def report_relpath(path: str, model_name: str | None = None) -> str:
  resolved = model_name if model_name is not None else get_current_model()
  if not resolved:
    return path
  return relative_path_from_model_root(path, resolved)


def convert_prompt_image_ref_for_model_report(ref: str, model_name: str) -> str:
  value = ref.strip()
  if value.startswith("results/"):
    return relative_path_from_model_root(value, model_name)
  if value.startswith("images/"):
    return relative_path_from_model_root(value, model_name)
  if value.startswith("../") or value.startswith("http://") or value.startswith("https://"):
    return value
  if os.path.isabs(value):
    return relative_path_from_model_root(value, model_name)
  return value
