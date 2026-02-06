import os
import subprocess
import sys
from dataclasses import dataclass

from .ResultPaths import model_artifact_dir, result_path


def file_content_changed(filepath: str, new_content: str) -> bool:
  """Return True when a file is missing or its text content differs."""
  if not os.path.exists(filepath):
    return True
  try:
    with open(filepath, "r", encoding="utf-8") as f:
      return f.read() != new_content
  except Exception:
    return True


def write_if_changed(filepath: str, content: str) -> bool:
  """Write text only if changed. Returns True when a write occurred."""
  if file_content_changed(filepath, content):
    parent = os.path.dirname(filepath)
    if parent:
      os.makedirs(parent, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
      f.write(content)
    return True
  return False


@dataclass(frozen=True)
class ModelArtifactStore:
  """
  Small helper for model-scoped artifacts.

  All files are rooted at `results/models/<model>/artifacts`.
  """
  model_name: str

  def __post_init__(self):
    os.makedirs(model_artifact_dir(self.model_name), exist_ok=True)

  @property
  def root(self) -> str:
    return model_artifact_dir(self.model_name)

  def path(self, filename: str) -> str:
    return result_path(filename, self.model_name)

  def part_name(self, question_num: int, part_index: int) -> str:
    return f"{question_num}_{part_index}_{self.model_name}"

  def run_python_script_to_file(self,
                                script_path: str,
                                output_path: str,
                                python_executable: str | None = None) -> None:
    exe = python_executable or sys.executable
    with open(output_path, "w", encoding="utf-8") as f:
      subprocess.run([exe, os.path.basename(script_path)], cwd=self.root, stdout=f, check=False)


ArtifactStore = ModelArtifactStore
