from pathlib import Path

from LLMBenchCore.AiEngineCliWorkspace import largest_new_file, snapshot_workspace_files


def _write_text(path: Path, text: str) -> Path:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(text, encoding="utf-8")
  return path


def _write_bytes(path: Path, data: bytes) -> Path:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_bytes(data)
  return path


def test_prefers_only_mentioned_text_file_over_larger_executable(tmp_path):
  _write_text(tmp_path / "question.txt", "prompt")
  previous_files = snapshot_workspace_files(str(tmp_path))

  source_path = _write_text(tmp_path / "solution.py", "print('answer')\n")
  _write_bytes(tmp_path / "solution.exe", b"MZ" + (b"\x00" * 10000))

  selected = largest_new_file(str(tmp_path),
                              previous_files,
                              output_text="Wrote the final answer to solution.py")

  assert selected == str(source_path.resolve())


def test_prefers_largest_non_binary_mentioned_file(tmp_path):
  previous_files = snapshot_workspace_files(str(tmp_path))

  source_path = _write_text(tmp_path / "solver.cpp", "int main() { return 0; }\n")
  _write_bytes(tmp_path / "solver.exe", b"MZ" + (b"\x00" * 10000))
  _write_text(tmp_path / "notes.txt", "notes\n" * 1000)

  selected = largest_new_file(str(tmp_path),
                              previous_files,
                              output_text="Created solver.cpp and compiled solver.exe")

  assert selected == str(source_path.resolve())


def test_mentioned_executable_does_not_beat_unmentioned_text_source(tmp_path):
  previous_files = snapshot_workspace_files(str(tmp_path))

  source_path = _write_text(tmp_path / "solver.cpp", "int main() { return 0; }\n")
  _write_bytes(tmp_path / "solver.exe", b"MZ" + (b"\x00" * 10000))

  selected = largest_new_file(str(tmp_path),
                              previous_files,
                              output_text="Compilation succeeded and produced solver.exe")

  assert selected == str(source_path.resolve())


def test_ignores_larger_clear_input_data_when_no_file_is_mentioned(tmp_path):
  previous_files = snapshot_workspace_files(str(tmp_path))

  _write_text(tmp_path / "input.txt", "1 2 3\n" * 1000)
  answer_path = _write_text(tmp_path / "answer.py", "print(3)\n")

  selected = largest_new_file(str(tmp_path), previous_files)

  assert selected == str(answer_path.resolve())


def test_binary_fallback_skips_executables_and_honors_size_cap(tmp_path):
  previous_files = snapshot_workspace_files(str(tmp_path))

  image_path = _write_bytes(tmp_path / "plot.png", b"\x89PNG\r\n\x1a\n" + (b"\x00" * 100))
  _write_bytes(tmp_path / "tool.exe", b"MZ" + (b"\x00" * 1000))
  _write_bytes(tmp_path / "large.bin", b"\x00" * 200)

  selected = largest_new_file(str(tmp_path), previous_files, binary_size_cap=150)

  assert selected == str(image_path.resolve())
