"""look through 'results' and identify:
- any models that are missing a report.html
- any report.html that is over 100mb
- any report.html with dead image links.
- any report.html that doesn't end with the "Total Tests" / "Overall Score" footer.
- any empty outputs /raw/*.txt
- any __exception__ in a /raw/*.txt
- any scores > 100%
- any report.html that has less than the expected number of questions
- any question/subpass that no model scored > 0.

Outputs a report with:
- list of issues.
- for single question&model issues:
  - a copy/paste command to run just that question/model combination again.
- for entire model issues:
  - a copy/paste command to run all questions for that model again.


Note this runs in the root directory of the benchmark not LLMBenchCore itself.

"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

MAX_REPORT_BYTES = 100 * 1024 * 1024
PLACEBO_MODELS = {"naive", "naive-optimised", "best-published", "random", "human"}
INTENTIONAL_PARTIAL_REPORT_FILENAMES = ("report.partial.html", "report-partial.html")
IMAGE_SRC_RE = re.compile(r"<img\b[^>]*\bsrc=['\"]([^'\"]+)['\"]", re.IGNORECASE)
QUESTION_ANCHOR_RE = re.compile(r"name=['\"]q(\d+)['\"]", re.IGNORECASE)
OVERALL_SUMMARY_RE = re.compile(
  r"<h2>Overall Summary</h2>.*?<tr[^>]*>.*?<td>(\d+)</td>.*?<td>([^<]+)</td>.*?<td>([0-9.]+)%</td>",
  re.IGNORECASE | re.DOTALL)
RAW_FILE_RE = re.compile(r"^(?P<model>.+)_(?P<question>\d+)_(?P<subpass>\d+)$")
BENCHMARK_ENTRY_CALL_RE = re.compile(r"run_benchmark_main\s*\([^\n]*__file__")


@dataclass(frozen=True)
class Issue:
  severity: str
  kind: str
  message: str
  model: str | None = None
  question: int | None = None
  subpass: int | None = None
  rerun_command: str | None = None


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", default=".")
  parser.add_argument("--min-systemic-zero-attempts", type=int, default=3)
  parser.add_argument("--show-info-details", action="store_true")
  return parser.parse_args()


def discover_contiguous_test_indices(root: Path) -> list[int]:
  numeric_files = {int(path.stem) for path in root.glob("*.py") if path.stem.isdigit()}
  discovered = []
  index = 1
  while index in numeric_files:
    discovered.append(index)
    index += 1
  return discovered


def load_json(path: Path) -> tuple[Any | None, str | None]:
  if not path.exists():
    return None, None
  try:
    with path.open("r", encoding="utf-8") as handle:
      return json.load(handle), None
  except Exception as exc:
    return None, str(exc)


def load_results_txt(path: Path) -> dict[str, float]:
  scores: dict[str, float] = {}
  if not path.exists():
    return scores
  with path.open("r", encoding="utf-8", errors="ignore") as handle:
    for raw_line in handle:
      line = raw_line.strip()
      if not line or ":" not in line:
        continue
      model, score_text = line.split(":", 1)
      model = model.strip()
      try:
        scores[model] = float(score_text.strip())
      except ValueError:
        continue
  return scores


def discover_benchmark_entry_script(root: Path) -> str:
  candidates: list[str] = []
  for path in sorted(root.glob("*.py")):
    if path.stem.isdigit():
      continue
    try:
      content = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
      continue
    if BENCHMARK_ENTRY_CALL_RE.search(content):
      candidates.append(path.name)
  if len(candidates) == 1:
    return candidates[0]
  if len(candidates) > 1:
    for candidate in candidates:
      if candidate.lower().endswith("benchmark.py"):
        return candidate
    return candidates[0]
  top_level_scripts = sorted(path.name for path in root.glob("*.py") if not path.stem.isdigit())
  if len(top_level_scripts) == 1:
    return top_level_scripts[0]
  return "<benchmark-script>.py"


def rerun_model_command(script_name: str, model: str) -> str:
  return f'python "{script_name}" -m "{model}" --ignore-cached-failures'


def rerun_question_command(script_name: str,
                           model: str,
                           question: int,
                           subpass: int | None = None) -> str:
  selector = f"{question}.{subpass}" if subpass is not None else str(question)
  return f'python "{script_name}" -m "{model}" -t {selector} --ignore-cached-failures'


def parse_raw_filename(path: Path) -> tuple[str, int, int] | None:
  match = RAW_FILE_RE.match(path.stem)
  if not match:
    return None
  try:
    return match.group("model"), int(match.group("question")), int(match.group("subpass"))
  except ValueError:
    return None


def model_issue_severity(model: str, leaderboard_models: set[str]) -> str:
  return "error" if model in leaderboard_models else "warning"


def issue_severity(model: str | None, leaderboard_models: set[str], kind: str) -> str:
  if model is None:
    return "error"
  if kind in {
      "missing_report", "missing_footer", "missing_questions_in_report",
      "report_shorter_than_run_summary", "incomplete_report"
  }:
    return "error"
  severity = model_issue_severity(model, leaderboard_models)
  if model in PLACEBO_MODELS and kind in {"empty_raw_output", "exception_raw_output"}:
    return "warning"
  return severity


def intentional_partial_report_path(model_dir: Path) -> Path | None:
  for filename in INTENTIONAL_PARTIAL_REPORT_FILENAMES:
    candidate = model_dir / filename
    if candidate.exists():
      return candidate
  return None


def read_report_metrics(report_path: Path, root: Path) -> dict[str, Any]:
  metrics: dict[str, Any] = {
    "size_bytes": report_path.stat().st_size,
    "question_ids": set(),
    "dead_images": [],
    "has_footer": False,
    "footer_total_tests": None,
    "footer_score_text": None,
    "footer_percentage": None,
  }
  if metrics["size_bytes"] > MAX_REPORT_BYTES:
    return metrics
  html = report_path.read_text(encoding="utf-8", errors="ignore")
  metrics["question_ids"] = {int(value) for value in QUESTION_ANCHOR_RE.findall(html)}
  footer_match = OVERALL_SUMMARY_RE.search(html)
  has_footer_strings = "Total Tests" in html and "Overall Score" in html and "Overall Summary" in html
  metrics["has_footer"] = bool(footer_match) or has_footer_strings
  if footer_match:
    metrics["footer_total_tests"] = int(footer_match.group(1))
    metrics["footer_score_text"] = footer_match.group(2).strip()
    try:
      metrics["footer_percentage"] = float(footer_match.group(3))
    except ValueError:
      metrics["footer_percentage"] = None
  dead_images: set[str] = set()
  for src in IMAGE_SRC_RE.findall(html):
    parsed = urlsplit(src)
    if parsed.scheme in {"http", "https", "data", "mailto"}:
      continue
    if src.startswith("#"):
      continue
    rel = unquote(parsed.path).strip()
    if not rel:
      continue
    if rel.startswith("/"):
      rel_parts = [part for part in Path(rel.lstrip("/")).parts if part not in {".", ""}]
      if rel_parts and rel_parts[0].lower() == root.name.lower():
        rel_parts = rel_parts[1:]
      candidate = root.joinpath(*rel_parts) if rel_parts else root
    else:
      candidate = report_path.parent / rel
    if not candidate.exists():
      dead_images.add(rel)
  metrics["dead_images"] = sorted(dead_images)
  return metrics


def scan_raw_file(path: Path) -> tuple[bool, bool]:
  try:
    size = path.stat().st_size
  except OSError:
    return False, False
  if size == 0:
    return True, False
  try:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
      head = handle.read(2048)
  except Exception:
    return False, False
  is_empty = not head.strip() and size <= 2048
  has_exception = "__exception__" in head
  return is_empty, has_exception


def score_issues_from_run_summary(run_summary: dict[str,
                                                    Any], model: str, leaderboard_models: set[str],
                                  benchmark_script: str) -> list[Issue]:
  issues: list[Issue] = []
  overall = run_summary.get("overall") if isinstance(run_summary, dict) else None
  severity = model_issue_severity(model, leaderboard_models)
  if isinstance(overall, dict):
    percentage = overall.get("percentage")
    accuracy = overall.get("accuracy")
    try:
      if percentage is not None and float(percentage) > 100:
        issues.append(
          Issue(severity=severity,
                kind="score_over_100",
                model=model,
                message=f"run_summary percentage is {float(percentage):.3f}%",
                rerun_command=rerun_model_command(benchmark_script, model)))
    except Exception:
      pass
    try:
      if accuracy is not None and float(accuracy) > 1:
        issues.append(
          Issue(severity=severity,
                kind="score_over_100",
                model=model,
                message=f"run_summary accuracy is {float(accuracy):.6f}",
                rerun_command=rerun_model_command(benchmark_script, model)))
    except Exception:
      pass
  tests = run_summary.get("tests") if isinstance(run_summary, dict) else None
  if not isinstance(tests, list):
    return issues
  for test in tests:
    if not isinstance(test, dict):
      continue
    question = test.get("test_index")
    subpasses = test.get("subpasses")
    if not isinstance(subpasses, list):
      continue
    for subpass in subpasses:
      if not isinstance(subpass, dict):
        continue
      try:
        score = float(subpass.get("score"))
      except Exception:
        continue
      if score > 1:
        subpass_index = subpass.get("subpass")
        rerun_command = None
        if isinstance(question, int):
          rerun_command = rerun_question_command(
            benchmark_script, model, question,
            subpass_index if isinstance(subpass_index, int) else None)
        issues.append(
          Issue(severity=severity,
                kind="subpass_score_over_100",
                model=model,
                question=question if isinstance(question, int) else None,
                subpass=subpass_index if isinstance(subpass_index, int) else None,
                message=f"subpass score is {score:.6f}",
                rerun_command=rerun_command or rerun_model_command(benchmark_script, model)))
  return issues


def build_scope(issue: Issue) -> str:
  parts = []
  if issue.model:
    parts.append(issue.model)
  if issue.question is not None:
    parts.append(f"Q{issue.question}")
  if issue.subpass is not None:
    parts.append(f"S{issue.subpass}")
  return " / ".join(parts) if parts else "global"


def print_issue_group(title: str, issues: list[Issue]) -> None:
  print(title)
  if not issues:
    print("  none")
    print()
    return
  for issue in issues:
    print(f"  - [{issue.kind}] {build_scope(issue)}: {issue.message}")
  print()


def print_issue_kind_summary(title: str, issues: list[Issue]) -> None:
  print(title)
  if not issues:
    print("  none")
    print()
    return
  counts = defaultdict(int)
  for issue in issues:
    counts[issue.kind] += 1
  for kind, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
    print(f"  - {kind}: {count}")
  print()


def print_info_details(intentional_partial_models: list[str], show_details: bool) -> None:
  print("Intentional partial runs")
  if not intentional_partial_models:
    print("  none")
    print()
    return
  print(
    f"  {len(intentional_partial_models)} model directories contain report.partial.html/report-partial.html."
  )
  if show_details:
    for entry in intentional_partial_models:
      print(f"  - {entry}")
  else:
    preview = intentional_partial_models[:12]
    for entry in preview:
      print(f"  - {entry}")
    remaining = len(intentional_partial_models) - len(preview)
    if remaining > 0:
      print(f"  - ... {remaining} more (use --show-info-details to list all)")
  print()


def collect_model_reruns(
    issues: list[Issue]) -> tuple[list[tuple[str, list[str]]], list[tuple[str, list[str]]]]:
  error_kinds_by_model: dict[str, set[str]] = defaultdict(set)
  warning_kinds_by_model: dict[str, set[str]] = defaultdict(set)
  for issue in issues:
    if not issue.model:
      continue
    target = error_kinds_by_model if issue.severity == "error" else warning_kinds_by_model
    target[issue.model].add(issue.kind)
  error_models = sorted((model, sorted(kinds)) for model, kinds in error_kinds_by_model.items())
  warning_models = sorted((model, sorted(kinds)) for model, kinds in warning_kinds_by_model.items()
                          if model not in error_kinds_by_model)
  return error_models, warning_models


def print_model_reruns(title: str, entries: list[tuple[str, list[str]]],
                       benchmark_script: str) -> None:
  print(title)
  if not entries:
    print("  none")
    print()
    return
  for model, kinds in entries:
    #print(f"  - {model}: {', '.join(kinds)}")
    print(f"{rerun_model_command(benchmark_script, model)}")
  print()


def main() -> int:
  args = parse_args()
  root = Path(args.root).resolve()
  if not (root / "results").exists():
    print(f"Expected to run from the benchmark root. Could not find: {root / 'results'}",
          file=sys.stderr)
    return 2
  benchmark_script = discover_benchmark_entry_script(root)
  test_indices = discover_contiguous_test_indices(root)
  expected_full_question_count = len(test_indices)
  expected_full_question_set = set(test_indices)
  results_dir = root / "results"
  models_dir = results_dir / "models"
  leaderboard_scores = load_results_txt(results_dir / "results.txt")
  leaderboard_models = set(leaderboard_scores)
  results_by_question_data, results_by_question_error = load_json(results_dir /
                                                                  "results_by_question.json")
  if not isinstance(results_by_question_data, dict):
    results_by_question_data = {}

  issues: list[Issue] = []
  intentional_partial_models: list[str] = []
  model_dirs = {
    path.name: path
    for path in models_dir.iterdir() if path.is_dir()
  } if models_dir.exists() else {}
  known_models = set(model_dirs) | leaderboard_models | set(results_by_question_data)

  for model in sorted(known_models):
    model_dir = model_dirs.get(model)
    report_path = model_dir / "report.html" if model_dir else None
    run_summary_path = model_dir / "run_summary.json" if model_dir else None
    partial_report_path = intentional_partial_report_path(model_dir) if model_dir else None
    run_summary: dict[str, Any] | None = None
    run_summary_error = None
    if run_summary_path and run_summary_path.exists():
      loaded, run_summary_error = load_json(run_summary_path)
      if isinstance(loaded, dict):
        run_summary = loaded
    if run_summary_error:
      issues.append(
        Issue(severity=issue_severity(model, leaderboard_models, "invalid_run_summary"),
              kind="invalid_run_summary",
              model=model,
              message=f"run_summary.json could not be parsed: {run_summary_error}",
              rerun_command=rerun_model_command(benchmark_script, model)))
    if model in leaderboard_scores and leaderboard_scores[model] > 1:
      issues.append(
        Issue(severity="error",
              kind="score_over_100",
              model=model,
              message=f"results.txt score is {leaderboard_scores[model]:.6f}",
              rerun_command=rerun_model_command(benchmark_script, model)))
    if run_summary is not None:
      issues.extend(
        score_issues_from_run_summary(run_summary, model, leaderboard_models, benchmark_script))
    question_data = results_by_question_data.get(model)
    question_map = question_data if isinstance(question_data, dict) else {}
    executed_questions = None
    if run_summary is not None and isinstance(run_summary.get("tests"), list):
      executed_questions = len(run_summary["tests"])
    elif question_map:
      executed_questions = len(question_map)
    if model_dir is None:
      if model in leaderboard_models:
        issues.append(
          Issue(severity="error",
                kind="missing_model_directory",
                model=model,
                message="model appears in results.txt but has no results/models directory",
                rerun_command=rerun_model_command(benchmark_script, model)))
      continue
    if report_path is None or not report_path.exists():
      raw_dir = model_dir / "raw"
      has_raw = raw_dir.exists() and any(raw_dir.glob("*.txt"))
      if partial_report_path is not None:
        intentional_partial_models.append(
          f"{model}: intentional partial report at {partial_report_path.name}")
      elif model in leaderboard_models or run_summary is not None or has_raw or any(
          model_dir.iterdir()):
        issues.append(
          Issue(severity=issue_severity(model, leaderboard_models, "missing_report"),
                kind="missing_report",
                model=model,
                message="report.html is missing",
                rerun_command=rerun_model_command(benchmark_script, model)))
      continue
    report_metrics = read_report_metrics(report_path, root)
    if report_metrics["size_bytes"] > MAX_REPORT_BYTES:
      issues.append(
        Issue(severity=issue_severity(model, leaderboard_models, "oversized_report"),
              kind="oversized_report",
              model=model,
              message=f"report.html is {report_metrics['size_bytes'] / (1024 * 1024):.1f} MB",
              rerun_command=rerun_model_command(benchmark_script, model)))
    if report_metrics["size_bytes"] <= MAX_REPORT_BYTES and not report_metrics["has_footer"]:
      issues.append(
        Issue(severity=issue_severity(model, leaderboard_models, "missing_footer"),
              kind="missing_footer",
              model=model,
              message="report.html is missing the Overall Summary footer",
              rerun_command=rerun_model_command(benchmark_script, model)))
    if report_metrics["footer_percentage"] is not None and report_metrics["footer_percentage"] > 100:
      issues.append(
        Issue(severity="error",
              kind="score_over_100",
              model=model,
              message=f"report footer percentage is {report_metrics['footer_percentage']:.3f}%",
              rerun_command=rerun_model_command(benchmark_script, model)))
    dead_images = report_metrics["dead_images"]
    if dead_images:
      display = ", ".join(dead_images[:5])
      if len(dead_images) > 5:
        display += f", ... ({len(dead_images)} total)"
      issues.append(
        Issue(severity=issue_severity(model, leaderboard_models, "dead_image_links"),
              kind="dead_image_links",
              model=model,
              message=f"report.html contains dead image links: {display}",
              rerun_command=rerun_model_command(benchmark_script, model)))
    report_questions = report_metrics["question_ids"]
    if model in leaderboard_models and report_questions and report_questions != expected_full_question_set:
      missing = sorted(expected_full_question_set - report_questions)
      issues.append(
        Issue(
          severity="error",
          kind="missing_questions_in_report",
          model=model,
          message=
          f"leaderboard model report covers {len(report_questions)}/{expected_full_question_count} questions; missing {missing[:12]}{'...' if len(missing) > 12 else ''}",
          rerun_command=rerun_model_command(benchmark_script, model)))
    elif executed_questions is not None and report_questions and len(
        report_questions) < executed_questions:
      issues.append(
        Issue(
          severity=model_issue_severity(model, leaderboard_models),
          kind="report_shorter_than_run_summary",
          model=model,
          message=
          f"report shows {len(report_questions)} question sections but run metadata suggests {executed_questions}",
          rerun_command=rerun_model_command(benchmark_script, model)))
    elif report_questions and len(report_questions) < expected_full_question_count:
      issues.append(
        Issue(
          severity=issue_severity(model, leaderboard_models, "incomplete_report"),
          kind="incomplete_report",
          model=model,
          message=
          f"report.html covers {len(report_questions)}/{expected_full_question_count} questions without using an intentional partial report file",
          rerun_command=rerun_model_command(benchmark_script, model)))

  scanned_raw_paths: set[Path] = set()
  raw_issue_groups: dict[tuple[str, str, int], list[tuple[int, Path]]] = defaultdict(list)
  raw_roots = [results_dir / "raw"]
  raw_roots.extend(path / "raw" for path in model_dirs.values())
  for raw_root in raw_roots:
    if not raw_root.exists():
      continue
    for path in raw_root.glob("*.txt"):
      resolved = path.resolve()
      if resolved in scanned_raw_paths:
        continue
      scanned_raw_paths.add(resolved)
      parsed = parse_raw_filename(path)
      if parsed is None:
        continue
      model, question, subpass = parsed
      empty_like, has_exception = scan_raw_file(path)
      if empty_like:
        raw_issue_groups[("empty_raw_output", model, question)].append((subpass, path))
      if has_exception:
        raw_issue_groups[("exception_raw_output", model, question)].append((subpass, path))

  for (kind, model, question), entries in sorted(raw_issue_groups.items()):
    entries = sorted(entries, key=lambda item: item[0])
    subpasses = [subpass for subpass, _ in entries]
    if len(entries) == 1:
      subpass, path = entries[0]
      message = (f"raw output file is empty: {path.relative_to(root)}" if kind == "empty_raw_output"
                 else f"raw output contains __exception__: {path.relative_to(root)}")
      rerun_command = rerun_question_command(benchmark_script, model, question, subpass)
      issue_subpass = subpass
    else:
      display_subpasses = ", ".join(str(value) for value in subpasses[:12])
      if len(subpasses) > 12:
        display_subpasses += ", ..."
      message = (
        f"{len(entries)} raw output files are empty across subpasses [{display_subpasses}]"
        if kind == "empty_raw_output" else
        f"{len(entries)} raw output files contain __exception__ across subpasses [{display_subpasses}]"
      )
      rerun_command = rerun_question_command(benchmark_script, model, question)
      issue_subpass = None
    issues.append(
      Issue(severity=issue_severity(model, leaderboard_models, kind),
            kind=kind,
            model=model,
            question=question,
            subpass=issue_subpass,
            message=message,
            rerun_command=rerun_command))

  if results_by_question_error:
    issues.append(
      Issue(severity="error",
            kind="invalid_results_by_question",
            message=f"results_by_question.json could not be parsed: {results_by_question_error}"))

  systemic_zero_issues: list[Issue] = []
  per_subpass_scores: dict[tuple[int, int], list[tuple[str, float]]] = defaultdict(list)
  question_titles: dict[int, str] = {}
  for model, model_data in results_by_question_data.items():
    if not isinstance(model_data, dict):
      continue
    for q_key, q_data in model_data.items():
      if not isinstance(q_data, dict):
        continue
      try:
        question = int(q_key)
      except Exception:
        continue
      title = q_data.get("title")
      if isinstance(title, str) and title:
        question_titles[question] = title
      subtasks = q_data.get("subtasks")
      if not isinstance(subtasks, dict):
        continue
      for s_key, value in subtasks.items():
        try:
          subpass = int(s_key)
          score = float(value)
        except Exception:
          continue
        per_subpass_scores[(question, subpass)].append((model, score))
  for (question, subpass), attempts in sorted(per_subpass_scores.items()):
    if len(attempts) < args.min_systemic_zero_attempts:
      continue
    best_score = max(score for _, score in attempts)
    if best_score > 0:
      continue
    title = question_titles.get(question, "unknown title")
    systemic_zero_issues.append(
      Issue(severity="warning",
            kind="systemic_zero_subpass",
            question=question,
            subpass=subpass,
            message=f"no positive score across {len(attempts)} attempted models for '{title}'"))

  seen_issue_keys: set[tuple[Any, ...]] = set()
  deduped_issues: list[Issue] = []
  for issue in issues + systemic_zero_issues:
    key = (issue.severity, issue.kind, issue.model, issue.question, issue.subpass, issue.message)
    if key in seen_issue_keys:
      continue
    seen_issue_keys.add(key)
    deduped_issues.append(issue)
  errors = sorted((issue for issue in deduped_issues if issue.severity == "error"),
                  key=lambda issue:
                  (issue.model or "", issue.question or -1, issue.subpass or -1, issue.kind))
  warnings = sorted((issue for issue in deduped_issues if issue.severity == "warning"),
                    key=lambda issue:
                    (issue.model or "", issue.question or -1, issue.subpass or -1, issue.kind))
  error_model_reruns, warning_model_reruns = collect_model_reruns(deduped_issues)

  print(f"Post-completion check: {root}")
  print(
    f"Benchmark tests discovered: {expected_full_question_count} (1-{expected_full_question_count if expected_full_question_count else 0})"
  )
  print(f"Model directories scanned: {len(model_dirs)}")
  print(f"Leaderboard models from results.txt: {len(leaderboard_models)}")
  print()
  print("Fairness rules")
  print("  - Models listed in results/results.txt are treated as intended full benchmark runs.")
  print(
    "  - Intentional partial runs are only recognized when the model directory contains report.partial.html or report-partial.html."
  )
  print(
    "  - A short or footerless report.html is treated as a broken full run, not an informational partial run."
  )
  print(
    f"  - 'No model scored > 0' is only flagged when at least {args.min_systemic_zero_attempts} models attempted that subpass."
  )
  print()
  print("Summary")
  print(f"  Errors: {len(errors)}")
  print(f"  Warnings: {len(warnings)}")
  print(f"  Intentional partial runs: {len(intentional_partial_models)}")
  print(f"  Benchmark entry script: {benchmark_script}")
  print()
  print_issue_kind_summary("Error counts by kind", errors)
  print_issue_kind_summary("Warning counts by kind", warnings)
  print_issue_group("Errors", errors)
  print_issue_group("Warnings", warnings)
  print_model_reruns("Suggested full reruns for warnings", warning_model_reruns, benchmark_script)
  print_model_reruns("Suggested full reruns for errors", error_model_reruns, benchmark_script)
  return 1 if errors or warnings else 0


if __name__ == "__main__":
  raise SystemExit(main())
