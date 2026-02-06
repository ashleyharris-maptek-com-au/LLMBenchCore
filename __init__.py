"""
LLMBenchCore - Abstract LLM Benchmarking Framework

This package provides the core infrastructure for running LLM benchmarks:
- BenchmarkRunner: Abstract base class for creating domain-specific benchmarks
- TestRunner: Core test execution, scoring, and HTML report generation
- CacheLayer: Response caching to avoid redundant API calls
- AI Engine adapters for major providers (OpenAI, Gemini, Anthropic, etc.)
"""

from .TestRunner import (
  BenchmarkRunner,
  run_benchmark_main,
  create_argument_parser,
  get_default_model_configs,
  run_model_config,
  runAllTests,
  runTest,
  parse_test_filter,
  run_setup,
  ALL_MODEL_CONFIGS,
  UNSKIP,
  IGNORE_CACHED_FAILURES,
  FORCE_ARG,
)

from .CacheLayer import CacheLayer
from .BatchOrchestrator import (
  BatchOrchestrator,
  BatchRequest,
  BatchResult,
  BatchJob,
  BatchStatus,
  run_batch_mode,
)
from .ResultPaths import (
  model_root,
  model_report_path,
  model_raw_path,
  model_prompt_path,
  model_cot_path,
  model_artifact_dir,
  model_artifact_path,
  result_path,
  report_relpath,
  legacy_raw_path,
  legacy_prompt_path,
  legacy_cot_path,
  set_current_model,
  reset_current_model,
  get_current_model,
)
from .ArtifactStore import (
  ArtifactStore,
  ModelArtifactStore,
)
from .CadArtifacts import (
  CadArtifactStore,
  write_if_changed,
)

__all__ = [
  'BenchmarkRunner',
  'run_benchmark_main',
  'create_argument_parser',
  'get_default_model_configs',
  'run_model_config',
  'runAllTests',
  'runTest',
  'parse_test_filter',
  'run_setup',
  'ALL_MODEL_CONFIGS',
  'UNSKIP',
  'IGNORE_CACHED_FAILURES',
  'FORCE_ARG',
  'CacheLayer',
  'BatchOrchestrator',
  'BatchRequest',
  'BatchResult',
  'BatchJob',
  'BatchStatus',
  'run_batch_mode',
  'model_root',
  'model_report_path',
  'model_raw_path',
  'model_prompt_path',
  'model_cot_path',
  'model_artifact_dir',
  'model_artifact_path',
  'result_path',
  'report_relpath',
  'legacy_raw_path',
  'legacy_prompt_path',
  'legacy_cot_path',
  'set_current_model',
  'reset_current_model',
  'get_current_model',
  'ArtifactStore',
  'ModelArtifactStore',
  'CadArtifactStore',
  'write_if_changed',
]
