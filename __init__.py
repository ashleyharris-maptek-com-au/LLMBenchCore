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
]
