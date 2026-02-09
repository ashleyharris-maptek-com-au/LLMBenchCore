import subprocess
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend to avoid tkinter threading issues

from typing import Dict, List, Any, Optional, Set
from abc import ABC, abstractmethod
import os
import base64
import hashlib
import html
import time
import argparse

from concurrent.futures import ThreadPoolExecutor, as_completed
from filelock import FileLock
from .CacheLayer import CacheLayer as cl
from . import ResultPaths as rp

global UNSKIP
UNSKIP = False
IGNORE_CACHED_FAILURES = False

# Global to track model configs for perfect score propagation
ALL_MODEL_CONFIGS = []

FORCE_ARG = False
# Optional override for per-request API timeout (seconds)
API_TIMEOUT_OVERRIDE: int | None = None

# Global reference to current benchmark runner instance (set by subclass)
_current_runner = None


def is_placebo_model(model_name: str) -> bool:
  for cfg in ALL_MODEL_CONFIGS:
    if cfg.get("name") == model_name:
      return cfg.get("engine") == "placebo"
  return False


class BenchmarkRunner(ABC):
  """
  Abstract base class for benchmark runners.
  
  Subclass this to create domain-specific benchmarks (e.g., spatial/geometry benchmarks).
  Override the abstract methods to customize benchmark behavior.
  """

  def __init__(self):
    global _current_runner
    _current_runner = self

  @abstractmethod
  def get_benchmark_title(self) -> str:
    """Return the benchmark title for reports and graphs."""
    pass

  @abstractmethod
  def get_benchmark_subtitle(self) -> str:
    """Return the benchmark subtitle for the landing page."""
    pass

  def get_benchmark_description(self) -> str:
    """Return additional HTML description for the landing page header."""
    return ""

  def get_model_configs(self) -> List[Dict[str, Any]]:
    """
    Return list of model configurations to run.
    Default implementation provides standard configs for major AI providers.
    Override to add, filter, or replace model configurations.
    """
    return get_default_model_configs()

  def get_additional_model_configs(self) -> List[Dict[str, Any]]:
    """
    Return additional model configurations to append to defaults.
    Override to add domain-specific models without replacing the defaults.
    """
    return []

  def filter_model_configs(self, configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter or modify the model configurations before use.
    Override to exclude certain models or modify their settings.
    
    Args:
        configs: The combined list of default + additional configs
        
    Returns:
        Filtered/modified list of configs
    """
    return configs

  def can_handle_custom_scoring(self, test_globals: dict) -> bool:
    """
    Check if this runner can handle custom scoring for a test.
    Override to detect domain-specific test patterns.
    
    Args:
        test_globals: The globals dict from the test module
        
    Returns:
        True if this runner should handle scoring for this test
    """
    return False

  def process_custom_scoring(self, index: int, subPass: int, result: Any, test_globals: dict,
                             aiEngineName: str) -> Dict[str, Any]:
    """
    Process custom scoring for a test result.
    Override to implement domain-specific scoring logic.
    
    Args:
        index: Test index
        subPass: Subpass number
        result: The AI's result
        test_globals: The globals dict from the test module
        aiEngineName: Name of the AI engine
        
    Returns:
        Dict with keys: score, output_image, output_additional_images, 
        reference_image, temp_dir, scoreExplanation, output_hyperlink
    """
    return {"score": 0, "scoreExplanation": "Custom scoring not implemented"}

  def run_setup_for_test(self, test_index: int, test_globals: dict) -> None:
    """
    Run any custom setup needed for a test during --setup mode.
    Override to implement domain-specific setup (e.g., building reference models).
    
    Args:
        test_globals: The globals dict from the test module
    """
    pass

  def add_arguments(self, parser: argparse.ArgumentParser) -> None:
    """
    Add additional command-line arguments.
    Override to add domain-specific CLI options.
    
    Args:
        parser: The argument parser to add arguments to
    """
    pass

  def handle_arguments(self, args: argparse.Namespace) -> None:
    """
    Handle custom command-line arguments after parsing.
    Override to process domain-specific CLI options.
    
    Args:
        args: Parsed command-line arguments
    """
    pass

  def get_final_model_configs(self) -> List[Dict[str, Any]]:
    """Get the final list of model configs after all processing."""
    configs = self.get_model_configs()
    configs.extend(self.get_additional_model_configs())
    return self.filter_model_configs(configs)

  def run(self,
          test_filter: Optional[Dict[int, Optional[Set[int]]]] = None,
          model_filter: Optional[Set[str]] = None) -> None:
    """
    Run the benchmark with the given filters.
    
    Args:
        test_filter: Optional dict mapping test indices to subtask filters.
                     None value means run all subtasks, Set[int] means specific subtasks.
        model_filter: Optional set of model names to run
    """
    global ALL_MODEL_CONFIGS

    configs = self.get_final_model_configs()
    ALL_MODEL_CONFIGS.clear()
    ALL_MODEL_CONFIGS.extend(configs)

    for config in configs:
      if model_filter and config["name"] not in model_filter:
        continue
      run_model_config(config, test_filter)


def get_default_model_configs() -> List[Dict[str, Any]]:
  """
  Returns the default list of model configurations for all major AI providers.
  """
  configs = []

  from .AiEnginePlacebo import get_placebo_model_configs
  configs.extend(get_placebo_model_configs())

  # OpenAI models
  openai_base_models = ["gpt-5-nano", "gpt-5-mini", "gpt-5.1", "gpt-5.2", "gpt-5.2-pro"]
  for model in openai_base_models:
    configs.append({
      "name": model,
      "engine": "openai",
      "base_model": model,
      "reasoning": False,
      "tools": False,
      "env_key": "OPENAI_API_KEY"
    })
    configs.append({
      "name": f"{model}-HighReasoning",
      "engine": "openai",
      "base_model": model,
      "reasoning": 10,
      "tools": False,
      "env_key": "OPENAI_API_KEY"
    })

    if "5.2-pro" not in model:
      configs.append({
        "name": f"{model}-Reasoning-Tools",
        "engine": "openai",
        "base_model": model,
        "reasoning": 10,
        "tools": True,
        "env_key": "OPENAI_API_KEY"
      })

  # Azure OpenAI (optional, configured via env)
  # Uses deployment name as base_model.
  configs.append({
    "name": "gpt-5.2-chat-azure",
    "engine": "azure_openai",
    "base_model": "gpt-5.2-chat",
    "reasoning": False,
    "tools": False,
    "env_key": "AZURE_OPENAI_API_KEY",
    "endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT"),
    "api_version": os.environ.get("AZURE_OPENAI_API_VERSION")
  })
  configs.append({
    "name": "gpt-5.2-chat-azure-Reasoning",
    "engine": "azure_openai",
    "base_model": "gpt-5.2-chat",
    "reasoning": 5,
    "tools": False,
    "env_key": "AZURE_OPENAI_API_KEY",
    "endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT"),
    "api_version": os.environ.get("AZURE_OPENAI_API_VERSION")
  })
  configs.append({
    "name": "gpt-5.2-chat-azure-Reasoning-Tools",
    "engine": "azure_openai",
    "base_model": "gpt-5.2-chat",
    "reasoning": 5,
    "tools": True,
    "env_key": "AZURE_OPENAI_API_KEY",
    "endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT"),
    "api_version": os.environ.get("AZURE_OPENAI_API_VERSION")
  })

  # Gemini models
  gemini_base_models = [
    "gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-3-pro-preview", "gemini-3-flash-preview"
  ]
  for model in gemini_base_models:
    configs.append({
      "name": model,
      "engine": "gemini",
      "base_model": model,
      "reasoning": False,
      "tools": False,
      "env_key": "GEMINI_API_KEY"
    })
    configs.append({
      "name": f"{model}-HighReasoning",
      "engine": "gemini",
      "base_model": model,
      "reasoning": 10,
      "tools": False,
      "env_key": "GEMINI_API_KEY"
    })
    configs.append({
      "name": f"{model}-Reasoning-Tools",
      "engine": "gemini",
      "base_model": model,
      "reasoning": 10,
      "tools": True,
      "env_key": "GEMINI_API_KEY"
    })

  # XAI/Grok models
  configs.append({
    "name": "grok-2-vision-1212",
    "engine": "xai",
    "base_model": "grok-2-vision-1212",
    "reasoning": False,
    "tools": False,
    "env_key": "XAI_API_KEY"
  })
  configs.append({
    "name": "grok-4-1-fast-non-reasoning",
    "engine": "xai",
    "base_model": "grok-4-1-fast-non-reasoning",
    "reasoning": False,
    "tools": False,
    "env_key": "XAI_API_KEY"
  })
  configs.append({
    "name": "grok-4-1-fast-reasoning",
    "engine": "xai",
    "base_model": "grok-4-1-fast-reasoning",
    "reasoning": 10,
    "tools": False,
    "env_key": "XAI_API_KEY"
  })
  configs.append({
    "name": "grok-4-0709-HighReasoning",
    "engine": "xai",
    "base_model": "grok-4-0709",
    "reasoning": 10,
    "tools": False,
    "env_key": "XAI_API_KEY"
  })
  configs.append({
    "name": "grok-4-0709",
    "engine": "xai",
    "base_model": "grok-4-0709",
    "reasoning": False,
    "tools": False,
    "env_key": "XAI_API_KEY"
  })
  configs.append({
    "name": "grok-4-0709-Reasoning-Tools",
    "engine": "xai",
    "base_model": "grok-4-0709",
    "reasoning": 10,
    "tools": True,
    "env_key": "XAI_API_KEY"
  })

  # Anthropic models
  anthropic_base_models = ["claude-sonnet-4-5", "claude-opus-4-5"]
  for model in anthropic_base_models:
    configs.append({
      "name": model,
      "engine": "anthropic",
      "base_model": model,
      "reasoning": False,
      "tools": False,
      "env_key": "ANTHROPIC_API_KEY"
    })
    configs.append({
      "name": f"{model}-HighReasoning",
      "engine": "anthropic",
      "base_model": model,
      "reasoning": 10,
      "tools": False,
      "env_key": "ANTHROPIC_API_KEY"
    })
    configs.append({
      "name": f"{model}-Reasoning-Tools",
      "engine": "anthropic",
      "base_model": model,
      "reasoning": 10,
      "tools": True,
      "env_key": "ANTHROPIC_API_KEY"
    })

  # Amazon Bedrock - Qwen models
  bedrock_qwen_models = [
    ("qwen3-32B", "qwen.qwen3-32b-v1:0"),
    ("qwen3-VL-235B-22B", "qwen.qwen3-vl-235b-a22b"),
  ]
  for name, model_id in bedrock_qwen_models:
    configs.append({
      "name": name,
      "engine": "bedrock",
      "base_model": model_id,
      "reasoning": False,
      "tools": False,
      "region": "us-east-1",
      "env_key": "AWS_ACCESS_KEY_ID"
    })
    configs.append({
      "name": name + "-HighReasoning",
      "engine": "bedrock",
      "base_model": model_id,
      "reasoning": 10,
      "tools": False,
      "region": "us-east-1",
      "env_key": "AWS_ACCESS_KEY_ID"
    })
    configs.append({
      "name": name + "-Reasoning-Tools",
      "engine": "bedrock",
      "base_model": model_id,
      "reasoning": 10,
      "tools": True,
      "region": "us-east-1",
      "env_key": "AWS_ACCESS_KEY_ID"
    })

  # Amazon Bedrock - Llama models
  bedrock_llama_models = [
    ("llama3-70b-bedrock", "meta.llama3-70b-instruct-v1:0"),
    ("llama3-1-405b-bedrock", "meta.llama3-1-405b-instruct-v1:0"),
  ]
  for name, model_id in bedrock_llama_models:
    configs.append({
      "name": name,
      "engine": "bedrock",
      "base_model": model_id,
      "reasoning": False,
      "tools": False,
      "region": "us-west-2",
      "env_key": "AWS_ACCESS_KEY_ID"
    })
    configs.append({
      "name": name + "-HighReasoning",
      "engine": "bedrock",
      "base_model": model_id,
      "reasoning": 10,
      "tools": False,
      "region": "us-west-2",
      "env_key": "AWS_ACCESS_KEY_ID"
    })
    configs.append({
      "name": name + "-Reasoning-Tools",
      "engine": "bedrock",
      "base_model": model_id,
      "reasoning": 10,
      "tools": True,
      "region": "us-west-2",
      "env_key": "AWS_ACCESS_KEY_ID"
    })

  # Amazon Bedrock - Mistral models
  bedrock_mistral_models = [("mistral-large-bedrock", "mistral.mistral-large-2402-v1:0"),
                            ("mistral-large-3-bedrock", "mistral.mistral-large-3-675b-instruct")]

  for name, model_id in bedrock_mistral_models:
    configs.append({
      "name": name,
      "engine": "bedrock",
      "base_model": model_id,
      "reasoning": False,
      "tools": False,
      "region": "us-east-1",
      "env_key": "AWS_ACCESS_KEY_ID"
    })
    configs.append({
      "name": name + "-HighReasoning",
      "engine": "bedrock",
      "base_model": model_id,
      "reasoning": 10,
      "tools": False,
      "region": "us-east-1",
      "env_key": "AWS_ACCESS_KEY_ID"
    })
    configs.append({
      "name": name + "-Reasoning-Tools",
      "engine": "bedrock",
      "base_model": model_id,
      "reasoning": 10,
      "tools": True,
      "region": "us-east-1",
      "env_key": "AWS_ACCESS_KEY_ID"
    })

  # Amazon Nova models
  nova_models = [("nova-lite", "amazon.nova-lite-v1:0"), ("nova-pro", "amazon.nova-pro-v1:0"),
                 ("nova-premier", "us.amazon.nova-premier-v1:0")]
  for name, model_id in nova_models:
    configs.append({
      "name": name,
      "engine": "bedrock",
      "base_model": model_id,
      "reasoning": False,
      "tools": False,
      "region": "us-east-1",
      "env_key": "AWS_ACCESS_KEY_ID"
    })
    configs.append({
      "name": name + "-HighReasoning",
      "engine": "bedrock",
      "base_model": model_id,
      "reasoning": 10,
      "tools": False,
      "region": "us-east-1",
      "env_key": "AWS_ACCESS_KEY_ID"
    })
    configs.append({
      "name": name + "-Reasoning-Tools",
      "engine": "bedrock",
      "base_model": model_id,
      "reasoning": False,
      "tools": True,
      "region": "us-east-1",
      "env_key": "AWS_ACCESS_KEY_ID"
    })

  # llama.cpp local server (optional, requires running server)
  # Users can add custom model configs by setting LLAMACPP_BASE_URL
  if os.environ.get("LLAMACPP_BASE_URL"):
    llamacpp_model = os.environ.get("LLAMACPP_MODEL_NAME", "llamacpp-local")
    configs.append({
      "name": llamacpp_model,
      "engine": "llamacpp",
      "base_model": llamacpp_model,
      "base_url": os.environ.get("LLAMACPP_BASE_URL"),
      "env_key": "LLAMACPP_BASE_URL"
    })

  return configs


def create_argument_parser(runner: BenchmarkRunner) -> argparse.ArgumentParser:
  """Create the standard argument parser with hooks for runner customization."""
  parser = argparse.ArgumentParser(description=f"Run {runner.get_benchmark_title()}",
                                   formatter_class=argparse.RawDescriptionHelpFormatter,
                                   epilog="""
Examples:
  python <script>.py                          # Run all tests on all available models
  python <script>.py --setup                  # Download/build reference data (run first!)
  python <script>.py --parallel               # Run all models in parallel
  python <script>.py --list-models            # List all available model names
  python <script>.py -t 1,2,3                 # Run only tests 1, 2, 3
  python <script>.py -t 5-10                  # Run tests 5 through 10
  python <script>.py -t 2.3                   # Run only subtask 3 of test 2
  python <script>.py -t 2.0,2.3,2.5           # Run subtasks 0, 3, 5 of test 2
  python <script>.py -m gpt-5-nano            # Run only gpt-5-nano model
  python <script>.py -m "gpt-5-nano,gpt-5.1"  # Run multiple specific models
  python <script>.py -m "nova-*"              # Run all models matching wildcard pattern
  python <script>.py -m "*-HighReasoning"     # Run all HighReasoning variants
  python <script>.py -t 1 -m gpt-5-nano       # Run test 1 on gpt-5-nano only
  python <script>.py --force -m gpt-5-nano    # Re-run without using cached AI responses
  python <script>.py --batch                  # Run in batch mode (50%% cheaper, async)
  python <script>.py --batch -m "gpt-*,claude-*"  # Batch mode for specific models
  python <script>.py --import-batch <batch_id> --import-model gpt-5.2  # Import from cancelled batch
  python <script>.py --import-batch results.jsonl --import-model gpt-5.2  # Import from JSONL file
    """)

  parser.add_argument("-t",
                      "--tests",
                      type=str,
                      help="Comma-separated list of test indices, ranges, or subtasks "
                      "(e.g., '1,2,3' or '5-10' or '2.3' for subtask 3 of test 2)")
  parser.add_argument(
    "-m",
    "--models",
    type=str,
    help="Comma-separated list of model names or patterns with wildcards (* and ?)")
  parser.add_argument("--list-models",
                      action="store_true",
                      help="List all available model names and exit")
  parser.add_argument("--force",
                      action="store_true",
                      help="Bypass AI response cache (still saves new responses to cache)")
  parser.add_argument("--api-timeout",
                      type=int,
                      default=None,
                      help="Override per-request API timeout in seconds (e.g., llama.cpp)")
  parser.add_argument("--offline",
                      action="store_true",
                      help="Only use cached results. Do not make any API calls.")
  parser.add_argument("--unskip", action="store_true", help="Run tests that are currently skipped.")
  parser.add_argument("--parallel", action="store_true", help="Run all models in parallel")
  parser.add_argument("--ignore-cached-failures",
                      action="store_true",
                      help="Ignore cached empty/error results")
  parser.add_argument("--setup",
                      action="store_true",
                      help="Download and build all reference data without running tests.")
  parser.add_argument("--batch",
                      action="store_true",
                      help="Run in batch mode: submit all prompts as batches, poll for results, "
                      "then process. Uses 50%% cheaper batch APIs where available.")
  parser.add_argument("--batch-poll-interval",
                      type=int,
                      default=300,
                      help="Seconds between batch status polls (default: 300)")
  parser.add_argument("--import-batch",
                      type=str,
                      metavar="ID_OR_FILE",
                      help="Import results from a failed/cancelled batch. Accepts a batch ID "
                      "(to fetch from provider) or a path to a JSONL file.")
  parser.add_argument("--import-model",
                      type=str,
                      metavar="MODEL",
                      help="Model name for the batch import (required with --import-batch)")

  # Allow runner to add custom arguments
  runner.add_arguments(parser)

  return parser


def run_benchmark_main(runner: BenchmarkRunner, script_file: str = None) -> None:
  """
  Main entry point for running a benchmark.
  Handles argument parsing and dispatches to the runner.
  
  Args:
      runner: The BenchmarkRunner instance
      script_file: The script file path (for --parallel mode), defaults to __file__ of caller
  """
  global UNSKIP, IGNORE_CACHED_FAILURES, FORCE_ARG
  import sys

  if script_file is None:
    script_file = sys.argv[0]

  parser = create_argument_parser(runner)
  args = parser.parse_args()

  # Handle global flags
  if args.ignore_cached_failures:
    from . import CacheLayer
    CacheLayer.IGNORE_CACHED_FAILURES = True
    IGNORE_CACHED_FAILURES = True
    print(
      "Ignore cached failures: Cached results that are empty, bailed or errored out will be ignored."
    )

  if args.force:
    from . import CacheLayer
    CacheLayer.FORCE_REFRESH = True
    FORCE_ARG = True
    print("Force mode: AI response cache will be bypassed (new responses still cached)")

  if args.offline:
    from . import CacheLayer
    CacheLayer.OFFLINE_MODE = True
    print("Offline mode: No API calls will be made, cache only.")

  if args.api_timeout is not None:
    global API_TIMEOUT_OVERRIDE
    API_TIMEOUT_OVERRIDE = args.api_timeout
    print(f"API timeout override set to {API_TIMEOUT_OVERRIDE} seconds")

  if args.unskip:
    UNSKIP = True

  # Let runner handle custom arguments
  runner.handle_arguments(args)

  # Handle --setup mode
  if args.setup:
    run_setup()
    sys.exit(0)

  # Get model configs
  all_configs = runner.get_final_model_configs()
  ALL_MODEL_CONFIGS.clear()
  ALL_MODEL_CONFIGS.extend(all_configs)

  # Handle --parallel mode
  if args.parallel:
    if args.models:
      print("--parallel and --models aren't compatible")
    cli_args = sys.argv[1:]
    cli_args.remove("--parallel")

    tasks = []
    for config in all_configs:
      tasks.append(subprocess.Popen([sys.executable, script_file, "-m", config["name"], *cli_args]))
    for task in tasks:
      task.wait()
    sys.exit(0)

  # Handle --list-models
  if args.list_models:
    print("Available models:")
    for config in all_configs:
      env_key = config.get("env_key")
      available = "+" if (env_key is None or os.environ.get(env_key)) else f"x (needs {env_key})"
      print(f"  {available} {config['name']}")
    sys.exit(0)

  # Parse test filter
  test_filter = None
  if args.tests:
    test_filter = parse_test_filter(args.tests)
    # Format filter for display
    filter_parts = []
    for t in sorted(test_filter.keys()):
      subtasks = test_filter[t]
      if subtasks is None:
        filter_parts.append(str(t))
      else:
        for s in sorted(subtasks):
          filter_parts.append(f"{t}.{s}")
    print(f"Running tests: {', '.join(filter_parts)}")

  # Parse model filter with wildcard support
  model_filter = None
  if args.models:
    import fnmatch

    if args.models == "best":
      args.models = "gpt-5.2-Reasoning-Tools,gemini-3-pro-preview-Reasoning-Tools,grok-4-0709-HighReasoning,claude-opus-4-5-Reasoning-Tools,qwen3-VL-235B-22B,llama3-1-405b-bedrock,mistral-large-bedrock,nova-premier-Reasoning-Tools"
    elif args.models == "worst":
      args.models = "nova-lite,llama3-70b-bedrock,qwen3-32B,claude-sonnet-4-5,grok-2-vision-1212,gemini-2.5-flash-lite,gpt-5-nano"

    all_model_names = [c["name"] for c in all_configs]
    patterns = [m.strip() for m in args.models.split(",")]
    matched_models = set()

    for pattern in patterns:
      if '*' in pattern or '?' in pattern:
        if pattern.startswith("^"):
          matches = [
            name for name in all_model_names
            if not fnmatch.fnmatch(name.lower(), pattern[1:].lower())
          ]
        else:
          matches = [
            name for name in all_model_names if fnmatch.fnmatch(name.lower(), pattern.lower())
          ]
        if not matches:
          print(f"Warning: Pattern '{pattern}' did not match any models")
        matched_models.update(matches)
      else:
        exact_match = next((name for name in all_model_names if name.lower() == pattern.lower()),
                           None)
        if exact_match:
          matched_models.add(exact_match)
        else:
          print(f"Error: Model '{pattern}' not found. Use --list-models to see available models.")

    if not matched_models:
      print("No models matched. Exiting.")
      sys.exit(1)

    model_filter = matched_models
    print(f"Running models: {sorted(model_filter)}")

  # Handle --import-batch mode
  if args.import_batch:
    if not args.import_model:
      print("Error: --import-model is required with --import-batch")
      sys.exit(1)
    # Find the model config
    model_config = next((c for c in all_configs if c["name"] == args.import_model), None)
    if not model_config:
      print(
        f"Error: Model '{args.import_model}' not found. Use --list-models to see available models.")
      sys.exit(1)
    from .BatchOrchestrator import import_batch_results
    import_batch_results(args.import_batch, model_config, runner)
    sys.exit(0)

  # Handle --batch mode
  if args.batch:
    from .BatchOrchestrator import run_batch_mode
    run_batch_mode(runner,
                   test_filter,
                   model_filter,
                   poll_interval=args.batch_poll_interval,
                   force_mode=args.force)
    sys.exit(0)

  # Run the benchmark
  runner.run(test_filter, model_filter)


def checkSavedPromptCache(aiEngineName: str, index: int, subPass: int, prompt: str):
  """
  Check if a saved prompt exists and matches the current prompt.
  If so, return the cached result. Otherwise return None.
  
  This is checked BEFORE the CacheLayer to avoid API calls when prompts haven't changed.
  """
  candidate_pairs = [
    (rp.model_prompt_path(aiEngineName, index,
                          subPass), rp.model_raw_path(aiEngineName, index, subPass)),
    (rp.legacy_prompt_path(aiEngineName, index,
                           subPass), rp.legacy_raw_path(aiEngineName, index, subPass)),
  ]

  selected_pair = None
  for prompt_file, result_file in candidate_pairs:
    if os.path.exists(prompt_file) and os.path.exists(result_file):
      selected_pair = (prompt_file, result_file)
      break

  if selected_pair is None:
    return None

  if is_placebo_model(aiEngineName):
    return None

  # Check both FORCE_ARG and the module's FORCE_REFRESH for robustness
  import sys
  cache_module = sys.modules.get('LLMBenchCore.CacheLayer')
  if FORCE_ARG or (cache_module and getattr(cache_module, 'FORCE_REFRESH', False)):
    return None

  prompt_file, result_file = selected_pair

  try:
    with open(prompt_file, "r", encoding="utf-8") as f:
      saved_prompt = f.read()

    # Compare prompts (strip to handle whitespace differences)
    if saved_prompt.strip() == str(prompt).strip():
      with open(result_file, "r", encoding="utf-8") as f:
        saved_result = f.read()

      if IGNORE_CACHED_FAILURES and len(saved_result) <= 10:
        print(
          f"Ignoring prompt cache - IGNORE_CACHED_FAILURES is set and result was '{saved_result}'.")
        return None

      print(f"Prompt cache hit for {aiEngineName} Q{index}/S{subPass}. {prompt_file}")
      # Results are saved with str(result), so use ast.literal_eval to parse
      import ast
      try:
        return ast.literal_eval(saved_result)
      except:
        return saved_result
  except Exception as e:
    print(f"Error checking saved prompt cache: {e}")

  return None


def getCompanyPrefix(model_name: str) -> str:
  """Extract company prefix from model name (part before first '-')."""
  if '-' in model_name:
    return model_name.split('-')[0]
  return model_name


def propogateUpwardsHack(aiEngineName: str, index: int, subPass: int, score: float):
  """
  HACK for development: If a subpass gets a perfect score, propagate the result
  to higher-grade LLMs from the same company.
  
  Same company = same prefix before '-' in model name.
  Higher grade = appears later in ALL_MODEL_CONFIGS list.
  """
  if score != 1.0:
    return

  company_prefix = getCompanyPrefix(aiEngineName)

  # Find current model's position in configs
  current_idx = -1
  for i, cfg in enumerate(ALL_MODEL_CONFIGS):
    if cfg.get('name') == aiEngineName:
      current_idx = i
      break

  if current_idx < 0:
    return

  # Copy results to higher-grade models from same company
  prompt_file = rp.model_prompt_path(aiEngineName, index, subPass)
  result_file = rp.model_raw_path(aiEngineName, index, subPass)
  cot_file = rp.model_cot_path(aiEngineName, index, subPass)

  if not os.path.exists(prompt_file) or not os.path.exists(result_file):
    prompt_file = rp.legacy_prompt_path(aiEngineName, index, subPass)
    result_file = rp.legacy_raw_path(aiEngineName, index, subPass)
    cot_file = rp.legacy_cot_path(aiEngineName, index, subPass)
    if not os.path.exists(prompt_file) or not os.path.exists(result_file):
      return

  for i in range(current_idx + 1, len(ALL_MODEL_CONFIGS)):
    other_name = ALL_MODEL_CONFIGS[i].get('name', '')
    other_prefix = getCompanyPrefix(other_name)

    if other_prefix != company_prefix:
      continue

    # Copy files to higher-grade model
    target_prompt = rp.model_prompt_path(other_name, index, subPass)
    target_result = rp.model_raw_path(other_name, index, subPass)
    target_cot = rp.model_cot_path(other_name, index, subPass)
    rp.ensure_model_dirs(other_name)

    # Only copy if target doesn't already exist
    if not os.path.exists(target_prompt):
      try:
        import shutil
        shutil.copy(prompt_file, target_prompt)
        shutil.copy(result_file, target_result)
        if os.path.exists(cot_file):
          shutil.copy(cot_file, target_cot)
        print(
          f"Propagated perfect score from {aiEngineName} to {other_name} for Q{index}/S{subPass}")
      except Exception as e:
        print(f"Failed to propagate result: {e}")


def runTest(index: int,
            aiEngineHook: callable,
            aiEngineName: str,
            subtask_filter: Optional[Set[int]] = None) -> Dict[str, Any]:
  """
    Run a test and return results including score and any generated images.
    
    Args:
        index: Test index number
        aiEngineHook: The AI engine hook function
        aiEngineName: Name of the AI engine
        subtask_filter: Optional set of subtask indices to run. If None, runs all subtasks.
    
    Returns a dictionary containing:
        - 'average_score': float - average score across all subpasses
        - 'total_score': float - sum of all subpass scores
        - 'subpass_count': int - number of subpasses completed
        - 'subpass_results': list of dicts with individual subpass results
    """
  # load test file, compile it, and get its globals in a map:
  g = {"__file__": str(index) + ".py"}

  if not os.path.exists(str(index) + ".py"):
    raise StopIteration

  t = time.time()

  try:
    code = open("" + str(index) + ".py", encoding="utf-8").read()
    compiled = compile(code, "" + str(index) + ".py", "exec")
    exec(compiled, g)
  except ImportError:
    print("Missing dependancy. Run 'pip install -r requirements.txt' before running!")

  t2 = time.time()
  if t2 - t > 1: print(f"Loading test {index} took {t2 - t:.2f} seconds")

  if "skip" in g and not UNSKIP:
    return {"average_score": 0, "total_score": 0, "subpass_count": 0, "subpass_results": []}

  prompts = []
  structure = g["structure"]

  if "prepareSubpassPrompt" in g:
    # get the prompt and structure from the globals:
    subPass = 0
    while True:
      try:
        prompts.append(g["prepareSubpassPrompt"](subPass))
        subPass += 1
      except StopIteration:
        break
  else:
    prompts.append(g["prompt"])

  # Helper to run a single prompt and save results
  def run_single_prompt(idx, prompt):
    # Check saved prompt cache first (before CacheLayer)
    cached_result = checkSavedPromptCache(aiEngineName, index, idx, prompt)
    if cached_result is not None:
      if structure is None:
        return cached_result
      if structure is not None:
        if isinstance(cached_result, dict):
          return cached_result

    try:
      r = aiEngineHook(prompt, structure, index, idx)
      if not isinstance(r, (tuple, list)):
        print("The following result from the AI engine is about to fail:")
        print(r)
      result, chainOfThought = r
    except Exception as e:
      print("Failed to get result for subpass " + str(idx) + " - " + str(e))
      result = {"__exception": str(e)}
      chainOfThought = "Exception was thrown: " + str(e)

    try:
      open(rp.model_raw_path(aiEngineName, index, idx), "w", encoding="utf-8").write(str(result))
    except Exception as e:
      print("Failed to save result for subpass " + str(idx) + " - " + str(e))

    try:
      open(rp.model_prompt_path(aiEngineName, index, idx), "w",
           encoding="utf-8").write(str(prompts[idx]))
    except Exception as e:
      print("Failed to save prompt for subpass " + str(idx) + " - " + str(e))

    try:
      open(rp.model_cot_path(aiEngineName, index, idx), "w",
           encoding="utf-8").write(str(chainOfThought))
    except Exception as e:
      print("Failed to save chain of thought for subpass " + str(idx) + " - " + str(e))

    return result

  earlyFail = "earlyFail" in g
  results = [None] * len(prompts)

  # Determine which subtasks to run
  def should_run_subtask(idx):
    return subtask_filter is None or idx in subtask_filter

  if subtask_filter:
    for i, prompt in enumerate(prompts):
      if should_run_subtask(i):
        results[i] = run_single_prompt(i, prompt)
  elif earlyFail and len(prompts) > 1:
    # Run first prompt sequentially
    if "earlyFailSubpassSampleCount" in g:
      for i in range(g["earlyFailSubpassSampleCount"]):
        if should_run_subtask(i):
          results[i] = run_single_prompt(i, prompts[i])
    else:
      if should_run_subtask(0):
        results[0] = run_single_prompt(0, prompts[0])
  elif "singleThreaded" in g:
    for i, prompt in enumerate(prompts):
      if should_run_subtask(i):
        results[i] = run_single_prompt(i, prompt)
  else:
    # Parallelize AI engine calls
    with ThreadPoolExecutor() as executor:
      future_to_index = {
        executor.submit(run_single_prompt, i, prompt): i
        for i, prompt in enumerate(prompts) if should_run_subtask(i)
      }
      for future in as_completed(future_to_index):
        idx = future_to_index[future]
        results[idx] = future.result()

  # Result processing and grading helper
  def process_subpass(subPass, result):
    score = 0
    subpass_data = {"subpass": subPass, "score": 0, "startProcessingTime": time.time()}

    # Check for content violation - always grade as 0
    if isinstance(result, dict) and result.get("__content_violation__"):
      print(f"Content violation for subpass {subPass} - grading as 0")
      subpass_data["score"] = 0
      subpass_data[
        "scoreExplanation"] = f"Content violation: {result.get('reason', 'Policy violation')}"
      subpass_data[
        "output_nice"] = "<strong style='color:red'>FALSE-POSITIVE CONTENT VIOLATION</strong><br>"\
          "This prompt was blocked by the AI provider's content policy. (LOL. Sad trombone failure "\
            "sound. This engine gets a well-deserved instant and eternal 0.)"
      subpass_data["endProcessingTime"] = time.time()
      return 0, subpass_data

    if isinstance(result, dict) and result.get("__exception__"):
      subpass_data["score"] = 0
      subpass_data["scoreExplanation"] = "Failed 3 times with an exception:"
      subpass_data[
        "output_nice"] = f"Exceptions thrown 3 times. Last exception was:<br>"\
          "{result.get('reason', 'Unknown exception')}"
      subpass_data["endProcessingTime"] = time.time()
      return 0, subpass_data

    if isinstance(result, dict) and "reasoning" in result:
      subpass_data["reasoning"] = result["reasoning"]
    elif isinstance(result, dict) and "reasoningAndDiscussion" in result:
      subpass_data["reasoningAndDiscussion"] = result["reasoningAndDiscussion"]

    if not result:
      print(f"No answer generated for subpass {subPass}")
      subpass_data["score"] = 0
    elif "resultToImage" in g:
      niceResult = None
      gaResult = g["gradeAnswer"](result, subPass, aiEngineName)
      if len(gaResult) == 2:
        score, explanation = gaResult
      else:
        score, explanation, niceResult = gaResult

      output_path = g["resultToImage"](result, subPass, aiEngineName)
      if niceResult is not None:
        subpass_data["output_nice"] = niceResult
      elif "resultToNiceReport" in g:
        subpass_data["output_nice"] = g["resultToNiceReport"](result, subPass, aiEngineName)
      else:
        subpass_data["output_text"] = result
        subpass_data["output_image"] = output_path

      if "getReferenceImage" in g:
        subpass_data["reference_image"] = g["getReferenceImage"](subPass, aiEngineName)
      subpass_data["score"] = score
      subpass_data["scoreExplanation"] = explanation

    elif _current_runner and _current_runner.can_handle_custom_scoring(g):
      # Custom scoring hook - allows subclasses to handle domain-specific scoring
      # (e.g., OpenSCAD volume comparison for spatial benchmarks)
      comparison_result = _current_runner.process_custom_scoring(index, subPass, result, g,
                                                                 aiEngineName)
      score = comparison_result["score"]
      subpass_data["score"] = score
      subpass_data["output_image"] = comparison_result.get("output_image")
      subpass_data["output_additional_images"] = comparison_result.get("output_additional_images")
      subpass_data["reference_image"] = comparison_result.get("reference_image")
      subpass_data["temp_dir"] = comparison_result.get("temp_dir")
      subpass_data["scoreExplanation"] = comparison_result.get("scoreExplanation")
      subpass_data["output_hyperlink"] = comparison_result.get("output_hyperlink")

    elif "gradeAnswer" in g:
      # Some tests require a custom grading function.

      try:
        gaResult = g["gradeAnswer"](result, subPass, aiEngineName)
      except Exception as e:
        print("Failed to grade subpass " + str(subPass) + " - " + str(e))
        score = 10000  # This should get attention!
        explanation = "Failed to grade subpass " + str(subPass) + " - " + str(e) + \
            "This is a framework error, not an AI error."
        gaResult = (score, explanation)

      niceResult = None
      if len(gaResult) == 2:
        score, explanation = gaResult
      else:
        score, explanation, niceResult = gaResult

      subpass_data["score"] = score
      subpass_data["scoreExplanation"] = explanation

      if niceResult is not None:
        subpass_data["output_nice"] = niceResult
      elif "resultToNiceReport" in g:
        try:
          subpass_data["output_nice"] = g["resultToNiceReport"](result, subPass, aiEngineName)
        except Exception as e:
          print("Failed to generate nice report for subpass " + str(subPass) + " - " + str(e))
          subpass_data["output_nice"] = "Failed to generate nice report for subpass " + str(
            subPass) + " - " + str(e)
      else:
        subpass_data["output_text"] = result

    subpass_data["endProcessingTime"] = time.time()

    if "extraGradeAnswerRuns" not in g:
      # HACK: Propagate perfect scores to higher-grade models from same company
      propogateUpwardsHack(aiEngineName, index, subPass, score)

    return score, subpass_data

  totalScore = 0
  subpass_results = [None] * len(prompts)
  earlyFailTriggered = False

  # When subtask_filter is specified, skip earlyFail logic - just run the specified subtasks
  if subtask_filter is not None:
    earlyFail = False

  if earlyFail and len(prompts) > 1:
    # Process first subpass sequentially
    resumeIndex = 1
    if "earlyFailSubpassSampleCount" in g:
      resumeIndex = g["earlyFailSubpassSampleCount"]
      for i in range(g["earlyFailSubpassSampleCount"]):
        passScore, first_subpass_data = process_subpass(i, results[i])
        totalScore += passScore
        subpass_results[i] = first_subpass_data
      first_score = totalScore / int(g["earlyFailSubpassSampleCount"])
    else:
      first_score, first_subpass_data = process_subpass(0, results[0])
      totalScore += first_score
      subpass_results[0] = first_subpass_data

    earlyFailThreshold = 0.5
    if "earlyFailThreshold" in g:
      earlyFailThreshold = g["earlyFailThreshold"]

    print(
      f"Early fail check. {first_score} < {earlyFailThreshold}? {first_score < earlyFailThreshold}")

    if first_score < earlyFailThreshold:
      # Early fail: assume all other subpasses will score 0
      earlyFailTriggered = True

      earlyFailScore = 0
      if "earlyFailTestsSameDifficulty" in g:
        earlyFailScore = first_score

      for subPass in range(resumeIndex, len(prompts)):
        subpass_results[subPass] = {
          "subpass":
          subPass,
          "score":
          earlyFailScore,
          "scoreExplanation":
          f"Skipped due to earlyFail (first subpass scored under {earlyFailScore*100}%)"
        }
    elif "singleThreaded" in g:
      for i in range(resumeIndex, len(prompts)):
        results[i] = run_single_prompt(i, prompts[i])
        passScore, first_subpass_data = process_subpass(i, results[i])
        totalScore += passScore
        subpass_results[i] = first_subpass_data
    else:
      # First subpass passed, run remaining prompts in parallel
      with ThreadPoolExecutor() as executor:
        future_to_index = {
          executor.submit(run_single_prompt, i, prompts[i]): i
          for i in range(resumeIndex, len(prompts))
        }
        for future in as_completed(future_to_index):
          idx = future_to_index[future]
          results[idx] = future.result()

      # Process remaining subpasses in parallel
      with ThreadPoolExecutor() as executor:
        future_to_subpass = {
          executor.submit(process_subpass, subPass, results[subPass]): subPass
          for subPass in range(resumeIndex, len(prompts))
        }
        for future in as_completed(future_to_subpass):
          subPass = future_to_subpass[future]
          score, subpass_data = future.result()
          totalScore += score
          subpass_results[subPass] = subpass_data

  elif "singleThreaded" in g:
    for i, result in enumerate(results):
      if result is None:
        continue  # Skip subtasks that weren't run (due to subtask_filter)
      score, subpass_data = process_subpass(i, result)
      totalScore += score
      subpass_results[i] = subpass_data
  else:
    with ThreadPoolExecutor() as executor:
      future_to_subpass = {
        executor.submit(process_subpass, subPass, result): subPass
        for subPass, result in enumerate(results) if result is not None
      }
      for future in as_completed(future_to_subpass):
        subPass = future_to_subpass[future]
        score, subpass_data = future.result()
        totalScore += score
        subpass_results[subPass] = subpass_data

  if "extraGradeAnswerRuns" in g:
    extraGradeAnswerRuns: list = g["extraGradeAnswerRuns"]
    if 0 in extraGradeAnswerRuns:
      extraGradeAnswerRuns.remove(0)
    gotAZero = False
    for subPass in extraGradeAnswerRuns:
      if gotAZero:
        subpass_data = {}
        subpass_data["score"] = 0
        subpass_data["subpass"] = subPass
        subpass_data["scoreExplanation"] = "Skipped due to earlier zero score."
        subpass_data["output_nice"] = "Skipped due to earlier zero score"
        subpass_results.append(subpass_data)
        continue

      print(f"Running extra subpass {subPass} for grading with engine {aiEngineName}")
      subpass_data = {}
      start = time.time()
      gaResult = g["gradeAnswer"](results[0], subPass, aiEngineName)
      execution_time = time.time() - start
      if execution_time > 1: print(f"Grade Answer {subPass} took {execution_time:.2f}s")
      niceResult = None
      if len(gaResult) == 2:
        score, explanation = gaResult
      else:
        score, explanation, niceResult = gaResult

      if score <= 0:
        gotAZero = True
      totalScore += score
      subpass_data["score"] = score
      subpass_data["subpass"] = subPass
      subpass_data["scoreExplanation"] = explanation

      if niceResult is not None:
        subpass_data["output_nice"] = niceResult
      else:
        report_start = time.time()
        subpass_data["output_nice"] = g["resultToNiceReport"](results[0], subPass, aiEngineName)
        report_time = time.time() - report_start
        if report_time > 1: print(f"Result to nice report {subPass} took {report_time:.2f}s")
      subpass_results.append(subpass_data)
      print()
  # Filter out None entries (subtasks that weren't run due to subtask_filter)
  actual_subpass_results = [r for r in subpass_results if r is not None]
  return {
    "average_score": totalScore / len(actual_subpass_results) if actual_subpass_results else 0,
    "total_score": max(0, totalScore),
    "subpass_count": len(actual_subpass_results),
    "subpass_results": actual_subpass_results
  }


def runAllTests(aiEngineHook: callable,
                aiEngineName: str,
                test_filter: Optional[Dict[int, Optional[Set[int]]]] = None):
  """
    Run all tests for an AI engine.
    
    Args:
        aiEngineHook: The AI engine hook function
        aiEngineName: Name of the AI engine
        test_filter: Optional dict mapping test indices to subtask filters.
                     If None, runs all tests. If a test maps to None, runs all subtasks.
                     If a test maps to a Set[int], runs only those subtasks.
    """
  rp.ensure_global_result_dirs()
  rp.ensure_model_dirs(aiEngineName)
  current_model_token = rp.set_current_model(aiEngineName)

  # Create a results file for the html results of this engines test run
  resultFilePath = rp.model_report_path(aiEngineName) if test_filter is None else os.path.join(
    rp.model_root(aiEngineName), "report.partial.html")
  results_file = open(resultFilePath, "w", buffering=1, encoding="utf-8")
  results_file.write("<html>\n<head>\n<style>\n")
  results_file.write("""
:root {
    --bg-color: #ffffff;
    --text-color: #333;
    --text-secondary: #666;
    --border-color: #ddd;
    --header-bg: #4CAF50;
    --header-text: white;
    --test-header-bg: #45a049;
    --prompt-bg: #f9f9f9;
    --subpass-bg: #ffffff;
    --img-border: #ccc;
    --score-good: #228B22;
    --score-bad: #dc3545;
}
@media screen and (prefers-color-scheme: dark) {
    :root {
        --bg-color: #1a1a1a;
        --text-color: #e0e0e0;
        --text-secondary: #aaa;
        --border-color: #444;
        --header-bg: #2d5a2d;
        --header-text: #e0e0e0;
        --test-header-bg: #3d6a3d;
        --prompt-bg: #2a2a2a;
        --subpass-bg: #1f1f1f;
        --img-border: #555;
        --score-good: #4CAF50;
        --score-bad: #ff6b6b;
    }
    a[href] { color: #55f}
}
body { background-color: var(--bg-color); color: var(--text-color); }
table { border-collapse: collapse; width: 100%; margin: 20px 0; }
th, td { border: 1px solid var(--border-color); padding: 12px; text-align: left; vertical-align: top; }
th { background-color: var(--header-bg); color: var(--header-text); }
.test-header { background-color: var(--test-header-bg); font-weight: bold; }
.prompt-row { background-color: var(--prompt-bg); font-style: italic; }
.subpass-row { background-color: var(--subpass-bg); }
img { max-width: 100%; height: auto; border: 1px solid var(--img-border); }
.score-good { color: var(--score-good); font-weight: bold; }
.score-bad { color: var(--score-bad); font-weight: bold; }
h1 { color: var(--text-color); }
h2 { color: var(--text-secondary); margin-top: 30px; }
""")
  results_file.write("</style>\n<meta charset='UTF-8'/>\n")

  # Add VizManager for WebGL context virtualization - only one active context at a time
  results_file.write("""
<script>
// VizManager: Virtualizes WebGL contexts to prevent "Too many active WebGL contexts" errors
// Only the visualization closest to the viewport center is active; all others are disposed
window.VizManager = (function() {
    const registeredViz = [];
    let activeVizId = null;
    let scrollTimeout = null;
    
    function getDistanceToViewportCenter(element) {
        const rect = element.getBoundingClientRect();
        const viewportCenter = window.innerHeight / 2;
        const elementCenter = rect.top + rect.height / 2;
        return Math.abs(elementCenter - viewportCenter);
    }
    
    function updateActiveVisualization() {
        let closestViz = null;
        let closestDistance = Infinity;
        
        for (const viz of registeredViz) {
            const container = document.getElementById(viz.containerId);
            if (!container) continue;
            
            // Check if container is in an open <details> element
            const details = container.closest('details');
            if (details && !details.open) continue;
            
            const distance = getDistanceToViewportCenter(container);
            if (distance < closestDistance) {
                closestDistance = distance;
                closestViz = viz;
            }
        }
        
        if (closestViz && closestViz.id !== activeVizId) {
            // Deactivate current
            if (activeVizId) {
                const currentViz = registeredViz.find(v => v.id === activeVizId);
                if (currentViz && currentViz.dispose) {
                    try { currentViz.dispose(); } catch(e) { console.warn('Dispose error:', e); }
                }
            }
            
            // Activate new
            activeVizId = closestViz.id;
            if (closestViz.activate) {
                try { closestViz.activate(); } catch(e) { console.warn('Activate error:', e); }
            }
        }
    }
    
    function onScrollOrResize() {
        if (scrollTimeout) clearTimeout(scrollTimeout);
        scrollTimeout = setTimeout(updateActiveVisualization, 100);
    }
    
    // Listen for scroll and resize
    window.addEventListener('scroll', onScrollOrResize, { passive: true });
    window.addEventListener('resize', onScrollOrResize, { passive: true });
    
    // Also listen for details toggle events
    document.addEventListener('toggle', function(e) {
        if (e.target.tagName === 'DETAILS') {
            setTimeout(updateActiveVisualization, 50);
        }
    }, true);
    
    return {
        register: function(vizConfig) {
            // vizConfig: { id, containerId, activate, dispose }
            registeredViz.push(vizConfig);
            // Trigger update after a short delay to allow DOM to settle
            setTimeout(updateActiveVisualization, 100);
        },
        
        forceUpdate: function() {
            updateActiveVisualization();
        },
        
        getActiveId: function() {
            return activeVizId;
        }
    };
})();
</script>
</head>
<body>
""")
  results_file.write("<h1>Benchmark Results for " + aiEngineName + "</h1>\n")
  results_file.write("<table>\n")

  testIndex = 1
  overall_total_score = 0
  overall_max_score = 0
  per_question_scores = {}  # {question_num: {"title": str, "score": float, "max": int}}

  longestProcessor = (None, None), 0

  while True:
    test_will_run = test_filter is None or testIndex in test_filter
    subtask_filter = test_filter.get(testIndex) if test_filter else None

    if not test_will_run:
      testIndex += 1
      if not os.path.exists(str(testIndex) + ".py"):
        break
      continue

    print("\n" + "=" * 60)
    print(f"TEST {testIndex} START")
    print("=" * 60)

    try:
      if not os.path.exists(str(testIndex) + ".py"):
        break
      test_result = {}
      # Load test metadata
      test_globals = {"__file__": str(testIndex) + ".py"}
      compiled = compile(
        open(str(testIndex) + ".py", encoding="utf-8").read(),
        str(testIndex) + ".py", "exec")
      exec(compiled, test_globals)

      test_was_run = test_filter is None or testIndex in test_filter
      if test_was_run:
        test_result = runTest(testIndex, aiEngineHook, aiEngineName, subtask_filter)
      else:
        test_result = {"total_score": 0, "subpass_count": 0, "subpass_results": []}

      current_test_index = testIndex
      testIndex += 1

      # Calculate max score for this test (1 point per subpass)
      max_score = test_result['subpass_count']
      overall_total_score += test_result['total_score']
      overall_max_score += max_score

      # Track per-question scores (only for tests that were actually run)
      if test_was_run:
        question_title = test_globals.get("title", f"Test {current_test_index}")
        # Build subtask scores dict: {subtask_num: score}
        subtask_scores = {}
        for subpass in test_result['subpass_results']:
          subtask_scores[subpass['subpass']] = subpass['score']
        per_question_scores[current_test_index] = {
          "title": question_title,
          "score": test_result['total_score'],
          "max": max_score,
          "subtasks": subtask_scores
        }

    except StopIteration:
      break

    # Console output
    print(f"Score: {test_result['total_score']:.2f} / {max_score}")
    print(f"Subpasses Completed: {test_result['subpass_count']}")
    print("\nSubpass Details:")

    for subpass in test_result['subpass_results']:
      print(f"  Subpass {subpass['subpass']}: Score = {subpass['score']:.4f}")
      #print(subpass)
      if 'endProcessingTime' in subpass and 'startProcessingTime' in subpass:
        timeTaken = subpass['endProcessingTime'] - subpass['startProcessingTime']
        if timeTaken > longestProcessor[1]:
          longestProcessor = ((testIndex - 1, subpass['subpass']), timeTaken)

    print("\n" + "=" * 60)
    print(f"TEST {testIndex-1} COMPLETED!")
    print("=" * 60)

    # HTML output

    # Header row: Test name, number of subpasses, score
    test_name = f"Test {testIndex-1}"

    # Extract test purpose from title or prompt (first line)
    test_purpose = ""
    if "title" in test_globals:
      test_purpose = test_globals["title"]
    elif "prompt" in test_globals:
      prompt_lines = test_globals["prompt"].strip().split("\n")
      test_purpose = prompt_lines[0] if prompt_lines else "No description available"
    else:
      test_purpose = "Unnamed test"

    score_class = "score-good" if test_result['total_score'] >= max_score * 0.7 else "score-bad"

    results_file.write("  <tr class='test-header'>\n")
    results_file.write(
      f"    <th colspan=3><a name='q{testIndex-1}'>{test_name}: {test_purpose}</a></th>\n")
    results_file.write(
      f"    <th class='{score_class}'>Score: {test_result['total_score']:.2f} / {max_score}</th>\n")
    results_file.write("  </tr>\n")

    # Prompt row: Typical prompt and how it changes
    results_file.write("  <tr class='prompt-row'>\n")
    results_file.write(
      "    <td colspan='3'><div style='overflow-x: auto;height: 200px;'><strong>Typical Prompt:</strong><br>"
    )

    def handlePromptImages(prompt: str) -> str:
      if "[[image:" not in prompt:
        return prompt

      out = prompt
      while "[[image:" in out and "]]" in out:
        start = out.find("[[image:")
        end = out.find("]]", start)
        if end < 0:
          break
        raw_ref = out[start + len("[[image:"):end].strip()
        mapped_ref = rp.convert_prompt_image_ref_for_model_report(raw_ref, aiEngineName)
        replacement = f"<img src='{html.escape(mapped_ref)}' width='200px'>"
        out = out[:start] + replacement + out[end + 2:]
      return out

    if "prepareSubpassPrompt" in test_globals:
      # Show first subpass prompt
      try:
        first_prompt = test_globals["prepareSubpassPrompt"](0)
        results_file.write(handlePromptImages(first_prompt).replace("\n", "<br>"))
      except:
        results_file.write("Dynamic prompt generation")
    elif "prompt" in test_globals:
      prompt_text = test_globals["prompt"].strip()
      results_file.write(handlePromptImages(prompt_text).replace("\n", "<br>"))
    else:
      results_file.write("No prompt available")

    results_file.write("</div></td>\n")
    results_file.write("    <td><strong>Prompt Changes:</strong><br>")

    if "promptChangeSummary" in test_globals:
      results_file.write(test_globals["promptChangeSummary"])
    elif "prepareSubpassPrompt" in test_globals:
      results_file.write("Prompt parameters change between subpasses (increasing difficulty)")
    else:
      results_file.write("Prompt remains constant")

    results_file.write("</td>\n")
    results_file.write("  </tr>\n")

    # Subpass rows
    for subpass in test_result['subpass_results']:
      results_file.write("  <tr class='subpass-row'>\n")

      # Subpass overview
      results_file.write(f"    <td rowspan=2><strong>Subpass {subpass['subpass']}</strong><br>")

      if "subpassParamSummary" in test_globals and subpass['subpass'] < len(
          test_globals["subpassParamSummary"]):
        results_file.write(
          f"Parameters: {test_globals['subpassParamSummary'][subpass['subpass']]}<br>")
      elif "prepareSubpassPrompt" in test_globals:
        try:
          subpass_prompt = test_globals["prepareSubpassPrompt"](subpass['subpass'])
          # Extract parameters from prompt
          results_file.write("Parameters: ")
          if "PARAM_" in subpass_prompt:
            results_file.write("(modified from base prompt)")
          else:
            results_file.write("(see typical prompt)")
        except:
          results_file.write("Subpass configuration")
      else:
        results_file.write("Same as typical prompt")

      results_file.write("<a href=\"prompts/" + aiEngineName + "_" + str(testIndex - 1) + "_" +
                         str(subpass['subpass']) + ".txt\">View exact prompt</a><br>")
      results_file.write("<a href=\"raw/" + aiEngineName + "_" + str(testIndex - 1) + "_" +
                         str(subpass['subpass']) + ".txt\">View raw AI output</a><br>")
      results_file.write("<a href=\"cot/" + aiEngineName + "_" + str(testIndex - 1) + "_" +
                         str(subpass['subpass']) + ".txt\">View chain of thought</a><br>")

      if "highLevelSummary" in test_globals and subpass['subpass'] == 0:
        results_file.write(test_globals['highLevelSummary'])

      results_file.write("</td>\n")

      if "reasoning" in subpass:
        results_file.write(
          f"<td colspan=2><div style='overflow-y: auto;max-height: 100px;'><strong>AI Reasoning: </strong>{html.escape(subpass['reasoning'])}</div></td>"
        )
      else:
        results_file.write("<td colspan=2></td>")

      # Score
      score_class = "score-good" if subpass['score'] >= 0.7 else "score-bad"
      results_file.write(
        f"    <td rowspan=2 class='{score_class}'><strong>{subpass['score']:.4f}</strong>")

      if "scoreExplanation" in subpass and subpass['scoreExplanation']:
        results_file.write(
          "<br><div style='font-size: 12px; font-style: italic; color: #666; margin-left: 20px; overflow-x: auto; max-width:200px;'>"
          + subpass['scoreExplanation'].replace("\n", "<br>") + "</div>")

      results_file.write("</td>\n")
      results_file.write("  </tr>\n")

      # Images
      if 'output_image' in subpass and subpass['output_image']:
        # Actual image
        results_file.write("    <td>")

        if "output_hyperlink" in subpass and subpass['output_hyperlink']:
          results_file.write(f"<a href='{subpass['output_hyperlink']}'>")
        if ('output_additional_images' in subpass and subpass['output_additional_images']
            and os.path.exists(subpass['output_additional_images'][0])
            and os.path.exists(subpass['output_image'])):
          # Build list of all images (main + additional)
          all_images = [subpass['output_image']] + [
            img for img in subpass['output_additional_images'] if os.path.exists(img)
          ]
          img_data_list = []
          for img_path in all_images:
            with open(img_path, 'rb') as img_file:
              img_data_list.append(base64.b64encode(img_file.read()).decode('utf-8'))

          # Generate unique viewer ID
          viewer_id = f"flickbook-{hashlib.md5(subpass['output_image'].encode()).hexdigest()[:12]}"
          radio_name = f"{viewer_id}-view"

          # Build radio inputs (hidden)
          inputs_html = []
          for idx in range(len(img_data_list)):
            checked = " checked" if idx == 0 else ""
            inputs_html.append(
              f'<input type="radio" name="{radio_name}" id="{viewer_id}-{idx}"{checked}>')

          # Build prev/next labels
          labels_html = []
          for idx in range(len(img_data_list)):
            prev_idx = (idx - 1) % len(img_data_list)
            next_idx = (idx + 1) % len(img_data_list)
            labels_html.append(
              f'<label class="fb-prev prev-{idx}" for="{viewer_id}-{prev_idx}">&#8592;</label>')
            labels_html.append(
              f'<label class="fb-next next-{idx}" for="{viewer_id}-{next_idx}">&#8594;</label>')

          # Build image tags
          image_tags = []
          for idx, img_b64 in enumerate(img_data_list):
            image_tags.append(
              f'<img src="data:image/png;base64,{img_b64}" class="fb-view view-{idx}" alt="Output">'
            )

          # Build CSS rules
          style_lines = [
            f'#{viewer_id} {{ display:flex; align-items:center; gap:8px; }}',
            f'#{viewer_id} input[type="radio"] {{ display:none; }}',
            f'#{viewer_id} .fb-frame {{ flex:1; text-align:center; order:1; }}',
            f'#{viewer_id} .fb-prev {{ order:0; cursor:pointer; font-size:18px; display:none; user-select:none; }}',
            f'#{viewer_id} .fb-next {{ order:2; cursor:pointer; font-size:18px; display:none; user-select:none; }}',
            f'#{viewer_id} .fb-view {{ display:none; max-width:100%; }}',
          ]
          for idx in range(len(img_data_list)):
            style_lines.append(
              f'#{viewer_id}-{idx}:checked ~ .fb-frame .view-{idx} {{ display:block; }}')
            style_lines.append(
              f'#{viewer_id}-{idx}:checked ~ .prev-{idx} {{ display:inline-flex; }}')
            style_lines.append(
              f'#{viewer_id}-{idx}:checked ~ .next-{idx} {{ display:inline-flex; }}')

          results_file.write(f'<div id="{viewer_id}" class="flickbook-viewer">'
                             f'<style>{" ".join(style_lines)}</style>'
                             f'{"".join(inputs_html)}'
                             f'{"".join(labels_html)}'
                             f'<div class="fb-frame">{"".join(image_tags)}</div>'
                             f'</div>')

        elif os.path.exists(subpass['output_image']):
          try:
            with open(subpass['output_image'], 'rb') as img_file:
              img_data = base64.b64encode(img_file.read()).decode('utf-8')
              results_file.write(f"<img src='data:image/png;base64,{img_data}' alt='Output'>")
          except:
            output_rel = rp.relative_path_from_model_root(subpass['output_image'], aiEngineName)
            results_file.write(f"<a href='{output_rel}'>View Output Image</a>")
        else:
          results_file.write("Image not found")

        if "output_hyperlink" in subpass and subpass['output_hyperlink']:
          results_file.write(f"</a>")
        results_file.write("</td>\n")

        # Reference image
        results_file.write("    <td>")
        if 'reference_image' in subpass and subpass['reference_image'] and os.path.exists(
            subpass['reference_image']):
          try:
            with open(subpass['reference_image'], 'rb') as img_file:
              img_data = base64.b64encode(img_file.read()).decode('utf-8')
              results_file.write(f"<img src='data:image/png;base64,{img_data}' alt='Reference'>")
          except:
            reference_rel = rp.relative_path_from_model_root(subpass['reference_image'],
                                                             aiEngineName)
            results_file.write(f"<a href='{reference_rel}'>View Reference Image</a>")
        else:
          results_file.write("No reference image")
        results_file.write("</td>\n")
      elif "output_nice" in subpass:
        # Nice preformatted output, if it contains table cells just display as is:
        if "</td><td>" in subpass["output_nice"]:
          results_file.write(subpass["output_nice"])
        else:
          results_file.write("    <td colspan='2'>" + subpass["output_nice"] + "</td>\n")
      elif "output_text" in subpass:
        # Text output only
        results_file.write(
          "    <td colspan='2'><pre style='max-width: 1000px; overflow-x: auto;'>" +
          html.escape(str(subpass["output_text"])) + "</pre></td>\n")
      else:
        # No images for this test type
        results_file.write("    <td>N/A (LLM did not answer)</td>\n")
        results_file.write("    <td>N/A (Test forfeited)</td>\n")

  # Overall summary
  results_file.write("<h2>Overall Summary</h2>\n")
  results_file.write("<table>\n")
  results_file.write("  <tr class='test-header'>\n")
  results_file.write("    <th>Total Tests</th>\n")
  results_file.write("    <th>Overall Score</th>\n")
  results_file.write("    <th>Percentage</th>\n")
  results_file.write("  </tr>\n")
  results_file.write("  <tr>\n")
  results_file.write(f"    <td>{testIndex - 1}</td>\n")
  results_file.write(f"    <td>{overall_total_score:.2f} / {overall_max_score}</td>\n")
  percentage = (overall_total_score / overall_max_score * 100) if overall_max_score > 0 else 0
  results_file.write(f"    <td>{percentage:.1f}%</td>\n")
  results_file.write("  </tr>\n")
  results_file.write("</table>\n")

  results_file.write("</body>\n</html>\n")
  results_file.close()

  print("\n" + "=" * 60)
  print("BENCHMARK COMPLETE")
  print("=" * 60)
  print(f"Total Score: {overall_total_score:.2f} / {overall_max_score} ({percentage:.1f}%)")
  print(f"Results saved to: {resultFilePath}")
  print("=" * 60)

  scores = {}

  # Use file lock to prevent race conditions when running parallel
  results_lock = FileLock("results/results.txt.lock")
  with results_lock:
    if not os.path.exists("results/results.txt"):
      with open("results/results.txt", "w", encoding="utf-8") as f:
        f.write("\n")

    with open("results/results.txt", "r", encoding="utf-8") as f:
      for line in f:
        if ":" in line:
          scores[line.split(":")[0].strip()] = line.split(":")[1].strip()

    if test_filter is None:
      # Don't overwrite scores if we're running a subset of tests.
      scores[aiEngineName] = overall_total_score / overall_max_score if overall_max_score > 0 else 0

    with open("results/results.txt", "w", encoding="utf-8") as f:
      for key, value in sorted(scores.items(), key=lambda item: float(item[1]), reverse=True):
        f.write(f"{key}: {value}\n")

  print("results.txt updated!")

  # Save per-question results to JSON
  import json
  per_question_file = "results/results_by_question.json"
  per_question_lock = FileLock("results/results_by_question.json.lock")
  with per_question_lock:
    all_per_question = {}
    if os.path.exists(per_question_file):
      with open(per_question_file, "r", encoding="utf-8") as f:
        try:
          all_per_question = json.load(f)
        except:
          all_per_question = {}

    # Merge with existing data for this engine (don't overwrite other questions/subtasks)
    if aiEngineName not in all_per_question:
      all_per_question[aiEngineName] = {}
    for q_num, q_data in per_question_scores.items():
      q_str = str(q_num)
      if q_str in all_per_question[aiEngineName]:
        # Merge subtasks instead of replacing whole question
        existing = all_per_question[aiEngineName][q_str]
        # Ensure existing has subtasks key (old data may not have it)
        if "subtasks" not in existing:
          existing["subtasks"] = {}
        existing["subtasks"].update(q_data.get("subtasks", {}))
        # Recalculate score and max from merged subtasks
        existing["score"] = sum(existing["subtasks"].values())
        existing["max"] = len(existing["subtasks"])
      else:
        all_per_question[aiEngineName][q_str] = q_data

    with open(per_question_file, "w", encoding="utf-8") as f:
      json.dump(all_per_question, f, indent=2)

  print("results_by_question.json updated!")

  if test_filter is not None:
    return

  # Generate a summary page of the results, suitable for use as a github landing page,
  # including a big graph of the results by engine name

  import matplotlib.pyplot as plt
  import pandas as pd

  df = pd.read_csv("results/results.txt", sep=":", header=None, names=["Engine", "Score"])

  di = df.to_dict()
  for i in range(df.shape[0]):
    e = di["Engine"][i]

    e = e.replace("-HighReasoning", "+R")
    e = e.replace("-Reasoning-Tools", "+RT")
    e = e.replace("-bedrock", "")
    di["Engine"][i] = e

  df = df.from_dict(di)

  # Use horizontal bar chart for better label readability
  fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.5)))
  ax.barh(df["Engine"], df["Score"], color='#1f77b4')
  ax.set_xlabel("Score")
  ax.set_ylabel("")
  ax.set_title(_current_runner.get_benchmark_title() if _current_runner else "Benchmark Results")
  ax.invert_yaxis()  # Highest score at top
  plt.tight_layout()
  plt.savefig("results/topLevelResults.png", dpi=600)
  plt.close()

  print("topLevelResults.png saved")

  # Generate per-question graphs
  question_graphs = {}  # {question_num: {"title": str, "filename": str}}

  # Get all question numbers from all engines
  all_questions = set()
  for engine_data in all_per_question.values():
    all_questions.update(int(q) for q in engine_data.keys())

  for q_num in sorted(all_questions):
    q_str = str(q_num)
    # Collect scores for this question from all engines
    engine_scores = []
    question_title = f"Question {q_num}"
    max_score = 1

    for engine_name, engine_data in all_per_question.items():
      if q_str in engine_data:
        q_data = engine_data[q_str]
        question_title = q_data.get("title", question_title)
        max_score = q_data.get("max", 1)
        score = q_data.get("score", 0)
        # Normalize to percentage of max
        normalized = score / max_score if max_score > 0 else 0
        engine_scores.append((engine_name, normalized))

    if not engine_scores:
      continue

    # Sort by score descending
    engine_scores.sort(key=lambda x: x[1], reverse=True)
    engines = [e[0] for e in engine_scores]
    scores_list = [e[1] for e in engine_scores]

    # Generate graph
    fig, ax = plt.subplots(figsize=(10, max(3, len(engines) * 0.4)))
    ax.barh(engines, scores_list, color='#667eea')
    ax.set_xlabel("Score (normalized)")
    ax.set_ylabel("")
    ax.set_title(f"Q{q_num}: {question_title}")
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    plt.tight_layout()

    filename = f"question_{q_num}.png"
    plt.savefig(f"results/{filename}", dpi=150)
    plt.close()

    best_engine, best_score = engine_scores[0] if engine_scores else ("", 0)
    if is_placebo_model(best_engine):
      best_engine, best_score = next(
        ((name, score) for name, score in engine_scores if not is_placebo_model(name)),
        ("", 0),
      )

    placebo_scores = [(name, score) for name, score in engine_scores if is_placebo_model(name)]
    placebo_engine, placebo_score = ("", 0)
    if placebo_scores:
      placebo_engine, placebo_score = max(placebo_scores, key=lambda item: item[1])

    question_graphs[q_num] = {
      "title": question_title,
      "filename": filename,
      "max": max_score,
      "best_engine": best_engine,
      "placebo_score": placebo_score,
      "placebo_engine": placebo_engine,
      "best_pct": best_score * 100
    }

  # Generate per-subtask graphs
  subtask_graphs = {
  }  # {(question_num, subtask_num): {"filename": str, "best_engine": str, "best_pct": float}}

  for q_num in sorted(all_questions):
    print(f"Updating graphs for {q_num}")
    q_str = str(q_num)
    # Get all subtask numbers for this question from all engines
    all_subtasks = set()
    for engine_data in all_per_question.values():
      if q_str in engine_data and "subtasks" in engine_data[q_str]:
        all_subtasks.update(int(s) for s in engine_data[q_str]["subtasks"].keys())

    for subtask_num in sorted(all_subtasks):
      subtask_str = str(subtask_num)
      # Collect scores for this subtask from all engines
      engine_scores = []

      for engine_name, engine_data in all_per_question.items():
        if q_str in engine_data and "subtasks" in engine_data[q_str]:
          subtasks = engine_data[q_str]["subtasks"]
          if subtask_str in subtasks:
            score = subtasks[subtask_str]
            engine_scores.append((engine_name, score))

      if not engine_scores:
        continue

      # Sort by score descending
      engine_scores.sort(key=lambda x: x[1], reverse=True)
      engines = [e[0] for e in engine_scores]
      scores_list = [e[1] for e in engine_scores]

      # Generate graph
      fig, ax = plt.subplots(figsize=(8, max(2, len(engines) * 0.35)))
      ax.barh(engines, scores_list, color='#10b981')
      ax.set_xlabel("Score")
      ax.set_ylabel("")
      ax.set_title(f"Q{q_num}.{subtask_num}")
      ax.set_xlim(0, 1)
      ax.invert_yaxis()
      plt.tight_layout()

      filename = f"question_{q_num}_subtask_{subtask_num}.png"
      plt.savefig(f"results/{filename}", dpi=100)
      plt.close()

      best_engine, best_score = engine_scores[0] if engine_scores else ("", 0)
      if is_placebo_model(best_engine):
        best_engine, best_score = next(
          ((name, score) for name, score in engine_scores if not is_placebo_model(name)),
          ("", 0),
        )

      subtask_graphs[(q_num, subtask_num)] = {
        "filename": filename,
        "best_engine": best_engine,
        "best_pct": best_score * 100
      }

  # Generate per-question subpass performance graphs (max, median, mean)
  import statistics
  subpass_perf_graphs = {}  # {q_num: {"max": filename, "median": filename, "mean": filename}}

  for q_num in sorted(all_questions):
    print(f"Updating subpass performance graphs for {q_num}")
    q_str = str(q_num)
    # Collect all subtask scores across all engines for this question
    # Structure: {subtask_num: [list of scores from all engines]}
    subtask_all_scores = {}

    for engine_name, engine_data in all_per_question.items():
      if is_placebo_model(engine_name): continue
      if q_str in engine_data and "subtasks" in engine_data[q_str]:
        for subtask_str, score in engine_data[q_str]["subtasks"].items():
          subtask_num = int(subtask_str)
          if subtask_num not in subtask_all_scores:
            subtask_all_scores[subtask_num] = []
          subtask_all_scores[subtask_num].append(score)

    if not subtask_all_scores:
      continue

    # Calculate max, median, mean for each subtask
    subtasks_sorted = sorted(subtask_all_scores.keys())
    max_scores = [max(subtask_all_scores[s]) for s in subtasks_sorted]
    median_scores = [statistics.median(subtask_all_scores[s]) for s in subtasks_sorted]
    mean_scores = [statistics.mean(subtask_all_scores[s]) for s in subtasks_sorted]

    subpass_perf_graphs[q_num] = {}

    # Generate max graph
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([str(s) for s in subtasks_sorted], max_scores, color='#10b981')
    ax.set_xlabel("Subpass")
    ax.set_ylabel("Score")
    ax.set_title(f"Q{q_num}: Max Score by Subpass")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    filename_max = f"question_{q_num}_subpass_max.png"
    plt.savefig(f"results/{filename_max}", dpi=100)
    plt.close()
    subpass_perf_graphs[q_num]["max"] = filename_max

    # Generate median graph
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([str(s) for s in subtasks_sorted], median_scores, color='#667eea')
    ax.set_xlabel("Subpass")
    ax.set_ylabel("Score")
    ax.set_title(f"Q{q_num}: Median Score by Subpass")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    filename_median = f"question_{q_num}_subpass_median.png"
    plt.savefig(f"results/{filename_median}", dpi=100)
    plt.close()
    subpass_perf_graphs[q_num]["median"] = filename_median

    # Generate mean graph
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([str(s) for s in subtasks_sorted], mean_scores, color='#f59e0b')
    ax.set_xlabel("Subpass")
    ax.set_ylabel("Score")
    ax.set_title(f"Q{q_num}: Mean Score by Subpass")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    filename_mean = f"question_{q_num}_subpass_mean.png"
    plt.savefig(f"results/{filename_mean}", dpi=100)
    plt.close()
    subpass_perf_graphs[q_num]["mean"] = filename_mean

  print("Done! Updating index.html page...")

  # Generate index.html landing page
  index_lock = FileLock("results/index.html.lock")
  with index_lock, open("results/index.html", "w", encoding="utf-8") as index_file:
    index_file.write(
      """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>""" +
      (_current_runner.get_benchmark_title() if _current_runner else "Benchmark Results") +
      """</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        h1 {
            margin: 0 0 10px 0;
            font-size: 2.5em;
        }
        .subtitle {
            opacity: 0.9;
            font-size: 1.1em;
        }
        .graph-container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .graph-container h2 {
            color: #333;
        }
        .graph-container img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 0 auto;
        }
        .results-table {
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th {
            background-color: #667eea;
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }
        td {
            padding: 15px;
            border-bottom: 1px solid #eee;
        }
        tr:hover {
            background-color: #f8f9fa;
        }
        tr:last-child td {
            border-bottom: none;
        }
        .score-cell {
            font-weight: bold;
            font-size: 1.1em;
        }
        .score-high {
            color: #10b981;
        }
        .score-medium {
            color: #f59e0b;
        }
        .score-low {
            color: #ef4444;
        }
        a {
            color: #667eea;
            text-decoration: none;
            font-weight: 500;
        }
        a:hover {
            text-decoration: underline;
        }
        .footer {
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            color: #666;
        }
        .badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.85em;
            font-weight: 600;
            margin-left: 10px;
        }
        .badge-best {
            background-color: #dcfce7;
            color: #166534;
        }
        @media (prefers-color-scheme: dark) {
            body {
                background-color: #1a1a2e;
                color: #e0e0e0;
            }
            .graph-container, .results-table {
                background: #16213e;
                box-shadow: 0 2px 4px rgba(0,0,0,0.3);
            }
            .graph-container h2 {
                color: #e0e0e0;
            }
            td {
                border-bottom-color: #2a2a4a;
            }
            tr:hover {
                background-color: #1f2b4a;
            }
            .footer {
                color: #888;
            }
            a {
                color: #8b9fea;
            }
            .badge-best {
                background-color: #1a4d2e;
                color: #6ee7a0;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>""" +
      (_current_runner.get_benchmark_title() if _current_runner else "Benchmark Results") + """</h1>
        <p class="subtitle">""" +
      (_current_runner.get_benchmark_subtitle() if _current_runner else "") + """</p>
        """ + (_current_runner.get_benchmark_description() if _current_runner else "") + """
    </div>
    
    <div class="graph-container">
        <h2 style="margin-top: 0;">Performance Overview</h2>
        <img src="topLevelResults.png" alt="Benchmark Results Graph">
    </div>
    
    <div class="results-table">
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Engine Name</th>
                    <th>Score</th>
                    <th>Percentage</th>
                    <th>Detailed Results</th>
                </tr>
            </thead>
            <tbody>
""")

    # Sort engines by score (descending)
    sorted_engines = sorted(scores.items(), key=lambda x: float(x[1]), reverse=True)
    best_score = float(sorted_engines[0][1]) if sorted_engines else 0

    for rank, (engine_name, score) in enumerate(sorted_engines, 1):
      score_float = float(score)
      percentage = score_float * 100

      # Determine score class
      if percentage >= 100:
        score_class = "score-low"
      elif percentage >= 70:
        score_class = "score-high"
      elif percentage >= 40:
        score_class = "score-medium"
      else:
        score_class = "score-low"

      # Add best badge
      badge = '<span class="badge badge-best">BEST</span>' if rank == 1 else ''

      index_file.write(f"""                <tr>
                    <td><strong>#{rank}</strong></td>
                    <td>{html.escape(engine_name)}{badge}</td>
                    <td class="score-cell {score_class}">{score_float:.4f}</td>
                    <td class="{score_class}">{percentage if percentage < 100 else 0:.1f}%</td>
                    <td><a href="{html.escape(rp.model_report_href(engine_name))}">View Details →</a></td>
                </tr>
""")

    index_file.write("""            </tbody>
        </table>
    </div>
""")

    # Add per-question sections
    if question_graphs:
      index_file.write("""
    <div class="graph-container">
        <h2 style="margin-top: 0;">Results by Question</h2>
    </div>
""")
      for q_num in sorted(question_graphs.keys()):
        # load test file, compile it, and get its globals in a map:
        g = {}

        if not os.path.exists(str(q_num) + ".py"):
          raise StopIteration

        exec(open("" + str(q_num) + ".py", encoding="utf-8").read(), g)

        if question_graphs[q_num]['placebo_score']:
          placebo_ratio = question_graphs[q_num]['best_pct'] /\
            (question_graphs[q_num]['placebo_score'] * 100)
          baseline_compare = "No Data"
          if placebo_ratio < 0.3:
            baseline_compare = "<p style='color:#0f0'> Placebo baseline is considerably better</p>"
          elif placebo_ratio < 0.9:
            baseline_compare = "<p style='color:#0f0'> Placebo baseline is better</p>"
          elif placebo_ratio < 0.99:
            baseline_compare = "<p style='color:#0f0'> Placebo baseline is marginally better</p>"
          elif placebo_ratio < 1.01:
            if question_graphs[q_num]['placebo_score'] == 0:
              baseline_compare = "<p style='color:#ff0'>Neither placebo baseline nor AI have solved this.</p>"
            elif question_graphs[q_num]['placebo_score'] == 1:
              baseline_compare = "<p style='color:#ff0'> Placebo baseline and the best AI have both mastered this.</p>"
            else:
              baseline_compare = "<p style='color:#ff0'> Placebo baseline and the best AI are equal.</p>"
          elif placebo_ratio < 1.5:
            baseline_compare = (
              "<p style='color:#f00'> The best AI is marginally better. "
              f"Placebo baseline scored {question_graphs[q_num]['placebo_score'] * 100:.1f}% </p>")
          else:
            baseline_compare = (
              "<p style='color:#f00'> The best AI is considerably better. "
              f"Placebo baseline scored {question_graphs[q_num]['placebo_score'] * 100:.1f}%</p>")
        else:
          if question_graphs[q_num]['best_pct'] == 0:
            baseline_compare = "<p style='color:#ff0'>Neither placebo baseline nor AI have solved this.</p>"
          else:
            baseline_compare = ("<p style='color:#f00'> The best AI is considerably better, "
                                "placebo baseline scored 0 or hasn't attempted.</p>")

        q_data = question_graphs[q_num]

        # Build subtask breakdown HTML
        subtask_html = ""
        q_subtasks = [(s, subtask_graphs[(q_num, s)])
                      for s in sorted([st for (q, st) in subtask_graphs.keys() if q == q_num])]
        if q_subtasks:
          subtask_rows = []
          for subtask_num, st_data in q_subtasks:
            score_color = "#10b981" if st_data[
              'best_pct'] >= 70 else "#f59e0b" if st_data['best_pct'] >= 40 else "#ef4444"
            subtask_rows.append(f"""
              <tr>
                <td><strong>{q_num}.{subtask_num}</strong></td>
                <td><a href="{html.escape(rp.model_report_href_with_anchor(st_data['best_engine'], f'q{q_num}'))}">{html.escape(st_data['best_engine'])}</a></td>
                <td style="color:{score_color}; font-weight:bold;">{st_data['best_pct']:.1f}%</td>
                <td><details style="display:inline"><summary style="cursor:pointer; color:#667eea; font-size:0.9em;">graph</summary><img src="{st_data['filename']}" style="max-width:400px; margin-top:5px;"></details></td>
              </tr>""")
          subtask_html = f"""
        <details style="margin-top:15px;">
            <summary style="cursor:pointer; color:#667eea;"><strong>Subtask Breakdown ({len(q_subtasks)} subtasks)</strong></summary>
            <table style="width:100%; margin-top:10px; font-size:0.9em;">
              <thead><tr style="background:#4a5568;"><th>Subtask</th><th>Best Engine</th><th>Score</th><th>Graph</th></tr></thead>
              <tbody>{"".join(subtask_rows)}</tbody>
            </table>
        </details>"""

        # Build subpass performance graphs HTML (max, median, mean)
        subpass_perf_html = ""
        if q_num in subpass_perf_graphs:
          perf = subpass_perf_graphs[q_num]
          subpass_perf_html = f"""
        <details style="margin-top:15px;">
            <summary style="cursor:pointer; color:#667eea;"><strong>Performance by Subpass (Max / Median / Mean)</strong></summary>
            <div style="display:flex; flex-wrap:wrap; gap:10px; margin-top:10px;">
              <div style="flex:1; min-width:250px;"><img src="{perf['max']}" style="width:100%;"></div>
              <div style="flex:1; min-width:250px;"><img src="{perf['median']}" style="width:100%;"></div>
              <div style="flex:1; min-width:250px;"><img src="{perf['mean']}" style="width:100%;"></div>
            </div>
        </details>"""

        index_file.write(f"""
    <div class="graph-container" id="q{q_num}">
        <img src="../images/{q_num}.png" style="float:right; max-width:400px">
        <a name="q{q_num}"><h2 style="margin-top: 0;color:#fff">Q{q_num}: {html.escape(q_data['title'])}</h2></a>
        {g.get("highLevelSummary","")}
        <p style='clear:both'><strong>Best result:</strong> <a href="{html.escape(rp.model_report_href_with_anchor(q_data['best_engine'], f'q{q_num}'))}">{html.escape(q_data['best_engine'])}</a> ({q_data['best_pct']:.1f}%)</p>
        {baseline_compare}
        <details>
            <summary style="cursor:pointer; color:#667eea;">Click to show comparison graph</summary>
            <img src="{q_data['filename']}" alt="Question {q_num} Results" style="margin-top:10px;">
        </details>
        {subpass_perf_html}
        {subtask_html}
    </div>
""")

    index_file.write("""
    <div class="footer">
        <p>Generated automatically by TestRunner.py</p>
        <p>Last updated: """ + __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M:%S") +
                     """</p>
    </div>
</body>
</html>
""")

  print(f"Landing page saved to: results/index.html")
  print(f"Longest processing time: {longestProcessor[1]} seconds for test {longestProcessor[0]}")
  rp.reset_current_model(current_model_token)


def run_model_config(config: dict, test_filter: Optional[Dict[int, Optional[Set[int]]]] = None):
  """Run tests for a single model configuration."""
  name = config["name"]
  engine_type = config["engine"]

  # Check if required API key is available
  env_key = config.get("env_key")
  if env_key and not os.environ.get(env_key):
    print(f"Skipping {name}: {env_key} not set")
    return

  if engine_type == "placebo":
    from .AiEnginePlacebo import PlaceboEngine
    placebo_id = config.get("placebo_id", name)
    engine = PlaceboEngine(placebo_id)
    runAllTests(engine.AIHook, name, test_filter)

  elif engine_type == "openai":
    from .AiEngineOpenAiChatGPT import OpenAIEngine
    timeout = config.get("timeout") or API_TIMEOUT_OVERRIDE or 3600
    engine = OpenAIEngine(config["base_model"],
                          config["reasoning"],
                          config["tools"],
                          timeout=timeout)
    cacheLayer = cl(engine.configAndSettingsHash, engine.AIHook, name)
    runAllTests(cacheLayer.AIHook, name, test_filter)

  elif engine_type in ("azure-openai", "azure_openai"):
    from .AiEngineAzureOpenAI import AzureOpenAIEngine
    timeout = config.get("timeout") or API_TIMEOUT_OVERRIDE or 3600
    engine = AzureOpenAIEngine(config["base_model"],
                               config["reasoning"],
                               config["tools"],
                               config.get("endpoint"),
                               config.get("api_version"),
                               timeout=timeout)
    cacheLayer = cl(engine.configAndSettingsHash, engine.AIHook, name)
    runAllTests(cacheLayer.AIHook, name, test_filter)

  elif engine_type == "gemini":
    from .AiEngineGoogleGemini import GeminiEngine
    timeout = config.get("timeout") or API_TIMEOUT_OVERRIDE or 3600
    engine = GeminiEngine(config["base_model"],
                          config["reasoning"],
                          config["tools"],
                          timeout=timeout)
    cacheLayer = cl(engine.configAndSettingsHash, engine.AIHook, name)
    runAllTests(cacheLayer.AIHook, name, test_filter)

  elif engine_type == "xai":
    from .AiEngineXAIGrok import GrokEngine
    timeout = config.get("timeout") or API_TIMEOUT_OVERRIDE or 3600
    engine = GrokEngine(config["base_model"], config["reasoning"], config["tools"], timeout=timeout)
    cacheLayer = cl(engine.configAndSettingsHash, engine.AIHook, name)
    runAllTests(cacheLayer.AIHook, name, test_filter)

  elif engine_type == "anthropic":
    from .AiEngineAnthropicClaude import ClaudeEngine
    timeout = config.get("timeout") or API_TIMEOUT_OVERRIDE or 3600
    engine = ClaudeEngine(config["base_model"],
                          config["reasoning"],
                          config["tools"],
                          timeout=timeout)
    cacheLayer = cl(engine.configAndSettingsHash, engine.AIHook, name)
    runAllTests(cacheLayer.AIHook, name, test_filter)

  elif engine_type == "bedrock":
    from .AiEngineAmazonBedrock import BedrockEngine
    timeout = config.get("timeout") or API_TIMEOUT_OVERRIDE or 3600
    engine = BedrockEngine(config["base_model"],
                           config["reasoning"],
                           config["tools"],
                           config.get("region", "us-east-1"),
                           timeout=timeout)
    cacheLayer = cl(engine.configAndSettingsHash, engine.AIHook, name)
    runAllTests(cacheLayer.AIHook, name, test_filter)

  elif engine_type == "llamacpp":
    from .AiEngineLlamaCpp import LlamaCppEngine
    base_url = config.get("base_url") or os.environ.get("LLAMACPP_BASE_URL")
    if not base_url:
      print(f"Skipping {name}: LLAMACPP_BASE_URL not set")
      return
    # Apply timeout override if provided
    timeout = config.get("timeout") or API_TIMEOUT_OVERRIDE or 3600
    engine = LlamaCppEngine(config["base_model"],
                            base_url,
                            timeout=timeout,
                            tools=config.get("tools", False))
    cacheLayer = cl(engine.configAndSettingsHash, engine.AIHook, name)
    runAllTests(cacheLayer.AIHook, name, test_filter)


def parse_test_filter(test_arg: str) -> Dict[int, Optional[Set[int]]]:
  """
  Parse test filter argument into a dict mapping test indices to subtask filters.
  
  Supports formats:
    - "2" -> run all subtasks of test 2
    - "2.3" -> run only subtask 3 of test 2
    - "2.3,2.5" -> run subtasks 3 and 5 of test 2
    - "2-4" -> run all subtasks of tests 2, 3, 4
    - "2,3.1,4" -> run all of test 2, subtask 1 of test 3, all of test 4
  
  Returns:
    Dict mapping test index to Optional[Set[int]]:
      - None means run all subtasks
      - Set[int] means run only those specific subtasks
  """
  tests: Dict[int, Optional[Set[int]]] = {}
  for part in test_arg.split(","):
    part = part.strip()
    if "." in part:
      # Subtask specification: "2.3"
      test_str, subtask_str = part.split(".", 1)
      test_idx = int(test_str)
      subtask_idx = int(subtask_str)
      if test_idx not in tests:
        tests[test_idx] = set()
      # If already set to None (all subtasks), keep it as specific subtasks
      if tests[test_idx] is None:
        tests[test_idx] = {subtask_idx}
      else:
        tests[test_idx].add(subtask_idx)
    elif "-" in part:
      # Range specification: "2-4"
      start, end = part.split("-", 1)
      for t in range(int(start), int(end) + 1):
        if t not in tests:
          tests[t] = None  # All subtasks
    else:
      # Single test: "2"
      test_idx = int(part)
      if test_idx not in tests:
        tests[test_idx] = None  # All subtasks
  return tests


def run_setup():
  """
  Execute all test files (1.py, 2.py, ...) to download and build reference data.
  This triggers any asset downloads, reference model builds, etc. without running AI tests.
  """
  print("=" * 60)
  print("SETUP MODE: Building reference data and downloading assets")
  print("=" * 60)

  test_index = 1
  successful = 0
  failed = 0

  while True:
    test_file = f"{test_index}.py"
    if not os.path.exists(test_file):
      break

    print(f"\n[{test_index}] Loading {test_file}...")

    try:
      # Execute the test file to trigger any module-level setup
      test_globals = {}
      exec(open(test_file, encoding="utf-8").read(), test_globals)

      # Get test title if available
      title = test_globals.get("title", f"Test {test_index}")
      print(f"    Title: {title}")

      # If there's a setup function, call it
      if "setup" in test_globals:
        print(f"    Running setup()...")
        test_globals["setup"]()

      # If there's custom setup needed, let the runner handle it
      if _current_runner:
        _current_runner.run_setup_for_test(test_index, test_globals)

      successful += 1
      print(f"    ✓ OK")

    except Exception as e:
      print(f"    ✗ Error: {e}")
      failed += 1

    test_index += 1

  print("\n" + "=" * 60)
  print(f"SETUP COMPLETE: {successful} tests loaded, {failed} failed")
  print("=" * 60)

  if failed > 0:
    print(f"\nWarning: {failed} test(s) had errors during setup.")
    print("Some reference data may not be available.")


if __name__ == "__main__":
  # TestRunner is now a library module.
  print("=" * 60)
  print("TestRunner.py is a library module.")
  print("")
  print("To create your own benchmark, subclass BenchmarkRunner and")
  print("call run_benchmark_main(runner, __file__) in your script.")
  print("")
  print("See README.md for documentation and examples.")
  print("=" * 60)
