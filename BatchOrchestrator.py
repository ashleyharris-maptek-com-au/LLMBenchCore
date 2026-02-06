"""
Batch Orchestrator for LLMBenchCore

Coordinates batch submission and polling across multiple AI providers.
Uses a cache-based approach: batch results are written to cache as they arrive,
then the normal test runner processes them as if they were cached results.

Supported providers:
- OpenAI: File upload + batch API
- Anthropic: Message Batches API
- Gemini: Inline batch requests
- xAI: Batch API with Chat objects

Providers without batch support fall back to synchronous API:
- AWS Bedrock (requires S3 setup)
- LlamaCpp (local server)
- Azure OpenAI (no batch API)
"""

import os
import json
import time
import importlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
from . import ResultPaths as rp

# Import CacheLayer as a module (avoid package attribute that may resolve to the class)
CacheModule = importlib.import_module("LLMBenchCore.CacheLayer")


class BatchStatus(Enum):
  PENDING = "pending"
  SUBMITTED = "submitted"
  PROCESSING = "processing"
  COMPLETED = "completed"
  FAILED = "failed"
  PARTIAL = "partial"  # Some requests completed, some failed


@dataclass
class BatchRequest:
  """A single request in a batch."""
  custom_id: str  # Unique ID: "{engine}_{testIndex}_{subPass}"
  prompt: str
  structure: Optional[dict]
  test_index: int
  sub_pass: int
  engine_name: str
  config_hash: str  # For cache key generation
  is_early_fail_check: bool = False  # True if this is an earlyFail initial request


@dataclass
class BatchResult:
  """Result of a single batch request."""
  custom_id: str
  success: bool
  result: Any  # The actual response (text or dict)
  chain_of_thought: str = ""
  error: Optional[str] = None


@dataclass
class BatchJob:
  """Tracks a batch job for a specific engine."""
  engine_name: str
  batch_id: str  # Provider-specific batch ID
  requests: List[BatchRequest]
  status: BatchStatus = BatchStatus.PENDING
  results: Dict[str, BatchResult] = field(default_factory=dict)
  provider_metadata: Dict[str, Any] = field(default_factory=dict)
  created_at: float = field(default_factory=time.time)
  completed_at: Optional[float] = None


@dataclass
class EarlyFailTestInfo:
  """Info about a test that uses earlyFail."""
  test_index: int
  test_globals: dict
  all_prompts: List[str]
  initial_count: int
  threshold: float


class BatchOrchestrator:
  """
  Orchestrates batch processing across multiple AI providers.
  
  Workflow:
  1. Collect all prompts from tests (gather_prompts)
  2. Submit batches to each provider (submit_batches)
  3. Poll for results and write to cache (poll_and_cache_results)
  4. Handle earlyFail by grading and submitting follow-up batches
  """

  # Map engine types to their batch support status
  BATCH_SUPPORTED_ENGINES = {
    "openai": True,
    "anthropic": True,
    "gemini": True,
    "xai": True,
    "bedrock": False,  # Requires S3 setup - fallback to sync
    "azure_openai": False,  # No batch API - fallback to sync
    "llamacpp": False,  # Local server - fallback to sync
    "placebo": False,  # Test engine - fallback to sync
  }

  def __init__(self, model_configs: List[Dict[str, Any]], force_refresh: bool = False):
    self.model_configs = {cfg["name"]: cfg for cfg in model_configs}
    self.batch_jobs: Dict[str, BatchJob] = {}  # engine_name -> BatchJob
    self.pending_requests: Dict[str, List[BatchRequest]] = {}  # engine_name -> requests
    self.poll_interval = 300  # seconds between polls
    self.max_poll_time = 24 * 60 * 60  # 24 hours max wait
    self.force_refresh = force_refresh
    self.skipped_cached = 0  # Count of requests skipped due to cache

  def supports_batch(self, engine_name: str) -> bool:
    """Check if an engine supports batch processing."""
    config = self.model_configs.get(engine_name, {})
    engine_type = config.get("engine", "unknown")
    return self.BATCH_SUPPORTED_ENGINES.get(engine_type, False)

  def get_engine_type(self, engine_name: str) -> str:
    """Get the engine type for a model config."""
    config = self.model_configs.get(engine_name, {})
    return config.get("engine", "unknown")

  def add_request(self, request: BatchRequest) -> bool:
    """
    Add a request to the pending batch for its engine.
    Returns True if added, False if skipped due to cache hit.
    """
    # Check if already cached - skip if so
    if CacheModule.is_cached(request.prompt, request.structure, request.config_hash,
                             self.force_refresh):
      self.skipped_cached += 1
      return False

    engine_name = request.engine_name
    if engine_name not in self.pending_requests:
      self.pending_requests[engine_name] = []
    self.pending_requests[engine_name].append(request)
    return True

  def write_result_to_cache(self, request: BatchRequest, result: Any) -> str:
    """Write a batch result to the cache file using CacheLayer."""
    return CacheModule.write_to_cache(request.prompt, request.structure, request.config_hash,
                                      result)

  def write_result_to_prompt_cache(self, request: BatchRequest, result: Any,
                                   chain_of_thought: str) -> None:
    """Write result to the prompt/raw/cot cache files (like CacheLayer does)."""
    rp.ensure_global_result_dirs()
    rp.ensure_model_dirs(request.engine_name)

    raw_file = rp.model_raw_path(request.engine_name, request.test_index, request.sub_pass)
    prompt_file = rp.model_prompt_path(request.engine_name, request.test_index, request.sub_pass)
    cot_file = rp.model_cot_path(request.engine_name, request.test_index, request.sub_pass)

    with open(raw_file, "w", encoding="utf-8") as f:
      f.write(str(result))
    with open(prompt_file, "w", encoding="utf-8") as f:
      f.write(str(request.prompt))
    with open(cot_file, "w", encoding="utf-8") as f:
      f.write(str(chain_of_thought))

  def submit_batches(self) -> Dict[str, str]:
    """
    Submit all pending requests as batches to their respective providers.
    Returns dict of engine_name -> batch_id for successfully submitted batches.
    """
    submitted = {}

    for engine_name, requests in self.pending_requests.items():
      if not requests:
        continue

      if not self.supports_batch(engine_name):
        print(f"[Batch] {engine_name} doesn't support batching, will use sync API")
        continue

      engine_type = self.get_engine_type(engine_name)
      config = self.model_configs.get(engine_name, {})

      try:
        print(f"[Batch] Submitting {len(requests)} requests for {engine_name}...")

        # Call the engine's submit_batch function
        if engine_type == "openai":
          from .AiEngineOpenAiChatGPT import submit_batch
        elif engine_type == "anthropic":
          from .AiEngineAnthropicClaude import submit_batch
        elif engine_type == "gemini":
          from .AiEngineGoogleGemini import submit_batch
        elif engine_type == "xai":
          from .AiEngineXAIGrok import submit_batch
        else:
          print(f"[Batch] Unknown engine type: {engine_type}")
          continue

        batch_id = submit_batch(config, requests)

        if batch_id:
          self.batch_jobs[engine_name] = BatchJob(engine_name=engine_name,
                                                  batch_id=batch_id,
                                                  requests=requests,
                                                  status=BatchStatus.SUBMITTED)
          submitted[engine_name] = batch_id
          print(f"[Batch] Submitted batch {batch_id} for {engine_name}")

      except Exception as e:
        print(f"[Batch] Failed to submit batch for {engine_name}: {e}")

    # Clear pending requests for successfully submitted batches
    for engine_name in submitted:
      self.pending_requests[engine_name] = []

    return submitted

  def poll_all_batches(self, callback=None) -> Dict[str, BatchStatus]:
    """
    Poll all submitted batches until completion or timeout.
    
    Args:
      callback: Optional function called with (engine_name, results_count) as results arrive
      
    Returns:
      Dict of engine_name -> final BatchStatus
    """
    start_time = time.time()
    final_statuses = {}

    while self.batch_jobs:
      # Check for timeout
      if time.time() - start_time > self.max_poll_time:
        print("[Batch] Max poll time exceeded, stopping")
        for engine_name, job in self.batch_jobs.items():
          final_statuses[engine_name] = BatchStatus.FAILED
        break

      completed_engines = []

      for engine_name, job in list(self.batch_jobs.items()):
        engine_type = self.get_engine_type(engine_name)
        config = self.model_configs.get(engine_name, {})

        try:
          # Get the engine-specific poll_batch function
          poll_batch = None
          if engine_type == "openai":
            from .AiEngineOpenAiChatGPT import poll_batch
          elif engine_type == "anthropic":
            from .AiEngineAnthropicClaude import poll_batch
          elif engine_type == "gemini":
            from .AiEngineGoogleGemini import poll_batch
          elif engine_type == "xai":
            from .AiEngineXAIGrok import poll_batch

          if poll_batch is None:
            status = BatchStatus.FAILED
            results = []
          else:
            status_str, results_list = poll_batch(job.batch_id, job.requests)
            # Convert status string to BatchStatus
            if status_str == "completed":
              status = BatchStatus.COMPLETED
            elif status_str == "failed":
              status = BatchStatus.FAILED
            else:
              status = BatchStatus.PROCESSING
            # Convert result dicts to BatchResult objects
            results = [
              BatchResult(custom_id=r["custom_id"],
                          success=r["success"],
                          result=r["result"],
                          chain_of_thought=r.get("chain_of_thought", ""),
                          error=r.get("error")) for r in results_list
            ]

          # Process any new results
          for result in results:
            if result.custom_id not in job.results:
              job.results[result.custom_id] = result
              # Find matching request and write to cache
              for req in job.requests:
                if req.custom_id == result.custom_id:
                  if result.success:
                    # Write as tuple (result, chain_of_thought) to match sync API cache format
                    cache_value = (result.result, result.chain_of_thought or "")
                    self.write_result_to_cache(req, cache_value)
                    self.write_result_to_prompt_cache(req, result.result, result.chain_of_thought)
                    print(f"[Batch] Cached result for {result.custom_id}")
                  break

              if callback:
                callback(engine_name, len(job.results))

          job.status = status

          if status in [BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.PARTIAL]:
            completed_engines.append(engine_name)
            final_statuses[engine_name] = status
            job.completed_at = time.time()
            success_count = len([r for r in job.results.values() if r.success])
            non_empty = len([r for r in job.results.values() if r.success and r.result])
            print(f"[Batch] {engine_name} batch {status.value}: "
                  f"{success_count}/{len(job.requests)} succeeded, {non_empty} non-empty")

        except Exception as e:
          print(f"[Batch] Error polling {engine_name}: {e}")

      # Remove completed jobs
      for engine_name in completed_engines:
        del self.batch_jobs[engine_name]

      if self.batch_jobs:
        print(f"[Batch] Waiting {self.poll_interval}s before next poll...")
        time.sleep(self.poll_interval)

    return final_statuses

  def get_non_batch_requests(self) -> Dict[str, List[BatchRequest]]:
    """Get requests for engines that don't support batching."""
    return {
      engine: requests
      for engine, requests in self.pending_requests.items()
      if not self.supports_batch(engine) and requests
    }

  def clear_pending(self) -> None:
    """Clear all pending requests."""
    self.pending_requests.clear()


def run_batch_mode(runner, test_filter=None, model_filter=None, poll_interval=60, force_mode=False):
  """
  Run the benchmark in batch mode.
  
  1. Gather all prompts from all tests (initial prompts only for earlyFail tests)
  2. Submit batches to providers that support it
  3. Poll for results and cache them
  4. Handle earlyFail by grading initial results and submitting follow-ups
  5. Run engines without batch support synchronously
  6. Finally, run normal test processing (results will hit cache)
  
  Args:
    runner: The BenchmarkRunner instance
    test_filter: Optional set of test indices to run
    model_filter: Optional set of model names to run
    poll_interval: Seconds between batch status polls (default: 60)
    force_mode: Whether --force was specified (will be disabled after batch completes)
  """
  from . import CacheLayer
  from .TestRunner import ALL_MODEL_CONFIGS

  print("\n" + "=" * 60)
  print("BATCH MODE ENABLED")
  print("=" * 60 + "\n")

  # Get all model configs
  configs = runner.get_final_model_configs()
  ALL_MODEL_CONFIGS.clear()
  ALL_MODEL_CONFIGS.extend(configs)

  # Filter configs
  if model_filter:
    configs = [c for c in configs if c["name"] in model_filter]

  orchestrator = BatchOrchestrator(configs, force_refresh=force_mode)
  orchestrator.poll_interval = poll_interval

  # Phase 1: Gather initial prompts (respecting earlyFail)
  print("[Batch] Phase 1: Gathering prompts from all tests...")
  prompts_gathered, early_fail_tests = gather_all_prompts(orchestrator, configs, test_filter)
  print(f"[Batch] Gathered {prompts_gathered} prompts to submit")
  if orchestrator.skipped_cached > 0:
    print(f"[Batch] Skipped {orchestrator.skipped_cached} prompts (already cached)")
  if early_fail_tests:
    print(
      f"[Batch] {len(early_fail_tests)} tests use earlyFail - follow-ups will be submitted after grading"
    )

  # If everything was cached, skip to final phase
  if prompts_gathered == 0:
    print("[Batch] All prompts already cached, skipping batch submission")
    CacheLayer.FORCE_REFRESH = False
    runner.run(test_filter, model_filter)
    print("\n" + "=" * 60)
    print("BATCH MODE COMPLETE (all from cache)")
    print("=" * 60)
    return

  # Phase 2: Submit batches for supported engines
  print("\n[Batch] Phase 2: Submitting batches...")
  submitted = orchestrator.submit_batches()
  print(f"[Batch] Submitted {len(submitted)} batches")

  # Fallback: if any engines still have pending requests (e.g., batch submission failed),
  # surface a warning instead of running them synchronously.
  leftover_pending = {
    engine: reqs
    for engine, reqs in orchestrator.pending_requests.items() if reqs and engine not in submitted
  }
  if leftover_pending:
    print(f"[Batch] WARNING: {len(leftover_pending)} engine(s) have unsubmitted batches. "
          "These will not be executed to avoid unexpected API calls.")
    print(
      "[Batch] Aborting batch run due to unsubmitted batches. Nothing further will be executed.")
    return

  # Phase 3: Run non-batch engines synchronously
  non_batch = orchestrator.get_non_batch_requests()
  if non_batch:
    print(f"\n[Batch] Phase 3: Running {len(non_batch)} engines without batch support...")
    run_non_batch_engines_sync(non_batch, configs)

  # Phase 4: Poll batches until complete
  if submitted:
    print("\n[Batch] Phase 4: Polling for batch results...")
    final_statuses = orchestrator.poll_all_batches()

    # Report final status
    for engine, status in final_statuses.items():
      print(f"[Batch] {engine}: {status.value}")

    # If any batch is not completed, abort to avoid sync API calls in normal run
    incomplete = [e for e, s in final_statuses.items() if s != BatchStatus.COMPLETED]
    if incomplete:
      print(f"[Batch] Incomplete batches for engines: {', '.join(incomplete)}. "
            "Aborting to avoid synchronous API calls.")
      return

  # Phase 5: Handle earlyFail - grade initial results and submit follow-ups
  if early_fail_tests:
    print("\n[Batch] Phase 5: Processing earlyFail tests...")
    follow_up_count = handle_early_fail_follow_ups(orchestrator, configs, early_fail_tests,
                                                   test_filter)
    if follow_up_count > 0:
      print(f"[Batch] Submitted {follow_up_count} follow-up requests")
      # Poll again for follow-ups
      submitted = orchestrator.submit_batches()
      if submitted:
        final_statuses = orchestrator.poll_all_batches()
        for engine, status in final_statuses.items():
          print(f"[Batch] {engine} follow-ups: {status.value}")

  # Phase 6: Disable force mode so cached results are used
  print("\n[Batch] Phase 6: Processing results (cache mode)...")
  CacheLayer.FORCE_REFRESH = False

  # Phase 7: Run normal benchmark (will hit cache)
  runner.run(test_filter, model_filter)

  print("\n" + "=" * 60)
  print("BATCH MODE COMPLETE")
  print("=" * 60)


def gather_all_prompts(orchestrator: BatchOrchestrator,
                       configs: List[Dict],
                       test_filter=None) -> Tuple[int, List[EarlyFailTestInfo]]:
  """
  Gather all prompts from all tests for all engines.
  
  Returns:
    Tuple of (total_prompts_count, list of EarlyFailTestInfo for tests using earlyFail)
  """
  import os

  total_prompts = 0
  early_fail_tests = []
  test_index = 1

  while True:
    if not os.path.exists(f"{test_index}.py"):
      break

    if test_filter and test_index not in test_filter:
      test_index += 1
      continue

    # Load test file
    try:
      g = {"__file__": f"{test_index}.py"}
      code = open(f"{test_index}.py", encoding="utf-8").read()
      compiled = compile(code, f"{test_index}.py", "exec")
      exec(compiled, g)
    except Exception as e:
      print(f"[Batch] Error loading test {test_index}: {e}")
      test_index += 1
      continue

    # Skip if test has skip flag
    if "skip" in g:
      test_index += 1
      continue

    # Get prompts
    structure = g.get("structure")
    prompts = []

    if "prepareSubpassPrompt" in g:
      sub_pass = 0
      while True:
        try:
          prompts.append(g["prepareSubpassPrompt"](sub_pass))
          sub_pass += 1
        except StopIteration:
          break
    else:
      prompts.append(g.get("prompt", ""))

    # Determine if earlyFail and how many initial prompts to send
    early_fail = "earlyFail" in g
    early_fail_count = g.get("earlyFailSubpassSampleCount", 1) if early_fail else len(prompts)
    early_fail_threshold = g.get("earlyFailThreshold", 0.5)

    # Track earlyFail tests for follow-up processing
    if early_fail and len(prompts) > early_fail_count:
      early_fail_tests.append(
        EarlyFailTestInfo(test_index=test_index,
                          test_globals=g,
                          all_prompts=prompts,
                          initial_count=early_fail_count,
                          threshold=early_fail_threshold))

    # Add prompts for each engine
    for config in configs:
      engine_name = config["name"]

      # Create engine to get config hash
      engine = create_engine_instance(config)
      if not engine:
        continue

      config_hash = engine.configAndSettingsHash

      # Add initial prompts (or all if not earlyFail)
      prompts_to_add = prompts[:early_fail_count] if early_fail else prompts

      for sub_pass, prompt in enumerate(prompts_to_add):
        request = BatchRequest(custom_id=f"{engine_name}_{test_index}_{sub_pass}",
                               prompt=prompt,
                               structure=structure,
                               test_index=test_index,
                               sub_pass=sub_pass,
                               engine_name=engine_name,
                               config_hash=config_hash,
                               is_early_fail_check=early_fail and sub_pass < early_fail_count)
        if orchestrator.add_request(request):
          total_prompts += 1

    test_index += 1

  return total_prompts, early_fail_tests


def handle_early_fail_follow_ups(orchestrator: BatchOrchestrator,
                                 configs: List[Dict],
                                 early_fail_tests: List[EarlyFailTestInfo],
                                 test_filter=None) -> int:
  """
  Grade initial earlyFail results and submit follow-up batches if they pass.
  
  Returns the number of follow-up requests submitted.
  """
  follow_up_count = 0

  for ef_test in early_fail_tests:
    test_index = ef_test.test_index
    g = ef_test.test_globals
    all_prompts = ef_test.all_prompts
    initial_count = ef_test.initial_count
    threshold = ef_test.threshold
    structure = g.get("structure")

    print(f"[Batch] Grading earlyFail test {test_index}...")

    for config in configs:
      engine_name = config["name"]

      # Check if initial results passed the threshold
      # Read the cached results for initial subpasses
      total_initial_score = 0
      initial_results_found = 0

      for sub_pass in range(initial_count):
        result_file_candidates = [
          rp.model_raw_path(engine_name, test_index, sub_pass),
          rp.legacy_raw_path(engine_name, test_index, sub_pass),
        ]
        result_file = next((p for p in result_file_candidates if os.path.exists(p)), None)
        if result_file:
          try:
            with open(result_file, "r", encoding="utf-8") as f:
              result_str = f.read()

            # Parse result
            import ast
            try:
              result = ast.literal_eval(result_str)
            except:
              result = result_str

            # Grade the result
            if "gradeAnswer" in g:
              try:
                grade_result = g["gradeAnswer"](result, sub_pass, engine_name)
                if len(grade_result) >= 2:
                  score = grade_result[0]
                  total_initial_score += score
                  initial_results_found += 1
              except Exception as e:
                print(
                  f"[Batch] Error grading {engine_name} test {test_index} subpass {sub_pass}: {e}")
          except Exception as e:
            print(f"[Batch] Error reading result for {engine_name} test {test_index}: {e}")

      # Calculate average score
      if initial_results_found > 0:
        avg_score = total_initial_score / initial_results_found
        print(
          f"[Batch] {engine_name} test {test_index}: avg score {avg_score:.3f} (threshold {threshold})"
        )

        if avg_score >= threshold:
          # Passed! Submit follow-up prompts
          engine = create_engine_instance(config)
          if not engine:
            continue

          config_hash = engine.configAndSettingsHash

          for sub_pass in range(initial_count, len(all_prompts)):
            request = BatchRequest(custom_id=f"{engine_name}_{test_index}_{sub_pass}",
                                   prompt=all_prompts[sub_pass],
                                   structure=structure,
                                   test_index=test_index,
                                   sub_pass=sub_pass,
                                   engine_name=engine_name,
                                   config_hash=config_hash,
                                   is_early_fail_check=False)
            orchestrator.add_request(request)
            follow_up_count += 1

          print(
            f"[Batch] {engine_name} test {test_index}: queued {len(all_prompts) - initial_count} follow-ups"
          )
        else:
          print(
            f"[Batch] {engine_name} test {test_index}: FAILED earlyFail check, skipping remaining subpasses"
          )

  return follow_up_count


def create_engine_instance(config: Dict):
  """Create an AI engine instance from config."""
  engine_type = config.get("engine", "unknown")

  if engine_type == "openai":
    from .AiEngineOpenAiChatGPT import OpenAIEngine
    return OpenAIEngine(config["base_model"], config.get("reasoning", False),
                        config.get("tools", False))
  elif engine_type == "anthropic":
    from .AiEngineAnthropicClaude import ClaudeEngine
    return ClaudeEngine(config["base_model"], config.get("reasoning", False),
                        config.get("tools", False))
  elif engine_type == "gemini":
    from .AiEngineGoogleGemini import GeminiEngine
    return GeminiEngine(config["base_model"], config.get("reasoning", False),
                        config.get("tools", False))
  elif engine_type == "xai":
    from .AiEngineXAIGrok import GrokEngine
    return GrokEngine(config["base_model"], config.get("reasoning", False),
                      config.get("tools", False))
  elif engine_type == "bedrock":
    from .AiEngineAmazonBedrock import BedrockEngine
    return BedrockEngine(config["base_model"], config.get("reasoning", False),
                         config.get("tools", False), config.get("region", "us-east-1"))
  elif engine_type == "azure_openai":
    from .AiEngineAzureOpenAI import AzureOpenAIEngine
    return AzureOpenAIEngine(config["base_model"], config.get("reasoning", False),
                             config.get("tools", False), config.get("endpoint"),
                             config.get("api_version"))
  elif engine_type == "llamacpp":
    from .AiEngineLlamaCpp import LlamaCppEngine
    return LlamaCppEngine(config.get("base_url", "http://localhost:8080"))
  elif engine_type == "placebo":
    from .AiEnginePlacebo import PlaceboEngine
    return PlaceboEngine(config["name"])
  return None


def run_non_batch_engines_sync(requests_by_engine: Dict[str, List[BatchRequest]],
                               configs: List[Dict]) -> None:
  """Run engines without batch support synchronously."""
  from .CacheLayer import CacheLayer

  config_map = {c["name"]: c for c in configs}

  for engine_name, requests in requests_by_engine.items():
    config = config_map.get(engine_name)
    if not config:
      continue

    print(f"[Batch] Running {len(requests)} requests for {engine_name} (sync)...")

    engine = create_engine_instance(config)
    if not engine:
      continue

    cache = CacheLayer(engine.configAndSettingsHash, engine.AIHook, engine_name)

    for req in requests:
      try:
        result = cache.AIHook(req.prompt, req.structure, req.test_index, req.sub_pass)
        print(f"[Batch] Completed {req.custom_id}")
      except Exception as e:
        print(f"[Batch] Error on {req.custom_id}: {e}")
