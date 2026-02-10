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
from ._prompt_utils import apply_prompt_prefix, resolve_prompt_prefix

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
  meta: Optional[dict] = None
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

  def write_result_to_prompt_cache(self,
                                   request: BatchRequest,
                                   result: Any,
                                   chain_of_thought: str,
                                   meta: Optional[dict] = None) -> None:
    """Write result to the prompt/raw/cot cache files (like CacheLayer does)."""
    rp.ensure_global_result_dirs()
    rp.ensure_model_dirs(request.engine_name)

    raw_file = rp.model_raw_path(request.engine_name, request.test_index, request.sub_pass)
    prompt_file = rp.model_prompt_path(request.engine_name, request.test_index, request.sub_pass)
    cot_file = rp.model_cot_path(request.engine_name, request.test_index, request.sub_pass)
    meta_file = rp.model_meta_path(request.engine_name, request.test_index, request.sub_pass)

    with open(raw_file, "w", encoding="utf-8") as f:
      f.write(str(result))
    with open(prompt_file, "w", encoding="utf-8") as f:
      f.write(str(request.prompt))
    with open(cot_file, "w", encoding="utf-8") as f:
      f.write(str(chain_of_thought))
    if isinstance(meta, dict):
      with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    elif os.path.exists(meta_file):
      os.remove(meta_file)

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
                          meta=r.get("meta"),
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
                    if isinstance(result.meta, dict):
                      cache_value = (result.result, result.chain_of_thought or "", result.meta)
                    else:
                      cache_value = (result.result, result.chain_of_thought or "")
                    self.write_result_to_cache(req, cache_value)
                    self.write_result_to_prompt_cache(req, result.result, result.chain_of_thought,
                                                      result.meta)
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
    from .TestRunner import NO_EARLY_FAIL
    early_fail = "earlyFail" in g and not NO_EARLY_FAIL
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
      prompt_prefix = resolve_prompt_prefix(config)

      # Create engine to get config hash
      engine = create_engine_instance(config)
      if not engine:
        continue

      config_hash = engine.configAndSettingsHash

      # Add initial prompts (or all if not earlyFail)
      prompts_to_add = prompts[:early_fail_count] if early_fail else prompts

      for sub_pass, prompt in enumerate(prompts_to_add):
        effective_prompt = apply_prompt_prefix(str(prompt), prompt_prefix)
        request = BatchRequest(custom_id=f"{engine_name}_{test_index}_{sub_pass}",
                               prompt=effective_prompt,
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
      prompt_prefix = resolve_prompt_prefix(config)

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
            effective_prompt = apply_prompt_prefix(str(all_prompts[sub_pass]), prompt_prefix)
            request = BatchRequest(custom_id=f"{engine_name}_{test_index}_{sub_pass}",
                                   prompt=effective_prompt,
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
    return OpenAIEngine(config["base_model"],
                        config.get("reasoning", False),
                        config.get("tools", False),
                        max_output_tokens=config.get("max_output_tokens"),
                        temperature=config.get("temperature"),
                        emit_meta=True)
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
    return AzureOpenAIEngine(config["base_model"],
                             config.get("reasoning", False),
                             config.get("tools", False),
                             config.get("endpoint"),
                             config.get("api_version"),
                             max_output_tokens=config.get("max_output_tokens"),
                             temperature=config.get("temperature"),
                             emit_meta=True)
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


def import_batch_results(batch_id_or_file: str, model_config: dict, runner) -> int:
  """
  Import results from a failed/cancelled batch or a JSONL file.
  
  Args:
    batch_id_or_file: Either a batch ID (to fetch from provider) or path to a JSONL file
    model_config: The model configuration dict
    runner: The BenchmarkRunner instance (for creating engine instances)
    
  Returns:
    Number of results imported
  """
  engine_name = model_config["name"]
  engine_type = model_config.get("engine", "unknown")

  print(f"\n[Import] Importing batch results for {engine_name}...")

  # Create engine instance to get config hash
  engine = create_engine_instance(model_config)
  if not engine:
    print(f"[Import] Error: Could not create engine instance for {engine_name}")
    return 0

  config_hash = engine.configAndSettingsHash

  # Detect if input is a file path or batch ID
  is_file = os.path.exists(batch_id_or_file) and batch_id_or_file.endswith('.jsonl')

  results = []

  if is_file:
    # Parse JSONL file directly
    print(f"[Import] Reading from file: {batch_id_or_file}")
    results = _parse_jsonl_file(batch_id_or_file, engine_type)
  else:
    # Fetch from provider API
    print(f"[Import] Fetching from batch ID: {batch_id_or_file}")
    results = _fetch_batch_results(batch_id_or_file, engine_type, model_config)

  if not results:
    print("[Import] No results found to import")
    return 0

  # Group results by test_index to minimize test file reloading
  results_by_test = {}
  for result in results:
    custom_id = result.get("custom_id", "")
    if not custom_id:
      continue

    # Parse custom_id format: "{engine}_{testIndex}_{subPass}"
    parts = custom_id.rsplit("_", 2)
    if len(parts) < 3:
      print(f"[Import] Warning: Invalid custom_id format: {custom_id}")
      continue

    try:
      test_index = int(parts[-2])
      sub_pass = int(parts[-1])
    except ValueError:
      print(f"[Import] Warning: Could not parse test/subpass from: {custom_id}")
      continue

    if not result.get("success", False):
      print(f"[Import] Skipping failed result: {custom_id}")
      continue

    if test_index not in results_by_test:
      results_by_test[test_index] = []
    results_by_test[test_index].append((sub_pass, result))

  # Process results grouped by test
  imported_count = 0
  test_cache = {}  # Cache loaded test globals and prompts

  for test_index, test_results in sorted(results_by_test.items()):
    # Load test file to get prompts and structure
    if test_index not in test_cache:
      test_file = f"{test_index}.py"
      if not os.path.exists(test_file):
        print(f"[Import] Warning: Test file {test_file} not found, skipping")
        continue

      try:
        g = {"__file__": test_file}
        code = open(test_file, encoding="utf-8").read()
        compiled = compile(code, test_file, "exec")
        exec(compiled, g)

        # Generate all prompts for this test
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

        test_cache[test_index] = {"prompts": prompts, "structure": g.get("structure")}
      except Exception as e:
        print(f"[Import] Error loading test {test_index}: {e}")
        continue

    cached_test = test_cache[test_index]
    prompts = cached_test["prompts"]
    structure = cached_test["structure"]

    for sub_pass, result in test_results:
      if sub_pass >= len(prompts):
        print(f"[Import] Warning: subpass {sub_pass} exceeds prompt count for test {test_index}")
        continue

      prompt = prompts[sub_pass]
      result_data = result.get("result")
      chain_of_thought = result.get("chain_of_thought", "")

      # Write to cache with correct prompt and structure for proper cache key
      cache_value = (result_data, chain_of_thought)
      CacheModule.write_to_cache(prompt, structure, config_hash, cache_value)

      # Write to prompt/raw/cot files
      rp.ensure_global_result_dirs()
      rp.ensure_model_dirs(engine_name)

      raw_file = rp.model_raw_path(engine_name, test_index, sub_pass)
      cot_file = rp.model_cot_path(engine_name, test_index, sub_pass)
      prompt_file = rp.model_prompt_path(engine_name, test_index, sub_pass)

      with open(raw_file, "w", encoding="utf-8") as f:
        f.write(str(result_data))
      with open(cot_file, "w", encoding="utf-8") as f:
        f.write(str(chain_of_thought))
      with open(prompt_file, "w", encoding="utf-8") as f:
        f.write(str(prompt))

      print(f"[Import] Cached: test {test_index} subpass {sub_pass}")
      imported_count += 1

  print(f"\n[Import] Successfully imported {imported_count} results")
  return imported_count


def _parse_jsonl_file(file_path: str, engine_type: str) -> list:
  """Parse a JSONL file into a list of result dicts."""
  results = []

  with open(file_path, "r", encoding="utf-8") as f:
    for line_num, line in enumerate(f, 1):
      line = line.strip()
      if not line:
        continue

      try:
        obj = json.loads(line)
        result = _parse_result_object(obj, engine_type)
        if result:
          results.append(result)
      except json.JSONDecodeError as e:
        print(f"[Import] Warning: Failed to parse line {line_num}: {e}")

  return results


def _parse_result_object(obj: dict, engine_type: str) -> dict:
  """
  Parse a single result object from JSONL based on engine type.
  Returns a normalized result dict with: custom_id, success, result, chain_of_thought, error
  """
  custom_id = obj.get("custom_id", "")

  if engine_type == "openai":
    # OpenAI format: {"custom_id": "...", "response": {"body": {"output": [...]}}}
    response = obj.get("response", {})
    body = response.get("body", {})

    text_output = ""
    cot = ""

    if "output" in body:
      for item in body.get("output", []):
        if item.get("type") == "message":
          for content_item in item.get("content", []):
            if content_item.get("type") in ("text", "output_text"):
              text_output = content_item.get("text", "")
        elif item.get("type") == "reasoning":
          for summary in item.get("summary", []):
            if summary.get("type") == "summary_text":
              cot += summary.get("text", "") + "\n"

    # Try to parse JSON
    result_data = text_output
    try:
      json_text = text_output.strip()
      if json_text.startswith("```json"):
        json_text = json_text[7:]
      if json_text.startswith("```"):
        json_text = json_text[3:]
      if json_text.endswith("```"):
        json_text = json_text[:-3]
      result_data = json.loads(json_text.strip())
    except:
      pass

    return {
      "custom_id": custom_id,
      "success": bool(text_output),
      "result": result_data,
      "chain_of_thought": cot.strip(),
      "error": obj.get("error")
    }

  elif engine_type == "anthropic":
    # Anthropic format: {"custom_id": "...", "result": {"type": "succeeded", "message": {...}}}
    result_obj = obj.get("result", {})

    if result_obj.get("type") != "succeeded":
      return {
        "custom_id": custom_id,
        "success": False,
        "result": None,
        "chain_of_thought": "",
        "error": str(result_obj)
      }

    message = result_obj.get("message", {})
    text_output = ""
    cot = ""

    for block in message.get("content", []):
      if block.get("type") == "text":
        text_output += block.get("text", "")
      elif block.get("type") == "thinking":
        cot += block.get("thinking", "") + "\n"

    # Try to parse JSON
    result_data = text_output
    try:
      json_text = text_output.strip()
      if json_text.startswith("```json"):
        json_text = json_text[7:]
      if json_text.startswith("```"):
        json_text = json_text[3:]
      if json_text.endswith("```"):
        json_text = json_text[:-3]
      result_data = json.loads(json_text.strip())
    except:
      pass

    return {
      "custom_id": custom_id,
      "success": True,
      "result": result_data,
      "chain_of_thought": cot.strip(),
      "error": None
    }

  else:
    # Generic fallback - try to extract text from common structures
    text_output = obj.get("text", obj.get("content", obj.get("result", "")))
    if isinstance(text_output, dict):
      text_output = json.dumps(text_output)

    return {
      "custom_id": custom_id,
      "success": bool(text_output),
      "result": text_output,
      "chain_of_thought": "",
      "error": None
    }


def _fetch_batch_results(batch_id: str, engine_type: str, config: dict) -> list:
  """Fetch batch results from provider API."""
  results = []

  try:
    if engine_type == "openai":
      from openai import OpenAI
      client = OpenAI(timeout=3600)

      batch_status = client.batches.retrieve(batch_id)
      output_file_id = batch_status.output_file_id

      if not output_file_id:
        # Check for error file
        error_file_id = batch_status.error_file_id
        if error_file_id:
          error_content = client.files.content(error_file_id)
          print(f"[Import] OpenAI error file:\n{error_content.text[:2000]}")
        print(f"[Import] Batch status: {batch_status.status}, no output file available")
        return []

      content = client.files.content(output_file_id)
      for line in content.text.strip().split("\n"):
        if line.strip():
          obj = json.loads(line)
          result = _parse_result_object(obj, engine_type)
          if result:
            results.append(result)

    elif engine_type == "anthropic":
      from anthropic import Anthropic
      client = Anthropic()

      for result in client.messages.batches.results(batch_id):
        custom_id = result.custom_id

        if result.result.type == "succeeded":
          message = result.result.message
          text_output = ""
          cot = ""

          for block in message.content:
            if block.type == "text":
              text_output += block.text
            elif block.type == "thinking":
              cot += block.thinking + "\n"

          # Try to parse JSON
          result_data = text_output
          try:
            json_text = text_output.strip()
            if json_text.startswith("```json"):
              json_text = json_text[7:]
            if json_text.startswith("```"):
              json_text = json_text[3:]
            if json_text.endswith("```"):
              json_text = json_text[:-3]
            result_data = json.loads(json_text.strip())
          except:
            pass

          results.append({
            "custom_id": custom_id,
            "success": True,
            "result": result_data,
            "chain_of_thought": cot.strip(),
            "error": None
          })
        else:
          results.append({
            "custom_id": custom_id,
            "success": False,
            "result": None,
            "chain_of_thought": "",
            "error": str(result.result)
          })

    elif engine_type == "gemini":
      from google import genai
      client = genai.Client()

      batch_job = client.batches.get(name=batch_id)

      if batch_job.dest and hasattr(batch_job.dest, 'inlined_responses'):
        for i, inline_response in enumerate(batch_job.dest.inlined_responses or []):
          custom_id = f"gemini_{i}"  # Gemini doesn't preserve custom_id well

          if inline_response.error:
            results.append({
              "custom_id": custom_id,
              "success": False,
              "result": None,
              "chain_of_thought": "",
              "error": str(inline_response.error)
            })
            continue

          text_content = ""
          cot = ""

          if inline_response.response:
            try:
              text_content = inline_response.response.text
            except AttributeError:
              if hasattr(inline_response.response, 'candidates'):
                for candidate in inline_response.response.candidates:
                  if hasattr(candidate, 'content') and candidate.content:
                    for part in candidate.content.parts:
                      if hasattr(part, 'thought') and part.thought:
                        cot += part.text + "\n"
                      elif hasattr(part, 'text'):
                        text_content += part.text

          # Try to parse JSON
          result_data = text_content
          try:
            json_text = text_content.strip()
            if json_text.startswith("```json"):
              json_text = json_text[7:]
            if json_text.startswith("```"):
              json_text = json_text[3:]
            if json_text.endswith("```"):
              json_text = json_text[:-3]
            result_data = json.loads(json_text.strip())
          except:
            pass

          results.append({
            "custom_id": custom_id,
            "success": True,
            "result": result_data,
            "chain_of_thought": cot.strip(),
            "error": None
          })

    elif engine_type == "xai":
      from xai_sdk import Client
      client = Client(timeout=3600)

      pagination_token = None
      while True:
        page = client.batch.list_batch_results(batch_id=batch_id,
                                               limit=100,
                                               pagination_token=pagination_token)

        for result in page.succeeded:
          custom_id = result.batch_request_id
          response = result.response

          text_content = response.content if hasattr(response, 'content') else ""
          cot = response.reasoning_content if hasattr(response, 'reasoning_content') else ""

          # Try to parse JSON
          result_data = text_content
          try:
            json_text = text_content.strip()
            if "```json" in json_text:
              json_text = json_text.split("```json", 1)[1].split("```", 1)[0]
            elif "```" in json_text:
              json_text = json_text.split("```", 1)[1].split("```", 1)[0]
            result_data = json.loads(json_text.strip())
          except:
            pass

          results.append({
            "custom_id": custom_id,
            "success": True,
            "result": result_data,
            "chain_of_thought": cot or "",
            "error": None
          })

        for result in page.failed:
          results.append({
            "custom_id": result.batch_request_id,
            "success": False,
            "result": None,
            "chain_of_thought": "",
            "error": result.error_message
          })

        if page.pagination_token is None:
          break
        pagination_token = page.pagination_token

    else:
      print(f"[Import] Unsupported engine type for batch import: {engine_type}")

  except Exception as e:
    print(f"[Import] Error fetching batch results: {e}")
    import traceback
    traceback.print_exc()

  return results
