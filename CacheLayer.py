import os
import tempfile
import json
import hashlib
import datetime
import time
import random
from typing import Callable, Optional

from .ContentViolationHandler import (is_prompt_blocked, block_prompt, is_violation_response)

# Global flag to bypass cache reading (still writes to cache)
FORCE_REFRESH = False

# Global flag to keep us offline.
OFFLINE_MODE = False

# Try very hard to get a cache hit, even if it means using old
# results months or years old.
POOR_MODE = True

# If the cache contains an empty result, ignore the cache and try again.
IGNORE_CACHED_FAILURES = False

PromptCacheKeyFunc = Callable[[str], str]


def default_prompt_cache_key(prompt: str) -> str:
  """Default cache prompt normalization: preserve existing stripped-prompt behavior."""
  return str(prompt).strip()


def default_cached_prompts_match(saved_prompt: str, current_prompt: str) -> bool:
  """Default saved-prompt comparator used when a benchmark does not customize cache matching."""
  return default_prompt_cache_key(saved_prompt) == default_prompt_cache_key(current_prompt)


def _cache_prompt_key(prompt: str, prompt_cache_key: Optional[PromptCacheKeyFunc] = None) -> str:
  if prompt_cache_key is None:
    return default_prompt_cache_key(prompt)
  return str(prompt_cache_key(str(prompt)))


def get_cache_file_path(prompt: str,
                        structure,
                        config_hash: str,
                        cache_date=None,
                        prompt_cache_key: Optional[PromptCacheKeyFunc] = None) -> str:
  """
  Generate the cache file path for a prompt/structure/config combination.
  This is the canonical cache path calculation used by both CacheLayer and BatchOrchestrator.
  """
  if cache_date is None:
    cache_date = datetime.datetime.now()

  cache_key_prompt = _cache_prompt_key(prompt, prompt_cache_key)
  h = (hashlib.sha256(cache_key_prompt.encode()).hexdigest(),
       hashlib.sha256(str(structure).encode()).hexdigest(), config_hash,
       cache_date.strftime("%b %Y"))
  h = hashlib.sha256(str(h).encode()).hexdigest()
  return os.path.join(tempfile.gettempdir(), "cache_" + str(h) + ".txt")


def is_cached(prompt: str,
              structure,
              config_hash: str,
              force_refresh: bool = False,
              prompt_cache_key: Optional[PromptCacheKeyFunc] = None) -> bool:
  """
  Check if a prompt/structure/config combination is already cached.
  Respects POOR_MODE to search back in time for older cache entries.
  
  Returns True if a valid cached response exists.
  """
  if force_refresh:
    return False

  cache_date = datetime.datetime.now()

  while True:
    cache_file = get_cache_file_path(prompt, structure, config_hash, cache_date, prompt_cache_key)

    if os.path.exists(cache_file):
      # Verify the cache file has valid content
      try:
        with open(cache_file, "r", encoding="utf-8") as f:
          cached_json = json.load(f)
        # Check if it's not an empty/error result
        if len(str(cached_json)) > 10:
          return True
        elif not IGNORE_CACHED_FAILURES:
          return True  # Even short results count as cached unless we're ignoring failures
      except:
        pass  # Invalid cache file, continue searching

    if not POOR_MODE:
      break

    cache_date -= datetime.timedelta(days=25)
    if cache_date < datetime.datetime(2025, 11, 30):
      break

  return False


def write_to_cache(prompt: str,
                   structure,
                   config_hash: str,
                   result,
                   prompt_cache_key: Optional[PromptCacheKeyFunc] = None) -> str:
  """
  Write a result to the cache. Returns the cache file path.
  """
  cache_file = get_cache_file_path(prompt,
                                   structure,
                                   config_hash,
                                   prompt_cache_key=prompt_cache_key)
  with open(cache_file, "w", encoding="utf-8") as f:
    json.dump(result, f)
  return cache_file


class CacheLayer:

  def __init__(self,
               configAndSettingsHash,
               aiEngineHook,
               engineName: str = "Unknown",
               prompt_cache_key: Optional[PromptCacheKeyFunc] = None):
    self.hash = configAndSettingsHash
    self.aiEngineHook = aiEngineHook
    self.engineName = engineName
    self.prompt_cache_key = prompt_cache_key
    self.temp_dir = tempfile.gettempdir()
    self.failCount = 0
    # Capture force_refresh at construction time via sys.modules to get the module, not the class
    import sys
    self.force_refresh = sys.modules[__name__].FORCE_REFRESH

  def AIHook(self, prompt: str, structure, index, subPass):
    # Check if this prompt is permanently blocked due to content violation
    if is_prompt_blocked(self.engineName, index, subPass, prompt):
      print(f"BLOCKED: Prompt for {self.engineName} Q{index}/S{subPass} is permanently blocked")
      if structure:
        return {
          "__content_violation__": True,
          "reason": "Previously blocked"
        }, "Content violation (blocked)"
      else:
        return "", "Content violation (blocked)"

    # Find cache file (searches back in time if POOR_MODE)
    cache_file = _find_cache_file(prompt, structure, self.hash, self.prompt_cache_key)

    if self.failCount > 3:
      if structure:
        return {}, "AI service has failed 9 times, assumed dead."
      else:
        return "", "AI service has failed 9 times, assumed dead."

    # Try to read from cache
    if not self.force_refresh and os.path.exists(cache_file):
      cached_result = _read_cache_file(cache_file)
      if cached_result is not None:
        return cached_result

    print(f"API Call ({self.engineName}): " + prompt[:100].replace("\n", " ") + "...")

    if OFFLINE_MODE:
      print("Offline mode: No API calls will be made, cache only.")
      return {}, ""

    print("Started at " + str(datetime.datetime.now()))
    result = self.aiEngineHook(prompt, structure)

    if not result and self.aiEngineHook.__name__ != "PlaceboAIHook":
      print("Empty result or Error 500, pausing and then retrying in a few minutes...")
      time.sleep(60 + random.randint(0, 120))
      result = self.aiEngineHook(prompt, structure)

      if not result:
        print(
          "Empty result or Error 500, pausing for a VERY LONG TIME and then retrying in a few minutes..."
        )
        time.sleep(600 + random.randint(0, 1200))
        result = self.aiEngineHook(prompt, structure)

    if not result:
      self.failCount += 1
      empty_result = {} if structure else ""
      write_to_cache(prompt, structure, self.hash, empty_result, self.prompt_cache_key)
      return empty_result, "AI didn't respond after 3 retries - failing test"

    print("Finished at " + str(datetime.datetime.now()))

    # Check if result indicates a content violation
    if result:
      result_data = result[0] if isinstance(result, tuple) else result
      if is_violation_response(result_data):
        reason = "Content violation detected"
        if isinstance(result_data, dict):
          reason = result_data.get("reason", reason)
        # For string violations, try to get reason from chain of thought (second tuple element)
        elif isinstance(result, tuple) and len(result) > 1 and result[1]:
          reason = str(result[1])
        block_prompt(self.engineName, index, subPass, prompt, reason)
        # Don't cache content violations - they're permanently blocked
        return result

    write_to_cache(prompt, structure, self.hash, result, self.prompt_cache_key)
    return result


def _find_cache_file(prompt: str,
                     structure,
                     config_hash: str,
                     prompt_cache_key: Optional[PromptCacheKeyFunc] = None) -> str:
  """Find the cache file, searching back in time if POOR_MODE is enabled."""
  cache_date = datetime.datetime.now()

  while True:
    cache_file = get_cache_file_path(prompt, structure, config_hash, cache_date, prompt_cache_key)

    if not POOR_MODE:
      break

    if os.path.exists(cache_file):
      break

    cache_date -= datetime.timedelta(days=25)

    if cache_date < datetime.datetime(2025, 11, 30):
      # Reset to current date for writing
      cache_file = get_cache_file_path(prompt,
                                       structure,
                                       config_hash,
                                       prompt_cache_key=prompt_cache_key)
      break

  return cache_file


def _read_cache_file(cache_file: str):
  """Read and validate a cache file. Returns None if invalid or should be skipped."""
  try:
    with open(cache_file, "r", encoding="utf-8") as f:
      cached_json = json.load(f)
      print("Using cached response from " + cache_file)

    if len(str(cached_json)) <= 10 and IGNORE_CACHED_FAILURES:
      print(f"IGNORE_CACHED_FAILURES set, cached result was too short: '{cached_json}'")
      try:
        os.unlink(cache_file)
      except:
        pass
      return None

    if "__exception__" in str(cached_json) and IGNORE_CACHED_FAILURES:
      print(f"IGNORE_CACHED_FAILURES set, cached result had an exception: '{cached_json}'")
      try:
        os.unlink(cache_file)
      except:
        pass
      return None

    # Catch cases where some thinking was kept, but the output was empty. This occurs in many
    # engines when you run out of API credits halfway through a call, or when an answer got
    # cut-off mid thought.
    if isinstance(cached_json, list) and len(cached_json) == 2 and isinstance(cached_json[0], str):
      if len(cached_json[0]) <= 2 and IGNORE_CACHED_FAILURES:
        print(f"IGNORE_CACHED_FAILURES set, cached result was too short: '{cached_json[0]}'")
        print(f"Cached answer had {len(cached_json[1])} bytes of reasoning - this may indicate ")
        print("an answer trimmed due to token limits - check the engine configuration.")
        try:
          os.unlink(cache_file)
        except:
          pass
        return None

    if len(cached_json) > 0:
      return cached_json
    return None
  except Exception as e:
    print("Failed to read cache file: " + cache_file + " - " + str(e))
    try:
      os.unlink(cache_file)
    except:
      pass
    return None
