import os
import tempfile
import json
import hashlib
import datetime
import time
import random

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


def get_cache_file_path(prompt: str, structure, config_hash: str, cache_date=None) -> str:
  """
  Generate the cache file path for a prompt/structure/config combination.
  This is the canonical cache path calculation used by both CacheLayer and BatchOrchestrator.
  """
  if cache_date is None:
    cache_date = datetime.datetime.now()

  h = (hashlib.sha256(prompt.strip().encode()).hexdigest(),
       hashlib.sha256(str(structure).encode()).hexdigest(), config_hash,
       cache_date.strftime("%b %Y"))
  h = hashlib.sha256(str(h).encode()).hexdigest()
  return os.path.join(tempfile.gettempdir(), "cache_" + str(h) + ".txt")


def is_cached(prompt: str, structure, config_hash: str, force_refresh: bool = False) -> bool:
  """
  Check if a prompt/structure/config combination is already cached.
  Respects POOR_MODE to search back in time for older cache entries.
  
  Returns True if a valid cached response exists.
  """
  if force_refresh:
    return False

  cache_date = datetime.datetime.now()

  while True:
    cache_file = get_cache_file_path(prompt, structure, config_hash, cache_date)

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


def write_to_cache(prompt: str, structure, config_hash: str, result) -> str:
  """
  Write a result to the cache. Returns the cache file path.
  """
  cache_file = get_cache_file_path(prompt, structure, config_hash)
  with open(cache_file, "w", encoding="utf-8") as f:
    json.dump(result, f)
  return cache_file


class CacheLayer:

  def __init__(self, configAndSettingsHash, aiEngineHook, engineName: str = "Unknown"):
    self.hash = configAndSettingsHash
    self.aiEngineHook = aiEngineHook
    self.engineName = engineName
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
    cache_file = _find_cache_file(prompt, structure, self.hash)

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

    print("API Call: " + prompt[:100].replace("\n", " ") + "...")

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
      write_to_cache(prompt, structure, self.hash, empty_result)
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

    write_to_cache(prompt, structure, self.hash, result)
    return result


def _find_cache_file(prompt: str, structure, config_hash: str) -> str:
  """Find the cache file, searching back in time if POOR_MODE is enabled."""
  cache_date = datetime.datetime.now()

  while True:
    cache_file = get_cache_file_path(prompt, structure, config_hash, cache_date)

    if not POOR_MODE:
      break

    if os.path.exists(cache_file):
      break

    cache_date -= datetime.timedelta(days=25)

    if cache_date < datetime.datetime(2025, 11, 30):
      # Reset to current date for writing
      cache_file = get_cache_file_path(prompt, structure, config_hash)
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
