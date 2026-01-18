import hashlib
from typing import Optional, Union, Callable, Tuple

# Placebo data provider - must be set by consuming benchmark
_placebo_data_provider: Optional[Callable[[int, int], Tuple[Optional[Union[dict, str]],
                                                            str]]] = None


def set_placebo_data_provider(
    provider: Callable[[int, int], Tuple[Optional[Union[dict, str]], str]]) -> None:
  """
  Set the placebo data provider function.
  
  This should be called by the consuming benchmark to provide pre-defined
  responses for the "Human with tools" baseline.
  
  Args:
      provider: A function that takes (question_num, subpass) and returns
                the expected response, or None if no response is available.
  """
  global _placebo_data_provider
  _placebo_data_provider = provider


class PlaceboEngine:
  """
  Placebo AI Engine class for testing with pre-defined responses.
  
  This engine returns pre-defined "human with tools" responses for benchmark tests.
  It doesn't make any API calls.
  
  Consuming benchmarks must call set_placebo_data_provider() to register their
  response data before using this engine.
  """

  def __init__(self):
    self.configAndSettingsHash = hashlib.sha256(b"Placebo").hexdigest()

  def AIHook(self, prompt: str, structure: Optional[dict], questionNum: int, subPass: int):
    """Dispatch to the appropriate question module for placebo responses."""
    if _placebo_data_provider is not None:
      result, reasoning = _placebo_data_provider(questionNum, subPass)
      if result is not None:
        return result, reasoning

      # No response found for this question/subpass
      return "", reasoning

    # No response found for this question/subpass
    return "", ""
